"""Worker that runs the ChemEagle_Hybrid pipeline (molnextr + rxnim evidence
+ Azure-OpenAI orchestrate) over a JSON list of images, sharing one
ChemIEToolkit instance.

Invoked by eval/run_chemeagle_hybrid_suite.py inside .venv-chemeagle.

Inputs (argv):
    1. tasks_json: path to a JSON file containing
       {"deployment": str, "images": [{"image": str, "out": str}, ...]}.
       Each image is processed in order; per-image errors are caught and
       written into the output JSON so the run keeps going.

Required env:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_VERSION   (default 2024-12-01-preview if unset)

Output: writes one JSON file per image (Maarten's hybrid native shape, or
{"error": "..."} on failure).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# --- bootstrap path so chemietoolkit/molnextr/rxnim + Maarten's modules import ---
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RANDOM_TESTS = REPO.parent
CHEMEAGLE_DIR = RANDOM_TESTS / "ChemEagle"
HYBRID_DIR = RANDOM_TESTS / "Maarten_ChemEagle_Gemini" / "ChemEagle_Hybrid"

# Order matters: Maarten's `utils/` must shadow any other utils package.
sys.path.insert(0, str(HYBRID_DIR))
sys.path.insert(0, str(CHEMEAGLE_DIR))


def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _build_azure_client():
    from openai import AzureOpenAI
    endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip().strip('"').rstrip("/")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)


_USAGE_ACCUM: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def _reset_usage() -> None:
    _USAGE_ACCUM["prompt_tokens"] = 0
    _USAGE_ACCUM["completion_tokens"] = 0
    _USAGE_ACCUM["calls"] = 0


def _snapshot_usage() -> dict:
    return dict(_USAGE_ACCUM)


def _azure_chat_json(client, deployment: str, messages: list) -> str:
    """Call Azure chat-completions, returning the message content string.

    gpt-5 reasoning deployments reject `temperature` and use
    `max_completion_tokens` instead of `max_tokens`. We try the new shape
    first; if the deployment is an older chat model that only accepts
    `max_tokens`, we retry with the old shape.
    """
    common = dict(
        model=deployment,
        messages=messages,
        response_format={"type": "json_object"},
    )
    try:
        resp = client.chat.completions.create(max_completion_tokens=8192, **common)
    except TypeError:
        resp = client.chat.completions.create(max_tokens=8192, **common)
    except Exception as e:
        msg = str(e).lower()
        # Some Azure API versions reject one of the param names with a 400.
        if "max_completion_tokens" in msg or "unsupported parameter" in msg:
            resp = client.chat.completions.create(max_tokens=8192, **common)
        else:
            raise
    content = resp.choices[0].message.content or ""
    try:
        finish = resp.choices[0].finish_reason
        usage = getattr(resp, "usage", None)
        ut = getattr(usage, "total_tokens", None) if usage else None
        up = getattr(usage, "prompt_tokens", None) if usage else None
        uc = getattr(usage, "completion_tokens", None) if usage else None
        if up:
            _USAGE_ACCUM["prompt_tokens"] += int(up)
        if uc:
            _USAGE_ACCUM["completion_tokens"] += int(uc)
        _USAGE_ACCUM["calls"] += 1
        print(
            f"    [llm] finish={finish} content_len={len(content)} "
            f"prompt={up} completion={uc} total={ut}",
            flush=True,
        )
        if not content:
            # Reasoning models can place output in non-standard fields.
            msg_obj = resp.choices[0].message
            extras = {k: getattr(msg_obj, k, None) for k in
                      ("refusal", "reasoning_content", "tool_calls")}
            print(f"    [llm] empty content; message extras: {extras}", flush=True)
    except Exception:
        pass
    return content


def _orchestrate_azure(client, deployment, image_url, evidence, system_prompt):
    from gemini_extractor import _safe_json_parse  # noqa: E402
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    if len(evidence_json) > 16_000:
        evidence_json = evidence_json[:16_000] + "\n... (evidence truncated)"
    user_text = (
        "Evidence packet (use the SMILES verbatim; trust the rxnim partition "
        "unless you have clear reason to override):\n\n"
        f"{evidence_json}\n\n"
        "Produce the structured JSON for this figure."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
    ]
    # gpt-5-mini occasionally returns empty content on first call; one retry.
    for attempt in range(2):
        content = _azure_chat_json(client, deployment, messages)
        if content.strip():
            return _safe_json_parse(content)
        print(f"    [llm] empty content on attempt {attempt+1}, retrying...", flush=True)
    return _safe_json_parse(content)


def _stage2_azure(client, deployment, image_url, stage1, evidence, variant_prompt):
    from gemini_extractor import _safe_json_parse  # noqa: E402
    draft_json = json.dumps(stage1, ensure_ascii=False, indent=2)
    if len(draft_json) > 4000:
        draft_json = draft_json[:4000] + "\n... (truncated)"
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    if len(evidence_json) > 8_000:
        evidence_json = evidence_json[:8_000] + "\n... (truncated)"
    user_text = (
        "Stage-1 draft (use as a hint about the template; re-derive per-row "
        "substitutions from the evidence + image):\n\n"
        f"{draft_json}\n\n"
        "Evidence packet (use the SMILES verbatim):\n\n"
        f"{evidence_json}\n\n"
        "Re-extract the variant table with one reaction per row, both reactants "
        "AND products substituted."
    )
    messages = [
        {"role": "system", "content": variant_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
    ]
    for attempt in range(2):
        content = _azure_chat_json(client, deployment, messages)
        if content.strip():
            return _safe_json_parse(content)
        print(f"    [llm] empty content on stage2 attempt {attempt+1}, retrying...", flush=True)
    return _safe_json_parse(content)


def main(tasks_path: str) -> int:
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    deployment = tasks["deployment"]
    images = tasks["images"]

    # Imports deferred until after sys.path setup.
    from PIL import Image  # noqa: E402
    import torch  # noqa: E402
    from chemietoolkit import ChemIEToolkit  # noqa: E402
    from gemini_hybrid import _gather_evidence  # noqa: E402
    from gemini_extractor import (  # noqa: E402
        _load_image_as_data_url, _postprocess, _should_run_stage2,
    )

    hybrid_prompt = _load_text(HYBRID_DIR / "prompt" / "prompt_hybrid_orchestrate.txt")
    variant_prompt = _load_text(HYBRID_DIR / "prompt" / "prompt_gemini_variant_expand.txt")

    print(f"[init] device selection...", flush=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[init] device={device}", flush=True)

    print(f"[init] instantiating ChemIEToolkit (loads molnextr/rxnim/coref weights)...", flush=True)
    t_init = time.perf_counter()
    toolkit = ChemIEToolkit(device=device)
    # Touch lazy-loaded properties so the cost is paid up front.
    _ = toolkit.molnextr; _ = toolkit.rxnim; _ = toolkit.coref
    print(f"[init] toolkit ready in {time.perf_counter() - t_init:.1f}s", flush=True)

    azure_client = _build_azure_client()
    print(f"[init] Azure client built; deployment={deployment}", flush=True)

    n_total = len(images)
    n_ok = 0
    n_fail = 0
    t0 = time.perf_counter()

    for i, item in enumerate(images, start=1):
        img_path = item["image"]
        out_path = item["out"]
        tag = Path(img_path).name
        tic = time.perf_counter()
        _reset_usage()
        print(f"\n[{i:>2}/{n_total}] {tag}", flush=True)
        try:
            image_pil = Image.open(img_path).convert("RGB")
            image_url = _load_image_as_data_url(img_path)

            # Stage 0: deep-learning evidence
            try:
                evidence = _gather_evidence(image_pil, toolkit)
            except Exception as e:
                err = f"evidence_error: {type(e).__name__}: {e}"
                print(f"    [FAIL] {err}", flush=True)
                Path(out_path).write_text(json.dumps(
                    {"reactions": [], "_evidence_error": err}, indent=2,
                ))
                n_fail += 1
                continue

            n_mols = len(evidence.get("molecules") or [])
            n_rxnim = len(evidence.get("rxnim_reactions") or [])
            print(f"    [evidence] mols={n_mols} rxnim_reactions={n_rxnim}", flush=True)

            # Stage 1: orchestrate
            try:
                result = _orchestrate_azure(
                    azure_client, deployment, image_url, evidence, hybrid_prompt,
                )
            except Exception as e:
                err = f"orchestrate_error: {type(e).__name__}: {e}"
                traceback.print_exc()
                Path(out_path).write_text(json.dumps(
                    {"reactions": [], "_orchestrate_error": err, "_evidence": evidence},
                    indent=2,
                ))
                n_fail += 1
                continue

            if not isinstance(result, dict):
                Path(out_path).write_text(json.dumps(
                    {"reactions": [], "_orchestrate_error": "non-dict response"},
                    indent=2,
                ))
                n_fail += 1
                continue
            result.setdefault("reactions", [])

            # Stage 2: conditional variant expansion
            if _should_run_stage2(result):
                print(f"    [stage2] variant-table re-prompt", flush=True)
                try:
                    stage2 = _stage2_azure(
                        azure_client, deployment, image_url, result, evidence,
                        variant_prompt,
                    )
                    if isinstance(stage2, dict) and stage2.get("reactions"):
                        stage2["_stage2_applied"] = True
                        stage2.setdefault("image_kind",
                                          result.get("image_kind", "variant_table"))
                        result = stage2
                except Exception as e:
                    result["_stage2_error"] = f"{type(e).__name__}: {e}"

            # Post-process: wildcard renumber + (best-effort) smiles_fix + optional validator
            result = _postprocess(result, azure_client, deployment)

            Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
            n_rxns = len(result.get("reactions") or [])
            s2 = " +s2" if result.get("_stage2_applied") else ""
            print(
                f"    [OK]{s2} kind={result.get('image_kind','?')} "
                f"rxns={n_rxns}  {time.perf_counter() - tic:.1f}s",
                flush=True,
            )
            n_ok += 1
        except Exception as e:
            traceback.print_exc()
            try:
                Path(out_path).write_text(json.dumps(
                    {"error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()}, indent=2,
                ))
            except Exception:
                pass
            n_fail += 1
        finally:
            try:
                u = _snapshot_usage()
                usage_path = Path(out_path).with_suffix(".usage.json")
                usage_path.write_text(json.dumps({
                    "model": deployment,
                    "prompt_tokens": u["prompt_tokens"],
                    "completion_tokens": u["completion_tokens"],
                    "total_tokens": u["prompt_tokens"] + u["completion_tokens"],
                    "calls": u["calls"],
                }, indent=2))
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    print(
        f"\n[done] ok={n_ok} fail={n_fail} total={n_total} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: _run_hybrid_one.py <tasks.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
