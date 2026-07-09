"""Run ChemEagle-Gemini fork (cloud, Gemini via OpenAI-compat shim) on every
benchmark image and score against our gold set.

Setup expected:
- ChemEagle-Gemini fork at ../ChemEagle-Gemini/ (sibling of ChemEagle/), with
  the original .venv-chemeagle/ symlinked over plus modified .py files that
  swap AzureOpenAI for OpenAI(base_url=Gemini OpenAI shim).
- Env vars (loaded from Single_SDK_agent/.env automatically if present):
    GEMINI_API_KEY (required)
    GEMINI_MODEL   (default: gemini-3-flash-preview)
    GEMINI_BASE_URL (default: https://generativelanguage.googleapis.com/v1beta/openai/)

Output: eval/results/chemeagle_gemini_<ts>/{summary.json, <stem>.json (native),
<stem>.converted.json (our schema), <stem>.log}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate, summarise  # noqa: E402
from eval.usage import add_usage, empty_usage, load_usage_json  # noqa: E402

DEFAULT_BENCH_IMAGES = REPO / "eval" / "benchmark" / "images"
DEFAULT_GOLD_DIR = REPO / "eval" / "ground_truth"
RESULTS_DIR = REPO / "eval" / "results"
CHEMEAGLE_DIR = REPO.parent / "ChemEagle-Gemini"
CHEMEAGLE_PY = CHEMEAGLE_DIR / ".venv-chemeagle" / "bin" / "python"

# Auto-load .env from Single_SDK_agent/ (where GEMINI_API_KEY lives) if present.
_DOTENV = REPO.parent / "Single_SDK_agent" / ".env"
if _DOTENV.exists():
    for _line in _DOTENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        # don't overwrite anything already in the environment
        if _k and _k not in os.environ:
            # strip inline shell-style comments (e.g. "value  # note")
            if "#" in _v:
                _v = _v.split("#", 1)[0].strip()
            os.environ[_k] = _v


# --- ChemEagle native output → our ReactionRecord schema ----------------------
ROLE_MAP = {
    "reagent": "reagent",
    "solvent": "solvent",
    "catalyst": "catalyst",
    "temperature": "temperature",
    "time": "time",
    "yield": "yield",
    "atmosphere": "atmosphere",
    "loading": "loading",
    "ee": "ee",
    "dr": "dr",
    "pressure": "pressure",
    # ChemEagle uses add_info for stereochem outcomes (ee, dr) — split where possible
    "add_info": None,  # handled by parse_string_condition
}

YIELD_RE = re.compile(r"(\d+(?:\.\d+)?\s*%(?:\s*yield)?)", re.I)
EE_RE = re.compile(r"(\d+(?:\.\d+)?\s*%\s*ee)", re.I)
DR_RE = re.compile(r"(\d+(?::\d+)+\s*dr)", re.I)


def _split_string_condition(text: str) -> list[dict]:
    """ChemEagle sometimes emits conditions as a single string like
    '71% yield, 14:1 dr, 91% ee'. Split into separate role-typed entries."""
    out: list[dict] = []
    rem = text
    for m in YIELD_RE.finditer(text):
        out.append({"role": "yield", "text": m.group(1).strip(), "smiles": None})
        rem = rem.replace(m.group(0), "")
    for m in EE_RE.finditer(text):
        out.append({"role": "ee", "text": m.group(1).strip(), "smiles": None})
        rem = rem.replace(m.group(0), "")
    for m in DR_RE.finditer(text):
        out.append({"role": "dr", "text": m.group(1).strip(), "smiles": None})
        rem = rem.replace(m.group(0), "")
    rem = rem.strip(" ,;")
    if rem:
        # Anything left we can't classify — drop into reagent as a catch-all.
        out.append({"role": "reagent", "text": rem, "smiles": None})
    return out


def _has_wildcard(s: str | None) -> bool:
    return bool(s) and ("*" in s or "[*:" in s)


def chemeagle_to_our_schema(ce: dict, file_name: str) -> dict:
    """Convert ChemEagle's output dict to our ReactionRecord shape.

    Drops the first reaction if it's the wildcard template (id often "0_1" with
    `*` in SMILES). Converts string-form conditions into typed Condition entries.
    """
    rxns_in = ce.get("reactions", []) or []
    rxns_out: list[dict] = []
    for i, r in enumerate(rxns_in):
        reactants_raw = r.get("reactants", []) or []
        products_raw = r.get("products", []) or []
        # Skip pure-template entries (wildcards in every reactant AND product).
        if reactants_raw and products_raw and all(_has_wildcard(c.get("smiles")) for c in reactants_raw + products_raw):
            continue

        def _compound(c: dict) -> dict:
            return {
                "smiles": c.get("smiles"),
                "label": c.get("label"),
                "uncertain": False,
            }

        conds_raw = r.get("conditions", []) or []
        conds_out: list[dict] = []
        additional: list[str] = []
        for c in conds_raw:
            if isinstance(c, str):
                conds_out.extend(_split_string_condition(c))
            elif isinstance(c, dict):
                role = c.get("role")
                text = c.get("text") or c.get("label") or ""
                smiles = c.get("smiles")
                if role == "add_info":
                    # Best-effort: try to bucket as ee/dr/yield via regex; else additional_info.
                    parts = _split_string_condition(text or "")
                    if parts and all(p["role"] in {"yield", "ee", "dr"} for p in parts):
                        conds_out.extend(parts)
                    else:
                        additional.append(text)
                elif role in ROLE_MAP and ROLE_MAP[role]:
                    conds_out.append({
                        "role": ROLE_MAP[role],
                        "text": str(text),
                        "smiles": smiles,
                    })
                else:
                    additional.append(f"{role}: {text}")

        rxn_id = r.get("reaction_id", f"0_{i+1}")
        if not re.match(r"^\d+_\d+$", str(rxn_id)):
            rxn_id = f"0_{i+1}"

        rxns_out.append({
            "reaction_id": rxn_id,
            "reactants": [_compound(c) for c in reactants_raw],
            "products": [_compound(c) for c in products_raw],
            "conditions": conds_out,
            "additional_info": additional,
        })

    return {"file_name": file_name, "reactions": rxns_out}


# --- Driver ------------------------------------------------------------------

INFER_SCRIPT = r'''
import sys, json, os
sys.path.insert(0, os.path.abspath("."))

# Patch openai's Completions.create BEFORE ChemEagle imports it, so every
# chat-completion call accumulates token usage into _ACCUM. Captures both
# sync and async clients (Azure subclasses these same Completions classes).
_ACCUM = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "model": None}
try:
    from openai.resources.chat.completions import Completions, AsyncCompletions
    _orig_sync = Completions.create
    _orig_async = AsyncCompletions.create

    def _record(resp, kwargs):
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return
            _ACCUM["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            _ACCUM["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            _ACCUM["calls"] += 1
            if _ACCUM["model"] is None:
                _ACCUM["model"] = kwargs.get("model")
        except Exception:
            pass

    def _wrapped_sync(self, *args, **kwargs):
        resp = _orig_sync(self, *args, **kwargs)
        _record(resp, kwargs)
        return resp

    async def _wrapped_async(self, *args, **kwargs):
        resp = await _orig_async(self, *args, **kwargs)
        _record(resp, kwargs)
        return resp

    Completions.create = _wrapped_sync
    AsyncCompletions.create = _wrapped_async
except Exception as _e:
    print(f"[usage-tracker] disabled: {type(_e).__name__}: {_e}")

from main import ChemEagle

img = sys.argv[1]
out_path = sys.argv[2]
usage_path = out_path + ".usage.json" if not out_path.endswith(".json") else out_path[:-5] + ".usage.json"
try:
    result = ChemEagle(img)
    with open(out_path, "w") as f:
        json.dump(result if isinstance(result, dict) else {"_raw": str(result)}, f, indent=2, ensure_ascii=False, default=str)
    print(f"OK {out_path}")
except Exception as e:
    import traceback
    err = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    with open(out_path, "w") as f:
        json.dump(err, f, indent=2)
    print(f"ERR {out_path}: {type(e).__name__}: {e}")
    raise
finally:
    try:
        with open(usage_path, "w") as f:
            json.dump({
                "model": _ACCUM["model"],
                "prompt_tokens": _ACCUM["prompt_tokens"],
                "completion_tokens": _ACCUM["completion_tokens"],
                "total_tokens": _ACCUM["prompt_tokens"] + _ACCUM["completion_tokens"],
                "calls": _ACCUM["calls"],
            }, f, indent=2)
    except Exception:
        pass
'''


def _list_images(images_dir: Path, manifest_path: Path | None = None) -> list[Path]:
    if not images_dir.exists():
        return []
    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if manifest_path is None:
        return all_images
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    allowed = {entry["stem"] for entry in manifest if entry.get("valid", True)}
    return [p for p in all_images if p.stem in allowed]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--per-image-timeout", type=int, default=1800,
                        help="Hard wall-clock timeout per image (default 1800 = 30 min).")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_BENCH_IMAGES,
                        help="Directory of benchmark images (default: eval/benchmark/images).")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR,
                        help="Directory of ground-truth JSONs (default: eval/ground_truth).")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional manifest JSON; only stems flagged valid are scored.")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model id (default: gemini-3-flash-preview). "
                             "Always overrides GEMINI_MODEL in the subprocess env.")
    args = parser.parse_args()
    args.images_dir = args.images_dir.resolve()
    args.gold_dir = args.gold_dir.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()

    if not CHEMEAGLE_PY.exists():
        print(f"ChemEagle venv not found at {CHEMEAGLE_PY}", file=sys.stderr)
        return 1

    # GEMINI_API_KEY must be set; fork's _gemini.py also accepts API_KEY.
    env = os.environ.copy()
    if not env.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set (looked in env and Single_SDK_agent/.env)",
              file=sys.stderr)
        return 1
    # Defaults for the OpenAI-compat shim — base URL is overridable via env,
    # model is forced from --model (so we can run different models without
    # editing .env files that other suites depend on).
    env.setdefault("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    env["GEMINI_MODEL"] = args.model
    print(f"Using Gemini model: {env['GEMINI_MODEL']}")
    print(f"Using Gemini base URL: {env['GEMINI_BASE_URL']}")

    images = _list_images(args.images_dir, args.manifest)
    if args.start_from > 0:
        images = images[args.start_from:]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"No benchmark images found under {args.images_dir}", file=sys.stderr)
        return 1

    suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = RESULTS_DIR / f"chemeagle_gemini_{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Write the inference helper script into the ChemEagle dir so its imports resolve.
    helper = CHEMEAGLE_DIR / "_run_one_image.py"
    helper.write_text(INFER_SCRIPT)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    print(f"=== ChemEagle-Gemini suite {suite_id} ===")
    print(f"  {len(images)} images, timeout={args.per_image_timeout}s")
    print(f"  images_dir = {_rel(args.images_dir)}")
    print(f"  gold_dir   = {_rel(args.gold_dir)}")
    print(f"  output dir: {suite_dir.relative_to(REPO)}")
    print()

    per_image: list[dict] = []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name}")
        log_path = suite_dir / f"{img.stem}.log"
        native_path = suite_dir / f"{img.stem}.json"
        cmd = [str(CHEMEAGLE_PY), str(helper), str(img), str(native_path)]
        t0 = time.time()
        timed_out = False
        try:
            with log_path.open("w") as f:
                proc = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(CHEMEAGLE_DIR),
                    timeout=args.per_image_timeout, env=env,
                )
            rc = proc.returncode
            err = None
        except subprocess.TimeoutExpired:
            rc = -2
            timed_out = True
            err = f"timeout after {args.per_image_timeout}s"
            try:
                subprocess.run(["pkill", "-KILL", "-f", f"_run_one_image.py {img}"], capture_output=True)
            except Exception:
                pass
        except Exception as exc:
            rc = -1
            err = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0
        flag = "TIMEOUT" if timed_out else f"rc={rc}"
        print(f"    {flag}  {elapsed:5.0f}s")
        usage = load_usage_json(suite_dir / f"{img.stem}.usage.json")
        per_image.append({
            "image": img.name,
            "stem": img.stem,
            "rc": rc,
            "elapsed_s": round(elapsed, 1),
            "native_json": str(native_path.relative_to(REPO)) if native_path.exists() else None,
            "log": str(log_path.relative_to(REPO)),
            **({"error": err} if err else {}),
            **({"usage": usage} if usage else {}),
        })

    print("\n=== converting + scoring ===")
    gold_dir = args.gold_dir
    scores: list[dict] = []
    for entry in per_image:
        if not entry.get("native_json"):
            scores.append({"image": entry["image"], "metrics": None, "error": entry.get("error", "no output")})
            continue
        native = json.loads((REPO / entry["native_json"]).read_text())
        if "error" in native:
            scores.append({"image": entry["image"], "metrics": None, "error": native["error"]})
            continue
        converted = chemeagle_to_our_schema(native, entry["image"])
        conv_path = suite_dir / f"{entry['stem']}.converted.json"
        conv_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
        gold_path = gold_dir / f"{entry['stem']}.json"
        gold = None
        if gold_path.exists():
            try:
                gold = json.loads(gold_path.read_text())
            except Exception:
                gold = None
        metrics = evaluate(converted, gold)
        scores.append({"image": entry["image"], "metrics": metrics, "has_gold": gold is not None})
        print(f"\n[{entry['image']}]  ({entry['elapsed_s']:.0f}s)")
        print(summarise(metrics))

    def _safe(d, *path, default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        return cur

    rows = []
    for s in scores:
        m = s.get("metrics")
        rows.append({
            "image": s["image"],
            "schema_pass": bool(_safe(m, "schema_conformance", "valid", default=False)),
            "smiles_valid_rate": _safe(m, "smiles_validity_rate", "rate", default=0.0),
            "role_enum_rate": _safe(m, "role_enum_compliance", "rate", default=0.0),
            "reaction_count": _safe(m, "reaction_count", "count", default=0),
            "reaction_count_gold": _safe(m, "reaction_count_gold", "count", default=None),
            "reactant_iou": _safe(m, "reactant_iou", "iou", default=None),
            "product_iou": _safe(m, "product_iou", "iou", default=None),
            "condition_recall": _safe(m, "condition_coverage", "recall", default=None),
            "condition_precision": _safe(m, "condition_coverage", "precision", default=None),
            "soft_f1": _safe(m, "soft_match", "f1", default=None),
            "hard_f1": _safe(m, "hard_match", "f1", default=None),
            "constitution_f1": _safe(m, "constitution_match", "f1", default=None),
            "partial_f1": _safe(m, "partial_match", "f1", default=None),
            "partial_mean_jaccard": _safe(m, "partial_match", "mean_jaccard", default=None),
            "ged": _safe(m, "graph_edit_distance", "avg_ged", default=None),
        })

    def _mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else None

    means = {
        "schema_pass_rate": sum(1 for r in rows if r["schema_pass"]) / max(1, len(rows)),
        "mean_smiles_valid_rate": _mean([r["smiles_valid_rate"] for r in rows]),
        "mean_role_enum_rate": _mean([r["role_enum_rate"] for r in rows]),
        "mean_reactant_iou": _mean([r["reactant_iou"] for r in rows]),
        "mean_product_iou": _mean([r["product_iou"] for r in rows]),
        "mean_condition_recall": _mean([r["condition_recall"] for r in rows]),
        "mean_condition_precision": _mean([r["condition_precision"] for r in rows]),
        "mean_soft_f1": _mean([r["soft_f1"] for r in rows]),
        "mean_hard_f1": _mean([r["hard_f1"] for r in rows]),
        "mean_constitution_f1": _mean([r["constitution_f1"] for r in rows]),
        "mean_partial_f1": _mean([r["partial_f1"] for r in rows]),
        "mean_partial_jaccard": _mean([r["partial_mean_jaccard"] for r in rows]),
        "mean_ged": _mean([r["ged"] for r in rows]),
    }

    usage_total = empty_usage()
    for e in per_image:
        if e.get("usage"):
            usage_total = add_usage(usage_total, e["usage"])
    elapsed_total_s = round(sum(float(e.get("elapsed_s") or 0) for e in per_image), 1)

    summary = {
        "suite_id": suite_id,
        "kind": "chemeagle_gemini",
        "model": env.get("GEMINI_MODEL"),
        "base_url": env.get("GEMINI_BASE_URL"),
        "images_dir": _rel(args.images_dir),
        "gold_dir": _rel(args.gold_dir),
        "manifest": str(args.manifest.relative_to(REPO)) if args.manifest else None,
        "n_images": len(images),
        "per_image_timeout_s": args.per_image_timeout,
        "per_image": per_image,
        "scores": scores,
        "table": rows,
        "means": means,
        "usage_total": usage_total,
        "elapsed_total_s": elapsed_total_s,
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== aggregate (mean over scored runs) ===")
    for k, v in means.items():
        print(f"  {k:30s}: {v if v is not None else '--'}")
    print(f"\nsummary written to {suite_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
