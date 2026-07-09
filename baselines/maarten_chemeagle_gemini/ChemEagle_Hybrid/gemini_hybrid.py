"""Hybrid extraction pipeline: molnextr + rxnim for SMILES recognition,
Gemini for orchestration (role assignment, variant-table expansion, schema).

Why hybrid: pure-Gemini (gemini_extractor.ChemEagle_Gemini) regressed Avg
Tanimoto from 0.88 to 0.79 on the GT3 article benchmark — Gemini Vision is a
generalist that makes systematic SMILES transcription errors that the molnextr
specialist doesn't make. The original failure modes (variant-table reactant
collapse, image-9 crash, role mis-assignment) were agent-level, not
recognition-level. This module keeps molnextr+rxnim and replaces only the
LLM agents with one Gemini orchestrate call per image (plus an optional
stage-2 for variant-table expansion).

Public entry points:

    create_hybrid_cache(client, *, model, ttl_seconds=3600) -> str | None
    ChemEagle_Hybrid(image_path, *, toolkit, cache_name=None,
                     variant_cache_name=None, model="gemini-3-flash-preview",
                     api_key=None, base_url=None) -> dict

`toolkit` is a ChemIEToolkit instance owned by the caller (lazy-loaded models
take ~30s on first access; share one instance across all images).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from utils.llm_client import GeminiOpenAIShim, get_client

# Reuse helpers from the pure-Gemini module — same JSON parsing, post-processing,
# variant-trigger heuristics. Underscore-prefixed but Python doesn't enforce it.
from gemini_extractor import (
    _load_image_as_data_url,
    _postprocess,
    _safe_json_parse,
    _should_run_stage2,
)

_REPO_ROOT = Path(__file__).resolve().parent
_PROMPT_HYBRID = _REPO_ROOT / "prompt" / "prompt_hybrid_orchestrate.txt"
_PROMPT_VARIANT = _REPO_ROOT / "prompt" / "prompt_gemini_variant_expand.txt"


def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# --- public: cache lifecycle ------------------------------------------------


def create_hybrid_cache(
    client: GeminiOpenAIShim,
    *,
    model: str = "gemini-3-flash-preview",
    ttl_seconds: int = 3600,
) -> Optional[str]:
    """Cache the hybrid orchestrate system prompt. Returns name or None."""
    return client.create_cache(
        model=model,
        system_instruction=_load_text(_PROMPT_HYBRID),
        ttl_seconds=ttl_seconds,
        display_name="chemeagle_hybrid_v1",
    )


# --- public: per-image extraction -------------------------------------------


def ChemEagle_Hybrid(
    image_path: str,
    *,
    toolkit,                              # ChemIEToolkit instance
    cache_name: Optional[str] = None,
    variant_cache_name: Optional[str] = None,
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """Extract reactions from one chemical figure via molnextr + Gemini.

    Pipeline:
      0. Run tk.coref.predict_images and tk.rxnim.predict_images (GPU, no API).
      1. Build evidence packet: molecules with labels + SMILES, rxnim
         reactant/product/condition split.
      2. ONE Gemini orchestrate call: image + evidence → structured JSON.
      3. Conditional stage 2: if image_kind=variant_table and the heuristic
         indicates collapse, re-prompt for per-row expansion.
      4. Post-process: utils.smiles_fix.sanitize_reactions, optional
         utils.rxn_validator.validate_and_fix on RDKit-parse failures.

    Returns the structured dict; never raises. On errors: returns
    `{"reactions": [], "_<step>_error": "..."}`.
    """
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    os.environ.setdefault("CHEMEAGLE_BACKEND", "gemini")

    # Caller is responsible for image_context (per-image trace partitioning).
    client = get_client(base_url=base_url, api_key=api_key)

    image_pil = Image.open(image_path).convert("RGB")
    image_url = _load_image_as_data_url(image_path)

    # Stage 0: deep-learning evidence (GPU, no API)
    try:
        evidence = _gather_evidence(image_pil, toolkit)
    except Exception as e:
        return {"reactions": [], "_evidence_error": f"{type(e).__name__}: {e}"}

    # Stage 1: orchestrate
    system_prompt = _load_text(_PROMPT_HYBRID) if not cache_name else None
    try:
        result = _orchestrate(client, model, image_url, evidence, cache_name, system_prompt)
    except Exception as e:
        return {
            "reactions": [], "_orchestrate_error": f"{type(e).__name__}: {e}",
            "_evidence": evidence,
        }
    if not isinstance(result, dict):
        return {"reactions": [], "_orchestrate_error": "non-dict response"}
    result.setdefault("reactions", [])

    # Stage 2 (conditional, variant tables)
    if _should_run_stage2(result):
        variant_prompt = _load_text(_PROMPT_VARIANT) if not variant_cache_name else None
        try:
            stage2 = _stage2_variant_expand_hybrid(
                client, model, image_url, result, evidence,
                variant_cache_name, variant_prompt,
            )
            if isinstance(stage2, dict) and stage2.get("reactions"):
                stage2["_stage2_applied"] = True
                stage2.setdefault("image_kind", result.get("image_kind", "variant_table"))
                result = stage2
        except Exception as e:
            result["_stage2_error"] = f"{type(e).__name__}: {e}"

    # Post-process (RDKit canonicalize + optional validator)
    result = _postprocess(result, client, model)
    return result


# --- evidence gathering -----------------------------------------------------


def _gather_evidence(image_pil: Image.Image, toolkit) -> dict:
    """Run tk.coref + tk.rxnim. Build a structured evidence packet for Gemini.

    Output shape:
        {
            "molecules": [{"label": "1", "smiles": "...", "bbox": [x1,y1,x2,y2]}, ...],
            "rxnim_reactions": [
                {"reactants": [{"smiles": "...", "bbox": [...]}, ...],
                 "products":  [...],
                 "conditions":[{"text": "...", "bbox": [...]}, ...]},
                ...
            ]
        }
    """
    # tk.coref returns: {"bboxes": [...], "corefs": [[mol_idx, label_idx], ...]}
    try:
        coref_out = toolkit.coref.predict_images(
            [image_pil], molnextr=True, coref=True, ocr=True,
        )
        coref = coref_out[0] if coref_out else {"bboxes": [], "corefs": []}
    except Exception:
        coref = {"bboxes": [], "corefs": []}

    # tk.rxnim returns: per-image list of reaction dicts
    try:
        rxnim_out = toolkit.rxnim.predict_images(
            [image_pil], molnextr=True, ocr=True,
        )
        rxnim = rxnim_out[0] if rxnim_out else []
        if not isinstance(rxnim, list):
            rxnim = [rxnim] if rxnim else []
    except Exception:
        rxnim = []

    bboxes = coref.get("bboxes") or []
    corefs = coref.get("corefs") or []

    # Build molecule-index -> label-text map from coref pairs.
    label_for_mol: dict[int, str] = {}
    for pair in corefs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        mol_i, lbl_i = pair
        if 0 <= mol_i < len(bboxes) and 0 <= lbl_i < len(bboxes):
            label = bboxes[lbl_i].get("text") or bboxes[lbl_i].get("identifier") or ""
            if isinstance(label, list):
                label = " ".join(str(t) for t in label if t)
            label_for_mol[mol_i] = str(label).strip()

    molecules: list[dict] = []
    for i, b in enumerate(bboxes):
        if b.get("category") != "[Mol]":
            continue
        smi = b.get("smiles")
        if not smi:
            continue
        molecules.append({
            "label": label_for_mol.get(i, ""),
            "smiles": smi,
            "bbox": _round_bbox(b.get("bbox")),
        })

    rxnim_reactions: list[dict] = []
    for r in rxnim:
        if not isinstance(r, dict):
            continue
        entry = {"reactants": [], "products": [], "conditions": []}
        for side in ("reactants", "products"):
            for item in r.get(side, []) or []:
                if not isinstance(item, dict):
                    continue
                entry[side].append({
                    "smiles": item.get("smiles", "") or "",
                    "bbox": _round_bbox(item.get("bbox")),
                })
        for item in r.get("conditions", []) or []:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text if t)
            entry["conditions"].append({
                "text": str(text or "").strip(),
                "bbox": _round_bbox(item.get("bbox")),
            })
        rxnim_reactions.append(entry)

    return {
        "molecules": molecules,
        "rxnim_reactions": rxnim_reactions,
    }


def _round_bbox(b) -> list:
    if not isinstance(b, (list, tuple)) or len(b) < 4:
        return []
    try:
        return [round(float(x), 3) for x in b[:4]]
    except Exception:
        return []


# --- orchestrate (stage 1) --------------------------------------------------


def _orchestrate(
    client,
    model: str,
    image_url: str,
    evidence: dict,
    cache_name: Optional[str],
    system_prompt: Optional[str],
) -> dict:
    extra_body: dict[str, Any] = {
        "thinking_budget": int(os.environ.get("CHEMEAGLE_GEMINI_THINKING_HYBRID", "0"))
    }
    messages: list[dict] = []
    if cache_name:
        extra_body["cached_content"] = cache_name
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    # Cap evidence size — pathological figures with hundreds of bboxes shouldn't
    # blow the cache+image input budget. 16K chars is plenty for ~50 molecules.
    if len(evidence_json) > 16_000:
        evidence_json = evidence_json[:16_000] + "\n... (evidence truncated)"
    user_text = (
        "Evidence packet (use the SMILES verbatim; trust the rxnim partition "
        "unless you have clear reason to override):\n\n"
        f"{evidence_json}\n\n"
        "Produce the structured JSON for this figure."
    )
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    })
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=int(os.environ.get("CHEMEAGLE_MAX_OUTPUT_TOKENS", "8192")),
        temperature=0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    return _safe_json_parse(content)


# --- stage 2 (variant-table expansion) --------------------------------------


def _stage2_variant_expand_hybrid(
    client,
    model: str,
    image_url: str,
    stage1: dict,
    evidence: dict,
    variant_cache_name: Optional[str],
    variant_prompt: Optional[str],
) -> dict:
    extra_body: dict[str, Any] = {
        "thinking_budget": int(os.environ.get("CHEMEAGLE_GEMINI_THINKING_STAGE2", "2048"))
    }
    messages: list[dict] = []
    if variant_cache_name:
        extra_body["cached_content"] = variant_cache_name
    elif variant_prompt:
        messages.append({"role": "system", "content": variant_prompt})

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
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    })
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=int(os.environ.get("CHEMEAGLE_MAX_OUTPUT_TOKENS", "8192")),
        temperature=0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    return _safe_json_parse(content)
