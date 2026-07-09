"""Gemini-only chemical extraction pipeline (replacement for ChemEAGLE_OS).

Two public entry points:

    create_extraction_cache(client, model, ttl_seconds=3600) -> str | None
    create_variant_cache(client, model, ttl_seconds=3600)   -> str | None
    ChemEagle_Gemini(image_path, *, cache_name=None, variant_cache_name=None,
                     model="gemini-3-flash-preview", api_key=None, base_url=None) -> dict

Output dict shape (compatible with scripts/benchmark_eval.py):

    {
        "image_kind": "single_reaction" | "variant_table" | ...,
        "reactions": [
            {
                "reaction_id": "...",
                "reactants":  [{"smiles": "...", "label": "..."}, ...],
                "products":   [{"smiles": "...", "label": "..."}, ...],
                "conditions": [{"role": "...", "text": "...", "smiles": "..." | null}, ...],
                "additional_info": ["..."]
            }, ...
        ],
        "_stage2_applied": bool (debug; eval ignores)
    }
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from utils.llm_client import (
    GeminiOpenAIShim,
    get_client,
)

_REPO_ROOT = Path(__file__).resolve().parent
_PROMPT_EXTRACT = _REPO_ROOT / "prompt" / "prompt_gemini_extract.txt"
_PROMPT_VARIANT = _REPO_ROOT / "prompt" / "prompt_gemini_variant_expand.txt"


# --- public: cache lifecycle ------------------------------------------------


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def create_extraction_cache(
    client: GeminiOpenAIShim,
    *,
    model: str = "gemini-3-flash-preview",
    ttl_seconds: int = 3600,
) -> Optional[str]:
    """Cache the stage-1 system prompt. Returns cache name or None."""
    return client.create_cache(
        model=model,
        system_instruction=_load_text(_PROMPT_EXTRACT),
        ttl_seconds=ttl_seconds,
        display_name="chemeagle_gemini_extract_v2",
    )


def create_variant_cache(
    client: GeminiOpenAIShim,
    *,
    model: str = "gemini-3-flash-preview",
    ttl_seconds: int = 3600,
) -> Optional[str]:
    """Cache the stage-2 variant-expand system prompt. Returns name or None."""
    return client.create_cache(
        model=model,
        system_instruction=_load_text(_PROMPT_VARIANT),
        ttl_seconds=ttl_seconds,
        display_name="chemeagle_gemini_variant_v2",
    )


# --- public: per-image extraction -------------------------------------------


def ChemEagle_Gemini(
    image_path: str,
    *,
    cache_name: Optional[str] = None,
    variant_cache_name: Optional[str] = None,
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """Extract reactions from one chemical figure with Gemini Vision.

    Pipeline:
      1. Stage 1 — full-image vision call → JSON draft.
      2. Stage 2 — variant-table expansion (only when stage 1 marks
         image_kind=variant_table AND the reactant set looks collapsed
         or contains a known solvent-as-reactant).
      3. utils.smiles_fix.sanitize_reactions — RDKit canonicalize, chiral H repair,
         disconnected R-group reattachment.
      4. utils.rxn_validator.validate_and_fix — only when CHEMEAGLE_VALIDATE=1
         (default off) OR CHEMEAGLE_VALIDATE_ON_FAIL_ONLY=1 (default on) AND
         at least one SMILES is RDKit-unparseable after step 3.

    On any LLM failure: returns {"reactions": [], "_stage1_error": "..."} —
    never raises into the caller.
    """
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    os.environ.setdefault("CHEMEAGLE_BACKEND", "gemini")

    # NOTE: image context (set_image_context/clear_image_context) is the CALLER's
    # responsibility. Runners set it before calling so per-image token accounting
    # works; ad-hoc single-image users can leave it unset (traces then go into
    # `_no_image_context.jsonl`).
    client = get_client(base_url=base_url, api_key=api_key)
    image_url = _load_image_as_data_url(image_path)

    system_prompt = _load_text(_PROMPT_EXTRACT) if not cache_name else None
    variant_prompt = _load_text(_PROMPT_VARIANT) if not variant_cache_name else None

    # Stage 1
    try:
        result = _stage1(client, model, image_url, cache_name, system_prompt)
    except Exception as e:
        return {"reactions": [], "_stage1_error": f"{type(e).__name__}: {e}"}

    if not isinstance(result, dict):
        return {"reactions": [], "_stage1_error": "non-dict response"}
    result.setdefault("reactions", [])

    # Stage 2 (conditional)
    if _should_run_stage2(result):
        try:
            stage2 = _stage2_variant_expand(
                client, model, image_url, result,
                variant_cache_name, variant_prompt,
            )
            if isinstance(stage2, dict) and stage2.get("reactions"):
                stage2["_stage2_applied"] = True
                # Preserve top-level image_kind classification from stage 1
                stage2.setdefault("image_kind", result.get("image_kind", "variant_table"))
                result = stage2
        except Exception as e:
            result["_stage2_error"] = f"{type(e).__name__}: {e}"

    # Post-process
    result = _postprocess(result, client, model)
    return result


# --- stages ----------------------------------------------------------------


def _stage1(
    client,
    model: str,
    image_url: str,
    cache_name: Optional[str],
    system_prompt: Optional[str],
) -> dict:
    extra_body: dict[str, Any] = {
        "thinking_budget": int(
            os.environ.get("CHEMEAGLE_GEMINI_THINKING_STAGE1", "0")
        )
    }
    messages: list[dict] = []
    if cache_name:
        extra_body["cached_content"] = cache_name
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract reactions from this image as JSON."},
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


def _stage2_variant_expand(
    client,
    model: str,
    image_url: str,
    stage1: dict,
    variant_cache_name: Optional[str],
    variant_prompt: Optional[str],
) -> dict:
    extra_body: dict[str, Any] = {
        "thinking_budget": int(
            os.environ.get("CHEMEAGLE_GEMINI_THINKING_STAGE2", "2048")
        )
    }
    messages: list[dict] = []
    if variant_cache_name:
        extra_body["cached_content"] = variant_cache_name
    elif variant_prompt:
        messages.append({"role": "system", "content": variant_prompt})

    draft = json.dumps(stage1, ensure_ascii=False, indent=2)
    if len(draft) > 4000:
        draft = draft[:4000] + "\n... (truncated)"
    user_text = (
        "Stage-1 draft below — use ONLY as a hint about the template; re-derive "
        "the per-row substitutions directly from the image. Output one reaction "
        "per row of the variant table, with both reactants AND products substituted.\n\n"
        f"{draft}"
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


# --- stage-2 trigger heuristics --------------------------------------------


# Canonical SMILES (RDKit isomeric) of common bench solvents/conditions that
# stage 1 sometimes mis-classifies as a reactant.
_KNOWN_CONDITION_SMILES_CANON = {
    "CN(C)C=O",     # DMF
    "CS(C)=O",      # DMSO
    "C1CCOC1",      # THF
    "O",            # water
    "CO",           # MeOH
    "CCO",          # EtOH
    "ClCCl",        # DCM
    "CC#N",         # MeCN
    "C1COCCO1",     # 1,4-dioxane (RDKit canonical form)
    "CC(C)=O",      # acetone
}

_RGROUP_PLACEHOLDER_RE = re.compile(r"\[\d?\*\]|\[Ar\]|\[R\d*\]")


def _canonicalize_smiles(smi: str) -> Optional[str]:
    try:
        from rdkit import Chem  # lazy: rdkit import is slow on cold start

        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return None


def _has_condition_as_reactant(rxn: dict) -> bool:
    for r in rxn.get("reactants") or []:
        smi = (r.get("smiles") or "").strip()
        if not smi:
            continue
        canon = _canonicalize_smiles(smi)
        if canon and canon in _KNOWN_CONDITION_SMILES_CANON:
            return True
    return False


def _has_unresolved_rgroup(rxn: dict) -> bool:
    for side in ("reactants", "products"):
        for m in rxn.get(side) or []:
            smi = m.get("smiles") or ""
            if _RGROUP_PLACEHOLDER_RE.search(smi):
                return True
    return False


def _all_reactants_identical(rxns: list) -> bool:
    if len(rxns) < 2:
        return False
    first = tuple(sorted((r.get("smiles") or "") for r in (rxns[0].get("reactants") or [])))
    for rxn in rxns[1:]:
        cur = tuple(sorted((r.get("smiles") or "") for r in (rxn.get("reactants") or [])))
        if cur != first:
            return False
    return True


def _products_differ(rxns: list) -> bool:
    if len(rxns) < 2:
        return False
    seen = set()
    for rxn in rxns:
        prods = tuple(sorted((p.get("smiles") or "") for p in (rxn.get("products") or [])))
        seen.add(prods)
    return len(seen) > 1


def _should_run_stage2(stage1: dict) -> bool:
    if (stage1 or {}).get("image_kind") != "variant_table":
        return False
    rxns = stage1.get("reactions") or []
    if len(rxns) < 2:
        return False
    if any(_has_condition_as_reactant(r) for r in rxns):
        return True
    if _all_reactants_identical(rxns) and _products_differ(rxns):
        return True
    if any(_has_unresolved_rgroup(r) for r in rxns):
        return True
    return False


# --- post-process ----------------------------------------------------------


def _has_unparseable_smiles(result: dict) -> bool:
    try:
        from rdkit import Chem
    except Exception:
        return False
    for rxn in result.get("reactions") or []:
        for side in ("reactants", "products", "conditions"):
            for m in rxn.get(side) or []:
                smi = m.get("smiles")
                if smi and Chem.MolFromSmiles(smi) is None:
                    return True
    return False


_WC_INDEX_RE = re.compile(r"\[(\d+)\*\]")


def _renumber_wildcards(result: dict) -> dict:
    """Renumber non-standard `[N*]` placeholders (e.g. `[7*]`) to consecutive
    `[1*]/[2*]/[3*]` per reaction. Only fires when an index > 5 appears, so
    standard `[1*]/[2*]/[3*]` reactions are untouched. Mapping is consistent
    across reactants / products / conditions of the same reaction so the
    placeholder semantics survive.
    """
    for rxn in result.get("reactions") or []:
        # Collect all wildcard indices used in this reaction.
        indices: set[int] = set()
        for side in ("reactants", "products", "conditions"):
            for m in rxn.get(side) or []:
                smi = m.get("smiles") or ""
                for match in _WC_INDEX_RE.finditer(smi):
                    indices.add(int(match.group(1)))
        if not indices:
            continue
        if all(i <= 5 for i in indices):
            continue  # already standard

        # Map sorted unique indices to 1, 2, 3, ...
        mapping = {old: new for new, old in enumerate(sorted(indices), start=1)}
        for side in ("reactants", "products", "conditions"):
            for m in rxn.get(side) or []:
                smi = m.get("smiles") or ""
                if not smi:
                    continue
                m["smiles"] = _WC_INDEX_RE.sub(
                    lambda mt: f"[{mapping[int(mt.group(1))]}*]", smi
                )
    return result


def _postprocess(result: dict, client, model: str) -> dict:
    # Renumber non-standard wildcards (e.g. molnextr's [7*]) to consecutive
    # [1*]/[2*]/[3*] before SMILES canonicalization. Idempotent + cheap.
    try:
        result = _renumber_wildcards(result)
    except Exception as e:
        result.setdefault("_renumber_error", f"{type(e).__name__}: {e}")

    try:
        from utils.smiles_fix import sanitize_reactions
        result = sanitize_reactions(result)
    except Exception as e:
        result.setdefault("_smiles_fix_error", f"{type(e).__name__}: {e}")

    always = os.environ.get("CHEMEAGLE_VALIDATE", "0") == "1"
    on_fail_only = os.environ.get("CHEMEAGLE_VALIDATE_ON_FAIL_ONLY", "1") == "1"

    needs_validate = always or (on_fail_only and _has_unparseable_smiles(result))
    if needs_validate:
        try:
            from utils.rxn_validator import validate_and_fix
            result = validate_and_fix(
                result, client=client, model=model,
                max_calls=int(os.environ.get("CHEMEAGLE_VALIDATE_MAX_CALLS", "10")),
            )
        except Exception as e:
            result.setdefault("_validator_error", f"{type(e).__name__}: {e}")
    return result


# --- helpers ---------------------------------------------------------------


def _load_image_as_data_url(image_path: str) -> str:
    p = Path(image_path)
    suffix = p.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(suffix, "image/png")
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]\}])")


def _safe_json_parse(content: str) -> dict:
    if not content:
        return {"reactions": []}
    s = content.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        s2 = _TRAILING_COMMA_RE.sub(r"\1", s)
        try:
            out = json.loads(s2)
        except Exception:
            return {"reactions": [], "_parse_error": "JSONDecodeError"}
    if not isinstance(out, dict):
        return {"reactions": [], "_parse_error": "non-dict"}
    return out
