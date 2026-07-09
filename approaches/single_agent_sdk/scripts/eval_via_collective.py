"""Score our extraction runs using the Collective_autogen eval suite.

This bridges:
  - our `FigureExtraction` schema (pipeline/schema.py) →
  - the Collective_autogen `record` schema that `eval/metrics.py:evaluate()`
    expects (reactions[].conditions[] as a flat list with role/text/smiles)

Then runs `evaluate()` per image against the matching gold file under
`Collective_autogen/eval/ground_truth/`, aggregates micro/macro metrics
across all images, and prints a side-by-side report.

Usage:
    .venv/bin/python3 scripts/eval_via_collective.py \
        --runs benchmark_runs/run01_opus4.7=Opus 4.7 \
                benchmark_runs/run_gpt54=GPT-5.4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Optional

# Make the collective_autogen eval module importable.
COLL = Path(__file__).resolve().parent.parent.parent / "collective_autogen"
sys.path.insert(0, str(COLL))
from eval.metrics import evaluate  # noqa: E402

GOLD_DIR = COLL / "eval" / "ground_truth"

# ───── role mapping: our pipeline → Collective_autogen enum ─────
LEGAL_ROLES = {
    "reagent", "solvent", "catalyst", "temperature", "time", "yield",
    "atmosphere", "loading", "ee", "dr", "pressure",
}
ROLE_MAP = {
    "catalyst": "catalyst",
    "solvent": "solvent",
    # everything else our schema has — reductant/additive/trap/oxidant/base/other —
    # collapses to the generic "reagent" role.
    "reductant": "reagent",
    "additive": "reagent",
    "trap": "reagent",
    "oxidant": "reagent",
    "base": "reagent",
    "other": "reagent",
    "": "reagent",
    None: "reagent",
}


def _coerce_role(role: Optional[str]) -> str:
    if role and role.lower() in LEGAL_ROLES:
        return role.lower()
    return ROLE_MAP.get((role or "").lower(), "reagent")


import re as _re
_ATTACH_MARKER_RX = _re.compile(r"\[\*:\d+\]")

# ───── deterministic condition normalization (Tier 1) ─────
# Buckets that collapse the catalyst/reagent/additive role-label disagreement
# we observed between GT and our extractions. Anything in REAGENT_BUCKET maps
# to the same opaque role for matching purposes.
REAGENT_BUCKET = {"reagent", "catalyst", "additive", "base", "oxidant",
                  "reductant", "trap", "other"}

# Numeric-with-unit collapsing: "82 %" → "82%", "20 mol %" → "20 mol%",
# "5 mmol" stays "5 mmol", "45 °c" stays "45 °c" (the unit normalization is
# only for the trailing punctuation, not generic word units).
_NUM_PCT_RX  = _re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MOL_PCT_RX  = _re.compile(r"(\d+(?:\.\d+)?)\s*mol\s*%")
_EQUIV_RX    = _re.compile(r"(\d+(?:\.\d+)?)\s*equiv")
_PAREN_RX    = _re.compile(r"[()]")
_MULTISPACE  = _re.compile(r"\s+")


def _bucket_role(role: str) -> str:
    """Collapse catalyst/reagent/additive/etc. into a single matching bucket."""
    role = (role or "").lower()
    if role in REAGENT_BUCKET:
        return "reagent_bucket"
    return role


def _normalize_text_lenient(text: str) -> str:
    """Strip parentheses, collapse whitespace inside number+unit pairs."""
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = _PAREN_RX.sub(" ", t)
    t = _MOL_PCT_RX.sub(r"\1 mol%", t)
    t = _NUM_PCT_RX.sub(r"\1%", t)
    t = _EQUIV_RX.sub(r"\1 equiv", t)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


def _canonical_smiles_safe(s):
    """Canonicalise a SMILES; return None on failure."""
    if not s or not isinstance(s, str) or s == "None":
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _conditions_for_lenient(record: dict) -> list[dict]:
    """Flatten conditions with both bucketed role and lenient text + canon-SMILES."""
    out = []
    for r in record.get("reactions", []):
        for c in r.get("conditions", []) or []:
            out.append(
                {
                    "role": _bucket_role(c.get("role", "")),
                    "text": _normalize_text_lenient(c.get("text", "")),
                    "smiles": _canonical_smiles_safe(c.get("smiles")),
                }
            )
    # De-dupe identical signatures.
    seen = []
    for c in out:
        sig = (c["role"], c["text"], c["smiles"])
        if sig not in [(x["role"], x["text"], x["smiles"]) for x in seen]:
            seen.append(c)
    return seen


def lenient_condition_coverage(record: dict, gold: dict) -> dict:
    """Two-pass matcher:
       (1) SMILES-equal + same role bucket  →  match  (handles different naming)
       (2) Lenient-text-equal + same role bucket  →  match
       Recall and precision over the two-pass union."""
    pred = _conditions_for_lenient(record)
    g    = _conditions_for_lenient(gold)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched = 0

    # Pass 1: by SMILES (if both have one)
    for pi, p in enumerate(pred):
        if p["smiles"] is None:
            continue
        for gi, gc in enumerate(g):
            if gi in used_g or gc["smiles"] is None:
                continue
            if p["role"] == gc["role"] and p["smiles"] == gc["smiles"]:
                used_g.add(gi); used_p.add(pi); matched += 1
                break

    # Pass 2: by lenient text + role
    for pi, p in enumerate(pred):
        if pi in used_p:
            continue
        for gi, gc in enumerate(g):
            if gi in used_g:
                continue
            if p["role"] == gc["role"] and p["text"] == gc["text"] and p["text"]:
                used_g.add(gi); used_p.add(pi); matched += 1
                break

    return {
        "gold_size": len(g),
        "pred_size": len(pred),
        "matched": matched,
        "recall": (matched / len(g)) if g else 1.0,
        "precision": (matched / len(pred)) if pred else 1.0,
    }


def _strip_attachment_markers(smi: Optional[str]) -> Optional[str]:
    """Normalize colon-suffixed wildcards `[*:N]` → `[*]`.

    Their schema rejects `[*:N]` as a raw R-fragment marker, but we use it
    semantically as an atom-mapped wildcard equivalent to `[*]`. Strip the
    map number for conformance without changing the molecular meaning.
    """
    if not smi:
        return smi
    return _ATTACH_MARKER_RX.sub("[*]", smi)


def _conds_from_reaction(rxn: dict) -> list[dict]:
    """Build their flat conditions[] from our reagents[] + conditions{} + product yields."""
    out: list[dict] = []

    # Our reagents (catalyst/solvent/etc) → flat condition entries.
    for r in rxn.get("reagents") or []:
        loading = r.get("loading") or ""
        text = r.get("label") or r.get("name") or r.get("smiles") or ""
        if loading:
            text = f"{text} ({loading})".strip()
        out.append(
            {
                "role": _coerce_role(r.get("role")),
                "text": text,
                "smiles": _strip_attachment_markers(r.get("smiles")) or None,
            }
        )

    # Our scalar conditions (temperature/time/atmosphere/solvent/other) → entries.
    cdict = rxn.get("conditions") or {}
    if isinstance(cdict, dict):
        for key in ("temperature", "time", "atmosphere"):
            if cdict.get(key):
                out.append({"role": key, "text": cdict[key], "smiles": None})
        if cdict.get("solvent"):
            out.append({"role": "solvent", "text": cdict["solvent"], "smiles": None})
        if cdict.get("other"):
            out.append({"role": "reagent", "text": cdict["other"], "smiles": None})

    # Yields live on each product in our schema; their schema records yield as
    # a per-reaction condition row.
    for p in rxn.get("products") or []:
        if p.get("yield_pct") is not None:
            out.append(
                {"role": "yield", "text": f"{p['yield_pct']:g} %", "smiles": None}
            )
        elif p.get("yield_note"):
            out.append({"role": "yield", "text": p["yield_note"], "smiles": None})

    return out


def figure_extraction_to_record(fx: dict, file_name: str) -> dict:
    """Translate one of our FigureExtraction dicts into the Collective record schema.

    Their `reaction_id` field is constrained to `^\\d+_\\d+$` (e.g. `0_1`, `1_1`),
    while ours uses freeform `entry_id` labels (`"1"`, `"general"`, `"I"`, …).
    We synthesise positional ids and stash the original label in `additional_info`
    so it round-trips for downstream inspection.
    """
    reactions = []
    for i, r in enumerate(fx.get("reactions") or []):
        original_entry = r.get("entry_id")
        extra: list[str] = []
        if original_entry and str(original_entry) != f"{i}_1":
            extra.append(f"original_entry_id={original_entry}")
        if r.get("title"):
            extra.append(f"title={r['title']}")
        if r.get("notes"):
            extra.append(f"notes={r['notes']}")

        reactions.append(
            {
                "reaction_id": f"{i}_1",
                "reactants": [
                    {
                        "smiles": _strip_attachment_markers(x.get("smiles")) or None,
                        "label": x.get("label"),
                        "uncertain": False,
                    }
                    for x in r.get("reactants") or []
                ],
                "products": [
                    {
                        "smiles": _strip_attachment_markers(x.get("smiles")) or None,
                        "label": x.get("label"),
                        "uncertain": False,
                    }
                    for x in r.get("products") or []
                ],
                "conditions": _conds_from_reaction(r),
                "additional_info": extra,
            }
        )
    return {"file_name": file_name, "reactions": reactions}


# ───── per-run scoring ─────

def score_run(run_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """Score every prediction JSON in `run_dir` against the matching gold.

    Returns (per_image_metrics, list_of_missing_gold_stems).
    """
    out: dict[str, dict] = {}
    missing: list[str] = []
    for pred_path in sorted(run_dir.glob("*.json")):
        if pred_path.stem.endswith(("_insight", ".original")):
            continue
        gold_path = GOLD_DIR / f"{pred_path.stem}.json"
        if not gold_path.exists():
            missing.append(pred_path.stem)
            continue
        try:
            fx = json.loads(pred_path.read_text())
            gold = json.loads(gold_path.read_text())
        except Exception as e:
            print(f"  ! could not parse {pred_path.name}: {e}", file=sys.stderr)
            continue
        record = figure_extraction_to_record(fx, file_name=f"{pred_path.stem}.png")
        try:
            metrics = evaluate(record, gold)
        except Exception as e:
            print(f"  ! evaluate failed on {pred_path.stem}: {e}", file=sys.stderr)
            continue
        # Tier-1 lenient condition matcher — runs alongside the strict one.
        try:
            metrics["condition_coverage_lenient"] = lenient_condition_coverage(record, gold)
        except Exception as e:
            print(f"  ! lenient condition matcher failed on {pred_path.stem}: {e}", file=sys.stderr)
        out[pred_path.stem] = metrics
    return out, missing


# ───── aggregation across images ─────

def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def aggregate(per_image: dict[str, dict]) -> dict:
    """Micro-aggregate across images for the headline numbers."""
    if not per_image:
        return {}
    schema_pass = sum(1 for m in per_image.values() if m["schema_conformance"]["valid"])

    # SMILES validity and role compliance — micro-average over total atoms checked.
    sv_valid = sum(m["smiles_validity_rate"]["valid"] for m in per_image.values())
    sv_total = sum(m["smiles_validity_rate"]["total"] for m in per_image.values())
    re_legal = sum(m["role_enum_compliance"]["legal"] for m in per_image.values())
    re_total = sum(m["role_enum_compliance"]["total"] for m in per_image.values())

    # Reaction count
    pred_count = sum(m["reaction_count"]["count"] for m in per_image.values())
    gold_count = sum(m.get("reaction_count_gold", {}).get("count", 0) for m in per_image.values())

    # IoU / coverage / F1 metrics — macro-average across images that have them.
    def _macro(parent, child):
        vals = [m[parent].get(child) for m in per_image.values() if parent in m]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return mean(vals) if vals else None

    return {
        "n_images": len(per_image),
        "schema_pass_rate": schema_pass / len(per_image),
        "smiles_validity": _safe_div(sv_valid, sv_total),
        "role_enum_compliance": _safe_div(re_legal, re_total),
        "reaction_count_pred": pred_count,
        "reaction_count_gold": gold_count,
        "reactant_iou":          _macro("reactant_iou", "iou"),
        "product_iou":           _macro("product_iou", "iou"),
        "condition_coverage":    _macro("condition_coverage", "recall"),
        "condition_coverage_lenient":   _macro("condition_coverage_lenient", "recall"),
        "condition_precision_lenient":  _macro("condition_coverage_lenient", "precision"),
        "soft_match_f1":         _macro("soft_match", "f1"),
        "hard_match_f1":         _macro("hard_match", "f1"),
        "constitution_match_f1": _macro("constitution_match", "f1"),
        "partial_match_f1":      _macro("partial_match", "f1"),
        "graph_edit_distance":   _macro("graph_edit_distance", "ged_per_reaction"),
    }


# ───── reporting ─────

KEYS_TO_SHOW = [
    ("schema_pass_rate",      "schema pass",         "{:>5.0%}"),
    ("smiles_validity",       "SMILES validity",     "{:>5.0%}"),
    ("role_enum_compliance",  "role compliance",     "{:>5.0%}"),
    ("reaction_count_pred",   "reactions pred",      "{:>5.0f}"),
    ("reaction_count_gold",   "reactions gold",      "{:>5.0f}"),
    ("reactant_iou",          "reactant IoU",        "{:>5.2f}"),
    ("product_iou",           "product IoU",         "{:>5.2f}"),
    ("condition_coverage",    "cond recall (strict)", "{:>5.0%}"),
    ("condition_coverage_lenient",  "cond recall (lenient)", "{:>5.0%}"),
    ("condition_precision_lenient", "cond precision (lenient)", "{:>5.0%}"),
    ("soft_match_f1",         "soft match F1",       "{:>5.2f}"),
    ("hard_match_f1",         "hard match F1",       "{:>5.2f}"),
    ("constitution_match_f1", "constitution F1",     "{:>5.2f}"),
    ("partial_match_f1",      "partial F1 (≥0.5)",   "{:>5.2f}"),
    ("graph_edit_distance",   "GED per reaction",    "{:>5.2f}"),
]


def fmt(val, fmtspec) -> str:
    if val is None:
        return "  —  "
    if "%" in fmtspec:
        return fmtspec.format(val)
    return fmtspec.format(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run dirs as `path=label` (e.g. benchmark_runs/run_gpt54=GPT-5.4).")
    args = ap.parse_args()

    runs: list[tuple[str, Path]] = []
    for spec in args.runs:
        if "=" not in spec:
            path, label = spec, Path(spec).name
        else:
            path, label = spec.split("=", 1)
        runs.append((label.strip(), Path(path.strip())))

    # Score each run.
    aggregates: dict[str, dict] = {}
    per_image_per_run: dict[str, dict[str, dict]] = {}
    for label, path in runs:
        print(f"\nScoring: {label}  ({path})")
        per_image, missing = score_run(path)
        if missing:
            print(f"  no gold for: {', '.join(missing[:3])}{' …' if len(missing) > 3 else ''}")
        aggregates[label] = aggregate(per_image)
        per_image_per_run[label] = per_image

    # Print side-by-side aggregate comparison.
    labels = [lbl for lbl, _ in runs]
    print()
    print(f"\n{'metric':<22}  " + "  ".join(f"{l:>16}" for l in labels))
    print("-" * (22 + 18 * len(labels)))
    for key, label, fmtspec in KEYS_TO_SHOW:
        row = f"{label:<22}  " + "  ".join(
            f"{fmt(aggregates[l].get(key), fmtspec):>16}" for l in labels
        )
        print(row)
    print("-" * (22 + 18 * len(labels)))
    n = aggregates[labels[0]].get("n_images", 0)
    print(f"  ({n} images)")


if __name__ == "__main__":
    main()
