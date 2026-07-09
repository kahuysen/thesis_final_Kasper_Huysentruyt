"""Per-image failure-mode analysis for the seven single-SDK variants.

Reads predicted JSONs from Single_SDK_agent/benchmark_runs/<variant>/ and the
matching gold JSONs from Collective_autogen/eval/ground_truth/. For each variant
computes per-image metrics (uses the existing `evaluate` from eval.metrics and
the `figure_extraction_to_record` translator from
Single_SDK_agent/scripts/eval_via_collective.py), identifies the three lowest
Product-IoU images and the lowest Partial-F1 image (if it differs), then dumps
the bipartite-best-matched predicted-vs-gold SMILES pairs plus a few coarse
signals (stereo-stripped equality, atom-count delta, schema/step-ceiling state)
so a human can finalise the failure-category label.

Why this helper lives here (Collective_autogen/eval/): the canonicalisation and
metric implementations live in `eval/metrics.py` and the per-image record
translator lives in `Single_SDK_agent/scripts/eval_via_collective.py`. This
script imports both rather than reimplementing either, so failure-mode tags are
consistent with the benchmark's headline scoreboard. It writes only to its
arguments — no source-of-truth files are modified.

Usage:
    python eval/failure_mode_analysis.py \\
        --bench-root /Users/kasperhuysentruyt/Documents/thesis/5.Code/Single_SDK_agent/benchmark_runs \\
        --out /Users/kasperhuysentruyt/Documents/thesis/5.Code/outputs/failure_modes_sdk_variants.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make eval and the SDK eval bridge importable.
REPO_ROOT = Path("/Users/kasperhuysentruyt/Documents/thesis/5.Code")
COLLECTIVE = REPO_ROOT / "Collective_autogen"
SDK_AGENT = REPO_ROOT / "Single_SDK_agent"
sys.path.insert(0, str(COLLECTIVE))
sys.path.insert(0, str(SDK_AGENT))

from eval.metrics import (  # noqa: E402
    evaluate,
    _canonical,
    _canonical_no_stereo,
)
from scripts.eval_via_collective import figure_extraction_to_record  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFMCS  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

GOLD_DIR = COLLECTIVE / "eval" / "ground_truth"

VARIANTS = [
    ("Claude Opus 4.7", "run01_opus4.7"),
    ("Claude Opus 4.6", "run_opus46"),
    ("GPT-5.4", "run_gpt54"),
    ("GPT-5.5", "run_gpt55"),
    ("Gemini 3.1 Pro Preview", "run_gemini31propreview"),
    ("Gemini 3 Flash Preview", "run_gemini3flash"),
    ("xAI Grok 4.3", "run_grok43"),
]


def _atom_counts(smiles: str | None) -> tuple[int, int] | None:
    """Return (heavy_atoms, bonds) for a SMILES, or None if unparseable."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return (m.GetNumHeavyAtoms(), m.GetNumBonds())


def _carbon_count(smiles: str | None) -> int | None:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C")


def _ged_pair(s1: str | None, s2: str | None) -> int:
    """Approximate graph-edit distance (MCS-derived)."""
    m1 = Chem.MolFromSmiles(s1) if s1 else None
    m2 = Chem.MolFromSmiles(s2) if s2 else None
    if m1 is None or m2 is None:
        return 10_000
    if Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2):
        return 0
    try:
        mcs = rdFMCS.FindMCS([m1, m2], timeout=5)
    except Exception:
        return 10_000
    if mcs.canceled or mcs.numAtoms == 0:
        return (m1.GetNumAtoms() + m1.GetNumBonds()) + (m2.GetNumAtoms() + m2.GetNumBonds())
    common = mcs.numAtoms + mcs.numBonds
    s1_size = m1.GetNumAtoms() + m1.GetNumBonds()
    s2_size = m2.GetNumAtoms() + m2.GetNumBonds()
    return (s1_size - common) + (s2_size - common)


def _best_match_smiles(pred: list[str], gold: list[str]) -> list[tuple[str | None, str | None, int]]:
    """Hungarian-aligned (pred_smi, gold_smi, ged) triples."""
    n_p, n_g = len(pred), len(gold)
    n = max(n_p, n_g)
    if n == 0:
        return []
    PEN = 100
    cost = np.full((n, n), PEN, dtype=float)
    for i in range(n_p):
        for j in range(n_g):
            cost[i, j] = _ged_pair(pred[i], gold[j])
    r, c = linear_sum_assignment(cost)
    out: list[tuple[str | None, str | None, int]] = []
    for i, j in zip(r, c):
        pi = pred[i] if i < n_p else None
        gi = gold[j] if j < n_g else None
        out.append((pi, gi, int(cost[i, j])))
    return out


def _all_smiles_for_role(record: dict, role: str) -> list[str]:
    field = "reactants" if role == "reactant" else "products"
    out: list[str] = []
    for r in record.get("reactions") or []:
        for c in r.get(field) or []:
            s = c.get("smiles")
            if s:
                out.append(s)
    return out


def _classify_pair(pred_smi: str | None, gold_smi: str | None) -> tuple[str, str]:
    """Coarse heuristic classifier. Returns (category, note)."""
    if pred_smi is None and gold_smi is None:
        return ("match", "both empty")
    if pred_smi is None:
        return ("missing_reactions", "gold molecule has no predicted match")
    if gold_smi is None:
        return ("hallucinated_reactions", "predicted molecule has no gold match")
    c_pred = _canonical(pred_smi)
    c_gold = _canonical(gold_smi)
    if c_pred is None:
        return ("schema_failure", "pred SMILES unparseable")
    if c_gold is None:
        return ("other", "gold SMILES unparseable (skipped)")
    if c_pred == c_gold:
        return ("match", "exact")
    ns_pred = _canonical_no_stereo(pred_smi)
    ns_gold = _canonical_no_stereo(gold_smi)
    if ns_pred and ns_gold and ns_pred == ns_gold:
        return ("stereo_error", "constitution identical, stereo differs")
    cp = _carbon_count(pred_smi)
    cg = _carbon_count(gold_smi)
    if cp is not None and cg is not None and abs(cp - cg) == 1:
        # Most likely a CH2 shift; verify other atom counts match.
        mp = Chem.MolFromSmiles(pred_smi)
        mg = Chem.MolFromSmiles(gold_smi)
        # Count non-C atoms
        def _non_c(mol):
            return tuple(sorted((a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "C")))
        if _non_c(mp) == _non_c(mg):
            return ("methylene_shift", f"|ΔC|=1 (pred {cp}, gold {cg})")
    ap = _atom_counts(pred_smi)
    ag = _atom_counts(gold_smi)
    if ap and ag and ap[0] == ag[0]:
        # Same heavy-atom count: could be connectivity OR substituent_position.
        # Distinguish: substituent_position is when both are aromatic rings with
        # same substituent set but different ring positions. Heuristic: same
        # heavy atoms, same molecular formula, both contain aromatic ring.
        mp = Chem.MolFromSmiles(pred_smi)
        mg = Chem.MolFromSmiles(gold_smi)
        if mp.GetNumAtoms() == mg.GetNumAtoms():
            formula_p = sorted(a.GetSymbol() for a in mp.GetAtoms())
            formula_g = sorted(a.GetSymbol() for a in mg.GetAtoms())
            if formula_p == formula_g and any(a.GetIsAromatic() for a in mp.GetAtoms()):
                return ("substituent_position", "same formula, aromatic — likely ring substituent shift")
        return ("connectivity_misread", "same heavy-atom count, different connectivity")
    if ap and ag:
        return ("connectivity_misread", f"heavy atoms differ ({ap[0]} vs {ag[0]})")
    return ("other", "unclassified")


def _check_step_ceiling(meta_path: Path, steps_limit: int = 16) -> bool:
    if not meta_path.exists():
        return False
    try:
        m = json.loads(meta_path.read_text())
    except Exception:
        return False
    return m.get("steps", 0) >= steps_limit


def analyse_image(pred_path: Path, gold_path: Path) -> dict:
    fx = json.loads(pred_path.read_text())
    gold = json.loads(gold_path.read_text())
    record = figure_extraction_to_record(fx, file_name=f"{pred_path.stem}.png")
    metrics = evaluate(record, gold)

    # Step-ceiling detection
    meta_path = pred_path.with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = pred_path.parent / f"{pred_path.stem}.meta.json"
    step_ceiling = _check_step_ceiling(meta_path)

    # Compute bipartite-best SMILES pairs for products + reactants
    pred_prod = _all_smiles_for_role(record, "product")
    gold_prod = _all_smiles_for_role(gold, "product")
    pred_react = _all_smiles_for_role(record, "reactant")
    gold_react = _all_smiles_for_role(gold, "reactant")
    prod_pairs = _best_match_smiles(pred_prod, gold_prod)
    react_pairs = _best_match_smiles(pred_react, gold_react)

    classified_prod = [
        {
            "pred": p,
            "gold": g,
            "ged": d,
            "category": _classify_pair(p, g)[0],
            "note": _classify_pair(p, g)[1],
        }
        for p, g, d in prod_pairs
    ]
    classified_react = [
        {
            "pred": p,
            "gold": g,
            "ged": d,
            "category": _classify_pair(p, g)[0],
            "note": _classify_pair(p, g)[1],
        }
        for p, g, d in react_pairs
    ]

    return {
        "image": pred_path.stem,
        "pred_path": str(pred_path),
        "gold_path": str(gold_path),
        "product_iou": metrics["product_iou"]["iou"],
        "reactant_iou": metrics["reactant_iou"]["iou"],
        "partial_f1": metrics["partial_match"]["f1"],
        "partial_mean_jaccard": metrics["partial_match"]["mean_jaccard"],
        "soft_f1": metrics["soft_match"]["f1"],
        "schema_valid": metrics["schema_conformance"]["valid"],
        "pred_count": metrics["reaction_count"]["count"],
        "gold_count": metrics["reaction_count_gold"]["count"],
        "step_ceiling_hit": step_ceiling,
        "condition_recall": metrics["condition_coverage"]["recall"],
        "product_pairs": classified_prod,
        "reactant_pairs": classified_react,
    }


def analyse_variant(variant_label: str, run_dir: Path) -> dict:
    all_results: list[dict] = []
    for pred_path in sorted(run_dir.glob("*.json")):
        if pred_path.name.endswith(".meta.json"):
            continue
        gold_path = GOLD_DIR / f"{pred_path.stem}.json"
        if not gold_path.exists():
            continue
        try:
            r = analyse_image(pred_path, gold_path)
        except Exception as e:
            print(f"  ! failed on {pred_path.stem}: {e}", file=sys.stderr)
            continue
        all_results.append(r)
    # Rank by product_iou (low = worst), break ties by partial_f1.
    by_prod = sorted(all_results, key=lambda r: (r["product_iou"], r["partial_f1"]))
    by_partial = sorted(all_results, key=lambda r: r["partial_f1"])
    lowest3 = by_prod[:3]
    lowest_partial = by_partial[0] if by_partial else None
    return {
        "variant": variant_label,
        "run_dir": str(run_dir),
        "n_images": len(all_results),
        "all": all_results,
        "lowest3_product_iou": lowest3,
        "lowest_partial_f1": lowest_partial,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path,
                    default=SDK_AGENT / "benchmark_runs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out_payload = {"variants": []}
    for label, dirname in VARIANTS:
        run_dir = args.bench_root / dirname
        if not run_dir.exists():
            print(f"!! missing run dir: {run_dir}", file=sys.stderr)
            continue
        print(f"scoring {label} ({run_dir})")
        v = analyse_variant(label, run_dir)
        out_payload["variants"].append(v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
