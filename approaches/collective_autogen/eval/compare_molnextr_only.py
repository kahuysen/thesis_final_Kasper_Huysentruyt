"""Compare MolNexTR-only flat SMILES extraction against the full pipeline.

For each benchmark image, computes the canonical-SMILES set produced by:
- MolNexTR-only: union of predictions.json entries from
  eval/benchmark/molnextr_crops/{stem}/predictions.json
- Full pipeline: union of reactant + product SMILES from the latest suite's
  result.json (default: suite_20260428_085137)

Both are scored against the ground truth's flat reactant ∪ product set
(eval/ground_truth/{stem}.json) using set IoU + precision/recall/F1.

This is a *flat-SMILES* metric — it ignores reaction structure entirely. The
question it answers: does the detect+recognise stage alone find the same
molecules the full pipeline does?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import _canonical

REPO = Path(__file__).resolve().parent.parent
GROUND_TRUTH = REPO / "eval" / "ground_truth"
CROPS_DIR = REPO / "eval" / "benchmark" / "molnextr_crops"
DEFAULT_SUITE = REPO / "eval" / "results" / "suite_20260428_085137"


def _gt_smiles_set(gold: dict) -> set[str]:
    out: set[str] = set()
    for r in gold.get("reactions", []) or []:
        for c in (r.get("reactants") or []) + (r.get("products") or []):
            s = _canonical(c.get("smiles"))
            if s:
                out.add(s)
    return out


def _pipeline_smiles_set(record: dict) -> set[str]:
    return _gt_smiles_set(record)  # same shape


def _molnextr_smiles_set(predictions: dict) -> set[str]:
    out: set[str] = set()
    for e in predictions.get("entries", []) or []:
        if e.get("error"):
            continue
        s = _canonical(e.get("smiles"))
        if s:
            out.add(s)
    return out


def _score(pred: set[str], gold: set[str]) -> dict:
    inter = pred & gold
    union = pred | gold
    p = (len(inter) / len(pred)) if pred else 0.0
    r = (len(inter) / len(gold)) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    iou = (len(inter) / len(union)) if union else 1.0
    return {
        "pred_size": len(pred),
        "gold_size": len(gold),
        "intersection": len(inter),
        "iou": iou,
        "precision": p,
        "recall": r,
        "f1": f1,
        "missing_in_pred": sorted(gold - pred),
        "extra_in_pred": sorted(pred - gold),
    }


def _suite_run_dirs(suite_dir: Path) -> dict[str, Path]:
    summary = json.loads((suite_dir / "summary.json").read_text())
    out: dict[str, Path] = {}
    for entry in summary.get("per_image", []):
        rd = entry.get("run_dir")
        stem = entry.get("stem")
        if rd and stem:
            out[stem] = REPO / rd
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE,
                        help=f"Suite dir whose result.json files to compare against (default: {DEFAULT_SUITE.name}).")
    parser.add_argument("--out", type=Path,
                        default=REPO / "eval" / "results" / "molnextr_only_vs_pipeline.json",
                        help="Where to write the per-image + aggregate JSON summary.")
    args = parser.parse_args()

    if not args.suite.exists():
        raise SystemExit(f"suite dir not found: {args.suite}")
    run_dirs = _suite_run_dirs(args.suite)

    gt_files = sorted(GROUND_TRUTH.glob("*.json"))
    rows = []
    for gt_path in gt_files:
        if gt_path.name == "README.md":
            continue
        gold = json.loads(gt_path.read_text())
        stem = gt_path.stem
        gold_set = _gt_smiles_set(gold)

        # MolNexTR-only predictions
        pred_path = CROPS_DIR / stem / "predictions.json"
        if not pred_path.exists():
            molnextr = None
        else:
            molnextr = _molnextr_smiles_set(json.loads(pred_path.read_text()))

        # Full pipeline result
        run_dir = run_dirs.get(stem)
        result_path = run_dir / "result.json" if run_dir else None
        if not result_path or not result_path.exists():
            pipeline = None
        else:
            try:
                rec = json.loads(result_path.read_text())
                pipeline = _pipeline_smiles_set(rec) if isinstance(rec, dict) else set()
            except json.JSONDecodeError:
                pipeline = set()  # invalid run_id raw output

        row = {
            "stem": stem,
            "gold_size": len(gold_set),
            "molnextr_only": _score(molnextr, gold_set) if molnextr is not None else None,
            "pipeline": _score(pipeline, gold_set) if pipeline is not None else None,
        }
        rows.append(row)

    # Aggregate
    def _avg(rows, src, key):
        vals = [r[src][key] for r in rows if r.get(src) is not None]
        return sum(vals) / len(vals) if vals else None

    aggregate = {
        "suite": args.suite.name,
        "n_images": len(rows),
        "molnextr_only_mean": {
            "iou": _avg(rows, "molnextr_only", "iou"),
            "precision": _avg(rows, "molnextr_only", "precision"),
            "recall": _avg(rows, "molnextr_only", "recall"),
            "f1": _avg(rows, "molnextr_only", "f1"),
        },
        "pipeline_mean": {
            "iou": _avg(rows, "pipeline", "iou"),
            "precision": _avg(rows, "pipeline", "precision"),
            "recall": _avg(rows, "pipeline", "recall"),
            "f1": _avg(rows, "pipeline", "f1"),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"aggregate": aggregate, "per_image": rows}, indent=2, ensure_ascii=False))

    # Console table
    hdr = f"{'stem':45s}  {'gold':>4s}  | {'mn_F1':>6s} {'mn_P':>5s} {'mn_R':>5s} {'mn_IoU':>6s} | {'pp_F1':>6s} {'pp_P':>5s} {'pp_R':>5s} {'pp_IoU':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mn = r["molnextr_only"] or {"f1": None, "precision": None, "recall": None, "iou": None}
        pp = r["pipeline"] or {"f1": None, "precision": None, "recall": None, "iou": None}
        def f(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
        print(f"{r['stem'][:45]:45s}  {r['gold_size']:>4d}  | {f(mn['f1']):>6s} {f(mn['precision']):>5s} {f(mn['recall']):>5s} {f(mn['iou']):>6s} "
              f"| {f(pp['f1']):>6s} {f(pp['precision']):>5s} {f(pp['recall']):>5s} {f(pp['iou']):>6s}")
    print("-" * len(hdr))
    a = aggregate
    def f(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
    print(f"{'MEAN':45s}  {'':>4s}  | "
          f"{f(a['molnextr_only_mean']['f1']):>6s} {f(a['molnextr_only_mean']['precision']):>5s} {f(a['molnextr_only_mean']['recall']):>5s} {f(a['molnextr_only_mean']['iou']):>6s} | "
          f"{f(a['pipeline_mean']['f1']):>6s} {f(a['pipeline_mean']['precision']):>5s} {f(a['pipeline_mean']['recall']):>5s} {f(a['pipeline_mean']['iou']):>6s}")
    print(f"\nSaved: {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
