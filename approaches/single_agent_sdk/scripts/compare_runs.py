"""Side-by-side comparison of two extraction runs against the same ground truth.

Usage:
    .venv/bin/python3 scripts/compare_runs.py \
        --run-a benchmark_runs/run01 --label-a "Opus 4.7" \
        --run-b benchmark_runs/run_gemini3flash --label-b "Gemini 3 Flash" \
        --gt corpus/Benchmark_kasper_GT3_Maarten/ground_truth \
        --images corpus/Benchmark_kasper_GT3_Maarten
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.eval import evaluate_run, ImageScore


def fmt_recall(s: ImageScore) -> str:
    if s.notes:
        return "—"
    return f"{s.matched_reactions}/{s.gt_reactions}"


def fmt_pct(x: float) -> str:
    return f"{x*100:>4.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--label-a", default="Run A")
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--label-b", default="Run B")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    args = ap.parse_args()

    scores_a, summary_a = evaluate_run(pred_dir=args.run_a, gt_dir=args.gt, image_dir=args.images)
    scores_b, summary_b = evaluate_run(pred_dir=args.run_b, gt_dir=args.gt, image_dir=args.images)

    by_img_a = {s.image: s for s in scores_a}
    by_img_b = {s.image: s for s in scores_b}
    images = sorted(set(by_img_a) | set(by_img_b))

    la = args.label_a
    lb = args.label_b
    print()
    print(f"{'image':<55}  {la:>10}  {lb:>10}  {'Δ rxn':>8}")
    print("-" * 95)
    for img in images:
        sa = by_img_a.get(img)
        sb = by_img_b.get(img)
        ra = fmt_recall(sa) if sa else "—"
        rb = fmt_recall(sb) if sb else "—"
        # Numerical delta in reaction recall (matched - matched), only when both ran
        delta = ""
        if sa and sb and not sa.notes and not sb.notes:
            d = sb.matched_reactions - sa.matched_reactions
            delta = f"{d:+d}"
        marker = "  ★" if (sa and sb and not sa.notes and not sb.notes
                            and sb.matched_reactions > sa.matched_reactions) else ""
        print(f"{img:<55}  {ra:>10}  {rb:>10}  {delta:>8}{marker}")
    print("-" * 95)

    def line(label: str, s: dict) -> str:
        if not s:
            return f"  {label:<14}  (no data)"
        return (
            f"  {label:<14}  "
            f"rxn-recall={fmt_pct(s['reaction_recall'])}  "
            f"prod-recall={fmt_pct(s['product_recall'])}  "
            f"prod-prec={fmt_pct(s['product_precision'])}  "
            f"yld-acc={fmt_pct(s['yield_accuracy'])}  "
            f"smiles-valid={fmt_pct(s['smiles_validity'])}  "
            f"({s['n_images']} imgs)"
        )

    print(f"\nAggregate")
    print(line(la, summary_a))
    print(line(lb, summary_b))


if __name__ == "__main__":
    main()
