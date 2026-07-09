"""Pipeline CLI.

Examples:
    # Extract every figure under ./corpus, write JSON + cards + CSV to ./results
    python cli.py extract corpus/ --out results/

    # Re-render cards from already-extracted JSON without re-calling the API
    python cli.py render results/ --out results/

    # Cheaper batch run
    python cli.py extract corpus/ --out results/ --model claude-sonnet-4-6 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.batch import DEFAULT_MODEL, run_batch
from pipeline.config import describe as describe_backend
from pipeline.eval import evaluate_run, format_report
from pipeline.flatten import figure_to_rows, write_csv
from pipeline.renderer import render_figure
from pipeline.schema import FigureExtraction


def cmd_extract(args: argparse.Namespace) -> int:
    print(f"Backend: {describe_backend()}")
    results = run_batch(
        corpus=args.corpus,
        out_root=args.out,
        model=args.model,
        concurrency=args.concurrency,
        skip_existing=not args.force,
        csv_name=args.csv_name,
    )
    n_ok = sum(1 for r in results if r.ok)
    n_err = sum(1 for r in results if not r.ok)
    print(f"\nDone: {n_ok} ok, {n_err} failed, {len(results)} total")
    return 0 if n_err == 0 else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run extraction on a benchmark folder and score against ground truth."""
    bench = Path(args.bench_dir)
    image_dir = bench
    gt_dir = bench / "ground_truth"
    if not gt_dir.exists():
        print(f"No ground_truth/ folder under {bench}")
        return 2
    out_dir = Path(args.out)

    print(f"Backend: {describe_backend()}")
    print(f"Images:  {image_dir}")
    print(f"GT:      {gt_dir}")
    print(f"Out:     {out_dir}\n")

    if not args.skip_extract:
        run_batch(
            corpus=image_dir,
            out_root=out_dir,
            model=args.model,
            concurrency=args.concurrency,
            skip_existing=not args.force,
            csv_name="reactions.csv",
        )

    scores, summary = evaluate_run(pred_dir=out_dir, gt_dir=gt_dir, image_dir=image_dir)
    report = format_report(scores, summary)
    print("\n" + report)
    (out_dir / "evaluation.txt").write_text(report)
    print(f"\nReport saved to {out_dir / 'evaluation.txt'}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    in_dir = Path(args.json_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files in {in_dir}")
        return 1

    all_rows = []
    for jf in json_files:
        try:
            fx = FigureExtraction.model_validate_json(jf.read_text())
        except Exception as e:
            print(f"  SKIP {jf.name}: {e}")
            continue
        cards = render_figure(fx, out_dir / f"{jf.stem}_cards")
        all_rows.extend(figure_to_rows(fx, source_image=jf.stem))
        print(f"  {jf.name}: {len(cards)} card(s)")

    if all_rows:
        csv_path = write_csv(all_rows, out_dir / args.csv_name)
        print(f"Wrote {len(all_rows)} rows → {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rxn-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Run vision agent over a corpus.")
    pe.add_argument("corpus", help="Folder of figure images.")
    pe.add_argument("--out", default="results", help="Output directory.")
    pe.add_argument("--model", default=DEFAULT_MODEL)
    pe.add_argument("--concurrency", type=int, default=4)
    pe.add_argument("--force", action="store_true", help="Re-extract even if JSON exists.")
    pe.add_argument("--csv-name", default="reactions.csv")
    pe.set_defaults(func=cmd_extract)

    pr = sub.add_parser("render", help="Re-render cards/CSV from existing JSON.")
    pr.add_argument("json_dir", help="Folder of per-figure JSON files.")
    pr.add_argument("--out", default="results", help="Output directory.")
    pr.add_argument("--csv-name", default="reactions.csv")
    pr.set_defaults(func=cmd_render)

    pb = sub.add_parser("benchmark", help="Extract a benchmark folder and score vs ground_truth/.")
    pb.add_argument("bench_dir", help="Folder containing images and a ground_truth/ subdir.")
    pb.add_argument("--out", default="benchmark_runs/latest", help="Output directory.")
    pb.add_argument("--model", default=DEFAULT_MODEL)
    pb.add_argument("--concurrency", type=int, default=4)
    pb.add_argument("--force", action="store_true", help="Re-extract even if JSON exists.")
    pb.add_argument("--skip-extract", action="store_true",
                    help="Only score the existing predictions in --out (don't call the API).")
    pb.set_defaults(func=cmd_benchmark)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
