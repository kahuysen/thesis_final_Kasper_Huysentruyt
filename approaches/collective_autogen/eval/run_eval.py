"""Run evaluation metrics against ReactionRecord runs.

Modes:
    # 1. Score a specific run dir against its ground-truth file
    python eval/run_eval.py --run runs/20260426_001601_selector

    # 2. Score the latest run in `runs/` against ground truth
    python eval/run_eval.py --latest

    # 3. Score all runs in `runs/` and print a comparison table
    python eval/run_eval.py --all

    # 4. Score a specific result.json against a specific gold file
    python eval/run_eval.py --result path/to/result.json --gold path/to/gold.json

The `--baseline-from` flag pins a baseline run for regression checking; the
exit code is non-zero if the new run regresses on key metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate, summarise  # noqa: E402

RUNS_DIR = REPO / "runs"
GOLD_DIR = REPO / "eval" / "ground_truth"
RESULTS_DIR = REPO / "eval" / "results"


def _load_record(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARN could not parse {path}: {exc}", file=sys.stderr)
        return None


def _gold_for(record: dict) -> dict | None:
    """Find the ground-truth file matching a record's `file_name`."""
    fn = record.get("file_name") or ""
    stem = Path(fn).stem
    if not stem:
        return None
    candidate = GOLD_DIR / f"{stem}.json"
    if candidate.exists():
        return _load_record(candidate)
    return None


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "result.json").exists()])


def _score_run(run_dir: Path, baseline: dict | None = None) -> dict:
    result_path = run_dir / "result.json"
    record = _load_record(result_path)
    if record is None:
        return {"run": run_dir.name, "error": "could not parse result.json", "metrics": None}
    gold = _gold_for(record)
    metrics = evaluate(record, gold)
    out = {"run": run_dir.name, "image": record.get("file_name"), "has_gold": gold is not None, "metrics": metrics}
    if baseline:
        out["regressions"] = _detect_regressions(metrics, baseline)
    return out


# Keys we treat as regression-sensitive. All are higher-is-better numbers in [0, 1].
_REG_KEYS = (
    ("schema_conformance", "valid"),
    ("smiles_validity_rate", "rate"),
    ("role_enum_compliance", "rate"),
    ("reactant_iou", "iou"),
    ("product_iou", "iou"),
    ("condition_coverage", "recall"),
)


def _detect_regressions(new: dict, baseline: dict, tol: float = 1e-9) -> list[str]:
    regs: list[str] = []
    for parent, child in _REG_KEYS:
        if parent not in new or parent not in baseline:
            continue
        nv = new[parent].get(child)
        bv = baseline[parent].get(child)
        if isinstance(nv, bool):
            if bv and not nv:
                regs.append(f"{parent}.{child}: {bv} -> {nv}")
        elif isinstance(nv, (int, float)) and isinstance(bv, (int, float)):
            if nv + tol < bv:
                regs.append(f"{parent}.{child}: {bv:.3f} -> {nv:.3f}")
    return regs


def _print_one(scored: dict) -> None:
    print(f"\n=== {scored['run']}  (image: {scored.get('image')})  has_gold={scored.get('has_gold')} ===")
    if scored.get("error"):
        print(f"  ERROR: {scored['error']}")
        return
    print(summarise(scored["metrics"]))
    if scored.get("regressions"):
        print("  REGRESSIONS vs baseline:")
        for r in scored["regressions"]:
            print(f"    - {r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--run", type=Path, help="Score this run dir.")
    g.add_argument("--latest", action="store_true", help="Score the most recent run in runs/.")
    g.add_argument("--all", action="store_true", help="Score every run in runs/.")
    parser.add_argument("--result", type=Path, help="Path to a result.json (use with --gold).")
    parser.add_argument("--gold", type=Path, help="Path to a ground-truth JSON.")
    parser.add_argument("--baseline-from", type=Path, help="Compare against this baseline run dir.")
    parser.add_argument("--save", action="store_true", help="Write a JSON results file under eval/results/.")
    args = parser.parse_args()

    baseline_metrics: dict | None = None
    if args.baseline_from:
        b = _score_run(args.baseline_from)
        baseline_metrics = b.get("metrics")

    scored: list[dict] = []
    if args.result:
        record = _load_record(args.result)
        gold = _load_record(args.gold) if args.gold else (_gold_for(record) if record else None)
        if record is None:
            print(f"could not parse {args.result}", file=sys.stderr)
            return 2
        metrics = evaluate(record, gold)
        s = {"run": args.result.parent.name, "image": record.get("file_name"), "has_gold": gold is not None, "metrics": metrics}
        if baseline_metrics:
            s["regressions"] = _detect_regressions(metrics, baseline_metrics)
        scored = [s]
    elif args.run:
        scored = [_score_run(args.run, baseline_metrics)]
    elif args.latest:
        runs = _list_runs()
        if not runs:
            print("no runs found", file=sys.stderr)
            return 2
        scored = [_score_run(runs[-1], baseline_metrics)]
    elif args.all:
        scored = [_score_run(r, baseline_metrics) for r in _list_runs()]
    else:
        parser.error("pass one of --run / --latest / --all / --result")

    for s in scored:
        _print_one(s)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
        print(f"\nwrote {out_file}")

    # Exit non-zero on any regression.
    if any(s.get("regressions") for s in scored):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
