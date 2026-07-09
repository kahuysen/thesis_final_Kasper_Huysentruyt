"""Re-score every suite under `eval/results/suite_*` against the current
gold (ChemEagle-derived as of Step B). Writes a `summary_rescored.json`
alongside each existing summary, and prints a phase-by-phase trajectory
table on stdout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate  # noqa: E402

SUITES_DIR = REPO / "eval" / "results"
GOLD_DIR = REPO / "eval" / "ground_truth"
RUNS_DIR = REPO / "runs"


def _safe(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def rescore_suite(suite_dir: Path) -> dict:
    summary_path = suite_dir / "summary.json"
    if not summary_path.exists():
        return {"suite": suite_dir.name, "error": "no summary.json"}
    summary = json.loads(summary_path.read_text())
    rows = []
    for entry in summary.get("per_image", []):
        if not entry.get("run_dir"):
            rows.append({"image": entry["image"], "skipped": "no run_dir"})
            continue
        run_dir = REPO / entry["run_dir"]
        rj = run_dir / "result.json"
        if not rj.exists() or rj.stat().st_size == 0:
            rows.append({"image": entry["image"], "skipped": "no result.json"})
            continue
        try:
            record = json.loads(rj.read_text())
        except Exception as e:
            rows.append({"image": entry["image"], "skipped": f"parse: {e}"})
            continue
        gold_path = GOLD_DIR / f"{Path(entry['image']).stem}.json"
        gold = json.loads(gold_path.read_text()) if gold_path.exists() else None
        m = evaluate(record, gold)
        rows.append({
            "image": entry["image"],
            "schema_pass": _safe(m, "schema_conformance", "valid", default=False),
            "reaction_count": _safe(m, "reaction_count", "count", default=0),
            "reaction_count_gold": _safe(m, "reaction_count_gold", "count", default=None),
            "reactant_iou": _safe(m, "reactant_iou", "iou", default=None),
            "product_iou": _safe(m, "product_iou", "iou", default=None),
            "condition_recall": _safe(m, "condition_coverage", "recall", default=None),
            "soft_f1": _safe(m, "soft_match", "f1", default=None),
            "hard_f1": _safe(m, "hard_match", "f1", default=None),
            "ged": _safe(m, "graph_edit_distance", "avg_ged", default=None),
        })

    means = {
        "schema_pass_rate": sum(1 for r in rows if r.get("schema_pass")) / max(1, len(rows)),
        "mean_reactant_iou": _mean([r.get("reactant_iou") for r in rows]),
        "mean_product_iou": _mean([r.get("product_iou") for r in rows]),
        "mean_condition_recall": _mean([r.get("condition_recall") for r in rows]),
        "mean_soft_f1": _mean([r.get("soft_f1") for r in rows]),
        "mean_hard_f1": _mean([r.get("hard_f1") for r in rows]),
        "mean_ged": _mean([r.get("ged") for r in rows]),
    }
    out = {"suite": suite_dir.name, "rows": rows, "means": means}
    (suite_dir / "summary_rescored.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> int:
    suites = sorted(p for p in SUITES_DIR.glob("suite_*") if (p / "summary.json").exists())
    if not suites:
        print("no suites found", file=sys.stderr)
        return 2

    print(f"\nRe-scored {len(suites)} suites against current ground truth (ChemEagle-derived).\n")
    print(f"  {'suite':30}  {'sPass':>5}  {'rIoU':>5}  {'pIoU':>5}  {'cRec':>5}  {'sF1':>5}  {'hF1':>5}  {'GED':>5}")
    print("  " + "-" * 80)
    for s in suites:
        result = rescore_suite(s)
        means = result["means"]
        def f(v, w=5, p=3):
            return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else "  -- "
        print(f"  {result['suite']:30}  "
              f"{f(means['schema_pass_rate'], 5, 2):>5}  "
              f"{f(means['mean_reactant_iou']):>5}  "
              f"{f(means['mean_product_iou']):>5}  "
              f"{f(means['mean_condition_recall'], 5, 2):>5}  "
              f"{f(means['mean_soft_f1']):>5}  "
              f"{f(means['mean_hard_f1']):>5}  "
              f"{f(means['mean_ged'], 5, 1):>5}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
