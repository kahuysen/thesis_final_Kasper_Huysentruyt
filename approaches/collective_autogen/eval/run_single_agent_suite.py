"""Run single_agent.py on every benchmark image, then aggregate scores.

Mirrors eval/run_benchmark_suite.py but invokes single_agent.py (one
AssistantAgent with all tools) instead of main.py (six-agent pipeline).

Output: eval/results/single_<ts>/{summary.json, <stem>.log, ...}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate, summarise  # noqa: E402
from eval.usage import add_usage, empty_usage, load_usage_json  # noqa: E402

BENCH_IMAGES = REPO / "eval" / "benchmark" / "images"
RUNS_DIR = REPO / "runs"
RESULTS_DIR = REPO / "eval" / "results"


def _list_images() -> list[Path]:
    if not BENCH_IMAGES.exists():
        return []
    return sorted([p for p in BENCH_IMAGES.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def _run_dir_from_log(log_path: Path) -> Path | None:
    """Parse single_agent.py's stdout for 'Results saved to <path>'. Printed
    unconditionally before exit (even on crash), so it works for crashed runs."""
    if not log_path.exists():
        return None
    try:
        for line in reversed(log_path.read_text().splitlines()):
            if line.startswith("Results saved to "):
                p = Path(line[len("Results saved to "):].strip())
                return p if p.exists() else None
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-model", default="gpt-5.4")
    parser.add_argument("--max-tool-iterations", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--per-image-timeout", type=int, default=2700,
                        help="Hard wall-clock timeout per image in seconds (default 2700 = 45 min).")
    parser.add_argument("--max-validate-retries", type=int, default=2)
    args = parser.parse_args()

    images = _list_images()
    if args.start_from > 0:
        images = images[args.start_from:]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print("No benchmark images found under eval/benchmark/images/", file=sys.stderr)
        return 1

    suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = RESULTS_DIR / f"single_{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== single-agent suite {suite_id} ===")
    print(f"  {len(images)} images, vision={args.vision_model!r}, max_tool_iter={args.max_tool_iterations}")
    print(f"  per-image logs under {suite_dir.relative_to(REPO)}")
    print()

    per_image: list[dict] = []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name}")
        log_path = suite_dir / f"{img.stem}.log"
        cmd = [
            sys.executable, "-u", "single_agent.py", str(img),
            "--vision-model", args.vision_model,
            "--max-validate-retries", str(args.max_validate_retries),
            "--max-tool-iterations", str(args.max_tool_iterations),
        ]

        t0 = time.time()
        timed_out = False
        try:
            with log_path.open("w") as f:
                proc = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO),
                    timeout=args.per_image_timeout,
                )
            rc = proc.returncode
            err = None
        except subprocess.TimeoutExpired:
            rc = -2
            timed_out = True
            err = f"timeout after {args.per_image_timeout}s — killed"
            try:
                subprocess.run(["pkill", "-KILL", "-f", f"single_agent.py {img}"], capture_output=True)
            except Exception:
                pass
        except Exception as exc:
            rc = -1
            err = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0
        run_dir = _run_dir_from_log(log_path)
        entry = {
            "image": img.name,
            "stem": img.stem,
            "rc": rc,
            "elapsed_s": round(elapsed, 1),
            "run_dir": str(run_dir.relative_to(REPO)) if run_dir else None,
            "log": str(log_path.relative_to(REPO)),
        }
        if err:
            entry["error"] = err
        usage = load_usage_json(run_dir / "usage.json") if run_dir else None
        if usage is not None:
            entry["usage"] = usage
        flag = "TIMEOUT" if timed_out else f"rc={rc}"
        print(f"    {flag}  {elapsed:5.0f}s  run={run_dir.name if run_dir else 'NONE'}")
        per_image.append(entry)

    print("\n=== scoring ===")
    gold_dir = REPO / "eval" / "ground_truth"
    scores: list[dict] = []
    for entry in per_image:
        if not entry.get("run_dir"):
            scores.append({"image": entry["image"], "metrics": None, "error": entry.get("error", "no run dir")})
            continue
        run_dir = REPO / entry["run_dir"]
        rj = run_dir / "result.json"
        if not rj.exists():
            scores.append({"image": entry["image"], "metrics": None, "error": "no result.json"})
            continue
        try:
            record = json.loads(rj.read_text())
        except Exception as e:
            scores.append({"image": entry["image"], "metrics": None, "error": f"parse: {e}"})
            continue
        gold_path = gold_dir / f"{entry['stem']}.json"
        gold = None
        if gold_path.exists():
            try:
                gold = json.loads(gold_path.read_text())
            except Exception:
                gold = None
        metrics = evaluate(record, gold)
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
            "ged": _safe(m, "graph_edit_distance", "avg_ged", default=None),
        })

    print("\n=== summary ===")
    head = f"  {'image':45} {'schema':6} {'rxn':5} {'rIoU':>5} {'pIoU':>5} {'cRec':>5} {'sF1':>5} {'hF1':>5} {'GED':>6}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        rxn = f"{r['reaction_count']}/{r['reaction_count_gold']}" if r['reaction_count_gold'] is not None else f"{r['reaction_count']}"
        def _fmt(v, fmt="6.3f", dash="  --  "):
            return f"{v:{fmt}}" if v is not None else dash
        print(f"  {r['image']:45} {('PASS' if r['schema_pass'] else 'FAIL'):6} "
              f"{rxn:>5} {_fmt(r['reactant_iou'], '5.3f', '  -- '):>5} "
              f"{_fmt(r['product_iou'], '5.3f', '  -- '):>5} "
              f"{_fmt(r['condition_recall'], '5.2f', '  -- '):>5} "
              f"{_fmt(r['soft_f1'], '5.3f', '  -- '):>5} "
              f"{_fmt(r['hard_f1'], '5.3f', '  -- '):>5} "
              f"{_fmt(r['ged'], '6.2f', '   -- '):>6}")

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
        "mean_ged": _mean([r["ged"] for r in rows]),
    }
    print("\n  aggregate (mean over scored runs):")
    for k, v in means.items():
        if v is None:
            print(f"    {k:30}: --")
        elif isinstance(v, float):
            print(f"    {k:30}: {v*100:.1f}%" if "rate" in k or "iou" in k.lower() or "recall" in k else f"    {k:30}: {v:.3f}")

    usage_total = empty_usage(model=args.vision_model)
    for e in per_image:
        if e.get("usage"):
            usage_total = add_usage(usage_total, e["usage"])
    elapsed_total_s = round(sum(float(e.get("elapsed_s") or 0) for e in per_image), 1)

    summary = {
        "suite_id": suite_id,
        "kind": "single_agent",
        "vision_model": args.vision_model,
        "max_tool_iterations": args.max_tool_iterations,
        "per_image": per_image,
        "scores": scores,
        "table": rows,
        "means": means,
        "usage_total": usage_total,
        "elapsed_total_s": elapsed_total_s,
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary written to {suite_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
