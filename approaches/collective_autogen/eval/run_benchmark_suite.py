"""Run main.py on every benchmark image, then aggregate scores.

Usage:
    python eval/run_benchmark_suite.py [--vision-model gpt-5.4] [--mini-model MODEL]
                                       [--no-molnextr] [--limit N]

    --mini-model defaults to None, which routes data_structure + verifier to the
    same gpt-5.4 client as the vision specialists. Pass e.g. `--mini-model gpt-5-mini`
    to fall back to the cheaper text-only model.

Each image runs sequentially. Per-image stdout is saved under
eval/results/suite_<ts>/<image_stem>.log. After all runs complete, the script
calls eval/run_eval.py --all and writes a summary JSON to
eval/results/suite_<ts>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate, summarise  # noqa: E402
from eval.usage import add_usage, empty_usage, load_usage_json  # noqa: E402

DEFAULT_BENCH_IMAGES = REPO / "eval" / "benchmark" / "images"
DEFAULT_GOLD_DIR = REPO / "eval" / "ground_truth"
RUNS_DIR = REPO / "runs"
RESULTS_DIR = REPO / "eval" / "results"


def _telegram_notify(text: str) -> None:
    """Best-effort Telegram ping. Reads TELEGRAM_TOKEN / TELEGRAM_CHAT_ID from env.
    Silent no-op if env vars are unset; never raises (suite runs unaffected by network)."""
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(url, data=data, timeout=10).read()
    except Exception:
        pass


def _list_images(images_dir: Path, manifest_path: Path | None = None) -> list[Path]:
    if not images_dir.exists():
        return []
    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if manifest_path is None:
        return all_images
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    allowed = {entry["stem"] for entry in manifest if entry.get("valid", True)}
    filtered = [p for p in all_images if p.stem in allowed]
    missing = sorted(allowed - {p.stem for p in filtered})
    if missing:
        print(f"  WARN: {len(missing)} manifest stems have no image on disk; first few: {missing[:5]}",
              file=sys.stderr)
    return filtered


def _run_dir_from_log(log_path: Path) -> Path | None:
    """Parse main.py's stdout for the 'Results saved to <path>' line. Printed
    unconditionally before exit (even when the run crashes or validation fails),
    so it works for crashed runs whose result.json is empty."""
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


def _latest_run_for(image_stem: str, since: float) -> Path | None:
    """Fallback: find the most recent run dir created after `since` whose
    result.json contains a matching file_name. Used only if the log-based
    lookup fails (e.g. log was truncated). Crashed runs have empty result.json
    so this fallback won't find them — that's why _run_dir_from_log is preferred."""
    candidates = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and p.stat().st_mtime >= since], reverse=True)
    for p in candidates:
        rj = p / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
            fn = data.get("file_name", "")
            if Path(fn).stem == image_stem:
                return p
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-model", default="gpt-5.4")
    parser.add_argument("--mini-model", default=None,
                        help="Optional cheaper text-only model for data_structure + verifier. "
                             "If unset, they share the vision client (gpt-5.4).")
    parser.add_argument("--no-molnextr", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="If >0, run only the first N images.")
    parser.add_argument("--start-from", type=int, default=0, help="Skip the first N images (0-indexed).")
    parser.add_argument("--per-image-timeout", type=int, default=900,
                        help="Hard wall-clock timeout per image in seconds (default 900 = 15 min).")
    parser.add_argument("--max-validate-retries", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to a manifest JSON (e.g. eval/benchmark/manifest_subset_30.json). "
                             "If set, only images whose stem appears in the manifest are run.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_BENCH_IMAGES,
                        help="Directory containing benchmark images (default: eval/benchmark/images).")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR,
                        help="Directory containing ground-truth JSONs (default: eval/ground_truth).")
    args = parser.parse_args()

    if args.manifest is not None:
        args.manifest = args.manifest.resolve()
    args.images_dir = args.images_dir.resolve()
    args.gold_dir = args.gold_dir.resolve()
    images = _list_images(args.images_dir, args.manifest)
    if args.start_from > 0:
        images = images[args.start_from:]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"No benchmark images found under {args.images_dir}", file=sys.stderr)
        return 1

    suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = RESULTS_DIR / f"suite_{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_label = str(args.manifest.relative_to(REPO)) if args.manifest else "(full directory)"
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)
    print(f"=== benchmark suite {suite_id} ===")
    print(f"  {len(images)} images from {manifest_label}, vision={args.vision_model!r}, mini={args.mini_model!r}, molnextr={not args.no_molnextr}")
    print(f"  images_dir = {_rel(args.images_dir)}")
    print(f"  gold_dir   = {_rel(args.gold_dir)}")
    print(f"  per-image logs under {suite_dir.relative_to(REPO)}")
    print()
    _telegram_notify(
        f"🚀 suite_{suite_id} started\n"
        f"{len(images)} images · vision={args.vision_model} · molnextr={not args.no_molnextr}\n"
        f"images_dir={_rel(args.images_dir)}"
    )

    per_image: list[dict] = []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name}")
        log_path = suite_dir / f"{img.stem}.log"
        cmd = [
            sys.executable, "-u", "main.py", str(img),
            "--vision-model", args.vision_model,
            "--max-validate-retries", str(args.max_validate_retries),
        ]
        if args.mini_model:
            cmd.extend(["--mini-model", args.mini_model])
        if not args.no_molnextr:
            cmd.append("--use-molnextr")

        t0 = time.time()
        since = t0
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
            # Make sure no orphaned main.py / subprocess descendants linger.
            try:
                subprocess.run(["pkill", "-KILL", "-f", f"main.py {img}"], capture_output=True)
            except Exception:
                pass
        except Exception as exc:
            rc = -1
            err = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0
        run_dir = _run_dir_from_log(log_path) or _latest_run_for(img.stem, since)
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
        if timed_out:
            icon = "⏰"
            status = f"TIMEOUT after {args.per_image_timeout}s"
        elif rc == 0:
            icon = "✅"
            status = f"rc=0  {elapsed:.0f}s"
        else:
            icon = "❌"
            status = f"rc={rc}  {elapsed:.0f}s"
        _telegram_notify(
            f"{icon} [{i}/{len(images)}] {img.name}\n"
            f"{status}\n"
            f"run={run_dir.name if run_dir else 'NONE'}"
        )

    # Score every successful run.
    print("\n=== scoring ===")
    gold_dir = args.gold_dir
    scores: list[dict] = []
    agg_keys = ("schema_conformance", "smiles_validity_rate", "role_enum_compliance",
                "reactant_iou", "product_iou", "condition_coverage")
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

    # Aggregate.
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

    usage_total = empty_usage(
        model=args.vision_model + (f"+{args.mini_model}" if args.mini_model else "")
    )
    for e in per_image:
        if e.get("usage"):
            usage_total = add_usage(usage_total, e["usage"])
    elapsed_total_s = round(sum(float(e.get("elapsed_s") or 0) for e in per_image), 1)

    summary = {
        "suite_id": suite_id,
        "vision_model": args.vision_model,
        "mini_model": args.mini_model,
        "use_molnextr": not args.no_molnextr,
        "manifest": str(args.manifest.relative_to(REPO)) if args.manifest else None,
        "images_dir": _rel(args.images_dir),
        "gold_dir": _rel(args.gold_dir),
        "n_images": len(images),
        "per_image": per_image,
        "scores": scores,
        "table": rows,
        "means": means,
        "usage_total": usage_total,
        "elapsed_total_s": elapsed_total_s,
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary written to {suite_dir / 'summary.json'}")
    n_ok = sum(1 for e in per_image if e.get("rc") == 0)
    def _fmt_mean(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "--"
    _telegram_notify(
        f"🏁 suite_{suite_id} done\n"
        f"{n_ok}/{len(per_image)} images succeeded\n"
        f"sF1={_fmt_mean(means.get('mean_soft_f1'))}  hF1={_fmt_mean(means.get('mean_hard_f1'))}  GED={_fmt_mean(means.get('mean_ged'))}\n"
        f"summary: {_rel(suite_dir / 'summary.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
