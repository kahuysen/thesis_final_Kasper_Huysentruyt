"""Run Maarten's ChemEagle_Hybrid pipeline on every benchmark image, with the
Gemini orchestrate stage swapped for an Azure-OpenAI call (deployment is
configurable; default = gpt-5-mini).

Pipeline per image:
  1. molnextr + rxnim + coref via the user's existing ../ChemEagle/.venv-chemeagle
     install (same toolkit the ChemEagle suite uses) -> evidence packet.
  2. One chat-completion call to the Azure deployment using Maarten's
     prompt_hybrid_orchestrate.txt as the system prompt + the evidence + image.
  3. Conditional stage-2 variant-table re-prompt when the heuristic fires.
  4. Maarten's _postprocess (wildcard renumber + best-effort smiles_fix).
  5. chemeagle_to_our_schema (reused from run_chemeagle_suite) -> our schema.
  6. eval/metrics.evaluate against the gold-set JSONs.

Output: eval/results/chemeagle_hybrid_<ts>/{summary.json,
        <stem>.json (Maarten native), <stem>.converted.json (our schema),
        run.log}

Usage:
  python eval/run_chemeagle_hybrid_suite.py \
      --images-dir eval/Benchmark_kasper_GT3_Maarten \
      --gold-dir eval/Benchmark_kasper_GT3_Maarten/ground_truth \
      --manifest eval/Benchmark_kasper_GT3_Maarten/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate, summarise  # noqa: E402
from eval.run_chemeagle_suite import chemeagle_to_our_schema  # noqa: E402
from eval.usage import add_usage, empty_usage, load_usage_json  # noqa: E402

DEFAULT_IMAGES = REPO / "eval" / "Benchmark_kasper_GT3_Maarten"
DEFAULT_GOLD = REPO / "eval" / "Benchmark_kasper_GT3_Maarten" / "ground_truth"
DEFAULT_MANIFEST = REPO / "eval" / "Benchmark_kasper_GT3_Maarten" / "manifest.json"
RESULTS_DIR = REPO / "eval" / "results"
CHEMEAGLE_DIR = REPO.parent / "ChemEagle"
CHEMEAGLE_PY = CHEMEAGLE_DIR / ".venv-chemeagle" / "bin" / "python"
WORKER = REPO / "eval" / "_run_hybrid_one.py"


def _list_images(images_dir: Path, manifest_path: Path | None) -> list[Path]:
    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if manifest_path is None or not manifest_path.exists():
        return all_images
    manifest = json.loads(manifest_path.read_text())
    allowed = {entry["stem"] for entry in manifest if entry.get("valid", True)}
    return [p for p in all_images if p.stem in allowed]


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--deployment", default="gpt-5-mini",
                        help="Azure deployment name to use for orchestrate "
                             "(default gpt-5-mini; switch to gpt-5.4 if vision "
                             "is unsupported).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=3600,
                        help="Hard wall-clock timeout for the whole batch "
                             "(default 3600 = 1 h).")
    args = parser.parse_args()
    args.images_dir = args.images_dir.resolve()
    args.gold_dir = args.gold_dir.resolve()
    if args.manifest:
        args.manifest = args.manifest.resolve()

    if not CHEMEAGLE_PY.exists():
        print(f"[ERR] ChemEagle venv not found at {CHEMEAGLE_PY}", file=sys.stderr)
        return 1
    if not WORKER.exists():
        print(f"[ERR] worker not found at {WORKER}", file=sys.stderr)
        return 1

    images = _list_images(args.images_dir, args.manifest)
    if args.start_from > 0:
        images = images[args.start_from:]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"[ERR] no images in {args.images_dir}", file=sys.stderr)
        return 1

    suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = RESULTS_DIR / f"chemeagle_hybrid_{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    log_path = suite_dir / "run.log"

    tasks = {
        "deployment": args.deployment,
        "images": [
            {"image": str(p), "out": str(suite_dir / f"{p.stem}.json")}
            for p in images
        ],
    }
    tasks_path = suite_dir / "tasks.json"
    tasks_path.write_text(json.dumps(tasks, indent=2))

    # Pass through Azure env vars; map common alternative names.
    env = os.environ.copy()
    if not env.get("AZURE_OPENAI_API_KEY"):
        print("[ERR] AZURE_OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    if not env.get("AZURE_OPENAI_ENDPOINT"):
        print("[ERR] AZURE_OPENAI_ENDPOINT not set", file=sys.stderr)
        return 2
    env.setdefault("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    # ChemEagle's molnextr reads these globals at import time. Mirror the
    # mapping used by run_chemeagle_suite.py.
    env.setdefault("API_KEY", env["AZURE_OPENAI_API_KEY"])
    env.setdefault("AZURE_ENDPOINT",
                   env["AZURE_OPENAI_ENDPOINT"].strip().strip('"').rstrip("/"))
    env.setdefault("API_VERSION", env["AZURE_OPENAI_API_VERSION"])

    print(f"=== ChemEagle_Hybrid suite {suite_id} ===")
    print(f"  deployment = {args.deployment}")
    print(f"  images     = {len(images)}")
    print(f"  images_dir = {_rel(args.images_dir)}")
    print(f"  gold_dir   = {_rel(args.gold_dir)}")
    print(f"  output     = {_rel(suite_dir)}")
    print(f"  log        = {_rel(log_path)}")
    print()

    # Run worker. Tee stdout/stderr to log AND console.
    cmd = [str(CHEMEAGLE_PY), str(WORKER), str(tasks_path)]
    t0 = time.time()
    timed_out = False
    rc: int | None = None
    try:
        with log_path.open("w") as logf:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(CHEMEAGLE_DIR), env=env, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
            try:
                proc.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
            rc = proc.returncode
    except Exception as exc:
        print(f"[ERR] worker invocation failed: {exc}", file=sys.stderr)
        rc = -1
    elapsed = time.time() - t0
    print(f"\n[worker] rc={rc} timed_out={timed_out} elapsed={elapsed:.1f}s")

    print("\n=== converting + scoring ===")
    rows: list[dict] = []
    scores: list[dict] = []
    per_image: list[dict] = []

    for img in images:
        native_path = suite_dir / f"{img.stem}.json"
        if not native_path.exists():
            scores.append({"image": img.name, "metrics": None, "error": "no output"})
            per_image.append({"image": img.name, "stem": img.stem, "ok": False,
                              "error": "no output"})
            continue
        try:
            native = json.loads(native_path.read_text())
        except Exception as e:
            scores.append({"image": img.name, "metrics": None, "error": str(e)})
            per_image.append({"image": img.name, "stem": img.stem, "ok": False,
                              "error": str(e)})
            continue
        if "error" in native and "reactions" not in native:
            scores.append({"image": img.name, "metrics": None, "error": native["error"]})
            per_image.append({"image": img.name, "stem": img.stem, "ok": False,
                              "error": native["error"]})
            continue
        converted = chemeagle_to_our_schema(native, img.name)
        conv_path = suite_dir / f"{img.stem}.converted.json"
        conv_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))

        gold_path = args.gold_dir / f"{img.stem}.json"
        gold = None
        if gold_path.exists():
            try:
                gold = json.loads(gold_path.read_text())
            except Exception:
                gold = None
        metrics = evaluate(converted, gold)
        scores.append({"image": img.name, "metrics": metrics, "has_gold": gold is not None})
        usage = load_usage_json(suite_dir / f"{img.stem}.usage.json")
        per_image.append({"image": img.name, "stem": img.stem, "ok": True,
                          "stage2_applied": bool(native.get("_stage2_applied")),
                          "image_kind": native.get("image_kind"),
                          "n_reactions": len(native.get("reactions") or []),
                          **({"usage": usage} if usage else {})})
        print(f"\n[{img.name}]")
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
            "constitution_f1": _safe(m, "constitution_match", "f1", default=None),
            "partial_f1": _safe(m, "partial_match", "f1", default=None),
            "partial_mean_jaccard": _safe(m, "partial_match", "mean_jaccard", default=None),
            "ged": _safe(m, "graph_edit_distance", "avg_ged", default=None),
        })

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
        "mean_constitution_f1": _mean([r["constitution_f1"] for r in rows]),
        "mean_partial_f1": _mean([r["partial_f1"] for r in rows]),
        "mean_partial_jaccard": _mean([r["partial_mean_jaccard"] for r in rows]),
        "mean_ged": _mean([r["ged"] for r in rows]),
    }

    usage_total = empty_usage(model=args.deployment)
    for e in per_image:
        if e.get("usage"):
            usage_total = add_usage(usage_total, e["usage"])

    summary = {
        "suite_id": suite_id,
        "kind": "chemeagle_hybrid",
        "deployment": args.deployment,
        "images_dir": _rel(args.images_dir),
        "gold_dir": _rel(args.gold_dir),
        "manifest": _rel(args.manifest) if args.manifest else None,
        "n_images": len(images),
        "worker_rc": rc,
        "worker_timed_out": timed_out,
        "elapsed_s": round(elapsed, 1),
        "per_image": per_image,
        "scores": scores,
        "table": rows,
        "means": means,
        "usage_total": usage_total,
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== aggregate (mean over scored runs) ===")
    for k, v in means.items():
        print(f"  {k:30s}: {v if v is not None else '--'}")
    print(f"\nsummary: {_rel(suite_dir / 'summary.json')}")
    return 0 if rc == 0 and not timed_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
