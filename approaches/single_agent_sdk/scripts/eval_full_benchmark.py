"""Score benchmark_full runs with per-slice rules and bootstrap CIs.

Scores one or more run directories (produced by benchmark_openrouter.py etc.)
against `data/benchmark_full/ground_truth/`, using the slice tags from
`data/benchmark_full/manifest.json`.

Scoring rules (frozen before any full-benchmark run):

  1. Every scored image in the manifest counts. A missing or failed
     prediction is scored as an empty extraction (zeros) — never dropped.
  2. Structural metrics (soft/hard/constitution/partial F1, IoUs, GED) are
     computed with conditions stripped from BOTH prediction and gold, so
     slices whose gold has no/partial conditions (gt2, gt3, gt4) are scored
     on the same footing. The with-conditions hard match is additionally
     reported on gt1 (`hard_match_full`) for thesis comparability.
  3. Condition metrics (strict + lenient coverage) are computed only on
     images whose gold has at least one condition entry.
  4. Wildcard rule: if the gold for an image contains no wildcard atoms
     (`*`) in any reaction, predicted reactions containing wildcards are
     dropped before scoring (the model reported an R-group template the GT
     chose not to label — e.g. the gt2 `reaction_template`). If the gold
     itself contains wildcard reactions (gt1 headers, gt4 `0_1` rows), all
     predicted reactions are kept.

Reported groups: all / heldout (scored minus dev16) / dev16, and per-slice
(gt1..gt4) plus gt1_heldout. Bootstrap 95% CIs (10k resamples over images,
seed 42) for the headline macro F1s.

Usage:
    .venv/bin/python3 scripts/eval_full_benchmark.py \
        --runs benchmark_runs/full_qwen38="Qwen3.8-27B" \
               benchmark_runs/full_opus47="Opus 4.7"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # approaches/single_agent_sdk
REPO = ROOT.parent.parent                              # 5.Code_reorg
DATA = REPO / "data" / "benchmark_full"

sys.path.insert(0, str(ROOT / "scripts"))
import eval_via_collective as evc                      # noqa: E402  (bridge + metrics import)
from eval.metrics import evaluate                      # noqa: E402  (path added by evc)

BOOT_N = 10_000
BOOT_SEED = 42
HEADLINE = ["soft_match_f1", "hard_match_f1", "partial_match_f1"]


# ───── per-image preparation ─────

def _has_wildcard(rec: dict) -> bool:
    for r in rec.get("reactions") or []:
        for part in ("reactants", "products"):
            for x in r.get(part) or []:
                if "*" in (x.get("smiles") or ""):
                    return True
    return False


def _drop_wildcard_reactions(rec: dict) -> dict:
    kept = []
    for r in rec.get("reactions") or []:
        smis = [x.get("smiles") or "" for part in ("reactants", "products")
                for x in r.get(part) or []]
        if not any("*" in s for s in smis):
            kept.append(r)
    return {**rec, "reactions": kept}


def _strip_conditions(rec: dict) -> dict:
    return {**rec,
            "reactions": [{**r, "conditions": []} for r in rec.get("reactions") or []]}


def _gold_condition_count(gold: dict) -> int:
    return sum(len(r.get("conditions") or []) for r in gold.get("reactions") or [])


def score_image(pred_record: dict, gold: dict) -> dict:
    """Apply the frozen per-image rules; return the metric dict."""
    if not _has_wildcard(gold):
        pred_record = _drop_wildcard_reactions(pred_record)

    m = evaluate(_strip_conditions(pred_record), _strip_conditions(gold))

    if _gold_condition_count(gold) > 0:
        full = evaluate(pred_record, gold)
        m["condition_coverage"] = full["condition_coverage"]
        try:
            m["condition_coverage_lenient"] = evc.lenient_condition_coverage(
                pred_record, gold)
        except Exception:
            pass
        if gold.get("slice") == "gt1":
            m["hard_match_full"] = full["hard_match"]
    else:
        m.pop("condition_coverage", None)
    return m


# ───── run scoring ─────

def score_run(run_dir: Path, manifest: list[dict]) -> dict[str, dict]:
    per_image: dict[str, dict] = {}
    for row in manifest:
        if not row["scored"]:
            continue
        stem = row["stem"]
        gold = json.loads((DATA / "ground_truth" / f"{stem}.json").read_text())
        pred_path = run_dir / f"{stem}.json"
        if pred_path.exists():
            try:
                fx = json.loads(pred_path.read_text())
                record = evc.figure_extraction_to_record(fx, file_name=row["file_name"])
            except Exception as e:
                print(f"  ! unparseable prediction {stem}: {e} — scored as empty",
                      file=sys.stderr)
                record = {"file_name": row["file_name"], "reactions": []}
        else:
            record = {"file_name": row["file_name"], "reactions": []}
        try:
            m = score_image(record, gold)
        except Exception as e:
            print(f"  ! evaluate failed on {stem}: {e} — scored as empty",
                  file=sys.stderr)
            m = score_image({"file_name": row["file_name"], "reactions": []}, gold)
        m["_slice"] = row["slice"]
        m["_dev16"] = row["dev16"]
        m["_predicted"] = pred_path.exists()
        per_image[stem] = m
    return per_image


# ───── groups, aggregation, bootstrap ─────

def groups_of(per_image: dict[str, dict]) -> dict[str, dict[str, dict]]:
    def sel(pred):
        return {k: v for k, v in per_image.items() if pred(v)}
    g = {
        "all":     per_image,
        "heldout": sel(lambda m: not m["_dev16"]),
        "dev16":   sel(lambda m: m["_dev16"]),
    }
    for s in ("gt1", "gt2", "gt3", "gt4"):
        g[s] = sel(lambda m, s=s: m["_slice"] == s)
    g["gt1_heldout"] = sel(lambda m: m["_slice"] == "gt1" and not m["_dev16"])
    return g


def bootstrap_ci(values: list[float], n: int = BOOT_N, seed: int = BOOT_SEED):
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def aggregate_group(imgs: dict[str, dict]) -> dict:
    if not imgs:
        return {}
    agg = evc.aggregate(imgs)
    agg["n_missing_predictions"] = sum(1 for m in imgs.values() if not m["_predicted"])
    agg["hard_match_full_f1"] = (
        mean(v) if (v := [m["hard_match_full"]["f1"] for m in imgs.values()
                          if "hard_match_full" in m]) else None)
    parent = {"soft_match_f1": "soft_match", "hard_match_f1": "hard_match",
              "partial_match_f1": "partial_match"}
    agg["ci95"] = {}
    for key in HEADLINE:
        vals = [m[parent[key]]["f1"] for m in imgs.values() if parent[key] in m]
        agg["ci95"][key] = bootstrap_ci(vals)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run_dir=Label pairs, e.g. benchmark_runs/full_qwen38=Qwen3.8-27B")
    ap.add_argument("--out-suffix", default="full_eval.json",
                    help="written inside each run dir")
    args = ap.parse_args()

    manifest = json.loads((DATA / "manifest.json").read_text())

    for spec in args.runs:
        run_str, _, label = spec.partition("=")
        run_dir = (ROOT / run_str) if not Path(run_str).is_absolute() else Path(run_str)
        label = label or run_dir.name
        print(f"\n══ {label}  ({run_dir}) ══")
        per_image = score_run(run_dir, manifest)
        report = {"label": label, "run_dir": str(run_dir),
                  "groups": {}, "per_image": per_image}
        for gname, imgs in groups_of(per_image).items():
            report["groups"][gname] = aggregate_group(imgs)
        out_path = run_dir / args.out_suffix
        out_path.write_text(json.dumps(report, indent=1, default=float))
        for gname in ("heldout", "dev16", "gt1", "gt2", "gt3", "gt4"):
            a = report["groups"].get(gname) or {}
            if not a:
                continue
            ci = (a.get("ci95") or {}).get("partial_match_f1")
            ci_s = f"  CI95={ci}" if ci else ""
            print(f"  {gname:12s} n={a['n_images']:3d}  "
                  f"soft={a.get('soft_match_f1'):.3f}  "
                  f"hard={a.get('hard_match_f1'):.3f}  "
                  f"partial={a.get('partial_match_f1'):.3f}{ci_s}  "
                  f"missing={a.get('n_missing_predictions')}")
        print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
