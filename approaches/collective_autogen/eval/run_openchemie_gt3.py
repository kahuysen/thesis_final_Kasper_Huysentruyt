"""Score OpenChemIE on the Benchmark_kasper_GT3_Maarten 16-image set and
compare side-by-side with the canonical multi-agent run (suite_20260429_123624).

Adapted from compare_openchemie.py — different manifest shape (only `stem` /
`valid`) and benchmark-local images + ground_truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate  # noqa: E402
from tools.openchemie_tool import extract_reactions  # noqa: E402

BENCH_DIR = REPO / "eval" / "Benchmark_kasper_GT3_Maarten"
MANIFEST = BENCH_DIR / "manifest.json"
GROUND_TRUTH = BENCH_DIR / "ground_truth"
DEFAULT_PIPELINE_SUITE = REPO / "eval" / "results" / "suite_20260429_123624"
OUT_PATH = REPO / "eval" / "results" / "openchemie_gt3.json"

ALLOWED_ROLES = {"reagent", "solvent", "catalyst", "temperature", "time",
                 "yield", "atmosphere", "loading", "ee", "dr", "pressure"}


def _compound(c: dict) -> dict:
    s = c.get("smiles")
    if s in ("<invalid>", ""):
        s = None
    return {"smiles": s, "label": c.get("label") or None, "uncertain": False}


def _classify_condition_text(text: str) -> str:
    t = text.lower()
    if "yield" in t or ("%" in t and any(k in t for k in ("yield", "isolat"))):
        return "yield"
    if any(k in t for k in ("°c", "* c", " c,", " c.", " r.t", "rt,", " rt", "room temp", "reflux")):
        return "temperature"
    if any(k in t for k in (" h ", " h,", " h.", " min", " hour", " day", " week")):
        return "time"
    if "atm" in t or "n2" in t or "ar," in t or " air" in t:
        return "atmosphere"
    return "reagent"


def _condition(c: dict) -> dict | None:
    text_parts = c.get("text")
    if isinstance(text_parts, list):
        text = " ".join(str(t).strip() for t in text_parts if t).strip()
    else:
        text = str(text_parts or "").strip()
    if not text:
        return None
    cat = (c.get("category") or "").lower().strip()
    role = cat if cat in ALLOWED_ROLES else _classify_condition_text(text)
    smiles = c.get("smiles")
    if smiles in ("<invalid>", ""):
        smiles = None
    return {"role": role, "text": text, "smiles": smiles}


def to_record(image_name: str, oc_result: dict) -> dict:
    rxns = []
    for i, r in enumerate(oc_result.get("reactions") or [], start=1):
        rxns.append({
            "reaction_id": f"0_{i}",
            "reactants": [_compound(c) for c in (r.get("reactants") or [])],
            "products": [_compound(c) for c in (r.get("products") or [])],
            "conditions": [c for c in (_condition(x) for x in (r.get("conditions") or [])) if c],
            "additional_info": [],
        })
    return {"file_name": image_name, "reactions": rxns}


def _pipeline_per_image(suite_dir: Path) -> dict[str, dict]:
    """Map stem -> flattened per-image metrics from the multi-agent suite."""
    summary = suite_dir / "summary.json"
    if not summary.exists():
        return {}
    data = json.loads(summary.read_text())
    out: dict[str, dict] = {}
    for s in data.get("scores", []):
        stem = Path(s.get("image", "")).stem
        if not stem:
            continue
        m = s.get("metrics") or {}
        out[stem] = {
            "rIoU": m.get("reactant_iou", {}).get("iou"),
            "pIoU": m.get("product_iou", {}).get("iou"),
            "sF1": m.get("soft_match", {}).get("f1"),
            "hF1": m.get("hard_match", {}).get("f1"),
            "partial_F1": m.get("partial_match", {}).get("f1"),
            "GED": m.get("graph_edit_distance", {}).get("avg_ged"),
            "smiles_validity": m.get("smiles_validity_rate", {}).get("rate"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    ap.add_argument("--gold-dir", type=Path, default=GROUND_TRUTH)
    ap.add_argument("--suite", type=Path, default=DEFAULT_PIPELINE_SUITE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--molscribe-coref", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    pipeline_per_img = _pipeline_per_image(args.suite.resolve())

    rows = []
    for entry in manifest:
        if not entry.get("valid"):
            continue
        stem = entry["stem"]
        image_path = args.bench_dir / f"{stem}.png"
        gold_path = args.gold_dir / f"{stem}.json"
        if not image_path.exists() or not gold_path.exists():
            rows.append({"stem": stem, "skipped": "missing image or gold",
                         "openchemie": None, "pipeline": pipeline_per_img.get(stem)})
            print(f"[SKIP] {stem} (missing inputs)")
            continue

        gold = json.loads(gold_path.read_text())
        t0 = time.time()
        oc = extract_reactions(str(image_path),
                               molscribe_coref=args.molscribe_coref,
                               device=args.device)
        elapsed = time.time() - t0
        if oc.get("error"):
            print(f"[ERR ] {stem}: {oc['error'][:160]}")
            rows.append({"stem": stem, "openchemie_error": oc["error"],
                         "pipeline": pipeline_per_img.get(stem)})
            continue

        record = to_record(f"{stem}.png", oc)
        m = evaluate(record, gold)

        rows.append({
            "stem": stem,
            "elapsed_s": round(elapsed, 1),
            "n_reactions_pred": len(record["reactions"]),
            "n_reactions_gold": m.get("reaction_count_gold", {}).get("count"),
            "openchemie": {
                "rIoU": m.get("reactant_iou", {}).get("iou"),
                "pIoU": m.get("product_iou", {}).get("iou"),
                "sF1": m.get("soft_match", {}).get("f1"),
                "hF1": m.get("hard_match", {}).get("f1"),
                "partial_F1": m.get("partial_match", {}).get("f1"),
                "GED": m.get("graph_edit_distance", {}).get("avg_ged"),
                "smiles_validity": m.get("smiles_validity_rate", {}).get("rate"),
                "schema_ok": m.get("schema_conformance", {}).get("valid"),
            },
            "pipeline": pipeline_per_img.get(stem),
            "openchemie_cached": oc.get("cached", False),
        })
        print(f"[OK  ] {stem}  rxnP={len(record['reactions'])}  "
              f"sF1={rows[-1]['openchemie']['sF1']}  pIoU={rows[-1]['openchemie']['pIoU']}  "
              f"({elapsed:.0f}s, cached={oc.get('cached', False)})")

    # Aggregate
    def _mean(seq, src, key):
        vals = [r[src][key] for r in seq
                if r.get(src) is not None and isinstance(r[src].get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    common_keys = ("rIoU", "pIoU", "sF1", "hF1", "partial_F1", "GED", "smiles_validity")
    agg = {
        "n_images": len(rows),
        "openchemie_mean": {k: _mean(rows, "openchemie", k) for k in common_keys},
        "pipeline_mean": {k: _mean(rows, "pipeline", k) for k in common_keys},
        "pipeline_suite": args.suite.name,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"aggregate": agg, "per_image": rows},
                                   indent=2, ensure_ascii=False))

    def f(v, p=3):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  -  "

    print()
    hdr = (f"{'stem':42s} {'rP':>3s} {'rG':>3s} | "
           f"{'oc_sF1':>6s} {'oc_hF1':>6s} {'oc_GED':>6s} {'oc_pIoU':>7s} | "
           f"{'pp_sF1':>6s} {'pp_hF1':>6s} {'pp_GED':>6s} {'pp_pIoU':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        oc = r.get("openchemie") or {}
        pp = r.get("pipeline") or {}
        print(f"{r['stem'][:42]:42s} "
              f"{str(r.get('n_reactions_pred','-')):>3s} {str(r.get('n_reactions_gold','-')):>3s} | "
              f"{f(oc.get('sF1')):>6s} {f(oc.get('hF1')):>6s} {f(oc.get('GED'),1):>6s} {f(oc.get('pIoU')):>7s} | "
              f"{f(pp.get('sF1')):>6s} {f(pp.get('hF1')):>6s} "
              f"{f(pp.get('GED'),1):>6s} {f(pp.get('pIoU')):>7s}")
    print("-" * len(hdr))
    a_oc = agg["openchemie_mean"]
    a_pp = agg["pipeline_mean"]
    print(f"{'MEAN':42s} {'':>3s} {'':>3s} | "
          f"{f(a_oc['sF1']):>6s} {f(a_oc['hF1']):>6s} {f(a_oc['GED'],1):>6s} {f(a_oc['pIoU']):>7s} | "
          f"{f(a_pp['sF1']):>6s} {f(a_pp['hF1']):>6s} "
          f"{f(a_pp['GED'],1):>6s} {f(a_pp['pIoU']):>7s}")
    print(f"\nSaved: {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
