"""Score OpenChemIE.extract_reactions_from_figures on the 30-image subset and
compare against the agent pipeline (latest suite that has a summary.json).

Pipeline:
  1. Read eval/benchmark/manifest_subset_30.json
  2. For each image, call tools.openchemie_tool.extract_reactions (cache-backed)
  3. Convert OpenChemIE output → ReactionRecord shape
  4. Score against eval/ground_truth/{stem}.json via eval.metrics.evaluate
  5. Side-by-side with the chosen suite's per-image rescored summary

Output: eval/results/openchemie_vs_pipeline.json + a console table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO))

from eval.metrics import evaluate  # noqa: E402
from tools.openchemie_tool import extract_reactions  # noqa: E402

GROUND_TRUTH = REPO / "eval" / "ground_truth"
MANIFEST = REPO / "eval" / "benchmark" / "manifest_subset_30.json"
DEFAULT_SUITE = REPO / "eval" / "results" / "suite_20260428_145233"
OUT_PATH = REPO / "eval" / "results" / "openchemie_vs_pipeline.json"

ALLOWED_ROLES = {"reagent", "solvent", "catalyst", "temperature", "time",
                 "yield", "atmosphere", "loading", "ee", "dr", "pressure"}


def _compound(c: dict) -> dict:
    """OpenChemIE emits the literal string '<invalid>' for SMILES it can't parse.
    Treat that as missing so set-IoU isn't polluted with sentinel tokens."""
    s = c.get("smiles")
    if s in ("<invalid>", ""):
        s = None
    return {"smiles": s, "label": c.get("label") or None, "uncertain": False}


def _classify_condition_text(text: str) -> str:
    """OpenChemIE doesn't classify condition text into roles — `category` is
    layout-related (`[Txt]`, `[Mol]`). Best-effort role detection from the text
    so hard-match-F1 isn't pure noise."""
    t = text.lower()
    if "yield" in t or "%" in t and any(k in t for k in ("yield", "isolat")):
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
    if cat in ALLOWED_ROLES:
        role = cat
    else:
        role = _classify_condition_text(text)
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


def _suite_per_image(suite_dir: Path) -> dict[str, dict]:
    """Return {stem: per_image_metrics} from rescored summary, or empty if missing."""
    rescored = suite_dir / "summary_rescored.json"
    if not rescored.exists():
        return {}
    data = json.loads(rescored.read_text())
    out: dict[str, dict] = {}
    for row in data.get("rows", []):
        if row.get("skipped"):
            continue
        img = row.get("image") or ""
        stem = Path(img).stem
        out[stem] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE,
                        help="Suite dir with summary_rescored.json to compare against.")
    parser.add_argument("--device", default="cpu",
                        help="Device for OpenChemIE: cpu | mps | cuda. Default cpu.")
    parser.add_argument("--molscribe-coref", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())
    pipeline_per_img = _suite_per_image(args.suite.resolve())

    rows = []
    for entry in manifest:
        if not entry.get("valid"):
            continue
        stem = entry["stem"]
        image_path = REPO / entry["image"]
        gold_path = GROUND_TRUTH / f"{stem}.json"
        if not image_path.exists() or not gold_path.exists():
            rows.append({"stem": stem, "openchemie": None, "pipeline": pipeline_per_img.get(stem),
                         "skipped": "missing image or gold"})
            continue
        gold = json.loads(gold_path.read_text())

        oc = extract_reactions(str(image_path), molscribe_coref=args.molscribe_coref,
                               device=args.device)
        if oc.get("error"):
            rows.append({"stem": stem, "openchemie_error": oc["error"],
                         "pipeline": pipeline_per_img.get(stem)})
            continue

        record = to_record(entry.get("image", stem), oc)
        m = evaluate(record, gold)

        rows.append({
            "stem": stem,
            "n_reactions_pred": len(record["reactions"]),
            "n_reactions_gold": m.get("reaction_count_gold", {}).get("count"),
            "openchemie": {
                "rIoU": m.get("reactant_iou", {}).get("iou"),
                "pIoU": m.get("product_iou", {}).get("iou"),
                "sF1": m.get("soft_match_f1", {}).get("f1"),
                "hF1": m.get("hard_match_f1", {}).get("f1"),
                "GED": m.get("graph_edit_distance", {}).get("avg_ged"),
                "smiles_validity": m.get("smiles_validity_rate", {}).get("rate"),
            },
            "pipeline": pipeline_per_img.get(stem),
            "openchemie_cached": oc.get("cached", False),
        })

    # Aggregate.
    def _mean(seq, src, key):
        vals = [r[src][key] for r in seq
                if r.get(src) is not None and isinstance(r[src].get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    agg = {
        "n_images": len(rows),
        "openchemie_mean": {
            k: _mean(rows, "openchemie", k)
            for k in ("rIoU", "pIoU", "sF1", "hF1", "GED", "smiles_validity")
        },
        "pipeline_mean": {
            k: _mean(rows, "pipeline", k)
            for k in ("reactant_iou", "product_iou", "soft_match_f1", "hard_match_f1", "graph_edit_distance")
        },
        "suite": args.suite.name,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"aggregate": agg, "per_image": rows}, indent=2,
                                   ensure_ascii=False))

    # Console table.
    def f(v, p=3):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  -  "

    hdr = f"{'stem':38s} {'rxnP':>4s} {'rxnG':>4s} | {'oc_sF1':>6s} {'oc_hF1':>6s} {'oc_GED':>6s} {'oc_pIoU':>7s} | {'pp_sF1':>6s} {'pp_hF1':>6s} {'pp_GED':>6s} {'pp_pIoU':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        oc = r.get("openchemie") or {}
        pp = r.get("pipeline") or {}
        print(f"{r['stem'][:38]:38s} {str(r.get('n_reactions_pred','-')):>4s} {str(r.get('n_reactions_gold','-')):>4s} "
              f"| {f(oc.get('sF1')):>6s} {f(oc.get('hF1')):>6s} {f(oc.get('GED'),1):>6s} {f(oc.get('pIoU')):>7s} "
              f"| {f(pp.get('soft_match_f1')):>6s} {f(pp.get('hard_match_f1')):>6s} {f(pp.get('graph_edit_distance'),1):>6s} {f(pp.get('product_iou')):>7s}")
    print("-" * len(hdr))
    a_oc = agg["openchemie_mean"]
    a_pp = agg["pipeline_mean"]
    print(f"{'MEAN':38s} {'':>4s} {'':>4s} "
          f"| {f(a_oc['sF1']):>6s} {f(a_oc['hF1']):>6s} {f(a_oc['GED'],1):>6s} {f(a_oc['pIoU']):>7s} "
          f"| {f(a_pp['soft_match_f1']):>6s} {f(a_pp['hard_match_f1']):>6s} {f(a_pp['graph_edit_distance'],1):>6s} {f(a_pp['product_iou']):>7s}")
    print(f"\nSaved: {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
