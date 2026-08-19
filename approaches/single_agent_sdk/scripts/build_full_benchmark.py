"""Build `data/benchmark_full/` from the ChemEagle public benchmark release.

Input: `Benchmark.zip` from https://huggingface.co/datasets/CYF200127/ChemEagle
(85,805,044 bytes, md5 611d4c01e4ed67f6d560d92448422172). The zip holds 324
figure images plus four ground-truth files, disjoint by image:

  - GT1.json — 142 entries in the same record schema as `data/benchmark/`
    (reactions[].reactants/conditions/products with smiles+label). Used verbatim.
  - GT2.json — 78 entries for R-group-table figures: a `reaction_template`
    plus letter-keyed `detailed_reactions`, structures only (no conditions).
  - GT3.csv — 182 rows (38 images): reactant/product/catalyst SMILES only.
  - GT4.csv — 727 rows (66 images): structures plus catalyst_text, yield, ee,
    dr, er, solvent, temperature. Includes wildcard template rows (`0_1`).

All four are mapped into the record schema, tagged with a `slice` field so
evaluation can apply per-slice scoring rules (see eval_full_benchmark.py).
A handful of GT file_names are misspelled relative to the shipped images
(colon vs underscore, `iamge` typo, `CHEM_01_01` vs `CHEM01_01`); these are
repaired via the explicit FILENAME_FIXUPS map and logged.

GT4 numeric yield/ee/er values are fractions (0.13); the figures print
percentages, so they are rendered as percent text ("13%") for text matching.
The raw value is preserved in the condition entry under `raw_value`.

Output layout:

  data/benchmark_full/
  ├── images/           all 324 release images, filenames preserved
  ├── ground_truth/     <stem>.json per scoreable image (record schema + "slice")
  ├── manifest.json     one row per image: stem, file_name, slice, dev16, scored
  └── exclusions.json   every dropped/duplicated/repaired GT entry, logged

The 16-image thesis development subset (dev16) is read from
`data/benchmark/manifest.json`; those stems are flagged so evaluation can
report the held-out set (scored minus dev16) separately.

Usage:
    python3 scripts/build_full_benchmark.py --zip /path/to/Benchmark.zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # approaches/single_agent_sdk
REPO = ROOT.parent.parent                              # 5.Code_reorg
DEV16_MANIFEST = REPO / "data" / "benchmark" / "manifest.json"
OUT_DIR = REPO / "data" / "benchmark_full"

EXPECTED_MD5 = "611d4c01e4ed67f6d560d92448422172"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# GT file_name → actual image name in the zip (verified by hand).
FILENAME_FIXUPS = {
    "250_iamge_3_1.png": "250_image_3_1.png",
    "CHEM_01_01.png": "CHEM01_01.png",
    "10.1002:anie.202314701 example 1.png": "10.1002_anie.202314701 example 1.png",
    "10.1002:anie.202313578 example 1.png": "10.1002_anie.202313578 example 1.png",
}


def _fixup(fname: str | None, exclusions: list[dict], source: str) -> str | None:
    if fname in FILENAME_FIXUPS:
        fixed = FILENAME_FIXUPS[fname]
        exclusions.append({"source": source, "file_name": fname,
                           "reason": f"filename repaired → {fixed}"})
        return fixed
    return fname


def _pct(v: str | None) -> tuple[str | None, float | None]:
    """GT4 fraction → percent text (0.13 → '13%'); passthrough otherwise."""
    if v in (None, ""):
        return None, None
    try:
        f = float(v)
    except ValueError:
        return v, None
    if 0 < f <= 1:
        return f"{f * 100:g}%", f
    return f"{f:g}%", f


def _gt2_to_record(entry: dict, fname: str) -> dict:
    reactions = []
    for i, (letter, rxn) in enumerate(sorted(entry.get("detailed_reactions", {}).items())):
        reactions.append(
            {
                "reaction_id": f"{i}_1",
                "reactants": [{"smiles": s, "label": None} for s in rxn.get("reactants", [])],
                "conditions": [],
                "products": [{"smiles": s, "label": None} for s in rxn.get("products", [])],
                "additional_info": [f"gt2_variant={letter}"],
            }
        )
    return {
        "file_name": fname,
        "slice": "gt2",
        "reaction_template": entry.get("reaction_template"),
        "reactions": reactions,
    }


def _gt3_rows_to_record(fname: str, rows: list[dict]) -> dict:
    reactions = []
    for i, r in enumerate(rows):
        conditions = []
        if r.get("cat_1"):
            conditions.append({"role": "catalyst", "text": None, "smiles": r["cat_1"]})
        extra = []
        if r.get("label"):
            extra.append(f"gt3_label={r['label']}")
        reactions.append(
            {
                "reaction_id": f"{i}_1",
                "reactants": [{"smiles": r[k], "label": None}
                              for k in ("rec_1", "rec_2") if r.get(k)],
                "conditions": conditions,
                "products": [{"smiles": r["pro_1"], "label": None}] if r.get("pro_1") else [],
                "additional_info": extra,
            }
        )
    return {"file_name": fname, "slice": "gt3", "reactions": reactions}


def _gt4_rows_to_record(fname: str, rows: list[dict]) -> dict:
    reactions = []
    for i, r in enumerate(rows):
        conditions = []
        for cat_smiles, cat_text in ((r.get("cat_1"), r.get("catalyst_text")),
                                     (r.get("cat_2"), None)):
            if cat_smiles or cat_text:
                conditions.append({"role": "catalyst", "text": cat_text or None,
                                   "smiles": cat_smiles or None})
        if r.get("solvent"):
            conditions.append({"role": "solvent", "text": r["solvent"], "smiles": None})
        if r.get("temperature"):
            conditions.append({"role": "temperature", "text": r["temperature"]})
        for role in ("yield", "ee"):
            text, raw = _pct(r.get(role))
            if text is not None:
                conditions.append({"role": role, "text": text, "raw_value": raw})
        extra = []
        for k in ("dr", "er"):
            if r.get(k):
                extra.append(f"gt4_{k}={r[k]}")
        rid = r.get("reaction_id") or f"{i}_1"
        reactions.append(
            {
                "reaction_id": rid,
                "reactants": [{"smiles": r[k], "label": None}
                              for k in ("rec_1", "rec_2") if r.get(k)],
                "conditions": conditions,
                "products": [{"smiles": r[k], "label": None}
                             for k in ("pro_1", "pro_2") if r.get(k)],
                "additional_info": extra,
            }
        )
    return {"file_name": fname, "slice": "gt4", "reactions": reactions}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to the release Benchmark.zip")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        print(f"WARNING: md5 {md5} != expected {EXPECTED_MD5} — not the audited release?")

    dev16 = {e["stem"] + ".png" for e in json.loads(DEV16_MANIFEST.read_text())}

    img_dir = OUT_DIR / "images"
    gt_dir = OUT_DIR / "ground_truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    exclusions: list[dict] = []
    images: dict[str, str] = {}   # file_name -> stem

    zf = zipfile.ZipFile(zip_path)
    gt_raw: dict[str, object] = {}
    for n in [n for n in zf.namelist() if not n.endswith("/")]:
        base = Path(n).name
        if base in ("GT1.json", "GT2.json"):
            gt_raw[base] = json.loads(zf.read(n))
        elif base in ("GT3.csv", "GT4.csv"):
            # CSVs are cp1252/latin-1 encoded (degree signs etc.).
            text = zf.read(n).decode("latin-1")
            gt_raw[base] = list(csv.DictReader(io.StringIO(text)))
        elif Path(base).suffix.lower() in IMAGE_EXTS:
            (img_dir / base).write_bytes(zf.read(n))
            images[base] = Path(base).stem
    missing_gt = [k for k in ("GT1.json", "GT2.json", "GT3.csv", "GT4.csv") if k not in gt_raw]
    if missing_gt:
        print(f"missing ground-truth files in zip: {missing_gt}")
        return 1

    gt_records: dict[str, dict] = {}

    # ── GT1: verbatim; dedupe by file_name, drop malformed ──
    seen_gt1: dict[str, dict] = {}
    for idx, entry in enumerate(gt_raw["GT1.json"]):
        fname = _fixup(entry.get("file_name"), exclusions, "GT1")
        if not fname:
            exclusions.append({"source": "GT1", "index": idx, "reason": "no file_name"})
            continue
        if fname in seen_gt1:
            identical = (json.dumps(entry, sort_keys=True)
                         == json.dumps(seen_gt1[fname], sort_keys=True))
            exclusions.append(
                {"source": "GT1", "index": idx, "file_name": fname,
                 "reason": "duplicate file_name — kept first entry",
                 "duplicate_identical": identical})
            continue
        seen_gt1[fname] = entry
        if fname not in images:
            exclusions.append({"source": "GT1", "file_name": fname,
                               "reason": "image not in release zip"})
            continue
        gt_records[fname] = {**entry, "file_name": fname, "slice": "gt1"}

    # ── GT2 ──
    for idx, entry in enumerate(gt_raw["GT2.json"]):
        fname = _fixup(entry.get("file_name"), exclusions, "GT2")
        if not fname:
            exclusions.append({"source": "GT2", "index": idx, "reason": "no file_name"})
            continue
        if fname not in images:
            exclusions.append({"source": "GT2", "file_name": fname,
                               "reason": "image not in release zip"})
            continue
        if fname in gt_records:
            exclusions.append({"source": "GT2", "file_name": fname,
                               "reason": "already covered — kept earlier slice"})
            continue
        gt_records[fname] = _gt2_to_record(entry, fname)

    # ── GT3 / GT4: group CSV rows by image ──
    for src, mapper in (("GT3.csv", _gt3_rows_to_record), ("GT4.csv", _gt4_rows_to_record)):
        by_file: dict[str, list[dict]] = defaultdict(list)
        for row in gt_raw[src]:
            fname = _fixup(row.get("file_name"), exclusions, src)
            if fname:
                by_file[fname].append(row)
        for fname, rows in by_file.items():
            if fname not in images:
                exclusions.append({"source": src, "file_name": fname,
                                   "reason": "image not in release zip"})
                continue
            if fname in gt_records:
                exclusions.append({"source": src, "file_name": fname,
                                   "reason": "already covered — kept earlier slice"})
                continue
            gt_records[fname] = mapper(fname, rows)

    for fname, rec in gt_records.items():
        (gt_dir / f"{Path(fname).stem}.json").write_text(
            json.dumps(rec, indent=1, ensure_ascii=False))

    manifest = []
    for fname in sorted(images):
        rec = gt_records.get(fname)
        manifest.append(
            {
                "stem": Path(fname).stem,
                "file_name": fname,
                "slice": rec["slice"] if rec else None,
                "dev16": fname in dev16,
                "scored": rec is not None,
            })
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT_DIR / "exclusions.json").write_text(
        json.dumps(exclusions, indent=1, ensure_ascii=False))

    counts = {s: sum(1 for m in manifest if m["slice"] == s)
              for s in ("gt1", "gt2", "gt3", "gt4")}
    n_dev = sum(1 for m in manifest if m["dev16"])
    n_scored = sum(1 for m in manifest if m["scored"])
    n_rxn = sum(len(r["reactions"]) for r in gt_records.values())
    print(f"images: {len(manifest)}  |  scored: {n_scored}  {counts}")
    print(f"reactions in gold: {n_rxn}")
    print(f"dev16 present: {n_dev}  |  held-out: {n_scored - n_dev}")
    print(f"unscored images: {[m['file_name'] for m in manifest if not m['scored']]}")
    print(f"exclusions logged: {len(exclusions)}  →  {OUT_DIR / 'exclusions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
