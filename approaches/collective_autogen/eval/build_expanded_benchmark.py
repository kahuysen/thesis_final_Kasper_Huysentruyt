"""Build an expanded benchmark from the three JSON GT sources in `Benchmark/`.

Walks every JSON GT inside `Benchmark/{review,article,r_group_resolution_diagrams}.zip`,
converts each record to our `ReactionRecord` schema, validates it, extracts the
matching image, and writes per-image gold under `eval/ground_truth/`.

Skipped:
- CSV-only GT (article.zip/GT1.csv, review.zip/GT3.csv) — would need a custom
  row→ReactionRecord converter; defer until coverage gaps justify the work.
- Records that fail pydantic validation — logged in the manifest with the
  errors so we can come back and fix the converter if patterns emerge.

Outputs:
- `eval/benchmark/images/{stem}.{ext}`
- `eval/ground_truth/{stem}.json`
- `eval/benchmark/manifest.json` — list of {stem, image, source_zip, source_gt,
  reactions, valid, errors}; consumers (run_benchmark_suite, etc.) can filter
  on `valid: true`.
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pydantic import ValidationError  # noqa: E402

from schema import ReactionRecord  # noqa: E402
from eval.build_benchmark import _convert_rgroup_record, _normalise_record  # noqa: E402

BENCH_DIR = REPO / "Benchmark"
IMAGES_OUT = REPO / "eval" / "benchmark" / "images"
GOLD_OUT = REPO / "eval" / "ground_truth"
MANIFEST_PATH = REPO / "eval" / "benchmark" / "manifest.json"

# (zip_name, gt_filename_in_zip, converter_kind)
# Order matters for dedup: first occurrence of a given stem wins.
SOURCES: list[tuple[str, str, str]] = [
    ("r_group_resolution_diagrams.zip", "GT1.json", "rgroup"),
    ("article.zip", "GT3.json", "standard"),
    ("article.zip", "GT2.json", "standard"),
    ("review.zip", "GT1.json", "standard"),
    ("review.zip", "GT2.json", "standard"),
]


def _load_records(zip_path: Path, gt_name: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z, z.open(gt_name) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _extract_image(zip_path: Path, image_name: str, dest: Path) -> bool:
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            if entry.split("/")[-1] == image_name:
                with z.open(entry) as src, dest.open("wb") as dst:
                    dst.write(src.read())
                return True
    return False


def main() -> int:
    if not BENCH_DIR.exists():
        print(f"Benchmark dir not found at {BENCH_DIR}", file=sys.stderr)
        return 2

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    GOLD_OUT.mkdir(parents=True, exist_ok=True)

    seen_stems: set[str] = set()
    manifest: list[dict] = []

    for zip_name, gt_name, kind in SOURCES:
        zip_path = BENCH_DIR / zip_name
        if not zip_path.exists():
            print(f"  WARN missing {zip_path}")
            continue
        try:
            records = _load_records(zip_path, gt_name)
        except KeyError:
            print(f"  WARN {zip_name} has no member {gt_name}")
            continue

        for rec in records:
            file_name = rec.get("file_name") or ""
            if not file_name:
                continue
            stem = Path(file_name).stem
            ext = Path(file_name).suffix or ".png"

            if stem in seen_stems:
                continue  # dedup: first JSON source wins

            norm = _convert_rgroup_record(rec) if kind == "rgroup" else _normalise_record(rec)

            try:
                ReactionRecord.model_validate(norm)
                valid = True
                errors: list[str] = []
            except ValidationError as e:
                valid = False
                errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5]]

            if not valid:
                manifest.append({
                    "stem": stem, "image": file_name, "source_zip": zip_name,
                    "source_gt": gt_name, "valid": False, "errors": errors,
                    "reactions": len(norm.get("reactions", [])),
                })
                continue

            # Extract image; skip the record if image is missing in the zip.
            img_dest = IMAGES_OUT / f"{stem}{ext}"
            ok = _extract_image(zip_path, file_name, img_dest)
            if not ok:
                manifest.append({
                    "stem": stem, "image": file_name, "source_zip": zip_name,
                    "source_gt": gt_name, "valid": False,
                    "errors": [f"image {file_name} not found in {zip_name}"],
                    "reactions": len(norm["reactions"]),
                })
                continue

            gold_path = GOLD_OUT / f"{stem}.json"
            gold_path.write_text(json.dumps(norm, indent=2, ensure_ascii=False), encoding="utf-8")

            seen_stems.add(stem)
            manifest.append({
                "stem": stem,
                "image": str(img_dest.relative_to(REPO)),
                "gold": str(gold_path.relative_to(REPO)),
                "source_zip": zip_name,
                "source_gt": gt_name,
                "valid": True,
                "errors": [],
                "reactions": len(norm["reactions"]),
            })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Stats.
    n_valid = sum(1 for m in manifest if m["valid"])
    n_invalid = len(manifest) - n_valid
    by_source = Counter((m["source_zip"], m["source_gt"], m["valid"]) for m in manifest)

    print(f"\n{'source':50s}  valid  invalid")
    print("-" * 70)
    sources = sorted({(z, g) for (z, g, _) in by_source})
    for z, g in sources:
        v = by_source[(z, g, True)]
        i = by_source[(z, g, False)]
        print(f"  {z}/{g:30s}  {v:>5d}  {i:>7d}")
    print("-" * 70)
    print(f"  {'TOTAL':<48s}  {n_valid:>5d}  {n_invalid:>7d}")
    print(f"\nWrote manifest: {MANIFEST_PATH.relative_to(REPO)}")
    print(f"Images:  {IMAGES_OUT.relative_to(REPO)}/  ({n_valid} files)")
    print(f"Gold:    {GOLD_OUT.relative_to(REPO)}/    ({n_valid} files)")

    if n_invalid:
        print(f"\n{n_invalid} records skipped — first 10 errors:")
        bad = [m for m in manifest if not m["valid"]][:10]
        for m in bad:
            print(f"  {m['source_zip']}/{m['source_gt']}  {m['stem']}: {m['errors'][0] if m['errors'] else 'no error msg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
