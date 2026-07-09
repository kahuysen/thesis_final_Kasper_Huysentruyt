"""Build a small benchmark set from `benchmark/*.zip`.

Picks a handful of images from each zip, extracts them under
`eval/benchmark/images/`, and splits the corresponding ground-truth records
into per-image files under `eval/ground_truth/`. Schemas are normalised to
match `schema.ReactionRecord` (extra fields stripped, list/dict shapes
converted where the source is non-standard, e.g. r_group_resolution_diagrams).

Run from the repo root:
    python eval/build_benchmark.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pydantic import ValidationError  # noqa: E402

from schema import ReactionRecord  # noqa: E402

BENCH_DIR = REPO / "benchmark"
IMAGES_OUT = REPO / "eval" / "benchmark" / "images"
GOLD_OUT = REPO / "eval" / "ground_truth"

# (zip_name, list_of_(gt_filename_in_zip, images_to_take)).
# `images_to_take` is a list of basenames inside the zip — we'll match GT
# records by `file_name` field.
SAMPLES: dict[str, list[tuple[str, list[str]]]] = {
    "article.zip": [
        ("GT2.json", ["04_JACS.png"]),
        ("GT3.json", [
            "CEJ_2016.pdf_page001_picture_02_s0.74.png",
            "ACScat_2020.pdf_page003_picture_01_s0.93.png",
        ]),
    ],
    "r_group_resolution_diagrams.zip": [
        ("GT1.json", [
            "acs.joc.2c00176 example 2.png",
            "acs.joc.3c00062 example 1.png",
            "acs.joc.3c00062 example 3.png",
        ]),
    ],
    "review.zip": [
        ("GT1.json", ["104-1.jpg", "107.jpg"]),
        ("GT2.json", ["116-1.jpg"]),
    ],
}


# ---------- schema normalisation ----------

_ROLE_FALLBACK = {"reagents": "reagent", "solvents": "solvent", "catalysts": "catalyst"}


def _strip_compound(c: dict) -> dict:
    """Reduce to {smiles, label, uncertain} only."""
    return {
        "smiles": c.get("smiles") if c.get("smiles") not in ("", None) else None,
        "label": c.get("label") if c.get("label") not in ("", "None", None) else None,
        "uncertain": bool(c.get("uncertain", False)),
    }


def _strip_condition(c: dict) -> dict | None:
    """Reduce to {role, text, smiles} only. Returns None if it can't be coerced."""
    role = c.get("role")
    if not role and c.get("text", "").lower().endswith(("ee", "er", "dr")):
        # Some sources embed e.g. "75% ee" without an explicit role.
        text = c["text"].lower()
        for r in ("ee", "er", "dr"):
            if text.endswith(r):
                role = r
                break
    role = (role or "").strip().lower()
    role = _ROLE_FALLBACK.get(role, role)
    if not role:
        return None
    return {
        "role": role,
        "text": c.get("text", ""),
        "smiles": c.get("smiles") if c.get("smiles") not in ("", None) else None,
    }


def _flatten_additional_info(items) -> list[str]:
    """`additional_info` may be a list of strings, or a list of dicts shaped
    like {"text": "..."} (article GT2/GT3). Coerce to list[str]."""
    out: list[str] = []
    for it in items or []:
        if isinstance(it, str):
            if it.strip():
                out.append(it)
        elif isinstance(it, dict):
            txt = it.get("text") or ""
            if txt.strip():
                out.append(txt)
    return out


def _normalise_record(rec: dict) -> dict:
    """Coerce a record into our `ReactionRecord` shape (best-effort)."""
    out = {"file_name": rec.get("file_name", ""), "reactions": []}
    for r in rec.get("reactions", []) or []:
        rxn = {
            "reaction_id": r.get("reaction_id", "0_1"),
            "reactants": [_strip_compound(c) for c in (r.get("reactants") or [])],
            "products": [_strip_compound(c) for c in (r.get("products") or [])],
            "conditions": [c for c in (_strip_condition(x) for x in (r.get("conditions") or [])) if c],
            "additional_info": _flatten_additional_info(r.get("additional_info")),
        }
        out["reactions"].append(rxn)
    return out


def _convert_rgroup_record(rec: dict) -> dict:
    """The r_group dataset uses `reaction_template` + `detailed_reactions`
    (dict-of-dicts). Each detailed_reactions[letter] becomes one reaction in
    our schema. No conditions are present in this dataset."""
    detailed = rec.get("detailed_reactions") or {}
    reactions = []
    for idx, (letter, body) in enumerate(detailed.items(), start=1):
        reactions.append({
            "reaction_id": f"0_{idx}",
            "reactants": [
                {"smiles": s if s else None, "label": letter, "uncertain": False}
                for s in (body.get("reactants") or [])
            ],
            "products": [
                {"smiles": s if s else None, "label": letter, "uncertain": False}
                for s in (body.get("products") or [])
            ],
            "conditions": [],
            "additional_info": [],
        })
    return {"file_name": rec.get("file_name", ""), "reactions": reactions}


# ---------- main ----------

def _load_gt_from_zip(zip_path: Path, gt_name: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(gt_name) as f:
            data = json.load(f)
    if isinstance(data, dict):
        # Some GTs might be single-record dicts. Normalise to list-form.
        return [data]
    return data


def _extract_image(zip_path: Path, image_name: str, dest: Path) -> Path | None:
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            if entry.split("/")[-1] == image_name:
                with z.open(entry) as src, dest.open("wb") as dst:
                    dst.write(src.read())
                return dest
    return None


def main() -> int:
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    GOLD_OUT.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for zip_name, gt_specs in SAMPLES.items():
        zip_path = BENCH_DIR / zip_name
        if not zip_path.exists():
            print(f"  WARN missing {zip_path}", file=sys.stderr)
            continue
        is_rgroup = "r_group" in zip_name
        for gt_name, image_names in gt_specs:
            records = _load_gt_from_zip(zip_path, gt_name)
            by_name = {r.get("file_name"): r for r in records}
            for img in image_names:
                rec = by_name.get(img)
                if rec is None:
                    print(f"  SKIP {img!r}: no GT record in {zip_name}/{gt_name}")
                    continue
                # Convert / normalise.
                norm = _convert_rgroup_record(rec) if is_rgroup else _normalise_record(rec)
                # Validate against our pydantic schema.
                stem = Path(img).stem
                gold_path = GOLD_OUT / f"{stem}.json"
                try:
                    ReactionRecord.model_validate(norm)
                    valid = True
                    err = None
                except ValidationError as e:
                    valid = False
                    err = e.errors()
                # Extract image.
                ext = Path(img).suffix
                img_path = IMAGES_OUT / f"{stem}{ext}"
                _extract_image(zip_path, img, img_path)
                # Write the GT.
                gold_path.write_text(json.dumps(norm, indent=2, ensure_ascii=False), encoding="utf-8")
                summary.append({
                    "zip": zip_name,
                    "gt": gt_name,
                    "image": img,
                    "image_dest": str(img_path.relative_to(REPO)),
                    "gold_dest": str(gold_path.relative_to(REPO)),
                    "reactions": len(norm["reactions"]),
                    "valid": valid,
                    "errors": err,
                })

    # Print a summary table.
    print(f"\n{'-'*100}")
    print(f"{'zip':35} {'image':45} {'rxns':>4}  valid")
    print(f"{'-'*100}")
    for s in summary:
        flag = "OK " if s["valid"] else "ERR"
        print(f"{s['zip']:35} {s['image']:45} {s['reactions']:>4}  {flag}")
    print(f"{'-'*100}")
    n_ok = sum(1 for s in summary if s["valid"])
    print(f"{n_ok}/{len(summary)} GT files valid against ReactionRecord.")
    if n_ok < len(summary):
        print("\nValidation errors:")
        for s in summary:
            if not s["valid"]:
                print(f"  {s['gold_dest']}:")
                for e in s["errors"][:3]:
                    print(f"    {e}")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
