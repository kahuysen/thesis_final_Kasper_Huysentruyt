"""Rebuild eval/ground_truth/ from ChemEagle's official benchmark.

Source: HuggingFace `CYF200127/ChemEagle/Benchmark.zip` (or the unzipped
`/tmp/chemeagle_gt/{GT1,GT2}.json`). For each image we already have under
`eval/benchmark/images/`, pull the matching record out of the ChemEagle GT
files, normalise it to our `ReactionRecord` schema, and write it as
`eval/ground_truth/<stem>.json`.

Background — Step A finding:
  - GT1.json (139 images, 983 reactions): Research Article subset. Native
    schema is already `{file_name, reactions: [...]}` — same as ours. Just
    needs `extra="forbid"` field stripping (gold has a stray `label` on
    each condition).
  - GT2.json (78 images, 1007 variants): OpenChemIE subset. Schema is
    `{file_name, reaction_template, detailed_reactions}` — identical to
    the r_group_resolution_diagrams source we already convert. Reuse
    `_convert_rgroup_record` from build_benchmark.py. Note: GT2 has NO
    conditions in gold (by design of the upstream dataset).
  - GT3.csv / GT4.csv cover separate Review-subset images we don't have
    locally; not consumed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pydantic import ValidationError
from schema import ReactionRecord
from eval.build_benchmark import _convert_rgroup_record, _normalise_record  # noqa: E402

CHEMEAGLE_GT_DIR = Path("/tmp/chemeagle_gt")  # populated by `unzip /tmp/Benchmark.zip`
GT1_PATH = CHEMEAGLE_GT_DIR / "GT1.json"
GT2_PATH = CHEMEAGLE_GT_DIR / "GT2.json"
GOLD_OUT = REPO / "eval" / "ground_truth"
IMAGES_DIR = REPO / "eval" / "benchmark" / "images"


def _load_gt():
    if not GT1_PATH.exists() or not GT2_PATH.exists():
        raise FileNotFoundError(
            f"ChemEagle GT files not at {CHEMEAGLE_GT_DIR}. "
            "Run: unzip /tmp/Benchmark.zip 'Benchmark/GT*.json' -d /tmp/chemeagle_gt"
        )
    gt1 = json.loads(GT1_PATH.read_text())
    gt2 = json.loads(GT2_PATH.read_text())
    gt1_by_name = {r["file_name"]: r for r in gt1 if r.get("file_name")}
    gt2_by_name = {r["file_name"]: r for r in gt2 if r.get("file_name")}
    return gt1_by_name, gt2_by_name


def _find_image_files() -> list[Path]:
    if not IMAGES_DIR.exists():
        return []
    return sorted([p for p in IMAGES_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def _summarise(rec: dict) -> str:
    n_rxn = len(rec.get("reactions", []) or [])
    n_cond = sum(len(r.get("conditions", []) or []) for r in rec.get("reactions", []) or [])
    n_smi = sum(
        sum(1 for c in (r.get("reactants") or []) + (r.get("products") or []) if c.get("smiles"))
        for r in rec.get("reactions", []) or []
    )
    return f"{n_rxn} reactions, {n_cond} conditions, {n_smi} non-null compound SMILES"


def main() -> int:
    gt1, gt2 = _load_gt()
    images = _find_image_files()
    if not images:
        print("no benchmark images found", file=sys.stderr)
        return 2

    rows = []
    for img in images:
        name = img.name
        source, raw = None, None
        if name in gt1:
            source, raw = "GT1", gt1[name]
        elif name in gt2:
            source, raw = "GT2", gt2[name]
        if raw is None:
            rows.append({"image": name, "source": "MISSING", "valid": False,
                         "old_summary": "", "new_summary": "(no GT in ChemEagle)",
                         "errors": ["no matching record in GT1 or GT2"]})
            continue

        # Normalise to our pydantic schema.
        if source == "GT2":
            new = _convert_rgroup_record(raw)
        else:
            new = _normalise_record(raw)

        # Validate — note any errors but still write (we'll see them in the diff).
        try:
            ReactionRecord.model_validate(new)
            valid = True
            errors = []
        except ValidationError as e:
            valid = False
            errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:3]]

        # Diff against existing local gold.
        gold_path = GOLD_OUT / f"{img.stem}.json"
        old_summary = ""
        if gold_path.exists():
            try:
                old = json.loads(gold_path.read_text())
                old_summary = _summarise(old)
            except Exception:
                old_summary = "(parse error)"

        # Write the new gold.
        if valid:
            gold_path.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")

        rows.append({
            "image": name,
            "source": source,
            "valid": valid,
            "old_summary": old_summary,
            "new_summary": _summarise(new),
            "errors": errors,
        })

    print(f"\n{'image':50}  {'src':4}  {'valid':5}  old → new")
    print("-" * 110)
    for r in rows:
        valid_flag = "OK" if r["valid"] else "ERR"
        print(f"  {r['image']:50}  {r['source']:4}  {valid_flag:5}  {r['old_summary']!s:38} → {r['new_summary']!s}")
        for err in r.get("errors", []):
            print(f"    {err}")

    n_ok = sum(1 for r in rows if r["valid"])
    print(f"\n{n_ok}/{len(rows)} images now have ChemEagle-derived gold validated against ReactionRecord.")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
