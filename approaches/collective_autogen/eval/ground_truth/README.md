# Ground Truth Files

One JSON file per input image, named `<image_stem>.json` (e.g. `example1.json`
for `input/example1.png`). The schema is the same `ReactionRecord` the pipeline
produces, validated by `pydantic` via `schema.py`.

## How `eval/run_eval.py` matches them
The runner reads `result.json` from a run dir, looks at the `file_name` field,
strips the extension, and looks for a matching `<stem>.json` here. If found, it
reports reference-based metrics (IoU, condition coverage); otherwise it falls
back to reference-free metrics only (schema, SMILES validity, role enum).

## Authoring tips
- Run the pipeline once on the image to get a starter `result.json`, then
  hand-correct it. Use `python eval/run_eval.py --result <path>` to verify it
  passes schema validation.
- All SMILES are compared after RDKit canonicalisation, so equivalent SMILES
  written differently (e.g. `c1ccccc1C=O` vs `O=Cc1ccccc1`) are matched.
- Condition `text` is normalised (lower-cased, whitespace-collapsed) for
  set comparison; case and extra spaces don't matter.
- For an R-group / substrate-scope figure, one reaction entry per variant.

## Current files
- `example1.json` — DRAFT; the structural backbone (5 R-group variants, two
  consecutive steps with conditions and yields) is taken from a 2026-04-26 run.
  Reactant/product SMILES need verification against the image. Replace once
  hand-checked.
