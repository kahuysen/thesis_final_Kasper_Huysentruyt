# Benchmark set

Nine images (3 per source zip) with paired ground-truth files, sampled
from the larger collections in `benchmark/*.zip`. Built by
`eval/build_benchmark.py` so the selection is reproducible — re-run it
to regenerate or extend.

## Files

```
eval/benchmark/images/                 # the input images
eval/ground_truth/<image_stem>.json    # one ReactionRecord per image
```

The ground-truth files live in `eval/ground_truth/` (not under `benchmark/`)
so `eval/run_eval.py` finds them automatically by stem matching.

## Sources and selection

| Source zip | Image | Reactions | GT origin | Notes |
|---|---|---|---|---|
| `article.zip` | `04_JACS.png` | 5 | `GT2.json` | named-product variants (Bezafibrate, Gemfibrozil, Loxoprofen) |
| `article.zip` | `CEJ_2016.pdf_page001_picture_02_s0.74.png` | 3 | `GT3.json` | OCR'd page crop; small N |
| `article.zip` | `ACScat_2020.pdf_page003_picture_01_s0.93.png` | 8 | `GT3.json` | dr / ee values in additional_info |
| `r_group_resolution_diagrams.zip` | `acs.joc.2c00176 example 2.png` | 8 | `GT1.json` | substrate-scope; conditions absent in GT |
| `r_group_resolution_diagrams.zip` | `acs.joc.3c00062 example 1.png` | 13 | `GT1.json` | larger R-group enumeration |
| `r_group_resolution_diagrams.zip` | `acs.joc.3c00062 example 3.png` | 12 | `GT1.json` | sibling figure to above |
| `review.zip` | `104-1.jpg` | 4 | `GT1.json` | wildcard-encoded GT (`[1*]`, `[2*]` placeholders) |
| `review.zip` | `107.jpg` | 7 | `GT1.json` | wildcard-encoded GT |
| `review.zip` | `116-1.jpg` | 1 | `GT2.json` | single reaction; smallest item |

All nine pass `ReactionRecord.model_validate`.

Some `additional_info` originals were `[{text: "..."}]` (article zip);
the builder flattens those to the schema's `list[str]`. `r_group_resolution_diagrams`
uses a different parent shape (`reaction_template` + `detailed_reactions`);
the builder converts each variant under `detailed_reactions` into a
separate `Reaction` entry. Conditions in that source are empty by design
of the upstream dataset, so condition-coverage metrics will read 0/0 for
those items.

## Adding more

Edit the `SAMPLES` dict at the top of `eval/build_benchmark.py`, then
re-run from the repo root:

```bash
python eval/build_benchmark.py
```

The script overwrites existing `eval/ground_truth/<stem>.json` and
`eval/benchmark/images/<stem>.<ext>` paths, so adding images is a matter of
listing them. Each `image_name` must match a `file_name` in the named GT
file inside the zip.

## Caveats on these ground-truth files

- A handful of SMILES contain wildcards (`*N=C=S`, `[1*]C(...)=O`) that
  RDKit can't parse. They're left as-is so the ground truth matches the
  source dataset; expect <100% `smiles_validity_rate` on self-compare.
- `eval/ground_truth/example1.json` is the older hand-drafted file (still
  used by `python main.py input/example1.png`). Nothing in the benchmark
  builder touches it.
