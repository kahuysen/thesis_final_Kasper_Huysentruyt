# OpenChemIE vs Multi-Agent Pipeline — Benchmark_kasper_GT3_Maarten

**Date:** 2026-04-29
**Benchmark:** `eval/Benchmark_kasper_GT3_Maarten/` (16 images, GT3 ground truth)
**OpenChemIE run:** `eval/results/openchemie_gt3.json`, `eval/results/openchemie_gt3_run.log`
**Pipeline reference:** `eval/results/suite_20260429_123624/` (canonical 16-image multi-agent run)
**Runner:** `eval/run_openchemie_gt3.py`

## Aggregate (mean over 16 images)

| Metric                                | OpenChemIE | Multi-agent | Notes                          |
|---------------------------------------|-----------:|------------:|--------------------------------|
| Soft-match F1 (sF1)                   |  **0.071** |   **0.216** | Agent ~3.0× higher             |
| Hard-match F1 (hF1)                   |      0.009 |       0.000 | Both effectively zero          |
| Partial-match F1 (Jaccard ≥ 0.5)      |      0.071 |       0.370 | Agent ~5.2× higher             |
| Reactant set IoU                      |      0.232 |       0.393 |                                |
| Product set IoU (pIoU)                |      0.067 |       0.551 | Agent ~8.2× higher             |
| Graph edit distance (lower = better)  |       71.5 |        41.9 | Agent better by 29.6           |
| SMILES validity rate                  |      1.000 |       1.000 |                                |

OpenChemIE results are deterministic and cached at `cache/openchemie/`.
Pipeline metrics are taken directly from the canonical suite's `summary.json`.

## Per-image table

`rP` = predicted reactions, `rG` = gold reactions.

| stem                                         | rP | rG | oc_sF1 | oc_pIoU | oc_GED | pp_sF1 | pp_pIoU | pp_GED |
|----------------------------------------------|---:|---:|-------:|--------:|-------:|-------:|--------:|-------:|
| ACScat_2020.pdf_page002_table_01_s0.88       |  1 | 14 |  0.000 |   0.000 |  101.8 |  0.000 |   0.000 |  100.0 |
| ACScat_2020.pdf_page003_picture_01_s0.93     |  1 |  8 |  0.000 |   0.000 |  159.5 |  0.000 |   0.000 |  112.6 |
| CEJ_2016.pdf_page001_picture_02_s0.74        |  3 |  3 |  1.000 |   1.000 |    0.0 |  0.000 |   0.000 |    7.0 |
| CEJ_2016.pdf_page002_table_02_s0.91          |  1 | 14 |  0.000 |   0.000 |   64.3 |  0.000 |   0.000 |   39.6 |
| CEJ_2016.pdf_page004_table_02_s0.66          |  1 | 10 |  0.000 |   0.000 |   39.1 |  0.000 |   0.583 |   37.3 |
| CS_2016.pdf_page002_table_02_s0.88           |  1 |  7 |  0.000 |   0.000 |   79.0 |  0.000 |   0.000 |   14.7 |
| GC_2015.pdf_page002_picture_02_s0.56         |  1 |  7 |  0.000 |   0.000 |   16.3 |  0.000 |   1.000 |   14.9 |
| GC_2015.pdf_page002_table_01_s0.61           |  1 | 10 |  0.000 |   0.000 |   58.0 |  1.000 |   1.000 |    0.0 |
| GC_2015.pdf_page003_picture_03_s0.71         |  1 |  4 |  0.000 |   0.000 |   91.0 |  0.857 |   1.000 |   22.8 |
| GC_2015.pdf_page003_table_02_s0.78           |  1 |  5 |  0.000 |   0.000 |   57.0 |  0.889 |   1.000 |   11.8 |
| GC_2015.pdf_page004_table_01_s0.81           |  0 |  6 |  0.000 |   0.000 |   45.0 |  0.000 |   1.000 |   45.0 |
| GC_2015.pdf_page006_picture_01_s0.86         |  1 |  9 |  0.000 |   0.000 |  103.3 |  0.000 |   0.889 |   59.7 |
| GC_2015.pdf_page007_picture_02_s0.76         |  1 |  5 |  0.000 |   0.000 |   83.8 |  0.000 |   0.000 |   53.4 |
| GC_2015.pdf_page007_picture_03_s0.60         |  1 |  5 |  0.000 |   0.000 |   72.6 |  0.000 |   0.800 |   45.4 |
| NC_2017.pdf_page004_picture_01_s0.91         |  1 | 13 |  0.143 |   0.077 |   82.0 |    —   |     —   |    —   |
| NC_2017.pdf_page005_picture_01_s0.90         |  1 |  3 |  0.000 |   0.000 |   91.3 |  0.500 |   1.000 |   64.7 |
| **MEAN**                                     |    |    |  0.071 |   0.067 |   71.5 |  0.216 |   0.551 |   41.9 |

`NC_2017.pdf_page004_picture_01_s0.91` has no multi-agent row — that image was the
1-of-16 timeout in the canonical suite (it is excluded from the pipeline mean).

## Observations

- **OpenChemIE collapses dense R-group tables to a single reaction.** On 14 of 16
  images it predicts exactly 1 reaction where the gold has 3-14, so soft-match F1
  is bounded by `2/(1+rG)` per image. The mean of 0.071 is consistent with that
  ceiling. The pipeline expands R-group variants and recovers many more matches.
- **Only win for OpenChemIE: `CEJ_2016.pdf_page001_picture_02_s0.74`** — a clean
  3-reaction figure with no R-group enumeration. OpenChemIE got it perfect
  (sF1=1.0, pIoU=1.0, GED=0); the pipeline got it wrong (sF1=0).
- **Hard-match F1 is ~0 for both systems.** This metric requires reactants,
  products, *and* condition strings to all match exactly — neither system extracts
  conditions cleanly enough to clear the bar, but both are close on
  reactants/products as seen in the IoU columns.
- **SMILES validity is 100 % for OpenChemIE** (all emitted SMILES parse with
  RDKit). Pipeline is also 100 %.
- **Wall-clock:** OpenChemIE on CPU averaged ~7 s per image (110 s total for the
  16-image suite). The multi-agent pipeline averaged 286 s per image.

## Reproducing

```bash
source .venv/bin/activate
python eval/run_openchemie_gt3.py
# → eval/results/openchemie_gt3.json
# → eval/results/openchemie_gt3_run.log
```

OpenChemIE runs out-of-process via `.venv-openchemie/bin/python` and is
disk-cached by image mtime, so re-runs are free.
