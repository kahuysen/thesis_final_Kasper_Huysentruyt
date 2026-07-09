# Gemini 3.5 Flash on the 16-image benchmark

**Run date:** 2026-05-22
**Corpus:** `Collective_autogen/eval/Benchmark_kasper_GT3_Maarten/` (16 PNGs, GT3-Maarten gold set)
**Model:** `gemini-3.5-flash` (Google Standard tier)
**Agent harness:** Single-SDK agent — same `pipeline/extractor_gemini_parallel.py` as the existing `gemini-3-flash-preview` entry, identical system prompt, identical tool schemas (`validate_smiles`, `submit_extraction`), manual parallel-function-call loop. Only `--model` changed.
**Script:** `scripts/benchmark_gemini.py --max-steps 128 --concurrency 1 --delay 8.0`

## Headline numbers

| metric | value |
|---|---|
| **Partial F1 (Jaccard ≥ 0.5)** | **0.804** |
| Reaction recall (cli scorer) | 0.85 |
| Product recall / precision (cli scorer) | 0.85 / 0.84 |
| Yield accuracy | 0.93 |
| SMILES validity (RDKit-parseable) | 1.00 |
| Images succeeded | 16 / 16 |
| Total wall time | 637 s (39.8 s / image) |
| Total tokens | 386 473 (346 521 in + 39 952 out; 24 155 / image) |
| Total tool calls | 216 (13.5 / image) |
| Mean agent round-trips | 3.1 (max 11) |
| **Total cost (Google sticker)** | **$0.88** ($0.055 / image) |

Cost projection: $1.50 / M input + $9.00 / M output, from the Google Standard tier sticker (Gemini 3 family). Not empirical billing.

## Per-image breakdown

Ordered by Partial F1, descending:

| image | F1 | matched/gold | steps | wall (s) |
|---|---:|---:|---:|---:|
| ACScat_2020 p3 picture_01 | 1.000 | 8/8 | 2 | 54.8 |
| CEJ_2016 p1 picture_02 | 1.000 | 3/3 | 2 | 29.0 |
| CEJ_2016 p2 table_02 | 1.000 | 14/14 | 2 | 39.3 |
| CS_2016 p2 table_02 | 1.000 | 7/7 | 2 | 19.1 |
| GC_2015 p2 picture_02 | 1.000 | 7/7 | 3 | 34.3 |
| GC_2015 p3 picture_03 | 1.000 | 4/4 | 2 | 16.7 |
| GC_2015 p6 picture_01 | 1.000 | 9/9 | 3 | 44.7 |
| GC_2015 p7 picture_02 | 1.000 | 5/5 | 2 | 31.2 |
| GC_2015 p2 table_01 | 0.952 | 10/10 | 2 | 21.0 |
| GC_2015 p3 table_02 | 0.889 | 4/5 | 2 | 10.9 |
| NC_2017 p5 picture_01 | 0.857 | 3/3 | 11 | 78.0 |
| CEJ_2016 p4 table_02 | 0.800 | 8/10 | 4 | 45.6 |
| NC_2017 p4 picture_01 | 0.692 | 9/13 | 2 | 48.3 |
| GC_2015 p7 picture_03 | 0.600 | 3/5 | 2 | 47.6 |
| ACScat_2020 p2 table_01 | **0.071** | 1/14 | 2 | 37.7 |
| GC_2015 p4 table_01 | **0.000** | 0/6 | 7 | 78.7 |

- **8 / 16 images are scored perfectly (F1 = 1.0).**
- **2 / 16 images effectively fail** (F1 < 0.1). They drag the mean down by ~0.058 — the macro mean over the remaining 14 would be ≈ 0.917.

## Position relative to the other Single-SDK agents

Reading values from `progress_April_data.xlsx`:

| system | F1 | $ / image | tokens / image | wall / image |
|---|---:|---:|---:|---:|
| SDK Opus 4.7 | 0.854 | $0.220 | 20 875 | 41 s |
| **SDK Gemini 3.1 Pro** | **0.832** | **$0.134** | **92 667** | **105 s** |
| **SDK Gemini 3.5 Flash (this run)** | **0.804** | **$0.055** | **24 155** | **40 s** |
| SDK Gemini 3 Flash | 0.79 | $0.018 | 29 000 | 53 s |
| SDK Grok 4.3 | 0.78 | $0.257 | 66 133 | 444 s |
| SDK GPT-5.5 | 0.723 | $0.226 | 16 166 | 99 s |

Gemini 3.5 Flash sits in the third row from the top: **+0.014 F1 over Gemini 3 Flash, –0.050 F1 below Opus 4.7, at roughly a quarter of Opus 4.7's per-image cost and the same wall time**. It is ~3.1× more expensive per token than Gemini 3 Flash on the sticker price.

## Failure modes on the two collapsed images

1. **`GC_2015 p4 table_01` — F1 = 0.000 (0/6 reactions matched).** The CLI scorer reports 6/6 product recall and 100 % SMILES validity, so the right product molecules are present and parse — but every reaction's reactant+condition fingerprint fell below the Jaccard 0.5 threshold against gold. The image used 47 943 input tokens (≈ 4.6× the run average), suggesting the agent loop accumulated a long context before submitting. Indicates the model identified products but mis-assigned reactants or conditions row-by-row.
2. **`ACScat_2020 p2 table_01` — F1 = 0.071 (1/14 reactions matched).** The same Boc-amino-acid coupling table that thrashed on the earlier 32-step run with the `OCH2c1ccccc1` SMILES bug. On this re-run the model produced RDKit-valid SMILES (validity = 100 %) but only one of the 14 entries canonicalised to the same structure as gold — likely stereo/substituent confusion on a row-by-row basis.

## Notes & caveats

- **Step-cap was the prior bottleneck, not this one.** The 32-step run that I aborted earlier got stuck in a feedback loop on the OCH2 table; raising the cap to 128 let the run complete, but the actual max steps used by any image was **11**. The reason image 1 finished this time isn't more rope — it's that the model produced different (still mostly wrong) SMILES on this attempt. This is plain sampling nondeterminism; a re-run could flip image 1 back to a step-cap timeout.
- **Cost is projected, not billed.** I used Google's published Standard-tier sticker. The Opus 4.7 entry in the XLSX, by contrast, uses an empirical bill that ran ≈ 53 % above its own sticker, so the apparent cost gap to Opus 4.7 may shrink on real billing.
- **Same SDK harness as Gemini 3 Flash, only the model string changed.** The agent loop was originally written for Gemini 3 (parallel function calling, `thought_signature` preservation, explicit `id` echo on `FunctionResponse`). Whether 3.5 Flash uses every Gemini-3-specific affordance is not verified; sending unused fields is harmless.
- The 128-step cap is a kwarg now (`scripts/benchmark_gemini.py --max-steps N`) and defaults to 32, so the existing entries are not affected.

## Files

- Predictions: `*.json` (16 files, FigureExtraction schema)
- Per-image metadata: `*.meta.json` (backend, model, steps, tool_calls, tokens, elapsed)
- CLI scorer report: `evaluation.txt`
- Plots updated: `figures/fig2_{cost,tokens,walltime}_quality_with_qwen_and_gemini35.png`
