# run_qwen3vl_30b_a3b — full 16-image benchmark

**Model:** `qwen/qwen3-vl-30b-a3b-instruct` (via OpenRouter)
**Date:** 2026-05-18
**Backend:** OpenRouter (`https://openrouter.ai/api/v1`)
**Concurrency:** 2
**Wall-clock totals:** ~328 s sum across the 11 successful images
(29.8 s avg / image), 144 tool calls,
232 k input tokens, 33 k output tokens.

## Result summary

| | value |
|---|---|
| Images extracted | **11 of 16** |
| Images that hit the agent step cap (128) without submitting | **5 of 16** |
| Aggregate reaction recall | **0.00** |
| Aggregate product recall | **0.00** |
| Aggregate product precision | **0.00** |
| Aggregate yield accuracy | 0.00 |
| Aggregate SMILES validity | **1.00** |

Every image on which the model produced an extraction scored **0/N**
reactions and **0/N** products matched against the ground truth.

## Per-image outcomes

| image | result | matched rxn / GT rxn |
|---|---|---|
| ACScat_2020 p2 table_01 (optimization, Opus 4.6/4.7 both got 13/14) | OK | 0 / 14 |
| ACScat_2020 p3 picture_01 | ERR — step cap | — |
| CEJ_2016 p1 picture_02 | OK | 0 / 3 |
| CEJ_2016 p2 table_02 | OK | 0 / 14 |
| CEJ_2016 p4 table_02 | ERR — step cap | — |
| CS_2016 p2 table_02 | OK | 0 / 7 |
| GC_2015 p2 picture_02 (catalyst gallery) | OK | 0 / 7 |
| GC_2015 p2 table_01 | OK | 0 / 10 |
| GC_2015 p3 picture_03 | OK | 0 / 4 |
| GC_2015 p3 table_02 | OK | 0 / 5 |
| GC_2015 p4 table_01 | OK | 0 / 6 |
| GC_2015 p6 picture_01 | OK | 0 / 9 |
| GC_2015 p7 picture_02 | ERR — step cap | — |
| GC_2015 p7 picture_03 | ERR — step cap | — |
| NC_2017 p4 picture_01 (DATB scope, Opus 4.6 collapsed here too) | ERR — step cap | — |
| NC_2017 p5 picture_01 (peptide comparative) | OK | 0 / 3 |

OK = `submit_extraction` called within the per-image step cap.
ERR = agent did not call `submit_extraction` within 128 steps —
the model entered a `validate_smiles` → parse-error → retry loop and
never converged.

## Failure modes

Two distinct failure modes, both rooted in vision rather than reasoning:

**1. Systematic low-level perception errors (on the 11 OK images).**
Even when the model finishes the extraction cleanly, the SMILES it
emits are valid but structurally wrong. Three recurring substitutions
are visible across the run (verified by direct comparison against the
GT on the ACScat optimization-table image):

- **Boc** `–OC(=O)O–C(CH₃)₃` → **ethoxycarbonyl** `–OC(=O)O–CH₂CH₃`
  (two methyl branches of *tert*-butyl missed; Cbz/Boc/CO₂Et all read
  as the simplest carbamate).
- **Benzylamide** `–C(=O)NH–CH₂–Ph` → **anilide** `–C(=O)NH–Ph`
  (CH₂ linker dropped).
- **Phenylalanine** `–CH(R)–CH₂–Ph` → **homophenylalanine**
  `–CH(R)–CH₂–CH₂–Ph` (one CH₂ added).

These three errors, present on every Phe/Boc/Bn-containing image,
account for the 0/N scores. The same failure modes appear in
Opus 4.6 case studies 2 and 4 but only on hard images; on
Qwen3-VL-30B-A3B they appear on every image including the optimization
table that both Opus models scored 13/14 on.

**2. Step-cap loops (on the 5 ERR images).**
On denser figures the model emits chemist-shorthand strings like
`MeO2C[C@H]([1*])NHO` ("Me", "Et", "Bn", "OAc" as if they were atom
symbols), which `validate_smiles` rejects. The model then issues a
near-identical shorthand string, fails again, and the loop continues
until the per-image step cap is exhausted. This is the same failure
the 8B variant exhibited on every image
(see `../run_qwen3vl_8b/README.md`); the 30B-A3B exhibits it only on
the 5 hardest figures.

## How to interpret in the thesis

This run is suitable as a **capability-floor / cost-quality lower-bound
data point**, not as a competitive baseline. The natural framings:

- **Vision quality is the binding constraint.** The model produces
  valid SMILES grammar and a well-formed `FigureExtraction` object on
  most images, but reads molecular structure off the page incorrectly
  on every image where it doesn't time out. None of the agent
  scaffolding (`validate_smiles`, schema validation, retry on
  validation failure) can recover quality from an upstream perception
  failure.
- **The Qwen perception errors mirror Opus 4.6's errors but are
  unconditional.** Opus 4.6 makes the same Boc→Et / benzyl→phenyl /
  CH₂-count errors only on the harder case-study images (the DATB
  scope table and the peptide comparative study); Qwen3-VL-30B-A3B
  makes them on every image including ones with single-feature
  structures. This is consistent with vision being the bottleneck
  across the model class, not a specific limitation of one family.
- **SMILES validity is uninformative as a quality metric here.**
  Both the run and Opus 4.6/4.7 score 100% on SMILES validity but the
  underlying extraction quality differs by an order of magnitude.
  Reaction- and product-recall, not validity, are the metrics that
  separate the models.

## Files

- `*.json` / `*.meta.json` — the 11 successful extractions
  (predictions and per-image cost metadata).
- No file is written for the 5 ERR images.
- `README.md` — this file.
