# run_qwen3vl_8b — partial / aborted run

**Model:** `qwen/qwen3-vl-8b-instruct` (via OpenRouter)
**Date:** 2026-05-18
**Wall-clock before abort:** ~20 min
**Result:** **1 of 16 images extracted** before the run was killed.

## Failure mode

The agent loop did not converge on this model. The single image that
produced a JSON file (`ACScat_2020.pdf_page002_table_01_s0.88.json`)
finished within ~1 min of starting; every subsequent image got stuck.

Inspection of the run log
(`/tmp/run_qwen3vl_8b.log`, ~5 600 lines) shows the same pattern:

- Qwen3-VL-8B-Instruct emits **chemist shorthand strings** rather than
  RDKit-parseable SMILES — examples seen in the log include
  `MeO2C(*)NCHO`, with abbreviated groups like `Me`, `Et`, `Bn`,
  `OAc`, `OMs` etc. used as if they were atom symbols.
- Each such string is rejected by the `validate_smiles` tool
  (`SMILES Parse Error: syntax error while parsing: …`).
- The agent issues another `validate_smiles` call with a near-identical
  shorthand string, fails again, and the loop repeats up to the
  per-image step cap (`MAX_AGENT_STEPS = 12`) and retry cap
  (`max_retries = 5` in `scripts/benchmark_openrouter.py`).
- Net effect: each image consumes the full retry budget without
  producing a `submit_extraction` call.

This is a **capability floor** rather than an agent-loop bug. The
model can recognise that the figure is an amide-coupling table
(visible in its tool-call inputs) but cannot emit the canonical SMILES
grammar that the validator requires. Stronger models in the same
family (`qwen/qwen3-vl-30b-a3b-instruct`) and Claude/GPT-class models
do not exhibit this failure on the same images.

## Status of files in this folder

| file | meaning |
|---|---|
| `ACScat_2020.pdf_page002_table_01_s0.88.json` | one successful extraction — quality not assessed; may itself be partially invalid. |
| `ACScat_2020.pdf_page002_table_01_s0.88.meta.json` | run metadata sidecar for the one extraction. |
| `README.md` | this file. |

The run was **not retried** with this model. If used in the thesis,
report it as a lower-bound / capability-floor data point rather than
as a comparable benchmark score.
