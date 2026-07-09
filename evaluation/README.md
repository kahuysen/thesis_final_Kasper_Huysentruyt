# Evaluation — canonical results & how to reproduce them

This folder is the **highlighted, single source of truth** for the numbers and figures in the thesis.
The full evaluation *harness* (run scripts, metrics, ground truth) lives with the multi-agent pipeline
at [`../approaches/collective_autogen/eval/`](../approaches/collective_autogen/eval/); the curated
canonical outputs are copied here for quick reference.

## Canonical files

| File | What it is |
|---|---|
| `results/eval_summary.xlsx` | Master spreadsheet: per-image and aggregate scores for every system. Drives the heatmap (thesis Figure 5). |
| `results/cost_vs_f1/points.json` | Per-system partial-F1 with cost / wall-time / token axes. Drives Figures 6–8. |
| `results/cost_vs_f1/f1_vs_cost.png`, `f1_vs_time.png`, `f1_vs_tokens.png` | The generated cost/time/token-vs-F1 plots. |
| `results/eval_summary_heatmap_ugent.png`, `..._simplified_ugent.png` | The per-system heatmap. |
| `model_prices.json` | Per-model USD pricing used for the cost axis. |

**Verified against the submitted thesis PDF** (`../paper/Thesis_Kasper_Huysentruyt.pdf`):

| System | partial-F1 |
|---|---|
| single_agent (Claude Opus 4.7) | **0.854** |
| single_agent (Gemini 3.1 Pro) | 0.832 |
| single_agent (Gemini 3 Flash) | 0.790 |
| ChemEagle (baseline) | 0.394 |
| multi_agent (collective_autogen) | 0.370 |
| ChemEagle_Hybrid | 0.200 |

> Note: earlier, superseded spreadsheets exist elsewhere in the source tree (e.g. an older
> `GT3_Maarten` 3-system sheet). Those are **not** canonical — this file is.

## Where the raw runs live

- **Single-agent** (the winning system, all models): `../approaches/single_agent_sdk/benchmark_runs/`
  (e.g. `run_opus46/`, `opus47_new_prompt/`, `run01_opus4.7/`, `run_gemini31propreview/`).
- **Multi-agent + baselines** (suites): `../approaches/collective_autogen/eval/results/` (`suite_*`,
  `chemeagle_*`, `chemeagle_gemini_*`, `chemeagle_hybrid_*`).

## Reproduce from scratch

From `../approaches/collective_autogen/eval/` (venv + API keys set — see the top-level README):

```bash
# 1. Re-run a suite for a given system (example: single-agent over the benchmark)
python run_single_agent_suite.py       # or run_chemeagle_suite.py, run_chemeagle_gemini_suite.py, …

# 2. Rebuild the master spreadsheet from the run outputs
python build_eval_xlsx.py              # -> results/eval_summary.xlsx

# 3. Regenerate the cost / time / token vs partial-F1 plots
python plot_cost_vs_f1.py              # -> results/cost_vs_f1/*.png + points.json

# 4. (Optional) failure-mode breakdown used in the discussion
python failure_mode_analysis.py
```

The head-to-head pipeline comparison figure in [`../outputs/`](../outputs/) is regenerated with
`../outputs/pipeline_comparison_plot.py`.
