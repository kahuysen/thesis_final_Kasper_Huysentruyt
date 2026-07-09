# thesis_AgenticChem — Agentic extraction of chemical reactions from figures

Code, benchmarks and results for the master's thesis of **Kasper Huysentruyt** (Ghent University).
The full report is in [`paper/Thesis_Kasper_Huysentruyt.pdf`](paper/Thesis_Kasper_Huysentruyt.pdf).

> **Research question.** Reaction records that retrosynthesis, yield- and condition-prediction
> models depend on are still built by hand from figures and PDFs. Vision-language models inside
> tool-using agentic loops could automate this — but is extraction quality set by the *model* or by
> the *orchestration* around it? We build and benchmark four extraction pipelines on **sixteen
> reaction figures** from five chemistry journals, from a large multi-agent system down to a single
> model with two chemistry tools.
>
> **Finding.** *The model, not the framework, decides the outcome.* The simplest single-agent design
> (**Claude Opus 4.7, partial-F1 = 0.85**) surpassed every multi-step system.

This repository contains **everything needed to reproduce the thesis experiments**: the pipelines,
the baselines, the benchmark data, the canonical runs, and the scripts that turn those runs into the
paper's figures. The ~10 GB of model weights are the only thing not included — they are downloaded
separately (see [`models/README.md`](models/README.md)).

## Layout

```
thesis_AgenticChem/
├── approaches/            My own contributions (the four pipelines that are compared)
│   ├── single_agent_sdk/     Single Claude-vision agent + 2 tools (WINNER)  — canonical runs in benchmark_runs/
│   ├── collective_autogen/   Multi-agent AutoGen pipeline (6 specialists)   — eval harness in eval/
│   └── chemeagle_gemini/     ChemEagle with the LLM swapped to Gemini
├── baselines/            Third-party systems, vendored verbatim (LICENSE files kept)
│   ├── chemeagle/            Upstream ChemEagle (CYF2000127)
│   ├── rxn_insight/          Upstream Rxn-INSIGHT (mrodobbe)
│   └── maarten_chemeagle_gemini/  Maarten Dobbelaere's hybrid variant (used with permission — see ATTRIBUTION.md)
├── app/                  ReactionMiner — FastAPI + web UI demo of the single-agent pipeline
├── evaluation/          ⭐ Canonical results surface: eval_summary.xlsx, cost/F1 points, heatmaps
├── outputs/             Final paper figures generated from the runs (pipeline_comparison, failure modes)
├── paper/               The thesis PDF, extended abstract, and all thesis figures (paper/figures/)
├── data/                Benchmark images, test papers, SMIRKS DB
├── models/              Model weights — EMPTY on GitHub; download from Hugging Face (see README)
└── tests/               Layout / smoke checks (no weights or API calls needed)
```

## ⭐ Canonical results — what backs the paper

Because the experiments were run many times, this is the **single source of truth** for the numbers
and figures in the thesis:

| Artifact | Path | Backs |
|---|---|---|
| Per-system partial-F1 vs cost/time/tokens | [`evaluation/results/cost_vs_f1/points.json`](evaluation/results/cost_vs_f1/points.json) | Figures 6–8 |
| Master evaluation spreadsheet | [`evaluation/results/eval_summary.xlsx`](evaluation/results/eval_summary.xlsx) | Heatmap (Figure 5) |
| Model price table | [`evaluation/model_prices.json`](evaluation/model_prices.json) | Cost axes |
| Single-agent raw runs (Opus 4.7 = 0.85, etc.) | `approaches/single_agent_sdk/benchmark_runs/` | Winning system |
| Multi-agent + baseline suites | `approaches/collective_autogen/eval/results/` | Multi-step systems |
| Final figures as used in the thesis | [`paper/figures/`](paper/figures/) | All results figures |

Verified against the submitted PDF: Opus 4.7 → 0.854, Gemini 3 Flash → 0.790, Gemini 3.1 Pro → 0.832,
multi-agent → 0.370, ChemEagle → 0.394. See [`evaluation/README.md`](evaluation/README.md) for how to
regenerate everything from the raw runs.

## Quickstart

```bash
git clone https://github.com/kahuysen/thesis_AgenticChem.git
cd thesis_AgenticChem

# 1. (Only for ChemEagle-based pipelines) download the ~10 GB weights — see models/README.md
# 2. Create a venv per pipeline you want to run:
cd approaches/single_agent_sdk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# … repeat per project, using each one's requirements.txt / pyproject.toml

# 3. Copy and fill in API keys (see "Setup — API keys"):
cp approaches/single_agent_sdk/.env.example approaches/single_agent_sdk/.env

# 4. Sanity-check the layout (no weights / API calls needed):
bash tests/run_tests.sh
```

To try the web demo:

```bash
cd app
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # fill in one provider's keys
.venv/bin/uvicorn app.main:app --reload --port 8000   # open http://localhost:8000
```

## Setup — API keys

Each pipeline reads credentials from a `.env` in its own folder; a `.env.example` ships next to each.
**You don't need every key** — only the provider you actually plan to call.

| Pipeline | Which keys does it need? | What does it call? |
|---|---|---|
| `single_agent_sdk` | one of `ANTHROPIC_API_KEY`, `AZURE_ANTHROPIC_*`, `AZURE_OPENAI_*`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | Claude / GPT / Gemini (selected per request) |
| `collective_autogen` | `AZURE_OPENAI_*` **or** `OPENAI_API_KEY` (+ optional `AZURE_ANTHROPIC_*` for Claude vision) | 6 GPT-class agents over AutoGen |
| `chemeagle_gemini` | `GEMINI_API_KEY` | Gemini via OpenAI-compat endpoint |
| `baselines/chemeagle` | `API_KEY`, `AZURE_ENDPOINT`, `API_VERSION` (non-standard names — see file) | Azure OpenAI |
| `app` | any one provider (Anthropic / Azure / Gemini / OpenRouter) | single-agent pipeline |

## Setup — picking a model

| Pipeline | How to choose / change the model |
|---|---|
| `single_agent_sdk` | CLI flag `--model <id>`; default from `DEFAULT_MODEL` / `AZURE_DEPLOYMENT_NAME` / `GEMINI_MODEL` in `.env` |
| `collective_autogen` | CLI flags `--vision-model <id> --mini-model <id>` (for Azure, `<id>` = deployment name) |
| `chemeagle_gemini` | `.env`: `GEMINI_MODEL=<id>` (e.g. `gemini-3-flash-preview`) |
| `baselines/chemeagle` | **Hardcoded** — edit the `model=` literals in `baselines/chemeagle/main.py` |

## How the heavy weights are wired

ChemEagle's multi-GB checkpoints (`ner.ckpt`, `rxn.ckpt`, `moldet.ckpt`, `corefdet.ckpt`,
`molnextr.pth`, `biobert-large-cased/`, `cre_models_v0.1/`, `Tesseract-OCR/`) live in `models/` and are
referenced through **relative symlinks** from `baselines/chemeagle/` and `approaches/chemeagle_gemini/`.
They are excluded from Git; download them from Hugging Face — see [`models/README.md`](models/README.md).

## References & acknowledgements

This work builds on several open-source projects — please cite them if you reuse this code:

- **ChemEagle** — upstream baseline and source of the model weights. https://github.com/CYF2000127/ChemEagle
- **AutoGen** — multi-agent framework used by `approaches/collective_autogen`. https://github.com/microsoft/autogen
- **Rxn-INSIGHT** — reaction-classification library, vendored under `baselines/rxn_insight`.
  https://github.com/mrodobbe/Rxn-INSIGHT
  > M. R. Dobbelaere, I. Lengyel, C. V. Stevens, K. M. Van Geem, "Rxn-INSIGHT: fast chemical reaction
  > analysis using bond-electron matrices", *J. Cheminform.*, 16(1), Mar. 2024.
- **`baselines/maarten_chemeagle_gemini`** — hybrid variant by **Maarten Dobbelaere**, included with his
  permission. See its `ATTRIBUTION.md`.

## Not included (by design)

Model weights (`*.ckpt`, `*.pth`, …, ~10 GB — download separately), virtual environments (`.venv*`),
`node_modules/`, populated `.env` files, per-session scratch (`app/runs/`, caches), and extracted
archives. See `.gitignore` for the full list.
