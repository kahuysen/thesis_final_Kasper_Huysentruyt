# ReactionMiner

A vision-LLM agent that turns chemistry figures from journal PDFs into validated, structured reaction data — and renders one reaction card per equation it finds.

This repository is the implementation accompanying a master's thesis. It pairs an Anthropic Claude vision agent with [RDKit](https://www.rdkit.org/) for SMILES validation and [Rxn-INSIGHT](https://github.com/mrodobbe/Rxn-INSIGHT) for downstream classification, naming, and chemoinformatic analysis.

---

## What it does

Given a figure image (a "Table N. Scope of …" or "Figure N. Control experiments" cropped from a paper), the pipeline produces:

1. **A structured `FigureExtraction` JSON** — every reactant, reagent, product, condition, yield, and HRMS value.
2. **One PNG card per reaction** — reactants + reagents + arrow + conditions + products, drawn with RDKit's 2D depiction.
3. **A flat CSV** for downstream analysis.
4. **A Rxn-INSIGHT analysis** per reaction — class (e.g. *C-C Coupling*), named reaction (e.g. *Heck terminal vinyl*), functional groups, by-products, fingerprints, scaffold — written as a Section-5.9-compatible CSV.

Internally, an Anthropic Claude model runs an agent loop with three tools:

| Tool | Purpose |
|------|---------|
| `validate_smiles` | RDKit canonicalization + formula + exact mass |
| `compute_exact_mass` | Adduct-aware monoisotopic mass calc |
| `submit_extraction` | Terminator — its argument **is** the final structured payload (Pydantic-validated) |

Delivery surface:

- **CLI** (`cli.py`) — batch runs (`extract`, `render`, `benchmark`) for processing folders of figures and scoring against ground truth.

---

## Repository layout

```
cli.py                  Batch CLI (extract / render / benchmark)
make_cards.py           Early standalone card-rendering demo
smoketest.py            Offline pipeline test (no API call)

pipeline/
  schema.py             Pydantic models for FigureExtraction
  config.py             Backend selection (direct Anthropic / Azure-hosted Claude)
  extractor.py          Agent loop (streaming + sync entry points)
  tools.py              Tool definitions (validate_smiles, compute_exact_mass)
  renderer.py           JSON → PNG card rendering (matplotlib + RDKit)
  flatten.py            JSON → CSV flattener
  batch.py              Multi-image batch orchestrator
  eval.py               Bipartite scorer vs. ground-truth JSON
  rxn_insight.py        Wrapper around the Rxn-INSIGHT subprocess

subprocess_drivers/
  rxn_insight_runner.py Runs Rxn-INSIGHT inside its own venv (numpy 1.x / Python 3.12)

corpus/                 Benchmark images + ground-truth JSON
```

---

## Setup

```bash
# main env
python3.13 -m venv .venv
.venv/bin/pip install anthropic pydantic rdkit matplotlib pillow python-dotenv

# secondary env for Rxn-INSIGHT (its deps pin numpy<=1.26)
python3.12 -m venv .venv-rxn-insight
VIRTUAL_ENV=.venv-rxn-insight .venv-rxn-insight/bin/pip install \
    rxn-insight 'scipy>=1.11,<1.15' joblib
```

Copy `.env.example` to `.env` and fill in either:

- direct Anthropic: `ANTHROPIC_API_KEY=sk-ant-...`
- Azure-hosted Claude: `LLM_PROVIDER=azure` + `AZURE_ANTHROPIC_ENDPOINT` + `AZURE_ANTHROPIC_API_KEY` + `AZURE_DEPLOYMENT_NAME`

`.env` is gitignored — never commit a populated copy.

---

## Usage

```bash
# Batch extraction over a folder
.venv/bin/python3 cli.py extract corpus/ --out results/

# Re-render cards from existing JSON (no API call)
.venv/bin/python3 cli.py render results/ --out results/

# Run + score against ground truth
.venv/bin/python3 cli.py benchmark corpus/Benchmark_kasper_GT3_Maarten/ \
    --out benchmark_runs/run01
```

---

## Status

Best benchmark numbers across 16 figures from ACS Catalysis, Chemistry – A European Journal, Chemical Science, Green Chemistry, and Nature Communications:

| metric | value |
|---|---|
| Reaction recall | **0.88** |
| Product recall | **0.88** |
| Product precision | **0.95** |
| Yield accuracy | 0.65 |
| SMILES validity | **1.00** |

Six of sixteen figures extract perfectly (100% recall on every reaction). Remaining gaps are concentrated in two figures with known ground-truth annotation issues.

---

## License

This is research software accompanying an academic thesis; treat as MIT-licensed unless otherwise noted in individual files. The benchmark figures under `corpus/` are crops from published journal articles — used here for academic evaluation only.
