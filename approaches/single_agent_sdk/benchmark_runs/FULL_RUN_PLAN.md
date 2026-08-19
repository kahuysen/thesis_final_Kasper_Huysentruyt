# Full-benchmark run plan — frozen before any full run

**Purpose.** Extend the thesis result (single-agent pipeline, 16-image dev set)
to the full public ChemEagle benchmark, as a held-out test: the pipeline,
prompts, tools, and this plan are frozen at the commit that introduces this
file, before the first scored full-benchmark run.

## Benchmark

- Source: `Benchmark.zip` from https://huggingface.co/datasets/CYF200127/ChemEagle
  (md5 `611d4c01e4ed67f6d560d92448422172`), rebuilt deterministically by
  `scripts/build_full_benchmark.py` into `data/benchmark_full/`.
- 324 images; 321 scored (GT1=139, GT2=78, GT3=38, GT4=66; 3 images have no
  released GT), 2,899 gold reactions. All repairs/exclusions logged in
  `data/benchmark_full/exclusions.json`.
- The 16-image thesis subset is flagged `dev16`; the **headline set is the
  305 held-out images** (scored minus dev16).

## Systems under test

Single-agent pipeline (`pipeline/`, unchanged from the thesis) via OpenRouter
(`scripts/benchmark_openrouter.py`), one pass per model, **provider-default
reasoning settings** (no per-model prompt or parameter tuning):

| Run dir | OpenRouter model id |
|---|---|
| `benchmark_runs/full_qwen38`   | `qwen/qwen3.8-27b` |
| `benchmark_runs/full_gemini3flash` | `google/gemini-3-flash-preview` (thesis model) |
| `benchmark_runs/full_gpt54`    | `openai/gpt-5.4` |
| `benchmark_runs/full_opus47`   | `anthropic/claude-opus-4-7` |

All four runs use a uniform per-round completion cap of
`--max-completion-tokens 16384` (raised from the thesis-era 8192 after the
pre-run smoke test showed that reasoning-by-default models exhaust 8192 on
hidden thinking before their first tool call; the cap is a harness limit, not
a model setting, so it is raised uniformly). Where a model is served by
multiple OpenRouter providers at different quantizations, the run pins the
provider(s) serving unquantized weights via `--provider-pin`; the pin and the
per-request serving provider are recorded in the meta sidecars.

Execution order: cheap → expensive (as listed); each run is scored before the
next starts; the Opus pass only starts after explicit go-ahead.
Concurrency 2–4, in-runner retry with backoff on 429; after a full pass, one
re-invocation retries remaining failures (`skip_existing` resumes); anything
still failing is **scored as an empty extraction (zeros), never dropped**.
The serving provider for every request is recorded in each `*.meta.json`.

## Scoring (frozen rules)

`scripts/eval_full_benchmark.py`:

1. Every scored manifest image counts; missing/failed predictions = zeros.
2. Structural metrics (soft/hard/constitution/partial F1, IoUs, GED) computed
   with conditions stripped from both sides — uniform across slices.
   `hard_match_full` (with conditions) additionally reported on GT1.
3. Condition metrics only where gold has ≥1 condition entry.
4. Wildcard rule: predicted wildcard (R-group template) reactions are dropped
   iff the image's gold contains no wildcard reactions.
5. Bootstrap 95% CIs, 10k image resamples, seed 42.

Comparisons to ChemEagle's published numbers use their GT verbatim; the
thesis-continuity comparison uses `partial_match_f1` (identical definition —
verified: the canonical thesis Opus 4.7 run reproduces 0.854 on dev16 under
this scorer).

## Pre-registered analysis

Headline: per-model macro partial/soft F1 on held-out 305 with CIs, sliced by
GT1–GT4; secondary: cost/time/tokens per image from meta sidecars; findings
reported regardless of whether the thesis ordering holds.
