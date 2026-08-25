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
| `benchmark_runs/full_qwen3vl`  | `qwen/qwen3-vl-235b-a22b-instruct` (see note) |
| `benchmark_runs/full_gemini3flash` | `google/gemini-3-flash-preview` (thesis model) |
| `benchmark_runs/full_gpt54`    | `openai/gpt-5.4` |
| `benchmark_runs/full_opus47`   | `anthropic/claude-opus-4-7` |

**Open-weight slot substitution (documented deviation, pre-scoring).** The
slot was originally `qwen/qwen3.8-27b`. Smoke testing showed its
default-enabled thinking emits 16k–21k reasoning tokens per round (often
without terminating at a 16k cap) and ~500 s/round on the only unquantized
provider (AkashML bf16), extrapolating to Opus-class cost (~$65–100) and
15–20 h for one pass. It was replaced with the non-thinking open-weight
flagship `qwen/qwen3-vl-235b-a22b-instruct` before any scored Qwen run; the
Qwen3.8 diagnostic transcripts are reported in the paper as a finding about
reasoning-by-default open models in agentic loops.

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

**Amendment 2026-08-19.** The open-weight slot runs
`qwen/qwen3-vl-235b-a22b-instruct` (`benchmark_runs/full_qwen3vl`) instead of
`qwen/qwen3.8-27b`: under provider-default reasoning, Qwen3.8 spends 16–21k
thinking tokens per round (~$65–100 and 15–20 h for a full pass) and only
terminates reliably on the bf16 provider with a 32k cap — logged in the smoke
tests; the substitution was confirmed by the user before any scored Qwen run.

**Amendment 2026-08-20 (user decision).** The frontier-Claude slot runs
`anthropic/claude-opus-5` (`benchmark_runs/full_opus5`) — the successor of the
thesis winner, same price tier — instead of `anthropic/claude-opus-4-7`.
Consequence, stated before the run: the full-benchmark table reports the
current frontier model, while the thesis-continuity link for Opus rests on
the dev16 subset only (where 4.7's canonical run scores 0.854 under this
scorer). A full Opus 4.7 pass remains an optional budget-permitting follow-up.

**Amendment 2026-08-26 (Gemini 3.7 Flash dropped).** The conditional
fifth row is not run. A capped 2-image probe ($0.10) showed
`google/gemini-3.7-flash` emits exactly one tool call per LLM round in
this loop (its predecessor batches dozens), needing 40--60+ rounds per
dense figure; a full pass extrapolates to $60--80, outside the
authorized budget. Probe transcripts in `benchmark_runs/probe_gemini37`.
The single-call-per-round serving behaviour is reported in the paper as
a harness-economics observation.

**Amendment 2026-08-25 (cost control).** Remaining runs use
`--max-steps 48` (down from 128). Billing data showed that non-submitting
runs burn their full round budget on quadratically growing context and
write no sidecar, making failures the dominant hidden cost (Qwen3-VL's
real spend ran ~10× its sidecar estimate). Across all completed full
runs, successful extractions use a median of 2–5 rounds and p95 ≤ 38;
the cap affects at most the rare >48-round success (one observed in
867 completed images) while cutting worst-case failure cost ~7×.
Images already completed under the 128 cap are unaffected.

**Amendment 2026-08-20b (user decision).** Added `google/gemini-3.7-flash`
(`benchmark_runs/full_gemini37flash`) as a fifth row, budget permitting —
the current cheap tier, mirroring how Opus 5 modernizes the frontier tier
relative to the thesis-linked Gemini 3 Flash and Opus 4.7. The `:batch`
variant was considered and rejected: batch queuing is incompatible with the
sequential multi-round agent loop.
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
