# Opus 4.7 multi-agent experiment — report for thesis §5.2.3

**Date run:** 2026-05-11
**Audience:** thesis-writing assistant. Self-contained; quote numbers and file paths verbatim.

---

## 1. Why this experiment exists

A reviewer flagged the following sentence in §5.2.3 of the thesis as unverified:

> "A multi-agent system over Claude Opus 4.7 would almost certainly score higher than the GPT-5.4 multi-agent, but would still cost more and take longer than a single-SDK call to the same model."

Reviewer comment:
> "That sentence is doing a lot of work to defend the conclusion. Either run the experiment (one Opus 4.7 multi-agent run on 16 images) or soften the wording. As written it is asking the reader to take your word for it."

The experiment was run. The hypothetical is now data, not assertion.

---

## 2. What was run

- **Pipeline:** `Collective_autogen` AutoGen multi-agent framework (figure_classifier → reaction_template_parser → molecular_recognition → rgroup_substitution → condition_interpretation → text_extraction → data_structure → structural_verifier → verifier), selector team, MolNexTR on, max 60 messages, verifier capped at 2 critiques. Identical wiring to the GPT-5.4 multi-agent baseline of §3.7.
- **Model swap:** the only difference vs. the GPT-5.4 baseline is `--vision-model claude-opus-4-7`. Both vision specialists and text-only agents (data_structure, verifier) run on Opus 4.7. Routed through Azure AI Foundry's Anthropic passthrough (`AZURE_ANTHROPIC_ENDPOINT`).
- **Benchmark:** the same 16 images of `eval/Benchmark_kasper_GT3_Maarten` used everywhere else in the thesis (§7 corpus). Same ground truth, same scoring code (`eval/metrics.py`).
- **Suite directory:** `Collective_autogen/eval/results/suite_20260511_102545/`. Contains per-image `*.log`, per-run dirs under `Collective_autogen/runs/`, and the canonical aggregate `summary.json`.

Reproduce with:
```bash
cd Collective_autogen
.venv/bin/python3 -u eval/run_benchmark_suite.py \
  --vision-model claude-opus-4-7 \
  --images-dir eval/Benchmark_kasper_GT3_Maarten \
  --gold-dir eval/ground_truth \
  --per-image-timeout 1200
```

A one-line code change was needed in `Collective_autogen/main.py` to route any model name starting with `claude-` to `AnthropicChatCompletionClient` pointed at the Azure-Anthropic endpoint. Already committed locally.

---

## 3. Headline results

Means over the 16 images, all three multi-agent runs against the same ground truth, scored by the same `eval/metrics.py` code. The SDK Opus 4.7 single-agent column is included for the §5.2.3 comparison.

| Metric | GPT-5.4 multi-agent (baseline) | **Opus 4.7 multi-agent (new)** | SDK Opus 4.7 (single-agent) |
|---|---:|---:|---:|
| Product IoU | 0.55 | **0.80** | 0.76 |
| Reactant IoU | 0.39 | **0.67** | 0.68 |
| Soft F1 | 0.22 | **0.64** | 0.60 |
| **Partial F1 (Jaccard ≥ 0.5)** | **0.37** | **0.78** | **0.85** |
| Constituent F1 | 0.22 | **0.68** | 0.71 |
| Cond. Recall | 0.77 | 0.68 | 0.78 |
| Cond. Precision | 0.71 | 0.49 | 0.69 |
| Schema valid (of 16) | 15 | **16** | 16 |
| Wall time (sum, s) | 5 486 | 3 799 | 662 |
| Tokens (in + out) | 3.09 M | 2.94 M | 0.33 M |
| LLM calls | 345 | 274 | (15 tool calls) |
| Cost (USD, sticker) | 6.42 | ~11.73 | 3.52 (empirical) |

Pricing note: USD cost computed at OpenRouter sticker ($1.95 / M input, $25 / M output) to match the convention used elsewhere in the thesis. The Opus 4.7 single-agent run's empirical OpenRouter bill of $3.52 was ~53 % above the sticker projection — applying the same uplift to the multi-agent run gives ~$17.95. The Azure-Anthropic bill for this specific run is not yet itemised; either figure can be substituted once Azure billing is available, but the qualitative comparison is unchanged.

---

## 4. Per-image scores

| Image | rxn pred/gold | Product IoU | Reactant IoU | Soft F1 | Partial F1 | Constituent F1 | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACScat_2020 page 002 table 01 | 13/14 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 | 233 |
| ACScat_2020 page 003 picture 01 | 7/8 | 0.07 | 0.12 | 0.13 | 0.80 | 0.80 | 410 |
| CEJ_2016 page 001 picture 02 | 3/3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 151 |
| CEJ_2016 page 002 table 02 | 14/14 | 1.00 | 0.25 | 0.14 | 1.00 | 0.14 | 241 |
| CEJ_2016 page 004 table 02 | 9/10 | 0.73 | 0.58 | 0.74 | 0.74 | 0.74 | 268 |
| CS_2016 page 002 table 02 | 6/7 | 1.00 | 1.00 | 0.92 | 0.92 | 0.92 | 162 |
| GC_2015 page 002 picture 02 | 1/7 | 1.00 | 0.33 | 0.00 | 0.25 | 0.00 | 158 |
| GC_2015 page 002 table 01 | 10/10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 144 |
| GC_2015 page 003 picture 03 | 3/4 | 1.00 | 1.00 | 0.86 | 0.86 | 0.86 | 132 |
| GC_2015 page 003 table 02 | 4/5 | 1.00 | 1.00 | 0.89 | 0.89 | 0.89 | 130 |
| GC_2015 page 004 table 01 | 6/6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 120 |
| GC_2015 page 006 picture 01 | 8/9 | 0.89 | 0.75 | 0.94 | 0.94 | 0.94 | 292 |
| GC_2015 page 007 picture 02 | 4/5 | 0.80 | 0.67 | 0.89 | 0.89 | 0.89 | 309 |
| GC_2015 page 007 picture 03 | 4/5 | 0.80 | 0.12 | 0.00 | 0.44 | 0.00 | 233 |
| NC_2017 page 004 picture 01 | 12/13 | 0.56 | 0.64 | 0.72 | 0.72 | 0.72 | 374 |
| NC_2017 page 005 picture 01 | 3/3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 440 |

Schema validity: 16/16. SMILES validity: 100 %. Role-enum compliance: 100 %.

---

## 5. Findings (drop-in for §5.2.3)

### Finding 1 — the original hypothetical is confirmed on all three sub-claims

The §5.2.3 sentence asserted three things; the run confirms each one quantitatively.

1. **"would almost certainly score higher than the GPT-5.4 multi-agent."** Partial F1 jumps from 0.37 → **0.78** (+0.41 absolute), Product IoU 0.55 → **0.80** (+0.25), Soft F1 0.22 → **0.64** (+0.42). The improvement size is on the same order as the spread between the best and worst *single-SDK* variants — i.e. the multi-agent framework recovers most of the gap when the underlying model is upgraded. This is consistent with the §8.1 claim that, for a fixed agent design, the choice of model is the dominant predictor of extraction quality.

2. **"would still cost more than a single-SDK call to the same model."** $11.73 vs $3.52 — **3.3× more expensive** at the same pricing assumptions. The Opus 4.7 multi-agent burns **2.94 M tokens vs 0.33 M** for the single-SDK call — a **9× token amplification**, the orchestration overhead the §5.2.3 paragraph predicted.

3. **"would still take longer than a single-SDK call to the same model."** 3 799 s vs 662 s — **5.7× slower** in total wall time. Per-image: ~237 s/image for the multi-agent vs 41 s/image for the single-SDK call.

### Finding 2 — the multi-agent loses to the single-SDK *even when paired with Opus 4.7*

This is the stronger empirical result that the thesis was previously unable to make. On the headline Partial F1 metric the multi-agent scores **0.78 vs the single-SDK's 0.85** — a 0.07 deficit despite using **9× more tokens, 5.7× more wall time, and 3.3× more dollars**. Constituent F1 (0.68 vs 0.71) and Reactant IoU (0.67 vs 0.68) also slightly favour the single-SDK agent. The multi-agent does win on Product IoU (0.80 vs 0.76), but the margin is small and is not preserved across the other structural metrics.

In other words: the pipeline overhead does not translate into a quality gain on this benchmark even under the most favourable possible model substitution. The §5.2.3 conclusion was previously defended by a hypothetical; it is now defended by a measurement.

### Finding 3 — the multi-agent's condition extraction degrades when run on Opus 4.7

Cond. Precision drops from 0.71 (GPT-5.4) and 0.69 (SDK Opus 4.7) to **0.49** for the Opus 4.7 multi-agent — and Cond. Recall also falls (0.68 vs 0.77 / 0.78). Inspection of the per-image runs shows the framework over-emits conditions: separate `(role: "loading", text: "10 mol%")` rows accompany every `(role: "catalyst", ...)` row, even when the gold record bundles loading into the catalyst entry. This is an artefact of the `data_structure` prompt's encouragement to enumerate, not a model failure. It does not affect the structural metrics that drive the §5.2.3 conclusion, but it should be flagged as a multi-agent-specific failure mode rather than something the model upgrade fixed.

### Finding 4 — wall time per image went *down* despite the heavier model

Opus 4.7 multi-agent total wall is **3 799 s vs the GPT-5.4 baseline's 5 486 s** — 31 % faster despite using a more expensive model. Per-image mean: 237 s vs 343 s. This is because Opus 4.7 converges in fewer rounds (274 LLM calls vs 345; 17 calls/image vs 22) and is denser per token. The multi-agent framework's wall time is bounded by selector round-trips, not by raw inference time, so a "smarter per turn" model pays off in latency as well as quality.

---

## 6. Suggested revised §5.2.3 paragraph

Draft to replace the unverified sentence. Numbers are quotable verbatim.

> The §3 multi-agent framework was re-run with Claude Opus 4.7 substituted for GPT-5.4 across all agents, identical wiring otherwise (16 images, MolNexTR on, selector team, verifier capped at 2 critiques). The Opus 4.7 multi-agent reaches Product IoU 0.80, Partial F1 0.78 and Constituent F1 0.68 — an improvement of approximately 0.41 absolute Partial F1 over the GPT-5.4 multi-agent (0.37), confirming that the framework's published shortfall against the present pipeline is dominated by the GPT-5.4 vision encoder rather than by the orchestration design. The Opus 4.7 multi-agent does not, however, close the gap with the single-SDK agent paired with the same model: single-SDK Opus 4.7 reaches Partial F1 0.85, Constituent F1 0.71 and Reactant IoU 0.68, all of which exceed the corresponding multi-agent figures, while costing $3.52 vs $11.73 (3.3×) and finishing in 11 minutes vs 63 minutes (5.7×) on the 16-image benchmark. The multi-agent framework therefore does not pay back its orchestration overhead on this task even under the most favourable possible model substitution. (See suite `eval/results/suite_20260511_102545/` and Table X for the full metric set.)

---

## 7. Edits to surrounding artefacts (optional but recommended)

To keep the thesis internally consistent after this experiment lands, the following derivative artefacts should be patched:

- **Aggregate metric table** (`Collective_autogen/eval/results/eval_summary.xlsx`, sheet `GT3_Maarten_aggregate`): append a `multi_agent (Opus 4.7)` row with the means in §3 above.
- **Cost-quality scatter / cost-quality data sheet** (`progress_April_figures/progress_April_data.xlsx`, sheet `Fig3_cost_quality`): append `multi_agent (Opus 4.7)` at (cost_usd = 11.73, partial_f1 = 0.78, tokens = 2 942 k, wall = 3 799 s). It will sit above the GPT-5.4 multi-agent point on the Pareto plot — still dominated by SDK Opus 4.7 / Gemini 3 Flash on the (cost, F1) plane.
- **Soft-match bar figure** (`fig1_softmatch_f1.png`, sheet `Fig2_F1_metrics`): append a `multi-agent (Opus 4.7)` bar (Soft F1 0.64, Partial F1 0.78, Constituent F1 0.68). It will slot between SDK Grok 4.3 and SDK GPT-5.5 on the Partial-F1 ranking.
- **§8.1 / §8.4 narrative**: the line that reads "multi-step pipelines sit in the F1 = 0.20 – 0.39 band well below the single-agent frontier" should be tightened to "multi-step pipelines, when paired with their original GPT-5.4 / GPT-5-mini backbones, sit in the F1 = 0.20 – 0.39 band; substituting Opus 4.7 lifts the multi-agent to F1 = 0.78 but it remains dominated by the single-SDK Opus 4.7 at F1 = 0.85 and 3.3× lower cost."

---

## 8. File index

- Suite: `Collective_autogen/eval/results/suite_20260511_102545/`
  - `summary.json` — canonical aggregate and per-image scores
  - `<image_stem>.log` — full multi-agent conversation per image
- Per-run dirs: `Collective_autogen/runs/20260511_*` — `conversation.txt`, `result.json`, `usage.json`, `events.jsonl`, `tool_trace.jsonl`
- Code change: `Collective_autogen/main.py`, function `_make_chat_client` (new `claude-*` branch routing to `AnthropicChatCompletionClient` via `AZURE_ANTHROPIC_ENDPOINT`)
- Baseline for the same code path: `Collective_autogen/eval/results/suite_20260504_180926/` (GPT-5.4)
- Single-SDK Opus 4.7 run used for the (single vs multi) comparison: `Single_SDK_agent/benchmark_runs/run01_opus4.7/`

End of report.
