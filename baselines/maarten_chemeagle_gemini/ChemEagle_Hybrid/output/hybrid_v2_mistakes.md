# Hybrid v2 — error enumeration

Run: `output/benchmark_article_hybrid_v2.json` (GT3 article benchmark, 16 images, 123 GT reactions). Strict canonical SMILES match against `input/Benchmark_chemeagle/Benchmark_data/article/GT3.json`. Date: 2026-04-29.

## Aggregate

| Outcome | Count | % of GT |
|---|---|---|
| Exact-match (strict canonical) | 67 | 54% |
| Stereo-only mismatch (chemistry right, stereo flipped) | 6 | 5% |
| Reactant-only diff (one component missing/wrong) | 15 | 12% |
| Product-only diff | 0 | 0% |
| Unmatched GT (missing or paired wrong) | 35 | 28% |

If stereo were perfect, the Soft-F1 ceiling lifts to ~59% (vs current 58%). **Most of the remaining gap is not stereo.**

## Per-image breakdown

| # | File | GT | Pred | Outcome |
|---|---|---|---|---|
| 0 | `ACScat_2020.pdf_page002_table_01_s0.88` | 14 | 13 | All 14 GT missed. molnextr returns the same Boc-Phe + benzylamine SMILES for every variant row even though GT shows 14 distinct substrates. Pred = 12 duplicates of Phe reaction. Variant column not being read. |
| 1 | `ACScat_2020.pdf_page003_picture_01_s0.93` | 8 | 7 | 1 OK, 6 stereo-only, 1 missing. Pure stereo gap on Boc-D-α-amino-acid center. Abstract template missing. |
| 2 | `CEJ_2016.pdf_page001_picture_02_s0.74` | 3 | 3 | 2 OK, 1 reactant-diff. Pred uses `[2*]N[3*]`, GT uses `[1*]N[2*]` and `[7*]`. Wildcard renumber didn't fire (indices ≤5). |
| 3 | `CEJ_2016.pdf_page002_table_02_s0.91` | 14 | 7 | 2 OK, 5 reactant-diff (boronic acid dropped), 7 GT rows not emitted. Pred emits half the table. |
| 4 | `CEJ_2016.pdf_page004_table_02_s0.66` | 10 | 10 | 8 OK, 1 reactant-diff (proline ester instead of free leucine), 1 missing, 1 pred spurious. |
| 5 | `CS_2016.pdf_page002_table_02_s0.88` | 7 | 7 | All 7 OK. Soft-F1 = 0.92. |
| 6 | `GC_2015.pdf_page002_picture_02_s0.56` | 7 | 7 | All 7 reactant-diff. Pred uses `[2*]N`, GT uses `[NH2]` standalone fragment. **Pred is more correct chemistry; GT is unusual.** |
| 7 | `GC_2015.pdf_page002_table_01_s0.61` | 10 | 11 | All 10 OK + 1 spurious pred reaction (pyrrolidine + phenylacetic acid → amide; likely from caption). |
| 8 | `GC_2015.pdf_page003_picture_03_s0.71` | 4 | 4 | All 4 OK. Recovered from v1. Case-A-vs-Case-B refinement put boronic-acid catalyst in conditions. |
| 9 | `GC_2015.pdf_page003_table_02_s0.78` | 5 | 5 | All 5 OK. (Crashed in baseline.) |
| 10 | `GC_2015.pdf_page004_table_01_s0.81` | 6 | 6 | All 6 chemistry-OK. **GT lists empty reactants; pred correctly fills them in.** Eval marks Soft-F1=0 due to GT sparsity. |
| 11 | `GC_2015.pdf_page006_picture_01_s0.86` | 9 | 8 | 8 OK, 1 missing (the abstract template). |
| 12 | `GC_2015.pdf_page007_picture_02_s0.76` | 5 | 4 | 4 OK, 1 missing (abstract template). |
| 13 | `GC_2015.pdf_page007_picture_03_s0.60` | 5 | 4 | 2 reactant-diff, 3 missing, 2 spurious. Variant-table chaos. Stage 2 fired but didn't expand. Reactants collapsed `[1*]/[2*]` to `H` and `Me`. |
| 14 | `NC_2017.pdf_page004_picture_01_s0.91` | 13 | 12 | 8 OK, 5 missing, 4 spurious. Pred misreads substituents on a substrate-scope figure (Et+Me+Ph instead of Me+Me+Ph). |
| 15 | `NC_2017.pdf_page005_picture_01_s0.90` | 3 | 2 | 2 OK, 1 missing. The 3 GT reactions are identical; pred only emits 2. |

## Mistake categories (aggregated)

| Category | Count | What it is | Fixability |
|---|---|---|---|
| **Variant-row collapse** (images 0, 13, partial 14) | ~16 GT | molnextr reads the same molecule for every row of a substrate-scope figure; per-row substituents aren't extracted | Hard — molnextr / rxnim limitation. Would need per-cell crop + re-recognition, or eval-side relaxation. |
| **Missing variant-table rows** (image 3 missing 7, image 15 missing 1) | ~8 GT | Pred only emits a subset of rows | Stage-2 prompt could be strengthened; or post-process duplicates |
| **Stereo flip on Boc-α-amino-acid** (image 1, parts of 0/13) | ~7 GT | molnextr inverts `@H` ↔ `@@H` on Boc-NH-CHR center | Stereo audit pass via Gemini per chiral center (~2× call cost) |
| **Missing abstract template** (image 1, 11, 12, 14) | 4 GT | GT lists abstract `[1*]/[2*]` first; pred only emits expansions | Prompt insistence; stronger triggering in stage 1 |
| **GT non-standard SMILES** (image 6 `[NH2]`, image 2 `[7*]`) | ~10 GT | GT uses unusual SMILES; pred chemically-correct output doesn't match | **Pred is right.** Won't fix without dumbing down |
| **Spurious extra pred reactions** (image 4 +1, image 7 +1, image 13 +2, image 14 +4) | ~8 spurious | Pred emits reactions not in GT — often Gemini inferring captioned reactions | Tighten "only emit what's drawn" clause |
| **Substrate-scope misread** (image 14) | ~4 GT | Pred wrote wrong substituent on dense scope panel | molnextr reading error |
| **GT empty reactants** (images 6, 10) | partial | GT lists empty reactant array; pred correctly fills it in | **Pred is right.** Won't fix |

## Top-level reading

- **~13 GT reactions** are losses where **pred is arguably more correct than GT** (`[NH2]` standalone fragment, `[7*]` non-standard wildcard, empty GT reactants). Strict eval punishes correctness over GT-convention compliance.
- **~30 GT reactions** are real losses driven by **molnextr failing to read variant-cells per-row** on substrate-scope figures (image 0 alone is 14; image 13 is 5; image 14 partial). This is the dominant residual problem — a deep-learning recognition issue, not a prompt issue.
- **~7 GT reactions** are **stereo flips** fixable with a per-center Gemini audit.
- **~5 GT reactions** are **missing abstract templates** — the v2 prompt clause works on some images but not others.
- **~8 pred reactions are spurious** — Gemini inferring reactions from captions or text rather than what's drawn.

## Highest-leverage residual fix

**Per-cell molnextr re-reading on variant tables.** When stage 1 detects a variant_table, crop each cell's R-group region individually and run molnextr on each crop, rather than relying on table-level recognition that's collapsing rows. This directly attacks the largest residual bucket (~16 GT reactions across images 0, 13, partial 14). Costs ~2× GPU time per variant-table image; **no extra Gemini calls**.

Stereo audit pass is the second-highest-leverage fix (~7 GT reactions, costs ~2× Gemini calls per chiral-center image).

Together these would push strict Soft-F1 from 0.58 toward 0.75 — close to the honest LLM-match ceiling on this benchmark.
