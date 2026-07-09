# Automated SMIRKS Generalization and Validation via Large Language Models

## Overview

This document describes the methodology for automatically generalizing specific reaction templates (SMIRKS) into broad, class-level SMIRKS patterns using a large language model (LLM), followed by rigorous cross-class false-positive testing and iterative refinement. The pipeline operates in five sequential phases, each with persistent checkpointing to allow interruption and resumption.

**Terminology note.** Throughout this document, *LLM* refers to a large language model — a neural network trained on vast text corpora that can follow natural-language instructions to perform reasoning tasks. We use Google's Gemini model, though the methodology is model-agnostic. The term *fine-tuning* in Phase 4 refers to *iterative prompt-based refinement* (re-prompting the LLM with error feedback), **not** to updating the model's weights, which is a different procedure in machine learning.

---

## Phase 1: Stratified Train/Test Splitting

To enable unbiased evaluation, the reaction database is split into disjoint training (80%) and test (20%) sets using stratified sampling. The split is performed independently within each tier-3 reaction class to preserve class proportions.

**Procedure:**
1. Reactions are grouped by their tier-3 class label.
2. Classes with fewer than a configurable minimum number of reactions (default: 5) are excluded.
3. Within each class, reaction indices are randomly shuffled using a fixed random seed (default: 42) for reproducibility.
4. The first 20% of shuffled indices are assigned to the test set, the remaining 80% to the training set. For small classes (5–9 reactions), at least one reaction is guaranteed in the test set.
5. Only integer row indices are stored (not the data itself), ensuring the split is lightweight and reproducible.

---

## Phase 2: SMIRKS Generalization via Few-Shot Prompting

For each tier-3 reaction class, a generalized set of SMIRKS is produced from the class's most frequent specific templates using an LLM.

### 2.1 Template Screening

Before invoking the LLM, the most representative templates for each class are selected from the **training set only**:

1. Templates (extracted at radius 0 for reactants and radius 1 for products, without ring closure information) are ranked by frequency within the class.
2. The top-*N* most frequent templates are selected, starting at *N* = 10 and doubling iteratively until their cumulative frequency covers at least 90% of the class's training reactions, up to a maximum of 50 templates.
3. For each selected template, the shortest example reaction in the training set that uses that template is retrieved as a concrete illustration.

### 2.2 Prompt Design

The LLM receives two components:

**System prompt (persistent instructions).** A detailed, domain-specific instruction set that defines SMARTS notation, establishes rules for handling reaction center (RC) atoms, context atoms, and leaving groups, and prescribes a six-step reasoning protocol (A–F). The system prompt includes three fully worked *few-shot examples* — complete input/reasoning/output demonstrations for three representative reaction classes (alcohol oxidation, N-alkylation, and reductive amination). These examples teach the model the expected reasoning pattern and output format without any model weight updates. This technique is known as *few-shot prompting* or *in-context learning*: the model generalizes from the provided examples to new, unseen reaction classes.

**User prompt (per-class input).** For each reaction class, a user prompt is constructed containing:
- The full reaction class name (e.g., "7.1.1 - Alcohols to aldehydes: Oxidation of Primary Alcohols to Aldehydes")
- The numbered list of specific SMIRKS templates from screening
- The numbered list of example reactions

### 2.3 Structured Output

The LLM is configured to return its response as structured JSON conforming to a predefined schema with two fields:
- `reasoning`: A free-text explanation following the six-step protocol
- `smirks`: A list of one or more generalized SMIRKS strings

This is achieved through *constrained decoding* (also called *structured output* or *JSON mode*), where the model's output is forced to conform to a schema. This eliminates the need for fragile text parsing and ensures machine-readable results.

### 2.4 Sampling Temperature

The LLM's *temperature* parameter controls the randomness of token selection during text generation. A temperature of 0 produces deterministic (greedy) output; higher values introduce more variation. We use a temperature of 0.3 — low enough for consistent, reliable SMIRKS generation while allowing some flexibility for the model to explore alternative generalizations.

### 2.5 Validation and Iterative Retry

Each set of generated SMIRKS is immediately validated against the training example reactions:

1. For each example reaction, the SMIRKS is applied using RDKit's `RunReactants` with all reactant permutations and subsets. A SMIRKS is considered *correct* for a reaction if any permutation produces a product matching the expected product (canonicalized, non-isomeric SMILES comparison).
2. The *combined coverage* is computed as the fraction of example reactions correctly covered by the union of all generated SMIRKS.
3. If combined coverage falls below the acceptance threshold (default: 50%), the LLM is re-prompted with *error feedback* — a message appended to the conversation describing which example reaction failed and how (not applicable vs. applicable but incorrect product). This feedback-driven retry mechanism allows the model to self-correct common errors such as incorrect hydrogen counts or over-specific context atoms.
4. Up to 3 retry rounds are attempted before accepting the best result.

### 2.6 Checkpointing

Training results are saved to a checkpoint file after each class completes, enabling resumption from the last successfully processed class in case of interruption. The checkpoint is written atomically (write to a temporary file, then rename) to prevent corruption from mid-write failures.

### 2.7 Context Caching (Optional)

For large-scale runs, the system prompt — which is identical across all classes and contains ~4,000 tokens including few-shot examples — can be cached server-side using the LLM provider's *context caching* feature. This avoids re-transmitting and re-processing the system prompt for every API call, reducing both latency and cost.

---

## Phase 3: Cross-Class False-Positive Testing

After SMIRKS have been generated for all classes, a global false-positive (FP) test determines whether any class's SMIRKS incorrectly match reactions belonging to other classes.

### 3.1 Definition

A *false positive* occurs when a SMIRKS pattern from class A, applied to a reaction from class B (where A ≠ B), produces the correct product of that class-B reaction. This means the SMIRKS is insufficiently specific — it captures a transformation pattern shared across classes that should be distinguishable.

Note: We require `correct = True` (the SMIRKS fires **and** produces the right product), not merely `applicable = True` (the SMIRKS fires but produces a wrong product). Only correct false positives represent genuine class confusion.

### 3.2 Procedure

1. **Pre-parsing.** All training reactions are parsed once: reactant SMILES are converted to RDKit molecule objects, and expected products are canonicalized into a set of SMILES strings.

2. **Pre-compilation.** All SMIRKS are compiled into RDKit `ChemicalReaction` objects once, and the number of required reactant templates (`nreact`) is recorded for each.

3. **Pre-filtering.** Before attempting `RunReactants`, a cheap check skips any SMIRKS whose `nreact` exceeds the number of reactants in the reaction. This eliminates the majority of SMIRKS–reaction pairs without invoking the reaction engine.

4. **Parallel execution.** Training reactions are divided into chunks (default: 200 reactions per chunk) and distributed across multiple CPU cores using Python's `joblib` library. Each worker compiles the SMIRKS independently (RDKit reaction objects cannot be serialized across processes) and tests its chunk of reactions against all other-class SMIRKS.

5. **Aggregation.** False positives are grouped by the SMIRKS's originating class, producing a mapping from each class to its list of FP records (containing the true class, the reaction, and the offending SMIRKS).

---

## Phase 4: Iterative SMIRKS Refinement

Classes whose SMIRKS triggered false positives are refined through iterative re-prompting of the LLM with targeted error feedback.

### 4.1 Refinement Prompt

A separate *refinement system prompt* instructs the model to tighten existing SMIRKS while preserving recall. The per-class *refinement user prompt* contains:

- The current SMIRKS for the class
- A representative sample of training reactions that must remain covered (up to 10 examples)
- False-positive reactions grouped by their true class (up to 5 per true class, 20 total), with class names for chemical context

The model is instructed to apply one or more strategies: adding context atoms, tightening hydrogen/degree constraints, introducing ring-membership requirements, or splitting a single broad SMIRKS into multiple narrower patterns.

### 4.2 Iterative Loop with Rollback Protection

For each class with false positives:

1. The refinement prompt is sent to the LLM, which returns updated SMIRKS with reasoning.
2. **True-positive (TP) check:** The new SMIRKS are validated against the training examples. If the combined coverage drops below a configurable fraction of the original coverage (default: 80%), the refinement is **rolled back** to the previous SMIRKS. This prevents over-constraining — it is preferable to tolerate some false positives rather than lose substantial recall.
3. **Quick FP re-check:** The new SMIRKS are tested specifically against the false-positive examples from Phase 3 (not the full database). Remaining FPs are counted.
4. Steps 1–3 are repeated for up to 3 rounds (configurable) or until all FP examples are eliminated.

### 4.3 Optional Full Re-Test

After all classes have been refined, an optional global FP re-test (repeating Phase 3 with the updated SMIRKS) can detect new false positives introduced by the refinement process. This is controlled by a command-line flag and adds significant computation time.

---

## Phase 5: Held-Out Evaluation

The final SMIRKS (refined where available, original otherwise) are evaluated on the held-out test set that was not used during any training or refinement step.

### 5.1 Recall (Per-Class)

For each class, the final SMIRKS are tested against the class's test reactions. *Test recall* is the fraction of test reactions where at least one SMIRKS produces the correct product, measured as the union coverage across all SMIRKS for the class. This metric estimates the generalization performance of the SMIRKS on unseen reactions of the same class.

### 5.2 False Positives (Cross-Class on Test Set)

A global FP test identical to Phase 3 is run on the test set. For each class, the number of test-set reactions from other classes that are incorrectly matched is reported. This provides an unbiased estimate of cross-class confusion in production use.

### 5.3 Aggregate Metrics

The following summary statistics are computed:
- **Mean test recall** across all classes
- **Total false positives** before and after refinement (on the training set)
- **Test-set false positives** per class and in aggregate
- **Number of classes refined** and **number of rollbacks** (where refinement was rejected due to TP loss)

---

## Implementation Details

### Reproducibility

- The stratified split uses a fixed random seed, ensuring identical train/test partitions across runs.
- LLM temperature is set to 0.3, providing near-deterministic outputs. Full determinism is not guaranteed by LLM providers due to floating-point non-determinism in distributed inference.

### Computational Considerations

- Phase 2 (LLM calls) is rate-limited by the API and typically processes 1–3 classes per minute depending on template count.
- Phase 3 (FP testing) is CPU-bound. With *C* classes averaging *S* SMIRKS each and *N* training reactions, the theoretical number of `RunReactants` calls is approximately *N* x *C* x *S*. In practice, the `nreact` pre-filter eliminates the majority of calls, and the RDKit substructure matching within `RunReactants` fails fast for non-matching patterns.
- All phases support checkpointing: the pipeline can be interrupted and resumed without losing progress.

### Software Dependencies

The pipeline depends on RDKit (cheminformatics), Google Gemini API (LLM inference), Pydantic (structured output schema), pandas (data handling), NumPy (random number generation), joblib (parallelism), and tqdm (progress reporting). No dependency on Rxn-INSIGHT itself is required — all functions are self-contained.
