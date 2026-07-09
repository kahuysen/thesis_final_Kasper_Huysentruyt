# Pipeline Schematic: LLM-Guided SMIRKS Generalization

## High-Level Overview

```
  REACTION DATABASE                     ONTOLOGY
  (SMILES strings)                 (structured_mapping.json)
        |                                   |
        v                                   v
  +------------------+            +-------------------+
  | Rxn-INSIGHT      |            | Gemini LLM        |
  | Atom mapping     |            | Hierarchical      |
  | (RXNMapper)      |            | classification    |
  +------------------+            | (tier 1-2-3-4)    |
        |                         +-------------------+
        v                                   |
  +------------------+                      |
  | Rxn-INSIGHT      |                      |
  | Template         |                      |
  | extraction       |                      |
  | (BE/T matrices)  |                      |
  +------------------+                      |
        |                                   |
        v                                   v
  +-------------------------------------------------+
  |          CLASSIFIED REACTION DATABASE            |
  |  Each reaction has:                              |
  |    - Mapped SMILES                               |
  |    - Specific template (SMIRKS)                  |
  |    - Tier 1/2/3/4 class labels                   |
  +-------------------------------------------------+
                        |
                        v
          +=============================+
          |   VALIDATED GENERALIZATION   |
          |        PIPELINE             |
          |   (this work, 5 phases)     |
          +=============================+
                        |
                        v
              +-------------------+
              |  Generalized      |
              |  SMIRKS library   |
              +-------------------+
                        |
                        v
              +-------------------+
              |  Rxn-INSIGHT      |
              |  Reaction naming  |
              |  & classification |
              +-------------------+
```

## Detailed 5-Phase Pipeline

```
+============================================================================+
|                                                                            |
|  PHASE 1: STRATIFIED SPLIT                                                 |
|                                                                            |
|  Classified Reaction DB                                                    |
|  ~~~~~~~~~~~~~~~~~~~~~~                                                    |
|  | rxn_1  | 1.1.1 | template_A |    Stratified     +-------+  +-------+   |
|  | rxn_2  | 1.1.1 | template_B |    80/20 split     | TRAIN |  | TEST  |   |
|  | rxn_3  | 7.1.1 | template_C |  =============>   | (80%) |  | (20%) |   |
|  | rxn_4  | 7.1.1 | template_A |   per class,      | held  |  | held  |   |
|  | ...    | ...   | ...        |   fixed seed       | for   |  | out   |   |
|  +--------+-------+------------+                    | Ph2-4 |  | Ph5   |   |
|                                                     +-------+  +-------+   |
|                                                                            |
+============================================================================+
                              |
                              v
+============================================================================+
|                                                                            |
|  PHASE 2: LLM-BASED SMIRKS GENERALIZATION (training set only)              |
|                                                                            |
|  For each tier-3/4 class:                                                  |
|                                                                            |
|  Step A: Template Screening                                                |
|  ~~~~~~~~~~~~~~~~~~~~~~~~~~                                                |
|  Training reactions for class 7.1.1                                        |
|       |                                                                    |
|       v                                                                    |
|  Rank templates by frequency  -->  Top-N templates (>= 90% coverage)       |
|  [template_A: 450x]               + shortest example reaction per template |
|  [template_B: 230x]                                                        |
|  [template_C: 120x]                                                        |
|  ...                                                                       |
|                                                                            |
|  Step B: Few-Shot Prompting                                                |
|  ~~~~~~~~~~~~~~~~~~~~~~~~~~                                                |
|  +---------------------------+    +----------------------------+           |
|  | SYSTEM PROMPT             |    | USER PROMPT (per class)    |           |
|  |                           |    |                            |           |
|  | - SMARTS notation rules   |    | Class: 7.1.1 - Oxidation  |           |
|  | - RC/context/LG rules     |    |   of Primary Alcohols     |           |
|  | - 6-step reasoning (A-F)  |    |                            |           |
|  | - 3 worked examples:      |    | Templates:                |           |
|  |   1. RC-only (7.1.1)      |    |   1. [c;H0]-[C;H2;D2]-..  |           |
|  |   2. Halide merge (1.1.1) |    |   2. [O;H1]-[C;H2;D2]-..  |           |
|  |   3. Multiple SMIRKS      |    |   ...                      |           |
|  |      (1.2.2)              |    |                            |           |
|  +---------------------------+    | Examples:                  |           |
|               |                   |   1. CCO>>CC=O             |           |
|               |                   |   2. OCc1cc2s..>>O=Cc1..   |           |
|               +----->  GEMINI <---+                            |           |
|                          |        +----------------------------+           |
|                          v                                                 |
|                  +----------------+                                        |
|                  | Structured     |                                        |
|                  | JSON output:   |                                        |
|                  |  - reasoning   |                                        |
|                  |  - smirks[]    |                                        |
|                  +----------------+                                        |
|                          |                                                 |
|  Step C: Validation      |                                                 |
|  ~~~~~~~~~~~~~~~~~~      v                                                 |
|           +----------------------------+                                   |
|           | For each example reaction: |                                   |
|           |   RunReactants(SMIRKS)     |--> correct product?               |
|           +----------------------------+                                   |
|                    |              |                                         |
|                  pass           fail                                        |
|                    |              |                                         |
|                    v              v                                         |
|              Accept SMIRKS   Retry with error feedback                     |
|                               (up to 3 rounds)                             |
|                                                                            |
|  [Checkpoint saved after each class]                                       |
|                                                                            |
+============================================================================+
                              |
                              v
+============================================================================+
|                                                                            |
|  PHASE 3: CROSS-CLASS FALSE POSITIVE TESTING                               |
|                                                                            |
|  Question: Does class A's SMIRKS incorrectly match class B's reactions?    |
|                                                                            |
|  +----------+     +----------+     +----------+                            |
|  | Class A   |     | Class B   |     | Class C   |     ...                  |
|  | SMIRKS    |     | SMIRKS    |     | SMIRKS    |                          |
|  +----------+     +----------+     +----------+                            |
|       |                 |                 |                                  |
|       +--------+--------+--------+--------+                                |
|                |                                                            |
|                v                                                            |
|  For each training reaction:                                               |
|    - Pre-parse reactants (once)                                            |
|    - Canonicalize products (once)                                          |
|    - Test against ALL other-class SMIRKS                                   |
|    - Pre-filter: skip if nreact mismatch                                   |
|                                                                            |
|  Parallelized with joblib (chunked, multi-core)                            |
|                                                                            |
|  False positive = SMIRKS fires AND produces correct product                |
|  (not just substructure match -- must give right answer)                   |
|                                                                            |
|  Output: FPs categorized by hierarchy level                                |
|  +-----------------------------------------------------------+            |
|  | FP Type            | Meaning              | Action        |            |
|  |--------------------+------------------------+--------------|            |
|  | Same tier-2        | Near-identical class   | Expected     |            |
|  | Same tier-1        | Same superclass        | Ontology     |            |
|  |                    | (same mechanism)       | granularity  |            |
|  | Cross tier-1       | Different mechanism    | Genuine FP   |            |
|  +-----------------------------------------------------------+            |
|                                                                            |
+============================================================================+
                              |
                              v
+============================================================================+
|                                                                            |
|  PHASE 4: ITERATIVE REFINEMENT                                             |
|                                                                            |
|  For each class with genuine false positives:                              |
|                                                                            |
|  +--------------------+     +--------------------+                         |
|  | Current SMIRKS     |     | FP examples        |                         |
|  | for class A        |     | (grouped by true   |                         |
|  |                    |     |  class, max 20)     |                         |
|  +--------------------+     +--------------------+                         |
|           |                          |                                     |
|           +----------+---------------+                                     |
|                      |                                                     |
|                      v                                                     |
|             +------------------+                                           |
|             | GEMINI           |                                            |
|             | Refinement       |                                            |
|             | prompt:          |                                            |
|             | "Tighten these   |                                            |
|             |  SMIRKS to NOT   |                                            |
|             |  match these FP  |                                            |
|             |  reactions"      |                                            |
|             +------------------+                                           |
|                      |                                                     |
|                      v                                                     |
|  +------------------------------------------+                             |
|  | Validation gate:                         |                              |
|  |   1. TP check: still covers training?    |                              |
|  |      if TP < 80% of original --> ROLLBACK|                              |
|  |   2. FP check: FPs eliminated?           |                              |
|  +------------------------------------------+                             |
|            |                    |                                           |
|          pass                 fail                                          |
|            |                    |                                           |
|       Accept refined       Retry (up to 3 rounds)                          |
|       SMIRKS               or rollback to original                         |
|                                                                            |
+============================================================================+
                              |
                              v
+============================================================================+
|                                                                            |
|  PHASE 5: HELD-OUT EVALUATION                                              |
|                                                                            |
|  Using TEST SET (never seen during Phase 2-4):                             |
|                                                                            |
|  Per class:                                                                |
|  +-------+    Final SMIRKS    +----------------------------+               |
|  | Test  | =================> | Recall: fraction of own    |               |
|  | set   |                    |   test reactions matched    |               |
|  +-------+                    +----------------------------+               |
|                                                                            |
|  Cross-class:                                                              |
|  +-------+    All SMIRKS      +----------------------------+               |
|  | Test  | =================> | FP count: other-class test |               |
|  | set   |                    |   reactions matched         |               |
|  +-------+                    |   (by hierarchy level)     |               |
|                               +----------------------------+               |
|                                                                            |
|  Aggregate metrics:                                                        |
|    - Mean test recall across all classes                                   |
|    - Total FPs before/after refinement                                     |
|    - FPs by hierarchy (same tier-1/2/3 vs cross tier-1)                    |
|    - Number of rollbacks                                                   |
|                                                                            |
+============================================================================+
                              |
                              v
              +===============================+
              |     FINAL OUTPUT              |
              |                               |
              |  Per class:                   |
              |    - Initial & final SMIRKS   |
              |    - Recall (train + test)    |
              |    - FPs by hierarchy level   |
              |    - LLM reasoning trace      |
              |                               |
              |  Aggregate:                   |
              |    - Mean recall              |
              |    - FP reduction             |
              |    - Ontology insights        |
              +===============================+
                              |
                              v
              +===============================+
              |     RXN-INSIGHT INTEGRATION   |
              |                               |
              |  Generalized SMIRKS become    |
              |  the naming database:         |
              |                               |
              |  New reaction                 |
              |       |                       |
              |       v                       |
              |  Match against SMIRKS library |
              |       |                       |
              |       v                       |
              |  Reaction name + class        |
              +===============================+
```

## Key Design Principle

```
+------------------------------------------------------------------+
|                                                                    |
|   LLM                              CHEMINFORMATICS                |
|   (Gemini)                         (RDKit / Rxn-INSIGHT)          |
|                                                                    |
|   - Chemical reasoning             - Formal validation            |
|   - Pattern recognition            - RunReactants (ground truth)  |
|   - Generalization                 - Template extraction          |
|   - Natural language rules         - Atom mapping                 |
|                                                                    |
|        "What SHOULD the              "Does it ACTUALLY             |
|         SMIRKS look like?"            produce the right product?"  |
|                                                                    |
|   Creative  <------- feedback loop ------->  Rigorous             |
|                                                                    |
+------------------------------------------------------------------+
```
