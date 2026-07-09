# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Python Environment

**ALWAYS** use `"C:\Users\mrodobbe\.conda\envs\genrxn\python.exe"` to run Python. **NEVER** use `python`, `python3`, or any other Python executable. This applies to all Bash tool calls, all commands suggested to the user, and all scripts.

## Project Overview

Rxn-INSIGHT (`gen-rxn-insight` on PyPI) is a Python library for classifying and naming chemical reactions, and suggesting reaction conditions based on similarity and popularity. It uses Bond-Electron (BE) matrices and Transformation (T) matrices to analyze reaction centers. Publication: https://doi.org/10.1186/s13321-024-00834-z

## Build & Install

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n rxn-insight python=3.10
conda activate rxn-insight

# Install for development with test and doc dependencies
pip install -e ".[test,doc]"
```

## Testing

```bash
# Run all tox environments (tests + mypy + coverage)
tox

# Run tests only (with mypy first, then pytest)
tox -e py3

# Run tests with coverage report
tox -e py3-coverage

# Run style checks (pre-commit + build)
tox -e style

# Build docs
tox -e docs

# Run pytest directly (faster iteration)
pytest -vv

# Run a single test
pytest tests/test_classification.py::test_initialization -vv

# Type checking
mypy --config-file mypy.ini src tests
```

## Package Naming Convention

There is an important dual-naming pattern in this project:
- **PyPI package name**: `gen-rxn-insight` (with `gen-` prefix)
- **Import name**: `rxn_insight` (no prefix, underscores)
- **Source directory**: `src/gen-rxn-insight/` (hyphenated, with `gen-` prefix)
- **Hatch version file**: `src/gen_rxn_insight/__about__.py` (underscored, with `gen_` prefix)

All internal imports use `from gen_rxn_insight.<module>` (e.g., `from gen_rxn_insight.reaction import Reaction`). The public API is exposed via `import gen_rxn_insight as ri`.

## Architecture

### Core Pipeline

The core analysis flow for a reaction follows this path:

1. **`Reaction`** (`reaction.py`) - Main entry point. Takes a reaction SMILES string, orchestrates atom mapping, classification, naming, and condition suggestion. The `get_reaction_info()` method runs the full analysis pipeline and returns a dict with class, name, functional groups, rings, by-products, tag, and scaffold.

2. **`ReactionClassifier`** (`classification.py`) - The computational engine (~1500 lines). Constructs BE-matrices for reactants and products from atom-mapped reactions, computes the T-matrix (difference), identifies the reaction center, extracts reaction templates via RDChiral, classifies reactions into 11 superclasses (Acylation, C-C Coupling, Reduction, Oxidation, etc.), identifies functional groups and ring systems in the reaction center, and balances reactions to predict by-products. Uses RXNMapper for atom mapping when not already present.

3. **`naming.py`** - Names reactions by matching against a SMIRKS pattern database (`data/smirks.json`). Permutes reactants and runs RDKit reaction templates to check if predicted products match actual products. Also provides `get_class_name()` for resolving class codes to human-readable tier names via the bundled `structured_mapping.json`.

4. **`template.py`** - Handles custom reaction template extraction with configurable radius, stereochemistry preservation, and ring closure information. Includes `mendeleev()` for atomic number to symbol conversion and `check_template_accuracy()` for template validation.

5. **`utils.py`** - Utility functions including: atom mapping (`get_atom_mapping` via RXNMapper), reaction template extraction (RDChiral-based), fingerprints (MACCS/Morgan for molecules and reactions), Murcko scaffold extraction, ring system detection, similarity metrics (16 distance metrics from scipy), reaction drawing (SVG via RDKit), and condition ranking functions.

### Data Classes

- **`Database`** (`database.py`) - Batch processing of reaction datasets. Creates Rxn-INSIGHT-compatible databases from CSV files or DataFrames. Uses `joblib.Parallel` for multi-core processing. Outputs include reaction fingerprints, tags, templates, and classification data. Can save to parquet/CSV/Excel.

- **`Molecule`** (`molecule.py`) - Analyzes individual molecules: SMILES/InChI/InChIKey conversion, PubChem API lookups, MACCS/Morgan fingerprints, functional group detection, ring identification, scaffold extraction, and reaction search by product InChIKey or scaffold similarity.

- **`Compound`** (`molecule.py`) - Subclass of `Molecule` that initializes from a compound name rather than SMILES, using OPSIN or PubChem name-to-structure resolution.

- **`ORDDatabase`** (`ord.py`) - Extends `Database` to load reactions from Open Reaction Database protocol buffer files. Requires optional `ord_schema` dependency.

### Internal Data Files

The package bundles JSON data files accessed via `importlib.resources` from the `<package>.data` subpackage:
- `smirks.json` - SMIRKS patterns for reaction naming (528 hand-crafted, JSONL with `name`/`smirks` fields)
- `structured_mapping.json` - Hierarchical class code → name mapping (14,060 entries, tier_2 through tier_5)
- `functional_groups.json` - SMARTS patterns for functional group identification
- `named_rings.json` - Ring SMILES to common name mappings

### Key Dependencies

- **RDKit** - Core cheminformatics (molecule parsing, fingerprints, drawing, substructure matching)
- **RXNMapper** - Attention-based atom mapping (uses PyTorch/Transformers under the hood)
- **RDChiral** - Reaction template extraction
- **Pandas** - DataFrames for databases and results
- **NumPy** (<=1.26.4) - Matrix operations for BE/T matrices

### Tagging System

Reactions are tagged with SHA-256 hashes for grouping similar reactions:
- **TAG** (fine-grained): hash of class + functional groups + participating rings
- **TAG2** (broad): hash of class + functional groups only (used for broadened similarity search)

## Recent API Additions

### Asymmetric radius support for template extraction
`extract_reaction_template` accepts an optional `radius_products` parameter (default `None` = symmetric).
`extract_all_templates` / `extract_templates_batch` accept `radii` as a list of `int` **or** `(r_reactant, r_product)` tuples.
Column names in the output DataFrame use the new scheme `TEMPLATE_rr{r_r}rp{r_p}_ring{0|1}`.
`Reaction.get_detailed_template()` forwards `radius_products` to the classifier.

### Template accuracy measurement
- `measure_template_accuracy(rsmi, template)` — returns `{'correct': bool, 'applicable': bool, 'n_outcomes': int}`. Tries all reactant permutations; distinguishes "template doesn't fire" from "fires with wrong product".
- `measure_templates_batch(reactions, templates, ...)` in `database.py` — parallel batch version, returns a DataFrame.
- Both are exported from `gen_rxn_insight.__init__`.

### Relaxed context atoms
`relax_context=True` on `extract_reaction_template`, `extract_all_templates`, `extract_templates_batch`, and `Reaction.get_detailed_template()` makes context atoms (those not in the reaction center) use minimal SMARTS (`[<element>;+<charge>]` — element + aromaticity + formal charge only), while RC atoms keep full specificity (H, D, charge, stereo, ring info). Default `False` preserves legacy behaviour.

`get_detailed_template` has an automatic fallback: if `relax_context=True` produces a template that fails `check_template_accuracy`, it retries with `relax_context=False` before returning `""`.

### Reaction naming batch
`name_reactions_batch` accepts mapped SMILES and `reactants>agents>products` format — atom maps and agents are stripped automatically (sanitized once upfront, before dispatching to workers). A `progress: bool = True` parameter shows a tqdm progress bar. The parallel path converts the DataFrame to lightweight tuples `(name, smirks, nreact)` before dispatching to workers for fast serialization. Both `name_reaction` and the parallel worker use early-exit on first product match for performance.

### Class name lookup
`get_class_name(code, tier=None)` resolves a dot-separated class code to human-readable tier names using the bundled `structured_mapping.json`. Returns a dict `{"tier_1": ..., "tier_2": ..., ...}` or a single string when `tier` is specified. Tier-1 superclass names (11 classes: Acylation, C-C Coupling, Reduction, etc.) are hardcoded since they aren't in the mapping file. The mapping is loaded once and cached.

### SMIRKS validation in curate_smirks
`curate_smirks()` now validates each SMIRKS with `ReactionFromSmarts` and drops entries that RDKit cannot parse (e.g., overly complex recursive SMARTS). Emits a warning with the count of dropped entries.

### Custom SMIRKS databases
`name_reactions_batch` and `name_reaction` accept custom SMIRKS databases via the `smirks_db` parameter. A custom DB is a JSONL file with `name` and `smirks` fields per line. The `name` field can be a class code (e.g., `"3.1.1.2.1"`) instead of a descriptive name — use `get_class_name()` to resolve codes to tier names afterwards.

```python
smirks_db = curate_smirks(pd.read_json("custom_smirks_db.json", orient="records", lines=True))
codes = name_reactions_batch(reactions, n_jobs=8, smirks_db=smirks_db)
# Resolve codes to names
from gen_rxn_insight import get_class_name
tier2_names = [get_class_name(c, tier=2) if c != "OtherReaction" else c for c in codes]
```

### Useful quick-access patterns
```python
# Get sanitized unmapped reaction from a classifier instance
clf.sanitized_reaction          # fastest

# Get sanitized unmapped reaction from a mapped SMILES string
from gen_rxn_insight.classification import sanitize_mapped_reaction
sanitize_mapped_reaction(mapped_smiles)[1]   # returns unmapped SMILES

# Batch strip atom maps (fastest, no RDKit needed)
import re
df['SANITIZED'] = df['MAPPED'].str.replace(r':\d+', '', regex=True)
```

## Template Extraction Internals

Template extraction lives in `template.py` + `classification.py:extract_reaction_template`.

### Key functions
- `expand_atoms_by_radius_with_leaving_groups(mol, center_atoms, radius, atom_map_dict, ...)` — BFS expansion from RC atoms; always adds unmapped leaving-group atoms regardless of radius; optionally adds stereo reference atoms.
- `connect_reaction_centers(mol, rc_indices, included)` — finds the shortest path(s) between disconnected RC components and adds bridging atoms. Uses `Chem.GetShortestPath` (full molecule graph, not restricted to mapped atoms).
- `build_smarts_with_ring_closures(mol, atom_indices, sssr, ..., rc_indices=None)` — DFS over `atom_indices` only; ring closures are only formed between atoms **within the expanded set**. When `rc_indices` is provided, atoms not in that set get relaxed SMARTS.
- `find_ring_closures(mol, component_atoms, start)` — pre-computes DFS back-edges (ring closures) for `build_component_with_rings`.

### Phase 2 bidirectional sync
After Phase 1 (BFS expansion), Phase 2 syncs map numbers between sides:
- **Pass 1 (product → reactant)**: stereo reference atoms added only in the product side are also added to the reactant.
- **Pass 2 (reactant → product)**: ring-bridging atoms added by `connect_reaction_centers` on the reactant are mirrored to the product, preventing broken ring closures in the product template.

### Stereo detection for reaction center expansion
The `preserve_stereo` code in `extract_reaction_template` detects stereochemistry changes between reactant and product to add affected atoms to the reaction center. It uses **CIP codes** (R/S, canonical) as the primary comparison. Chiral tags (`GetChiralTag()`, CW/CCW) are only used as a fallback when neither side has a CIP code (e.g., newly created stereocenters).

**Why not chiral tags?** Chiral tags are encoding-dependent — they flip when neighbor ordering changes (e.g., a neighbor gains a new substituent via the reaction), even when the actual 3D configuration is preserved. Comparing chiral tags directly caused false positives that inflated the reaction center with stereo atoms whose configuration didn't actually change, leading to `@`/`@@` marks in the template that made `RunReactants` produce wrong stereochemistry (`applicable=True, correct=False`).

### Known limitation: mapper quality
When RXNMapper maps leaving-group atoms (assigns them map numbers), those atoms enter the reaction center, which makes the template incorrect. This is a mapper quality issue, not a bug in Rxn-INSIGHT. Symptoms: `applicable=True`, `correct=False`, reaction center contains far more atoms than expected (including the leaving group). No fix inside Rxn-INSIGHT without better upstream mapping.

### Python environment
Use `C:\Users\mrodobbe\.conda\envs\genrxn\python.exe` directly (not `python` or `python3`) — the conda environment is `genrxn`.

## Tier-Level Recommendation Script

`scripts/recommend_tier_levels.py` — standalone script (~690 lines, no gen-rxn-insight dependency) that analyzes the classification database to determine the optimal tier level (tier_3, tier_4, or tier_5) for each reaction class before running SMIRKS generalization.

### Purpose

Not all tier_3 classes are homogeneous: some have high template coverage (top-10 templates cover 90%+, stay at tier_3), but others are heterogeneous and benefit from splitting into finer subclasses. This script automates that decision.

### Recursive Decision Tree

For each tier_3 class, the algorithm applies these rules:

1. **Coverage >= threshold** (default 0.9) → keep at this tier
2. **At max depth** → keep (can't go deeper)
3. **Only 1 subclass** → look through to deeper tiers; if no benefit, keep parent
4. **Coverage gain < 0.05** → keep (weighted avg subclass coverage barely improves)
5. **Otherwise** → split into viable subclasses (>= `min_class_size`), recurse each

Small subclasses (< `min_class_size`) are **dropped** (not merged into the parent), avoiding parent-child overlap in the output. Dropped subclasses are tracked in the JSON output for transparency.

### Key Component: ClassIndex

Pre-groups template counts and parent-child relationships via `df.groupby()` once during initialization, making all subsequent lookups O(1) dict access. This reduces runtime from minutes to ~22 seconds on 665K reactions.

### CLI

```bash
python scripts/recommend_tier_levels.py \
    --database classification_database.parquet \
    --mapping structured_mapping.json \
    --output tier_recommendations \
    --coverage-threshold 0.9 \
    --min-class-size 5 \
    --top-n 10 \
    --max-tier 0 \
    --template-col TEMPLATE_rr0rp1_ring0 \
    --verbose
```

`--max-tier 0` auto-detects the finest tier in the data. `--template-col` defaults to `TEMPLATE_rr0rp1_ring0`.

### Output

- **CSV report** (`{output}_report.csv`) — one row per class at every tier level evaluated, with columns: `class_code`, `class_name`, `tier`, `n_reactions`, `n_templates`, `top_n_coverage`, `n_subclasses`, `n_viable_subclasses`, `coverage_gain`, `recommended_tier`, `merge_up`, `recommended_for_processing`, `reason`.
- **JSON class list** (`{output}_classes.json`) — directly consumable by `generalize_smirks_validated.py`:
  ```json
  {
    "metadata": { "coverage_threshold": 0.9, "min_class_size": 5, "top_n": 10 },
    "classes_by_tier": { "tier_3": [...], "tier_4": [...], "tier_5": [...] },
    "dropped": { "5.1.1.3.2": "5.1.1.3" },
    "all_classes": ["1.3.1", "5.1.1.3", "4.1.4.4.12", ...]
  }
  ```

### Integration with the Pipeline

```bash
python scripts/generalize_smirks_validated.py \
    --database classification_database.parquet \
    --mapping structured_mapping.json \
    --output validated_smirks.json \
    --api-key $GEMINI_API_KEY \
    --classes-file tier_recommendations_classes.json
```

`--classes-file` reads the JSON and pulls out the `all_classes` key. Mutually exclusive with `--classes`.

### Key Design Decisions

- **No overlap**: when a class is split, only the children appear in `all_classes`, never the parent
- **Self-referencing tier guard**: some reactions have the same class code in both a tier column and its child column (e.g., `tier_4 = "1.2.1.5"` AND `tier_5 = "1.2.1.5"`). `compute_weighted_subclass_coverage` skips children whose code equals the parent (`child_cls == cls`) to prevent a class from appearing as its own subclass, which would cause both the parent and its real children to end up in `all_classes`
- **Depth-independent**: handles tier columns containing codes with mismatched dot-depths (e.g., `"3.11.25"` in tier_4) via explicit `_source_depth` tracking
- **Coverage verification**: `_verify_coverage()` logs what fraction of reactions are covered by `all_classes` and identifies gaps from empty tier_3 codes and dropped tiny subclasses

## Validated SMIRKS Generalization Pipeline

`scripts/generalize_smirks_validated.py` — standalone script (~2000 lines, no gen-rxn-insight dependency) that generalizes specific reaction templates into broad SMIRKS patterns via Gemini LLM, with rigorous cross-class validation.

### 5 Phases with Checkpointing

1. **Stratified Split** — 80/20 train/test per class (seed 210995), configurable via `--class-column` / `--split-column`.
2. **Training** — per-class template screening + Gemini few-shot prompting + RDKit validation loop (retry with error feedback up to 3 rounds). Checkpoints after each class.
3. **False Positive Testing** — each class's SMIRKS tested against all other-class training reactions via `joblib.Parallel`. FP = SMIRKS fires AND produces correct product. Pre-filters by reactant count.
4. **Iterative Refinement** — Gemini re-prompted with FP examples to tighten SMIRKS. Rollback if TP drops below threshold (default 80% of original). On rollback, original SMIRKS are restored.
5. **Held-out Evaluation** — recall + FP count on test set, reported by hierarchy level.

### Mixed-Depth Class Processing

The pipeline supports processing classes at different tier depths in a single run. `--classes` accepts mixed-depth codes (e.g., `1.3.1 5.1.1.3 4.1.4.4.12`). When mixed depths are provided, `--class-column` is auto-set to `mixed`. The split is always performed at `--split-column` granularity (default: finest tier in data), while processing aggregates reactions per class code using `get_class_split()`. Wildcard expansion: `4.1.4.4.*` expands to direct children, `4.1.4.4.**` to all descendants.

### FP Hierarchy Categorization

`categorize_fps()` uses `_class_ancestors()` to compute shared prefix depth between SMIRKS class and FP true class:
- `n_same_tier2` — near-identical class (same parent)
- `n_same_tier1` — same superclass (same mechanism family)
- `n_cross_tier1` — genuinely different reaction type
- Works with any class depth (tier_3 `"1.4.2"`, tier_4 `"1.4.2.3"`, etc.)

### Checkpoint System

- Checkpoint stores `class_column`; if `--class-column` changes between runs, the split and all downstream results are invalidated and re-computed
- Atomic writes via `tempfile.mkstemp` + `os.replace` (no timestamped backups — previous checkpoint is overwritten)
- Checkpoints after every class in Phase 2 and Phase 4 for resumability

### Token Usage Tracking

`SmirksGeneralizer` tracks cumulative token usage via `response.usage_metadata` (prompt + completion tokens, call count). Logged after Phase 2 and Phase 4, saved in checkpoint and final output JSON under `summary.token_usage`.

### Key CLI Flags

```bash
python generalize_smirks_validated.py \
    --database classification_database.parquet \
    --mapping structured_mapping.json \
    --output validated_smirks.json \
    --api-key $GEMINI_API_KEY \
    --class-column tier_4 \
    --classes 1.3.1 5.1.1.3 4.1.4.4.12 \
    --split-column tier_5 \
    --phase all \
    --seed 210995 \
    --fp-n-jobs 8 \
    --fp-retest \
    --model gemini-3-flash-preview
```

`--phase` accepts: `all`, `split`, `train`, `fp_test`, `finetune`, `eval` (phases depend on each other sequentially). `--model` defaults to `gemini-3-flash-preview`.

### Supporting Files

- `scripts/generalize_smirks_standalone.py` — original single-class generalization script (reused inline)
- `scripts/merge_checkpoints.py` — merges extra classes from an old checkpoint into a new one (used to graft 117 optimized classes from Phase-3 work into the final Phase-5 checkpoint)
- `scripts/regenerate_class_smirks.py` — regenerates SMIRKS for specific classes with diverse template sampling + optional discrimination refinement against competing classes
- `scripts/methodology_smirks_generalization.md` — publication-ready methodology description
- `scripts/pipeline_schematic.md` — ASCII pipeline diagram

## SMIRKS DB Ordering Pipeline

`name_reaction` uses a **first-match-wins** strategy: it iterates the SMIRKS database in order and returns the first SMIRKS that fires correctly. This means false positives (a SMIRKS for class A firing on class B's reactions) can be eliminated by placing class B's SMIRKS **before** class A's, so B matches first.

### Key Insight

The Phase-3 FP graph (built from training data only) is **incomplete** — it misses many cross-class FP relationships present in the full dataset. Building the FP graph from the full reaction database is essential for accurate ordering.

### Scripts

#### `scripts/order_smirks_db.py`

Quick ordering using the Phase-3 FP graph from a checkpoint file. Useful as an initial ordering but insufficient for production accuracy.

```bash
python scripts/order_smirks_db.py \
    --checkpoint gemini_smirks_checkpoint.json \
    --output ordered_smirks_db.json \
    [--use-refined] [--include-class] [--stats]
```

#### `scripts/build_fp_graph_and_reorder.py`

Builds a **complete** FP graph by testing every reaction against ALL SMIRKS (not just until first match), then computes optimal ordering. This is the production-quality approach.

```bash
python scripts/build_fp_graph_and_reorder.py \
    --smirks-db ordered_smirks_db.json \
    --reactions reaction_db.parquet \
    --checkpoint gemini_smirks_checkpoint.json \
    --output reordered_smirks_db.json \
    --n-jobs 8
```

- Tests all 665K reactions × all SMIRKS (~2–3 hours with 8 workers)
- Discovers all cross-class FP edges (not just training-set FPs)
- Outputs: reordered JSONL, metadata JSON, full FP graph JSON

#### `scripts/evaluate_naming.py`

Evaluates a SMIRKS DB by running first-match naming on the reaction database and comparing against ground-truth tier columns.

```bash
python scripts/evaluate_naming.py \
    --smirks-db reordered_smirks_db.json \
    --reactions reaction_db.parquet \
    --output naming_eval.csv \
    --n-jobs 8 [--sample 1000]
```

- Reports exact match, ancestor match, mismatch, and unmatched rates
- Splits results by whether the reaction's true class is in the SMIRKS DB
- Ancestor matching: child predictions on parent reactions count as correct when parent has no own SMIRKS
- Class merges (`_CLASS_MERGES`): 16 pairs of structurally indistinguishable classes across superclasses are treated as equivalent (e.g., Buchwald-Hartwig/SNAr, cross-superclass pairs like 5.1.4.1/8.7.2.1)
- Outputs: per-reaction CSV + summary JSON

**Per-tier accuracy metrics:** Many classes only have tier_3-level SMIRKS (no tier_4/5 children). A prediction of `"1.7.3"` for a reaction with ground truth `"1.7.3.2"` is correct but less specific. Two ways to measure:
- **Strict**: truncate prediction to tier depth, require exact match → penalizes coarse predictions
- **Ancestor**: prediction is correct if it's a prefix of the actual tier value → fair metric when SMIRKS exist only at coarser levels

Final results (665K reactions, 4,964 SMIRKS, 3,497 classes — 94.21% correct overall):
| Tier | Strict | Ancestor |
|------|--------|----------|
| tier_1 | 97.9% | 97.9% |
| tier_2 | 96.5% | 96.5% |
| tier_3 | 90.4% | 90.5% |
| tier_4 | 63.4% | 87.1% |
| tier_5 | 42.9% | 84.9% |

Overall: 84.36% exact match, 9.85% ancestor match, 1.36% mismatch, 4.43% unmatched.
The gap between strict and ancestor at tier_4/5 quantifies the opportunity from generating finer-grained SMIRKS.

### Ordering Algorithm

1. Build directed FP graph: edge A → B means "A's SMIRKS fires correctly on B's reactions"
2. Find strongly-connected components (SCCs) via iterative Tarjan's
3. Topologically sort the DAG of SCCs (classes that get falsely matched come first)
4. Within each SCC (mutual FPs), use a greedy min-feedback-arc-set heuristic: iteratively place the node with highest (in_weight − out_weight) first

### Results

**Phase-3 graph only (incomplete):**
- 1,926,000 total FP instances across 11,353 class-pair edges
- Ordering eliminated 95.8% of FPs (1,844,646 eliminated, 81,354 remaining)
- 215 non-trivial SCCs involving 872 classes (unavoidable mutual FPs)
- Naming accuracy was only ~55% because the Phase-3 graph missed many edges

**Full FP graph (final):**
- 4,964 SMIRKS, 3,497 classes
- 94.21% correct (84.36% exact + 9.85% ancestor), 1.36% mismatch, 4.43% unmatched

### Data Locations

- Final checkpoint: `EPFL_Research/Reaction Classification/Results/final_liacpc19_results/merged_checkpoint.json`
- Final SMIRKS DB: `EPFL_Research/Reaction Classification/Results/final_liacpc19_results/final_reordered_smirks_db.json`
- Code-only SMIRKS DB: `EPFL_Research/Reaction Classification/Results/final_liacpc19_results/smirks_db.json` (name=class code)
- Class names lookup: `EPFL_Research/Reaction Classification/Results/final_liacpc19_results/class_names.json`
- Naming eval: `EPFL_Research/Reaction Classification/Results/final_liacpc19_results/final_naming_eval.csv` + `.summary.json`
- Reaction DB: `Gen-Rxn-INSIGHT/data/reaction_db.parquet` (665,901 reactions, 18 columns)

## Planned: Chemistry Reasoning Agent

A tool-use agent that reasons about organic chemistry grounded in the validated SMIRKS library and classification database. The SMIRKS library is essentially a machine-readable organic chemistry textbook — each entry encodes a transformation pattern validated against thousands of real reactions with measured recall.

### Architecture

A simple agent loop: LLM receives user question + tool definitions → decides which tools to call → tools query existing data → LLM reasons over results → returns grounded answer. No framework needed, uses the same `google-genai` SDK as the generalization pipeline.

### Tools

- **`lookup_class(name_or_code)`** — returns generalized SMIRKS, LLM reasoning from Phase 2, class hierarchy, example reactions, class size, recall metrics
- **`search_database(substrate_smarts, class_code, ...)`** — queries the classification database (parquet) by class, functional groups, or substrate structure
- **`compare_classes(class_a, class_b)`** — side-by-side SMIRKS diff + differentiating structural features
- **`draw_scheme(smirks)`** — renders generalized SMIRKS as a generic reaction scheme (SVG/PNG via RDKit)

### Use Cases

- **Reaction scheme from name**: "Show me the general scheme for Buchwald-Hartwig amination" → look up SMIRKS, render as reaction drawing with R-groups
- **Mechanistic comparison**: "What's the difference between Suzuki and Heck coupling?" → retrieve both SMIRKS, compare reaction center atoms, explain structural distinction
- **Reagent suggestion**: SMIRKS defines which functional groups participate; the database provides the most common reagents/catalysts/solvents for that class, ranked by popularity
- **Selectivity explanation**: "Why does my reaction give aldehyde instead of carboxylic acid?" → compare SMIRKS for tier-3 classes 7.1.1 vs 7.1.2, identify discriminating constraints
- **Feasibility prediction**: "Can I do Friedel-Crafts acylation on a pyridine?" → check if training reactions in that class contain electron-poor heterocycles, answer based on database evidence
- **Reaction novelty detection**: reactions matching no SMIRKS in the library are novel relative to the validated reaction space
- **Database curation**: flag reactions whose class label contradicts SMIRKS matching (labeled class A but matches class B's SMIRKS)
- **Ontology quality auditing**: the Phase 3 FP confusion matrix diagnoses whether class boundaries are well-defined

### Key Advantage

The agent doesn't rely on LLM parametric chemistry knowledge — it reasons over formally validated SMIRKS patterns and empirical database evidence. Answers are grounded in patterns tested against real reactions with measured recall and false-positive rates, not just memorized from training data.

## Linting & Style

- **Ruff** configured with Pyflakes, Pycodestyle, Pydocstyle (Google convention), pyupgrade, isort, and Pylint rules
- Tests are exempt from docstring and magic value rules (`D`, `PLR2004`)
- Pre-commit hooks: `check-added-large-files` (max 5MB) and `check-merge-conflict`
- Ruff linting is currently commented out in pre-commit config

## CI

GitHub Actions runs tox on push for Python 3.10 and 3.11 on Ubuntu.
