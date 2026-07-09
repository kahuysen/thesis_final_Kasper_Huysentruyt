"""Validated SMIRKS generalization with false-positive testing.

No dependency on gen-rxn-insight.  Only requires:
    pip install rdkit google-genai pydantic pandas tqdm joblib numpy

Five phases with checkpointing:
  1. Stratified 80/20 train/test split (at finest tier by default)
  2. Training: SMIRKS generalization per class (training set only)
  3. False-positive testing: each class's SMIRKS vs. all other-class reactions
  4. Fine-tuning: Gemini-based refinement to eliminate false positives
  5. Held-out evaluation: recall + FP count on test set

Usage:
    python generalize_smirks_validated.py \
        --database classification_database.parquet \
        --mapping structured_mapping.json \
        --output validated_smirks.json \
        --api-key $GEMINI_API_KEY \
        --phase all
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  REUSED FROM generalize_smirks_standalone.py                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ── test_smirks ──────────────────────────────────────────────────────────────


def test_smirks(rxn: str, smirks: str) -> dict:
    """Test whether a SMIRKS template fires on a reaction.

    Tries all reactant subsets and permutations so that rc_only templates
    (fewer reactant components than the actual reaction) are handled.

    Args:
        rxn: Reaction SMILES (``reactants>>products``).
        smirks: SMIRKS template string.

    Returns:
        Dict with ``applicable`` (bool), ``correct`` (bool), and
        ``products`` (set of predicted SMILES).
    """
    out: dict[str, Any] = {
        "applicable": False,
        "correct": False,
        "products": set(),
    }
    parts = rxn.split(">>")
    if len(parts) != 2:
        return out
    reactants = [Chem.MolFromSmiles(s) for s in parts[0].split(".")]
    expected = {
        Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
        for s in parts[1].split(".")
    }

    try:
        rxn_obj = AllChem.ReactionFromSmarts(smirks)
    except Exception:
        return out
    if rxn_obj is None:
        return out

    nreact = rxn_obj.GetNumReactantTemplates()
    for subset in itertools.combinations(reactants, nreact):
        for perm in itertools.permutations(subset):
            try:
                outcomes = rxn_obj.RunReactants(perm)
            except Exception:
                continue
            for prods in outcomes:
                try:
                    smi = Chem.MolToSmiles(prods[0], isomericSmiles=False)
                except Exception:
                    continue
                out["applicable"] = True
                out["products"].add(smi)
                if smi in expected:
                    out["correct"] = True

    return out


# ── screen_tier3 ─────────────────────────────────────────────────────────────


def get_example_reaction(df: pd.DataFrame, template: str) -> str:
    """Get the shortest reaction example for a given template."""
    dfc = df[df["TEMPLATE_rr0rp1_ring0"] == template].copy()
    series = dfc["SANITIZED_REACTION"].dropna()
    return series.loc[series.str.len().idxmin()]


def _parent_class(cls: str) -> str | None:
    """Derive the parent class code by dropping the last dotted segment.

    Examples: ``"1.4.2"`` -> ``"1.4"``, ``"1.4"`` -> ``"1"``, ``"1"`` -> None.
    """
    clean = cls.replace("CONFLICT:", "")
    parts = clean.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def tier_depth(cls: str) -> int:
    """Return the tier depth of a class code (number of dot-separated segments).

    Examples: ``"1.3.1"`` -> 3, ``"5.1.1.3"`` -> 4, ``"CONFLICT:1.3.1"`` -> 3.
    """
    return cls.replace("CONFLICT:", "").count(".") + 1


def tier_col_for_class(cls: str) -> str:
    """Return the DataFrame column name for a class code's tier depth.

    Examples: ``"1.3.1"`` -> ``"tier_3"``, ``"5.1.1.3"`` -> ``"tier_4"``.
    """
    return f"tier_{tier_depth(cls)}"


def detect_finest_tier(df: pd.DataFrame) -> str:
    """Find the deepest ``tier_N`` column in the DataFrame.

    Raises ``ValueError`` if no ``tier_N`` columns exist.
    """
    tier_cols = [
        c for c in df.columns
        if c.startswith("tier_") and c[5:].isdigit()
    ]
    if not tier_cols:
        raise ValueError("No tier_N columns found in DataFrame")
    return max(tier_cols, key=lambda c: int(c.split("_")[1]))


def get_class_split(
    fine_split: dict[str, dict[str, list[int]]],
    cls: str,
) -> dict[str, list[int]]:
    """Get aggregated train/test indices for a class from a fine-grained split.

    If *cls* is an exact key in the split, return it directly.
    Otherwise aggregate all keys that start with ``cls + "."``.

    Raises ``KeyError`` if *cls* is not found (exact or prefix).
    """
    if cls in fine_split:
        return fine_split[cls]

    prefix = cls + "."
    train: list[int] = []
    test: list[int] = []
    for key, indices in fine_split.items():
        if key.startswith(prefix):
            train.extend(indices["train"])
            test.extend(indices["test"])

    if not train and not test:
        raise KeyError(
            f"Class {cls!r} not found in split (exact or prefix)"
        )
    return {"train": train, "test": test}


def _class_in_split(cls: str, split: dict) -> bool:
    """Return True if *cls* is an exact key or any key starts with *cls* + '.'."""
    if cls in split:
        return True
    prefix = cls + "."
    return any(k.startswith(prefix) for k in split)


def _expand_class_wildcards(
    classes: list[str],
    split: dict[str, dict[str, list[int]]],
) -> list[str]:
    """Expand wildcard class specifications against the split keys.

    Supports two expansion forms:

    - ``"4.1.4.4.*"`` — expands to all split keys that start with
      ``"4.1.4.4."`` (direct children only, i.e. one level deeper).
    - ``"4.1.4.4.**"`` — expands to all split keys that start with
      ``"4.1.4.4."`` (all descendants at any depth).

    Plain class codes (no wildcard) are passed through unchanged.
    """
    expanded: list[str] = []
    for cls in classes:
        if cls.endswith(".**"):
            prefix = cls[:-2]  # "4.1.4.4."
            matches = sorted(k for k in split if k.startswith(prefix))
            if matches:
                expanded.extend(matches)
                logger.info(
                    f"Expanded {cls!r} -> {len(matches)} classes "
                    f"from split"
                )
            else:
                logger.warning(
                    f"Wildcard {cls!r} matched no classes in split"
                )
        elif cls.endswith(".*"):
            prefix = cls[:-1]  # "4.1.4.4."
            parent_depth = prefix.count(".")  # depth of children
            matches = sorted(
                k for k in split
                if k.startswith(prefix) and k.count(".") == parent_depth
            )
            if matches:
                expanded.extend(matches)
                logger.info(
                    f"Expanded {cls!r} -> {len(matches)} classes "
                    f"from split"
                )
            else:
                logger.warning(
                    f"Wildcard {cls!r} matched no classes in split"
                )
        else:
            expanded.append(cls)
    return expanded


def screen_tier3(
    llm_class: str,
    df: pd.DataFrame,
    named_dict: dict,
    top_n: int = 10,
    class_col: str = "tier_3",
) -> dict[str, Any]:
    """Screen a class: get top-N templates, examples, and coverage.

    Works with any tier column (tier_3, tier_4, etc.).  The parent tier
    name is derived by dropping the last segment of the class code.
    """
    dfc = df[df[class_col] == llm_class].copy()

    # Build hierarchical name from tier 2 down to current tier
    # e.g. "5.1.1.3" → ancestors ["5", "5.1", "5.1.1", "5.1.1.3"]
    # We skip tier 1 (just a number) and join tier 2+ names with ": "
    ancestors = _class_ancestors(llm_class)
    hier_names = []
    for anc in ancestors:
        if anc.count(".") >= 1:  # skip tier 1
            anc_name = named_dict.get(anc, "")
            if anc_name:
                hier_names.append(anc_name)
    if hier_names:
        name = f"{llm_class} - {': '.join(hier_names)}"
    else:
        name = f"{llm_class} - {named_dict.get(llm_class, llm_class)}"

    top_n_templates = Counter(dfc["TEMPLATE_rr0rp1_ring0"]).most_common(top_n)
    occs = sum(t[1] for t in top_n_templates)
    coverage = occs / len(dfc.index)
    top_n_examples = [get_example_reaction(dfc, t[0]) for t in top_n_templates]
    return {
        "reaction_class": name,
        "top_n_templates": [t[0] for t in top_n_templates],
        "top_n_examples": top_n_examples,
        "coverage": coverage,
    }


def screen_tier3_auto(
    llm_class: str,
    df: pd.DataFrame,
    named_dict: dict,
    min_coverage: float = 0.9,
    start_n: int = 10,
    max_n: int = 50,
    class_col: str = "tier_3",
) -> dict[str, Any]:
    """Screen a class, auto-increasing top_n until coverage >= min_coverage."""
    top_n = start_n
    while top_n <= max_n:
        data = screen_tier3(
            llm_class, df, named_dict, top_n=top_n, class_col=class_col,
        )
        if data["coverage"] >= min_coverage or top_n >= max_n:
            data["top_n"] = top_n
            return data
        top_n = min(top_n * 2, max_n)

    data["top_n"] = top_n
    return data


# ── Pydantic response model ─────────────────────────────────────────────────


class SmirksGeneralization(BaseModel):
    """Structured output from Gemini for SMIRKS generalization."""

    reasoning: str = Field(
        description="Step-by-step reasoning following steps A-F."
    )
    smirks: list[str] = Field(
        description="One or more generalized SMIRKS strings."
    )


# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert cheminformatician. Your task: given a set of specific reaction \
templates (SMIRKS) and example reactions for a single reaction class, produce one \
or more generalized SMIRKS that cover all the templates.

== SMARTS NOTATION REFERENCE ==
- [C] = aliphatic carbon, [c] = aromatic carbon, [#6] = any carbon
- [N] = aliphatic nitrogen, [n] = aromatic nitrogen, [#7] = any nitrogen
- H<n> = total hydrogen count (e.g. H2 = two hydrogens)
- D<n> = degree = number of explicit connections (e.g. D3 = three bonds)
- +0 = neutral charge
- :<n> = atom map number (tracks atom identity across reactants\u2192products)
- [F,Cl,Br,I] = list notation: matches any of the listed elements
- Atoms WITHOUT a map number are unmapped: they are part of a leaving group \
  or reagent fragment and do NOT appear on the other side of >>

== RULES ==
1. REACTION CENTER (RC) atoms are those whose bonds or properties change \
   between reactants and products. RC atoms MUST have map numbers and specific \
   descriptors (H, D, charge).
2. CONTEXT atoms are mapped atoms adjacent to the RC that do NOT change. \
   Context atoms should be GENERAL:
   - If context varies between aromatic (c) and aliphatic (C): use [#6;+0:n]
   - If context is always aromatic: use [c;H0;+0:n] or similar
   - If context is always aliphatic with consistent H-count: keep [C;H<n>;+0:n]
3. LEAVING GROUPS (atoms that depart) have NO map number on the reactant side \
   and do not appear in the product.
   - Halide leaving groups: merge with [F,Cl,Br,I;H0;+0] (no map number)
   - Complex leaving groups (e.g. mesylate, boronate ester): simplify to the \
     minimal connection pattern or omit if not needed for matching
4. When ALL templates show the SAME RC transformation and differ ONLY in context \
   atoms, produce a MINIMAL RC-only SMIRKS with no context atoms.
5. Produce MULTIPLE SMIRKS when templates fall into groups that differ in the \
   RC transformation itself (e.g. different H-counts/degrees on RC atoms, \
   primary vs secondary amine). One SMIRKS cannot cover both.
6. Keep +0 on all mapped atoms.
7. Every mapped atom in the products MUST appear in the reactants.
8. The unmapped carbonyl oxygen in a ketone/aldehyde that is lost during \
   reduction should have NO map number: [C;H0;D3;+0:n]=[O;H0;+0]
9. Product-side atoms: update H and D to reflect the new bonding state.

== REASONING STEPS (follow these in order) ==
A. IDENTIFY THE CORE TRANSFORMATION: What bond is formed? What bond is broken? \
   What functional group change occurs?
B. FIND THE REACTION CENTER ATOMS: Which atoms change H-count, degree, or \
   bonding partners? These are your RC atoms \u2014 they need map numbers and \
   specific H/D descriptors.
C. CHECK FOR DISTINCT PATTERNS: Do the templates fall into 2+ groups where \
   the RC itself differs (e.g. different starting H-count on nitrogen)? \
   If yes \u2192 plan multiple SMIRKS.
D. IDENTIFY VARIABLE POSITIONS: What differs across templates but serves the \
   same role? (leaving groups, context atom type, reagent fragments) \
   Generalize these.
E. DECIDE CONTEXT LEVEL: Are context atoms needed for correct matching, or \
   can you strip them for a minimal RC-only SMIRKS?
F. WRITE THE SMIRKS: For each pattern, write reactant>>product. Verify every \
   mapped atom appears on both sides. Verify H/D changes are consistent.

== FEW-SHOT EXAMPLES ==

--- EXAMPLE 1 (RC-only, minimal) ---

INPUT:
Reaction class: 7.1.1 - Alcohols to aldehydes: Oxidation of Primary Alcohols to Aldehydes

Templates:
1. [c;H0;+0:1]-[C;H2;D2;+0:2]-[O;H1;D1;+0:3]>>[c;H0;+0:1]-[C;H1;D2;+0:2]=[O;H0;D1;+0:3]
2. [O;H1;D1;+0:1]-[C;H2;D2;+0:2]-[c;H0;+0:3]>>[O;H0;D1;+0:1]=[C;H1;D2;+0:2]-[c;H0;+0:3]
3. [C;H2;+0:1]-[C;H2;D2;+0:2]-[O;H1;D1;+0:3]>>[C;H2;+0:1]-[C;H1;D2;+0:2]=[O;H0;D1;+0:3]
4. [C;H1;+0:1]-[C;H2;D2;+0:2]-[O;H1;D1;+0:3]>>[C;H1;+0:1]-[C;H1;D2;+0:2]=[O;H0;D1;+0:3]
5. [C;H0;+0:1]-[C;H2;D2;+0:2]-[O;H1;D1;+0:3]>>[C;H0;+0:1]-[C;H1;D2;+0:2]=[O;H0;D1;+0:3]
6. [O;H1;D1;+0:1]-[C;H2;D2;+0:2]-[C;H2;+0:3]>>[O;H0;D1;+0:1]=[C;H1;D2;+0:2]-[C;H2;+0:3]
7. [O;H1;D1;+0:1]-[C;H2;D2;+0:2]-[C;H1;+0:3]>>[O;H0;D1;+0:1]=[C;H1;D2;+0:2]-[C;H1;+0:3]
8. [O;H1;D1;+0:1]-[C;H2;D2;+0:2]-[C;H0;+0:3]>>[O;H0;D1;+0:1]=[C;H1;D2;+0:2]-[C;H0;+0:3]
9. [C;H1;+0:1]/[C;H2;D2;+0:2]-[O;H1;D1;+0:3]>>[C;H1;+0:1]/[C;H1;D2;+0:2]=[O;H0;D1;+0:3]
10. [O;H1;D1;+0:1]-[C;H2;D2;+0:2]/[C;H1;+0:3]>>[O;H0;D1;+0:1]=[C;H1;D2;+0:2]/[C;H1;+0:3]

Example reactions:
1. CSc1ccccc1CO>>CSc1ccccc1C=O
2. OCc1cc2sccc2s1>>O=Cc1cc2sccc2s1
3. CCCCCCCCCO>>CCCCCCCCC=O
4. C=CCC(CO)CCOC>>C=CCC(C=O)CCOC
5. CC(C)CCCCC#CCO>>CC(C)CCCCC#CC=O

REASONING:
A. CORE TRANSFORMATION: Primary alcohol (-CH2-OH) is oxidized to aldehyde (-CH=O). \
A C-H bond is broken and the C-O single bond becomes a C=O double bond. \
The oxygen loses its hydrogen.
B. REACTION CENTER ATOMS: Two atoms change:
   - Carbon: H2,D2 \u2192 H1,D2 (loses one H, keeps same degree because single bond \
     becomes double bond)
   - Oxygen: H1,D1 \u2192 H0,D1 (loses H, keeps D1 because single bond becomes \
     double bond)
C. DISTINCT PATTERNS: No. All 10 templates show the identical RC transformation \
   (CH2-OH \u2192 CH=O). They only differ in what is attached to the carbon.
D. VARIABLE POSITIONS: The neighbor of the RC carbon varies: [c;H0], [C;H2], \
   [C;H1], [C;H0], and sometimes has stereo notation (/). This is purely a \
   context atom.
E. CONTEXT LEVEL: Since the context atom is not needed to distinguish this \
   reaction (the RC itself is unique: CH2-OH \u2192 CH=O), we strip all context \
   for a minimal RC-only SMIRKS.
F. WRITE THE SMIRKS:
   [C;H2;D2;+0:1]-[O;H1;D1;+0:2]>>[C;H1;D2;+0:1]=[O;H0;D1;+0:2]
   Check: :1 appears on both sides \u2713, :2 appears on both sides \u2713, \
   C goes H2\u2192H1 (lost 1H) \u2713, O goes H1\u2192H0 (lost 1H) \u2713, \
   single bond \u2192 double bond \u2713

OUTPUT:
["[C;H2;D2;+0:1]-[O;H1;D1;+0:2]>>[C;H1;D2;+0:1]=[O;H0;D1;+0:2]"]

--- EXAMPLE 2 (halide merging + context generalization) ---

INPUT:
Reaction class: 1.1.1 - N-substitution with alkyl-X: Alkylation with Alkyl Halides or Sulfonates (SN2 type)

Templates:
1. [C;H2;+0:1]-[C;H2;D2;+0:2]-[Cl;H0;+0].[N;H1;D2;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]>>[C;H2;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
2. [C;H2;+0:1]-[C;H2;D2;+0:2]-[Br;H0;+0].[N;H1;D2;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]>>[C;H2;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
3. [C;H2;+0:1]-[N;H1;D2;+0:2]-[C;H2;+0:3].[C;H2;D2;+0:4](-[C;H2;+0:5])-[Br;H0;+0]>>[C;H2;+0:1]-[N;H0;D3;+0:2](-[C;H2;D2;+0:4]-[C;H2;+0:5])-[C;H2;+0:3]
4. [C;H2;+0:1]-[N;H1;D2;+0:2]-[C;H2;+0:3].[C;H2;D2;+0:4](-[C;H2;+0:5])-[Cl;H0;+0]>>[C;H2;+0:1]-[N;H0;D3;+0:2](-[C;H2;D2;+0:4]-[C;H2;+0:5])-[C;H2;+0:3]
5. [C;H0;+0:1]-[C;H2;D2;+0:2]-[Br;H0;+0].[N;H1;D2;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]>>[C;H0;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
6. [c;H0;+0:1]-[C;H2;D2;+0:2]-[Cl;H0;+0].[N;H1;D2;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]>>[c;H0;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
7. [C;H2;+0:1]-[N;H1;D2;+0:2]-[C;H2;+0:3].[C;H2;D2;+0:4](-[c;H0;+0:5])-[Cl;H0;+0]>>[C;H2;+0:1]-[N;H0;D3;+0:2](-[C;H2;D2;+0:4]-[c;H0;+0:5])-[C;H2;+0:3]
8. [N;H1;D2;+0:1](-[C;H2;+0:2])-[C;H2;+0:3].[C;H2;+0:4]-[C;H2;D2;+0:5]-[O;H0;+0]-[S;H0;+0](-[C;H3;+0])(=[O;H0;+0])=[O;H0;+0]>>[C;H2;+0:4]-[C;H2;D2;+0:5]-[N;H0;D3;+0:1](-[C;H2;+0:2])-[C;H2;+0:3]
9. [c;H0;+0:1]-[C;H2;D2;+0:2]-[Br;H0;+0].[N;H1;D2;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]>>[c;H0;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
10. [C;H2;+0:1]-[C;H2;D2;+0:2]-[I;H0;+0].[N;H1;D2;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]>>[C;H2;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]

Example reactions:
1. C1CCNCC1.OCCCCl>>OCCCN1CCCCC1
2. C1COCCN1.OCCCBr>>OCCCN1CCOCC1
3. CN1CCNCC1.OCCBr>>CN1CCN(CCO)CC1
4. CN1CCNCC1.OCCCl>>CN1CCN(CCO)CC1
5. C#CCBr.CN1CCNCC1>>C#CCN1CCN(C)CC1

REASONING:
A. CORE TRANSFORMATION: SN2 N-alkylation. A C-X bond breaks (X = halide leaving \
   group), a new C-N bond forms. The nitrogen loses one hydrogen.
B. REACTION CENTER ATOMS:
   - C (the alkyl carbon): bonded to halide in reactant, bonded to N in product. \
     Stays H2,D2 (halide replaced by N, same degree).
   - N (the amine nitrogen): H1,D2 \u2192 H0,D3 (loses one H, gains one bond to C).
   - The halide LEAVES \u2014 it has no map number and does not appear in the product.
C. DISTINCT PATTERNS: No. All templates show the same transformation: \
   secondary amine (N;H1;D2) + alkyl-X \u2192 tertiary amine (N;H0;D3). Template 8 \
   uses mesylate instead of halide, but the RC change is identical.
D. VARIABLE POSITIONS:
   - Leaving group: Cl (templates 1,4,6,7), Br (2,3,5,9), I (10), mesylate (8). \
     Merge halides to [F,Br,Cl,I;H0;+0]. Mesylate is an outlier \u2014 the halide \
     SMIRKS won't cover it, but it covers the vast majority.
   - Context atom on the alkyl side (:1 or :5): [C;H2], [C;H0], [c;H0]. \
     Varies between aliphatic and aromatic \u2192 generalize to [#6;+0].
   - Context atoms on the amine side (:3,:5 or :4,:5): consistently [C;H2;+0]. \
     Keep as-is.
E. CONTEXT LEVEL: We need at least one context atom on each side for correct \
   substructure matching:
   - Alkyl side: [#6;+0:1] (context) \u2014 [C;H2;D2;+0:2] (RC) \u2014 [F,Br,Cl,I] (LG)
   - Amine side: [C;H2;+0:3] \u2014 [N;H1;D2;+0:4] (RC) \u2014 [C;H2;+0:5]
F. WRITE THE SMIRKS:
   [#6;+0:1]-[C;H2;D2;+0:2]-[F,Br,Cl,I;H0;+0].[C;H2;+0:3]-[N;H1;D2;+0:4]-[C;H2;+0:5]>>[#6;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:4](-[C;H2;+0:3])-[C;H2;+0:5]
   Check: :1\u2013:5 all appear on both sides \u2713, halide has no map number \u2713, \
   N goes H1,D2\u2192H0,D3 \u2713, C:2 stays H2,D2 \u2713, leaving group merged \u2713

OUTPUT:
["[#6;+0:1]-[C;H2;D2;+0:2]-[F,Br,Cl,I;H0;+0].[C;H2;+0:3]-[N;H1;D2;+0:4]-[C;H2;+0:5]>>[#6;+0:1]-[C;H2;D2;+0:2]-[N;H0;D3;+0:4](-[C;H2;+0:3])-[C;H2;+0:5]"]

--- EXAMPLE 3 (multiple SMIRKS needed) ---

INPUT:
Reaction class: 1.2.2 - Reductive amination: Reductive Amination with ketones

Templates:
1. [C;H2;+0:1]-[N;H1;D2;+0:2]-[C;H2;+0:3].[C;H0;D3;+0:4](-[C;H2;+0:5])(-[C;H2;+0:6])=[O;H0;+0]>>[C;H2;+0:1]-[N;H0;D3;+0:2](-[C;H1;D3;+0:4](-[C;H2;+0:5])-[C;H2;+0:6])-[C;H2;+0:3]
2. [N;H1;D2;+0:1](-[C;H2;+0:2])-[C;H2;+0:3].[C;H2;+0:4]-[C;H0;D3;+0:5](-[C;H2;+0:6])=[O;H0;+0]>>[C;H2;+0:4]-[C;H1;D3;+0:5](-[C;H2;+0:6])-[N;H0;D3;+0:1](-[C;H2;+0:2])-[C;H2;+0:3]
3. [N;H2;D1;+0:1]-[C;H2;+0:2].[C;H2;+0:3]-[C;H0;D3;+0:4](-[C;H2;+0:5])=[O;H0;+0]>>[C;H2;+0:3]-[C;H1;D3;+0:4](-[C;H2;+0:5])-[N;H1;D2;+0:1]-[C;H2;+0:2]
4. [C;H2;+0:1]-[N;H2;D1;+0:2].[C;H0;D3;+0:3](-[C;H2;+0:4])(-[C;H2;+0:5])=[O;H0;+0]>>[C;H2;+0:1]-[N;H1;D2;+0:2]-[C;H1;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
5. [c;H0;+0:1]-[N;H2;D1;+0:2].[C;H0;D3;+0:3](-[C;H2;+0:4])(-[C;H2;+0:5])=[O;H0;+0]>>[c;H0;+0:1]-[N;H1;D2;+0:2]-[C;H1;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
6. [N;H2;D1;+0:1]-[c;H0;+0:2].[C;H2;+0:3]-[C;H0;D3;+0:4](-[C;H2;+0:5])=[O;H0;+0]>>[C;H2;+0:3]-[C;H1;D3;+0:4](-[C;H2;+0:5])-[N;H1;D2;+0:1]-[c;H0;+0:2]
7. [C;H1;+0:1]-[N;H2;D1;+0:2].[C;H0;D3;+0:3](-[C;H2;+0:4])(-[C;H2;+0:5])=[O;H0;+0]>>[C;H1;+0:1]-[N;H1;D2;+0:2]-[C;H1;D3;+0:3](-[C;H2;+0:4])-[C;H2;+0:5]
8. [N;H1;D2;+0:1](-[C;H2;+0:2])-[C;H2;+0:3].[C;H3;+0:4]-[C;H0;D3;+0:5](-[C;H3;+0:6])=[O;H0;+0]>>[C;H3;+0:4]-[C;H1;D3;+0:5](-[C;H3;+0:6])-[N;H0;D3;+0:1](-[C;H2;+0:2])-[C;H2;+0:3]
9. [N;H2;D1;+0:1]-[C;H1;+0:2].[C;H2;+0:3]-[C;H0;D3;+0:4](-[C;H2;+0:5])=[O;H0;+0]>>[C;H2;+0:3]-[C;H1;D3;+0:4](-[C;H2;+0:5])-[N;H1;D2;+0:1]-[C;H1;+0:2]
10. [N;H2;D1;+0:1]-[C;H2;+0:2].[C;H2;+0:3]-[C;H0;D3;+0:4](-[C;H3;+0:5])=[O;H0;+0]>>[C;H2;+0:3]-[C;H1;D3;+0:4](-[C;H3;+0:5])-[N;H1;D2;+0:1]-[C;H2;+0:2]

Example reactions:
1. O=C1CCC1.OC1CCNCC1>>OC1CCN(C2CCC2)CC1
2. C1CCNC1.O=C1CCCCC1>>C1CCC(N2CCCC2)CC1
3. CCC(=O)CC.NCc1ccccc1>>CCC(CC)NCc1ccccc1
4. NCCN.O=C1CCCCC1>>NCCNC1CCCCC1
5. Cc1oncc1N.O=C1CCCC1>>Cc1oncc1NC1CCCC1

REASONING:
A. CORE TRANSFORMATION: Reductive amination \u2014 a ketone C=O reacts with an amine. \
   The C=O is reduced: the oxygen leaves, the carbon gains an H, and a new C-N \
   bond forms. The nitrogen loses one H.
B. REACTION CENTER ATOMS:
   - Carbonyl carbon: H0,D3 \u2192 H1,D3 (gains one H, keeps D3 because the lost \
     C=O bond is replaced by C-N + C-H). The oxygen LEAVES (no map number).
   - Nitrogen: loses one H and gains one bond to the carbonyl carbon.
C. DISTINCT PATTERNS: YES \u2014 two groups:
   - Group A (templates 1,2,8): SECONDARY amine reacts. N starts at H1,D2, \
     ends at H0,D3. The amine has TWO carbon neighbors.
   - Group B (templates 3,4,5,6,7,9,10): PRIMARY amine reacts. N starts at \
     H2,D1, ends at H1,D2. The amine has ONE carbon neighbor.
   These cannot be merged: the N atom has different H-count and degree in each \
   group. We need TWO separate SMIRKS.
D. VARIABLE POSITIONS:
   - Context atom on the amine (attached to N): [C;H2], [c;H0], [C;H1]. \
     Varies between aromatic and aliphatic \u2192 generalize to [#6;+0].
   - Context atoms on the ketone (attached to carbonyl C): [C;H2], [C;H3]. \
     These are the ketone substituents. Since we want to cover any ketone, \
     we can strip these context atoms \u2014 the C=O pattern is sufficient.
   - The ketone oxygen =[O;H0;+0] leaves during reduction: no map number.
E. CONTEXT LEVEL:
   - Amine side: keep one context [#6;+0] on each side of N (Group A needs two, \
     Group B needs one).
   - Ketone side: the carbonyl C plus its unmapped =O is sufficient. Strip the \
     carbon substituents for maximum generality.
F. WRITE THE SMIRKS:
   Group A (secondary amine + ketone \u2192 tertiary amine):
   [#6;+0:1]-[N;H1;D2;+0:2]-[#6;+0:3].[C;H0;D3;+0:4]=[O;H0;+0]>>[#6;+0:1]-[N;H0;D3;+0:2](-[C;H1;D3;+0:4])-[#6;+0:3]
   Check: :1\u2013:4 on both sides \u2713, O has no map \u2713, N goes H1,D2\u2192H0,D3 \u2713, \
   C:4 goes H0\u2192H1 \u2713

   Group B (primary amine + ketone \u2192 secondary amine):
   [#6;+0:1]-[N;H2;D1;+0:2].[C;H0;D3;+0:3]=[O;H0;+0]>>[#6;+0:1]-[N;H1;D2;+0:2]-[C;H1;D3;+0:3]
   Check: :1\u2013:3 on both sides \u2713, O has no map \u2713, N goes H2,D1\u2192H1,D2 \u2713, \
   C:3 goes H0\u2192H1 \u2713

OUTPUT:
["[#6;+0:1]-[N;H1;D2;+0:2]-[#6;+0:3].[C;H0;D3;+0:4]=[O;H0;+0]>>[#6;+0:1]-[N;H0;D3;+0:2](-[C;H1;D3;+0:4])-[#6;+0:3]", \
"[#6;+0:1]-[N;H2;D1;+0:2].[C;H0;D3;+0:3]=[O;H0;+0]>>[#6;+0:1]-[N;H1;D2;+0:2]-[C;H1;D3;+0:3]"]

== END EXAMPLES ==
"""


# ── User prompt template ─────────────────────────────────────────────────────

_USER_PROMPT_TEMPLATE = """\
Generalize the following reaction templates into one or more SMIRKS.

Reaction class: {reaction_class}

Templates:
{numbered_templates}

Example reactions:
{numbered_examples}"""

_RETRY_FEEDBACK_TEMPLATE = """

The SMIRKS you produced failed validation. On example reaction \
"{failing_rxn}", test_smirks returned applicable={applicable}, correct={correct}. \
Please fix the SMIRKS. Common issues: wrong H-count on product atoms, \
missing map number, over-specific context atoms."""


def _build_user_prompt(
    reaction_class: str,
    templates: list[str],
    examples: list[str],
) -> str:
    """Build the per-class user prompt."""
    numbered_templates = "\n".join(
        f"{i + 1}. {t}" for i, t in enumerate(templates)
    )
    numbered_examples = "\n".join(
        f"{i + 1}. {e}" for i, e in enumerate(examples)
    )
    return _USER_PROMPT_TEMPLATE.format(
        reaction_class=reaction_class,
        numbered_templates=numbered_templates,
        numbered_examples=numbered_examples,
    )


def validate_smirks(
    smirks_list: list[str],
    examples: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Validate generalized SMIRKS against example reactions.

    Returns:
        Dict with ``valid`` (bool), ``combined_coverage`` (float),
        ``per_smirks`` (list of dicts), and ``failures`` (list of
        ``(smirks, rxn, applicable, correct)`` tuples).
    """
    per_smirks: list[dict[str, Any]] = []
    failures: list[tuple[str, str, bool, bool]] = []

    for smirks in smirks_list:
        scores = []
        first_failure = None
        for rxn in examples:
            try:
                r = test_smirks(rxn, smirks)
            except Exception:
                r = {"applicable": False, "correct": False}
            scores.append(r["correct"])
            if not r["correct"] and first_failure is None:
                first_failure = (smirks, rxn, r["applicable"], r["correct"])
        coverage = sum(scores) / len(scores) if scores else 0.0
        per_smirks.append({
            "smirks": smirks,
            "coverage": coverage,
            "n_correct": sum(scores),
            "n_total": len(scores),
        })
        if first_failure is not None and coverage < threshold:
            failures.append(first_failure)

    # Combined coverage: union across all SMIRKS
    combined_correct: set[int] = set()
    for smirks in smirks_list:
        for i, rxn in enumerate(examples):
            try:
                r = test_smirks(rxn, smirks)
            except Exception:
                r = {"correct": False}
            if r["correct"]:
                combined_correct.add(i)
    combined_coverage = (
        len(combined_correct) / len(examples) if examples else 0.0
    )

    return {
        "valid": combined_coverage >= threshold,
        "combined_coverage": combined_coverage,
        "per_smirks": per_smirks,
        "failures": failures,
    }


# ── SmirksGeneralizer class ─────────────────────────────────────────────────


class SmirksGeneralizer:
    """Generalizes reaction templates into broad SMIRKS via Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.3,
        use_cache: bool = False,
        timeout: int | None = None,
    ) -> None:
        from google import genai
        from google.genai import types as gtypes

        self._genai = genai
        self._gtypes = gtypes

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        self._cache_name: str | None = None
        if use_cache:
            self._cache_name = self._create_cache()

    def _create_cache(self) -> str | None:
        """Create a Gemini context cache for the system prompt."""
        try:
            cache = self.client.caches.create(
                model=self.model,
                config=self._gtypes.CreateCachedContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    ttl="3600s",
                ),
            )
            logger.info(f"Context cache created: {cache.name}")
            return cache.name
        except Exception as exc:
            logger.warning(
                f"Context caching not available for {self.model!r}: {exc}. "
                "Falling back to uncached mode."
            )
            return None

    def refresh_cache(self) -> None:
        """Renew the context cache TTL."""
        if self._cache_name is None:
            return
        try:
            self.client.caches.update(
                name=self._cache_name,
                config=self._gtypes.UpdateCachedContentConfig(ttl="3600s"),
            )
        except Exception as exc:
            logger.warning(f"Cache renewal failed: {exc}. Recreating.")
            self._cache_name = self._create_cache()

    @property
    def token_usage(self) -> dict[str, int]:
        """Return cumulative token usage statistics."""
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def _call(
        self,
        user_prompt: str,
        retries: int = 3,
        system_prompt: str | None = None,
    ) -> SmirksGeneralization:
        """Make one Gemini call with retries. Returns parsed Pydantic object.

        Args:
            user_prompt: The user prompt to send.
            retries: Number of retries on failure.
            system_prompt: Override system prompt (used for refinement).
        """
        sys_prompt = system_prompt or _SYSTEM_PROMPT
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": SmirksGeneralization,
            "temperature": self.temperature,
        }
        if self.timeout is not None:
            config_kwargs["httpOptions"] = self._gtypes.HttpOptions(
                timeout=self.timeout * 1000,  # seconds to milliseconds
            )
        if self._cache_name and system_prompt is None:
            config_kwargs["cached_content"] = self._cache_name
        else:
            config_kwargs["system_instruction"] = sys_prompt

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    config=self._gtypes.GenerateContentConfig(**config_kwargs),
                    contents=[user_prompt],
                )
                # Track token usage
                self.total_calls += 1
                meta = getattr(response, "usage_metadata", None)
                if meta is not None:
                    self.total_prompt_tokens += getattr(
                        meta, "prompt_token_count", 0
                    ) or 0
                    self.total_completion_tokens += getattr(
                        meta, "candidates_token_count", 0
                    ) or 0
                # response.parsed can be None if the model's output
                # didn't conform to the schema (e.g. empty, truncated,
                # or safety-filtered).  Treat as a retryable failure.
                if response.parsed is None:
                    raw = getattr(response, "text", "<no text>")
                    raise ValueError(
                        f"Model returned unparseable response: {raw[:200]}"
                    )
                return response.parsed
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed: {exc}. "
                    f"Retrying in {wait}s."
                )
                time.sleep(wait)
        raise RuntimeError(f"All {retries} attempts failed") from last_exc

    def generalize(
        self,
        reaction_class: str,
        templates: list[str],
        examples: list[str],
        max_retries: int = 3,
        validation_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Generalize templates for one reaction class.

        Returns:
            Dict with ``reaction_class``, ``smirks`` (list), ``reasoning``,
            ``valid`` (bool), ``combined_coverage`` (float), ``per_smirks``
            (list), ``attempts`` (int), and ``error`` (str or None).
        """
        user_prompt = _build_user_prompt(reaction_class, templates, examples)

        result = None
        val = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self._call(user_prompt)
            except Exception as exc:
                logger.error(
                    f"LLM call failed for {reaction_class}: {exc}"
                )
                return {
                    "reaction_class": reaction_class,
                    "smirks": [],
                    "reasoning": None,
                    "valid": False,
                    "combined_coverage": 0.0,
                    "per_smirks": [],
                    "attempts": attempt,
                    "error": str(exc),
                }

            val = validate_smirks(
                result.smirks, examples, threshold=validation_threshold
            )

            if val["valid"]:
                logger.info(
                    f"[{reaction_class}] Attempt {attempt}: PASS "
                    f"(coverage={val['combined_coverage']:.0%})"
                )
                return {
                    "reaction_class": reaction_class,
                    "smirks": result.smirks,
                    "reasoning": result.reasoning,
                    "valid": True,
                    "combined_coverage": val["combined_coverage"],
                    "per_smirks": val["per_smirks"],
                    "attempts": attempt,
                    "error": None,
                }

            logger.info(
                f"[{reaction_class}] Attempt {attempt}: FAIL "
                f"(coverage={val['combined_coverage']:.0%})"
            )
            if attempt < max_retries and val["failures"]:
                smirks_str, rxn, applicable, correct = val["failures"][0]
                feedback = _RETRY_FEEDBACK_TEMPLATE.format(
                    failing_rxn=rxn,
                    applicable=applicable,
                    correct=correct,
                )
                user_prompt += feedback

        return {
            "reaction_class": reaction_class,
            "smirks": result.smirks if result else [],
            "reasoning": result.reasoning if result else None,
            "valid": False,
            "combined_coverage": val["combined_coverage"] if val else 0.0,
            "per_smirks": val["per_smirks"] if val else [],
            "attempts": max_retries,
            "error": f"Validation failed after {max_retries} attempts",
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NEW: VALIDATED PIPELINE                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ── Checkpoint helpers ───────────────────────────────────────────────────────


def save_checkpoint(path: Path, data: dict) -> None:
    """Atomically save checkpoint to JSON (write-to-temp then rename)."""
    path = Path(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # On Windows, os.replace works atomically on same volume
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint from JSON. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Phase 1: Stratified Split ───────────────────────────────────────────────


def stratified_split(
    df: pd.DataFrame,
    class_col: str = "tier_3",
    test_frac: float = 0.2,
    min_size: int = 5,
    seed: int = 210995,
) -> dict[str, dict[str, list[int]]]:
    """Create a stratified 80/20 train/test split per class.

    Args:
        df: Database DataFrame.
        class_col: Column with class labels.
        test_frac: Fraction of data for test set.
        min_size: Minimum class size to include.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping class code -> {"train": [row indices], "test": [row indices]}.
        Indices are integer positions (iloc-style), stored as Python ints.
    """
    rng = np.random.RandomState(seed)
    split: dict[str, dict[str, list[int]]] = {}

    grouped = df.groupby(class_col)
    for cls, group in grouped:
        if len(group) < min_size:
            continue

        indices = group.index.tolist()
        rng.shuffle(indices)

        n_test = max(1, int(len(indices) * test_frac))
        test_idx = [int(i) for i in indices[:n_test]]
        train_idx = [int(i) for i in indices[n_test:]]
        split[cls] = {"train": train_idx, "test": test_idx}

    return split


# ── Phase 2: Training ───────────────────────────────────────────────────────


def train_all_classes(
    df: pd.DataFrame,
    split: dict[str, dict[str, list[int]]],
    named_dict: dict,
    generalizer: SmirksGeneralizer,
    ckpt_path: Path,
    ckpt: dict,
    classes: list[str] | None = None,
    min_coverage: float = 0.9,
    max_templates: int = 50,
    max_retries: int = 3,
    validation_threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Phase 2: generalize SMIRKS for each class using training data only.

    The tier column for each class is derived automatically from the class
    code depth via ``tier_col_for_class``.  The split may be at a finer
    granularity than the processing classes; ``get_class_split`` aggregates
    indices upward when needed.

    Saves checkpoint after each class so that it can resume.

    Returns:
        Dict mapping class code -> training result dict.
    """
    train_results = ckpt.get("train_results", {})
    target_classes = classes or sorted(split.keys())

    remaining = [c for c in target_classes if c not in train_results]
    logger.info(
        f"Phase 2: {len(remaining)} classes to train "
        f"({len(train_results)} already done)"
    )

    for cls in tqdm(remaining, desc="Phase 2: Training"):
        if not _class_in_split(cls, split):
            logger.warning(f"Class {cls} not in split, skipping")
            continue

        train_idx = get_class_split(split, cls)["train"]
        df_train = df.loc[train_idx]
        class_col = tier_col_for_class(cls)

        try:
            data = screen_tier3_auto(
                cls,
                df_train,
                named_dict,
                min_coverage=min_coverage,
                max_n=max_templates,
                class_col=class_col,
            )
        except Exception as exc:
            logger.warning(f"Screening failed for {cls}: {exc}")
            train_results[cls] = {
                "reaction_class": cls,
                "smirks": [],
                "valid": False,
                "error": f"Screening failed: {exc}",
            }
            ckpt["train_results"] = train_results
            save_checkpoint(ckpt_path, ckpt)
            continue

        result = generalizer.generalize(
            reaction_class=data["reaction_class"],
            templates=data["top_n_templates"],
            examples=data["top_n_examples"],
            max_retries=max_retries,
            validation_threshold=validation_threshold,
        )

        train_results[cls] = result

        # Checkpoint after each class
        ckpt["train_results"] = train_results
        save_checkpoint(ckpt_path, ckpt)

        # Refresh cache periodically
        generalizer.refresh_cache()

    return train_results


# ── Phase 3: False Positive Testing ──────────────────────────────────────────


def _test_smirks_precompiled(
    reactant_mols: list,
    expected_prods: set[str],
    rxn_obj: Chem.rdChemReactions.ChemicalReaction,
) -> bool:
    """Optimized FP test with pre-parsed Mols and pre-compiled SMIRKS.

    Returns True if the SMIRKS fires correctly (i.e., is a false positive
    when tested against a reaction from another class).
    """
    nreact = rxn_obj.GetNumReactantTemplates()
    for subset in itertools.combinations(reactant_mols, nreact):
        for perm in itertools.permutations(subset):
            try:
                outcomes = rxn_obj.RunReactants(perm)
            except Exception:
                continue
            for prods in outcomes:
                try:
                    smi = Chem.MolToSmiles(prods[0], isomericSmiles=False)
                except Exception:
                    continue
                if smi in expected_prods:
                    return True
    return False


def _compile_smirks(smirks: str) -> Chem.rdChemReactions.ChemicalReaction | None:
    """Compile a SMIRKS string into a reaction object. Returns None on failure."""
    try:
        rxn_obj = AllChem.ReactionFromSmarts(smirks)
        return rxn_obj
    except Exception:
        return None


def _parse_reaction(rxn_smi: str) -> tuple[list, set[str]] | None:
    """Parse a reaction SMILES into (reactant mols, expected product SMILES).

    Returns None if parsing fails.
    """
    parts = rxn_smi.split(">>")
    if len(parts) != 2:
        return None
    reactant_smiles = parts[0].split(".")
    product_smiles = parts[1].split(".")

    reactant_mols = []
    for s in reactant_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        reactant_mols.append(mol)

    expected = set()
    for s in product_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        expected.add(Chem.MolToSmiles(mol, isomericSmiles=False))

    return reactant_mols, expected


def _fp_test_chunk(
    chunk: list[tuple[str, str, list, set[str]]],
    all_smirks_compiled: list[tuple[str, str, Any, int]],
) -> list[dict[str, str]]:
    """Worker: test a chunk of reactions against all other-class SMIRKS.

    Args:
        chunk: List of (rxn_class, rxn_smi, reactant_mols, expected_prods).
        all_smirks_compiled: List of (smirks_class, smirks_str, rxn_obj, nreact).

    Returns:
        List of false-positive records: {smirks_class, true_class, rxn, smirks}.
    """
    fps = []
    for rxn_class, rxn_smi, reactant_mols, expected_prods in chunk:
        n_reactants = len(reactant_mols)
        for smirks_class, smirks_str, rxn_obj, nreact in all_smirks_compiled:
            # Skip same class (prefix-aware for mixed-depth tiers)
            if (
                smirks_class == rxn_class
                or rxn_class.startswith(smirks_class + ".")
                or smirks_class.startswith(rxn_class + ".")
            ):
                continue
            # Pre-filter: skip if wrong number of reactants
            if nreact > n_reactants:
                continue
            if _test_smirks_precompiled(reactant_mols, expected_prods, rxn_obj):
                fps.append({
                    "smirks_class": smirks_class,
                    "true_class": rxn_class,
                    "rxn": rxn_smi,
                    "smirks": smirks_str,
                })
    return fps


def run_fp_test_global(
    df: pd.DataFrame,
    split: dict[str, dict[str, list[int]]],
    train_results: dict[str, dict[str, Any]],
    n_jobs: int = 4,
    chunk_size: int = 200,
    use_set: str = "train",
) -> dict[str, list[dict[str, str]]]:
    """Phase 3: test each class's SMIRKS against all other-class reactions.

    Args:
        df: Full database DataFrame.
        split: The train/test split dict.
        train_results: Dict of class -> training result (with "smirks" key).
        n_jobs: Number of parallel workers.
        chunk_size: Reactions per worker chunk.
        use_set: "train" or "test" — which set to use as the reaction pool.

    Returns:
        Dict mapping smirks_class -> list of FP records.
    """
    # 1. Collect all SMIRKS with their class labels
    all_smirks: list[tuple[str, str]] = []
    for cls, result in train_results.items():
        if not result.get("smirks"):
            continue
        for s in result["smirks"]:
            all_smirks.append((cls, s))

    if not all_smirks:
        logger.warning("No SMIRKS to test for false positives")
        return {}

    # 2. Pre-compile all SMIRKS
    all_smirks_compiled: list[tuple[str, str, Any, int]] = []
    for cls, smirks_str in all_smirks:
        rxn_obj = _compile_smirks(smirks_str)
        if rxn_obj is not None:
            nreact = rxn_obj.GetNumReactantTemplates()
            all_smirks_compiled.append((cls, smirks_str, rxn_obj, nreact))

    logger.info(
        f"Phase 3: {len(all_smirks_compiled)} compiled SMIRKS, "
        f"testing against {use_set} reactions"
    )

    # 3. Collect all reactions with pre-parsed mols
    reactions: list[tuple[str, str, list, set[str]]] = []
    for cls, indices in split.items():
        idx_list = indices[use_set]
        for idx in idx_list:
            rxn_smi = df.at[idx, "SANITIZED_REACTION"]
            if pd.isna(rxn_smi):
                continue
            parsed = _parse_reaction(rxn_smi)
            if parsed is None:
                continue
            reactant_mols, expected_prods = parsed
            reactions.append((cls, rxn_smi, reactant_mols, expected_prods))

    logger.info(f"  {len(reactions)} reactions to test")

    # 4. Chunk reactions for parallel processing
    chunks = [
        reactions[i : i + chunk_size]
        for i in range(0, len(reactions), chunk_size)
    ]

    # 5. Run in parallel
    # Note: pre-compiled rxn_obj cannot be pickled, so we pass SMIRKS strings
    # and compile inside workers.
    smirks_for_workers = [
        (cls, s, nreact)
        for cls, s, _rxn_obj, nreact in all_smirks_compiled
    ]

    all_fps: list[dict[str, str]] = []
    if n_jobs == 1:
        for chunk in tqdm(chunks, desc="Phase 3: FP testing"):
            fps = _fp_test_chunk_serializable(chunk, smirks_for_workers)
            all_fps.extend(fps)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fp_test_chunk_serializable)(chunk, smirks_for_workers)
            for chunk in tqdm(chunks, desc="Phase 3: FP testing")
        )
        for fps in results:
            all_fps.extend(fps)

    # 6. Group by smirks_class
    fp_by_class: dict[str, list[dict[str, str]]] = {}
    for fp in all_fps:
        cls = fp["smirks_class"]
        if cls not in fp_by_class:
            fp_by_class[cls] = []
        fp_by_class[cls].append(fp)

    total_fps = sum(len(v) for v in fp_by_class.values())
    logger.info(
        f"  {total_fps} false positives across "
        f"{len(fp_by_class)} classes"
    )

    return fp_by_class


def _fp_test_chunk_serializable(
    chunk: list[tuple[str, str, list, set[str]]],
    smirks_list: list[tuple[str, str, int]],
) -> list[dict[str, str]]:
    """Serializable version that compiles SMIRKS inside the worker.

    Args:
        chunk: (rxn_class, rxn_smi, reactant_mols, expected_prods).
        smirks_list: (smirks_class, smirks_str, nreact).
    """
    # Compile SMIRKS once per worker invocation
    compiled = []
    for cls, smirks_str, nreact in smirks_list:
        rxn_obj = _compile_smirks(smirks_str)
        if rxn_obj is not None:
            compiled.append((cls, smirks_str, rxn_obj, nreact))

    return _fp_test_chunk(chunk, compiled)


# ── Phase 4: Fine-tuning ────────────────────────────────────────────────────

_REFINE_SYSTEM_PROMPT = """\
You are an expert cheminformatician refining SMIRKS reaction patterns.

Your task: given existing SMIRKS for a reaction class, plus FALSE POSITIVE \
examples where those SMIRKS incorrectly matched reactions from OTHER classes, \
produce refined SMIRKS that:
1. Still cover the training examples for this class (high recall)
2. Do NOT match the false positive examples (eliminate FPs)

Strategies to reduce false positives:
- ADD CONTEXT ATOMS: require specific neighbors of the reaction center
- TIGHTEN H/D CONSTRAINTS: be more specific about hydrogen counts and degrees
- SPLIT INTO MULTIPLE PATTERNS: if the current SMIRKS is too broad, split it \
  into multiple narrower patterns that together cover the training examples
- ADD RING MEMBERSHIP: use ring-bond notation or [R] / [r] constraints if the \
  FPs involve ring vs. chain differences

IMPORTANT: Do NOT over-constrain. Your refined SMIRKS must still match the \
training examples. If tightening would drop too much recall, prefer to accept \
some false positives.

Keep +0 on all mapped atoms. Every mapped atom in products MUST appear in reactants.
"""

_REFINE_USER_PROMPT_TEMPLATE = """\
Refine the SMIRKS for this reaction class to eliminate false positives.

Reaction class: {reaction_class}

Current SMIRKS:
{current_smirks}

Training examples that MUST still be covered:
{training_examples}

FALSE POSITIVE examples (reactions from OTHER classes that incorrectly matched):
{fp_examples}

Produce refined SMIRKS that cover the training examples but do NOT match the \
false positive examples. If impossible to eliminate all FPs without dropping \
recall below 80%, prioritize recall."""


def _build_refine_prompt(
    reaction_class: str,
    smirks: list[str],
    fps: list[dict[str, str]],
    train_examples: list[str],
    named_dict: dict,
    max_fp_per_class: int = 5,
    max_fp_total: int = 20,
) -> str:
    """Build a refinement prompt with FP examples grouped by true class."""
    current_smirks = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(smirks))

    training_examples = "\n".join(
        f"  {i+1}. {e}" for i, e in enumerate(train_examples[:10])
    )

    # Group FPs by true class
    fp_by_true: dict[str, list[str]] = {}
    for fp in fps:
        tc = fp["true_class"]
        if tc not in fp_by_true:
            fp_by_true[tc] = []
        fp_by_true[tc].append(fp["rxn"])

    fp_lines = []
    total = 0
    for tc, rxns in sorted(fp_by_true.items()):
        tc_name = named_dict.get(tc, tc)
        fp_lines.append(f"  From class {tc} ({tc_name}):")
        for rxn in rxns[:max_fp_per_class]:
            fp_lines.append(f"    - {rxn}")
            total += 1
            if total >= max_fp_total:
                break
        if total >= max_fp_total:
            break

    return _REFINE_USER_PROMPT_TEMPLATE.format(
        reaction_class=reaction_class,
        current_smirks=current_smirks,
        training_examples=training_examples,
        fp_examples="\n".join(fp_lines),
    )


def refine_class_smirks(
    cls: str,
    generalizer: SmirksGeneralizer,
    current_smirks: list[str],
    fps: list[dict[str, str]],
    train_examples: list[str],
    reaction_class_name: str,
    named_dict: dict,
    max_rounds: int = 3,
    tp_drop_limit: float = 0.8,
) -> dict[str, Any]:
    """Phase 4: iterative fine-tuning for one class.

    Args:
        cls: Class code (e.g. "7.1.1").
        generalizer: SmirksGeneralizer instance.
        current_smirks: Current SMIRKS list.
        fps: False positive records for this class.
        train_examples: Training example reactions.
        reaction_class_name: Full class name string.
        named_dict: Class code -> name mapping.
        max_rounds: Maximum refinement rounds.
        tp_drop_limit: Minimum TP ratio vs. original to accept new SMIRKS.

    Returns:
        Dict with refined SMIRKS and metadata.
    """
    # Measure original TP
    original_val = validate_smirks(current_smirks, train_examples, threshold=0.0)
    original_tp = original_val["combined_coverage"]

    best_smirks = list(current_smirks)
    remaining_fps = list(fps)
    rollback = False

    for round_num in range(1, max_rounds + 1):
        if not remaining_fps:
            logger.info(f"  [{cls}] Round {round_num}: no FPs remaining")
            break

        logger.info(
            f"  [{cls}] Round {round_num}: "
            f"{len(remaining_fps)} FPs, refining..."
        )

        prompt = _build_refine_prompt(
            reaction_class=reaction_class_name,
            smirks=best_smirks,
            fps=remaining_fps,
            train_examples=train_examples,
            named_dict=named_dict,
        )

        try:
            result = generalizer._call(
                prompt,
                retries=3,
                system_prompt=_REFINE_SYSTEM_PROMPT,
            )
        except Exception as exc:
            logger.warning(f"  [{cls}] Refinement LLM call failed: {exc}")
            break

        new_smirks = result.smirks

        # TP check: must still cover training examples
        new_val = validate_smirks(new_smirks, train_examples, threshold=0.0)
        new_tp = new_val["combined_coverage"]

        if new_tp < original_tp * tp_drop_limit:
            logger.warning(
                f"  [{cls}] Round {round_num}: TP dropped from "
                f"{original_tp:.0%} to {new_tp:.0%} (limit "
                f"{original_tp * tp_drop_limit:.0%}). Rolling back."
            )
            rollback = True
            best_smirks = list(current_smirks)
            remaining_fps = list(fps)
            break

        # Quick FP re-check against the specific FP examples
        new_remaining = []
        for fp in remaining_fps:
            parsed = _parse_reaction(fp["rxn"])
            if parsed is None:
                continue
            reactant_mols, expected_prods = parsed

            still_fp = False
            for s in new_smirks:
                rxn_obj = _compile_smirks(s)
                if rxn_obj is None:
                    continue
                if _test_smirks_precompiled(reactant_mols, expected_prods, rxn_obj):
                    still_fp = True
                    break
            if still_fp:
                new_remaining.append(fp)

        logger.info(
            f"  [{cls}] Round {round_num}: TP={new_tp:.0%}, "
            f"FPs: {len(remaining_fps)} -> {len(new_remaining)}"
        )

        best_smirks = new_smirks
        remaining_fps = new_remaining

    # On rollback, best_smirks == current_smirks, so tp_after == original_tp
    tp_after = (
        original_tp
        if rollback
        else validate_smirks(best_smirks, train_examples, threshold=0.0)[
            "combined_coverage"
        ]
    )

    return {
        "smirks_refined": best_smirks,
        "n_fp_before": len(fps),
        "n_fp_after": len(remaining_fps),
        "finetune_rounds": round_num if remaining_fps != fps else 0,
        "rollback": rollback,
        "tp_initial": original_tp,
        "tp_after": tp_after,
    }


# ── Phase 5: Held-out Evaluation ────────────────────────────────────────────


def evaluate_held_out(
    df: pd.DataFrame,
    split: dict[str, dict[str, list[int]]],
    final_smirks: dict[str, list[str]],
    n_jobs: int = 4,
) -> dict[str, dict[str, Any]]:
    """Phase 5: evaluate final SMIRKS on the held-out test set.

    For each class:
      - test_recall: fraction of own test reactions covered
      - test_fp: number of other-class test reactions incorrectly matched

    Args:
        df: Full database DataFrame.
        split: Train/test split dict.
        final_smirks: Dict of class -> list of SMIRKS.
        n_jobs: Number of parallel workers.

    Returns:
        Dict mapping class -> evaluation metrics.
    """
    eval_results: dict[str, dict[str, Any]] = {}

    # 1. Test recall: each class's SMIRKS vs. own test reactions
    logger.info("Phase 5: Evaluating recall on test set...")
    for cls, smirks_list in tqdm(
        final_smirks.items(), desc="Phase 5: Recall"
    ):
        if not smirks_list or not _class_in_split(cls, split):
            eval_results[cls] = {
                "test_recall": 0.0,
                "n_test": 0,
                "n_covered": 0,
            }
            continue

        test_idx = get_class_split(split, cls)["test"]
        test_rxns = [
            df.at[i, "SANITIZED_REACTION"]
            for i in test_idx
            if not pd.isna(df.at[i, "SANITIZED_REACTION"])
        ]

        if not test_rxns:
            eval_results[cls] = {
                "test_recall": 0.0,
                "n_test": 0,
                "n_covered": 0,
            }
            continue

        val = validate_smirks(smirks_list, test_rxns, threshold=0.0)
        eval_results[cls] = {
            "test_recall": val["combined_coverage"],
            "n_test": len(test_rxns),
            "n_covered": int(val["combined_coverage"] * len(test_rxns)),
        }

    # 2. Test FP: run global FP test on test set
    logger.info("Phase 5: Testing for false positives on test set...")
    # Build train_results-like dict for run_fp_test_global
    mock_results = {
        cls: {"smirks": smirks_list}
        for cls, smirks_list in final_smirks.items()
    }
    fp_test = run_fp_test_global(
        df, split, mock_results, n_jobs=n_jobs, use_set="test"
    )

    for cls in final_smirks:
        if cls not in eval_results:
            eval_results[cls] = {}
        cls_fps = fp_test.get(cls, [])
        eval_results[cls]["test_fp"] = len(cls_fps)
        eval_results[cls]["test_fp_unique"] = len({r["rxn"] for r in cls_fps})
        eval_results[cls]["test_fp_hierarchy"] = categorize_fps(
            cls_fps, cls
        )

    return eval_results


# ── FP hierarchy categorization ──────────────────────────────────────────────


def _class_ancestors(cls: str) -> list[str]:
    """Return all ancestor prefixes for a class code, shortest first.

    Example: ``"1.4.2.3"`` -> ``["1", "1.4", "1.4.2", "1.4.2.3"]``.
    Strips ``CONFLICT:`` prefix before splitting.
    """
    clean = cls.replace("CONFLICT:", "")
    parts = clean.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def categorize_fps(
    fps: list[dict[str, str]],
    smirks_class: str,
) -> dict[str, Any]:
    """Categorize FPs by hierarchy level relative to *smirks_class*.

    Works with any class depth (tier_3 ``"1.4.2"``, tier_4 ``"1.4.2.3"``,
    etc.).  For each FP, finds the deepest shared ancestor with
    *smirks_class* and assigns it to that level.

    Returns a dict with:
      - ``n_total``: total FP count
      - ``n_cross_tier1``: FPs from a different superclass (genuine FPs)
      - ``n_same_tier<N>``: FPs sharing tier-N ancestor (one key per level)
      - ``by_tier1``: Counter of {true_tier1: count}
      - ``by_true_class``: Counter of {true_class: count}
    """
    s_ancestors = _class_ancestors(smirks_class)

    # Buckets: depth 0 = same tier-1, depth 1 = same tier-2, ...
    # "cross_tier1" = no shared ancestor at all
    depth_counts: Counter = Counter()
    by_tier1: Counter = Counter()
    by_true_class: Counter = Counter()

    for fp in fps:
        tc = fp["true_class"]
        t_ancestors = _class_ancestors(tc)
        by_tier1[t_ancestors[0]] += 1
        by_true_class[tc] += 1

        # Find deepest shared ancestor
        shared_depth = 0
        for sa, ta in zip(s_ancestors, t_ancestors):
            if sa == ta:
                shared_depth += 1
            else:
                break

        if shared_depth == 0:
            depth_counts["cross_tier1"] += 1
        else:
            depth_counts[f"same_tier{shared_depth}"] += 1

    result: dict[str, Any] = {
        "n_total": len(fps),
        "n_cross_tier1": depth_counts.get("cross_tier1", 0),
    }
    # Add all same_tierN keys that exist
    max_depth = len(s_ancestors)
    for d in range(1, max_depth + 1):
        key = f"same_tier{d}"
        result[f"n_{key}"] = depth_counts.get(key, 0)

    result["by_tier1"] = dict(by_tier1.most_common())
    result["by_true_class"] = dict(by_true_class.most_common())

    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLI & MAIN                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PHASES = ["split", "train", "fp_test", "finetune", "eval"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validated SMIRKS generalization with false-positive testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--database",
        required=True,
        help="Parquet file with columns: tier_2, tier_3, "
        "TEMPLATE_rr0rp1_ring0, SANITIZED_REACTION.",
    )
    p.add_argument(
        "--mapping",
        required=True,
        help="Path to structured_mapping.json.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON file path for final results.",
    )
    p.add_argument(
        "--split-column",
        default=None,
        help="Column to split on (default: auto-detect finest tier_N "
        "in DataFrame). The split is always at this granularity; "
        "--class-column controls processing granularity.",
    )
    p.add_argument(
        "--class-column",
        default=None,
        help="DataFrame column controlling processing granularity "
        "(e.g. tier_3, tier_4). Auto-detected from --classes depth "
        "if omitted. Set to 'mixed' when --classes has mixed depths.",
    )
    p.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Class codes to process (may mix depths). Supports "
        "wildcards: '4.1.4.4.*' expands to direct children, "
        "'4.1.4.4.**' expands to all descendants. "
        "Default: all classes from --class-column.",
    )
    p.add_argument(
        "--classes-file",
        default=None,
        help="Path to a JSON file with an 'all_classes' key "
        "(e.g. output of recommend_tier_levels.py). "
        "Mutually exclusive with --classes.",
    )
    p.add_argument(
        "--phase",
        default="all",
        choices=["all"] + PHASES,
        help="Run a specific phase or 'all'.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var).",
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model ID (e.g. gemini-3-flash-preview, "
        "gemini-3.1-pro-preview).",
    )
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per Gemini API call (default: 300).",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable Gemini context caching.",
    )
    p.add_argument(
        "--min-class-size",
        type=int,
        default=5,
        help="Minimum reactions per class to include.",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for test set.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=210995,
        help="Random seed for stratified split.",
    )
    p.add_argument(
        "--min-template-coverage",
        type=float,
        default=0.9,
        help="Min template coverage for screen_tier3_auto.",
    )
    p.add_argument(
        "--max-templates",
        type=int,
        default=50,
        help="Max templates per class.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max LLM retries per class during training.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Min combined coverage to accept SMIRKS.",
    )
    p.add_argument(
        "--fp-n-jobs",
        type=int,
        default=4,
        help="Number of parallel workers for FP testing.",
    )
    p.add_argument(
        "--max-finetune-rounds",
        type=int,
        default=3,
        help="Max Gemini refinement rounds per class.",
    )
    p.add_argument(
        "--tp-drop-limit",
        type=float,
        default=0.8,
        help="Rollback if TP drops below this fraction of original.",
    )
    p.add_argument(
        "--fp-retest",
        action="store_true",
        help="Re-run global FP test after fine-tuning.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _should_run_phase(phase: str, target: str) -> bool:
    """Check if a phase should run given the --phase argument."""
    if target == "all":
        return True
    return phase == target


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading database...")
    df = pd.read_parquet(args.database)
    logger.info(f"  Reactions: {len(df):,}")

    logger.info("Loading structured mapping...")
    with open(args.mapping, encoding="utf-8") as f:
        named_dict = json.load(f)

    # ── Load classes from file if provided ──────────────────────────────
    if args.classes_file and args.classes:
        p_err = "Cannot specify both --classes and --classes-file."
        raise SystemExit(p_err)
    if args.classes_file:
        logger.info(f"Loading classes from {args.classes_file}")
        with open(args.classes_file, encoding="utf-8") as f:
            classes_data = json.load(f)
        if "all_classes" not in classes_data:
            raise SystemExit(
                f"JSON file {args.classes_file} has no 'all_classes' key."
            )
        args.classes = classes_data["all_classes"]
        logger.info(f"  Loaded {len(args.classes)} classes from file")

    # ── Auto-detect split_column (finest tier in DataFrame) ───────────
    if args.split_column is None:
        args.split_column = detect_finest_tier(df)
        logger.info(
            f"Auto-detected --split-column={args.split_column!r}"
        )

    # ── Auto-detect class_column from --classes depth ────────────────────
    if args.class_column is None:
        if args.classes:
            # Strip wildcard suffixes before computing depth;
            # wildcards are expanded later against the split keys.
            plain = [
                c.removesuffix(".**").removesuffix(".*")
                for c in args.classes
                if not c.endswith(".*")
            ]
            wildcard = [
                c for c in args.classes if c.endswith(".*")
            ]
            if plain:
                depths = {tier_depth(c) for c in plain}
            elif wildcard:
                # All entries are wildcards — class_column will be
                # determined after expansion; defer to "mixed" for now
                depths = set()
            else:
                depths = set()

            if len(depths) == 1:
                args.class_column = f"tier_{depths.pop()}"
                logger.info(
                    f"Auto-detected --class-column={args.class_column!r} "
                    f"from class code depth"
                )
            else:
                args.class_column = "mixed"
                logger.info(
                    f"Mixed/wildcard class depths, --class-column=mixed"
                )
        else:
            args.class_column = "tier_3"
            logger.info(
                "No --classes specified, defaulting to --class-column=tier_3"
            )

    # ── Checkpoint ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    ckpt_path = output_path.with_name(output_path.stem + "_checkpoint.json")
    ckpt = load_checkpoint(ckpt_path) or {}
    logger.info(
        f"Checkpoint: {ckpt_path} "
        f"({'loaded' if ckpt else 'new'})"
    )

    # ── Invalidation checks ──────────────────────────────────────────────
    # class_column change: invalidate training + downstream (NOT split)
    ckpt_class_col = ckpt.get("class_column")
    if (
        ckpt_class_col is not None
        and ckpt_class_col != args.class_column
    ):
        logger.info(
            f"class_column changed ({ckpt_class_col!r} -> "
            f"{args.class_column!r}), invalidating training and "
            f"downstream results..."
        )
        for key in ("train_results", "fp_results",
                    "finetune_results", "eval_results"):
            ckpt.pop(key, None)
    ckpt["class_column"] = args.class_column

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Stratified Split
    # ══════════════════════════════════════════════════════════════════════
    if _should_run_phase("split", args.phase):
        # Invalidate cached split if split_column changed
        ckpt_split_col = ckpt.get("split_column")
        split_valid = (
            "split" in ckpt
            and ckpt_split_col == args.split_column
        )
        if split_valid and args.phase == "all":
            logger.info("Phase 1: Split already done (from checkpoint)")
            split = ckpt["split"]
        else:
            if "split" in ckpt and ckpt_split_col != args.split_column:
                logger.info(
                    f"Phase 1: split_column changed "
                    f"({ckpt_split_col!r} -> {args.split_column!r}), "
                    f"re-splitting and clearing all downstream..."
                )
                # Clear everything downstream of the split
                for key in ("train_results", "fp_results",
                            "finetune_results", "eval_results"):
                    ckpt.pop(key, None)
            logger.info(
                f"Phase 1: Stratified split at {args.split_column!r}..."
            )
            split = stratified_split(
                df,
                class_col=args.split_column,
                test_frac=args.test_fraction,
                min_size=args.min_class_size,
                seed=args.seed,
            )
            ckpt["split"] = split
            ckpt["split_column"] = args.split_column
            save_checkpoint(ckpt_path, ckpt)
            logger.info(
                f"  {len(split)} classes with >= {args.min_class_size} "
                f"reactions"
            )

            # Summary stats
            n_train = sum(len(v["train"]) for v in split.values())
            n_test = sum(len(v["test"]) for v in split.values())
            logger.info(f"  Train: {n_train:,}, Test: {n_test:,}")
    else:
        split = ckpt.get("split")
        if split is None:
            logger.error("Phase 1 (split) must run first. Use --phase split")
            return

    # ── Determine target classes ─────────────────────────────────────────
    if args.classes:
        # Expand wildcards first (e.g. "4.1.4.4.*" -> all tier_5 children)
        expanded = _expand_class_wildcards(args.classes, split)
        target_classes = [
            c for c in expanded if _class_in_split(c, split)
        ]
        if len(target_classes) < len(expanded):
            missing = set(expanded) - set(target_classes)
            logger.warning(f"Classes not in split (too small?): {missing}")

        # Re-derive class_column after expansion if it was "mixed"
        if args.class_column == "mixed" and target_classes:
            depths = {tier_depth(c) for c in target_classes}
            if len(depths) == 1:
                args.class_column = f"tier_{depths.pop()}"
                logger.info(
                    f"After expansion: --class-column={args.class_column!r}"
                )
    elif args.class_column != "mixed":
        # Derive from DataFrame column, filtered to what's in the split
        candidates = sorted(df[args.class_column].dropna().unique())
        target_classes = [
            c for c in candidates if _class_in_split(c, split)
        ]
    else:
        # mixed without explicit --classes: process at split granularity
        target_classes = sorted(split.keys())

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Training
    # ══════════════════════════════════════════════════════════════════════
    if _should_run_phase("train", args.phase):
        logger.info("Phase 2: Training SMIRKS generalization...")
        generalizer = SmirksGeneralizer(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            use_cache=args.use_cache,
            timeout=args.timeout,
        )
        train_results = train_all_classes(
            df=df,
            split=split,
            named_dict=named_dict,
            generalizer=generalizer,
            ckpt_path=ckpt_path,
            ckpt=ckpt,
            classes=target_classes,
            min_coverage=args.min_template_coverage,
            max_templates=args.max_templates,
            max_retries=args.max_retries,
            validation_threshold=args.threshold,
        )

        n_valid = sum(1 for r in train_results.values() if r.get("valid"))
        usage = generalizer.token_usage
        logger.info(
            f"Phase 2 done: {n_valid}/{len(train_results)} "
            f"classes produced valid SMIRKS\n"
            f"  Token usage: {usage['total_calls']} calls, "
            f"{usage['prompt_tokens']:,} prompt + "
            f"{usage['completion_tokens']:,} completion = "
            f"{usage['total_tokens']:,} total tokens"
        )
        ckpt["token_usage_phase2"] = usage
        save_checkpoint(ckpt_path, ckpt)
    else:
        train_results = ckpt.get("train_results")
        if train_results is None:
            logger.error(
                "Phase 2 (train) must run first. Use --phase train"
            )
            return

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: False Positive Testing
    # ══════════════════════════════════════════════════════════════════════
    if _should_run_phase("fp_test", args.phase):
        logger.info("Phase 3: False positive testing...")
        fp_results = run_fp_test_global(
            df=df,
            split=split,
            train_results=train_results,
            n_jobs=args.fp_n_jobs,
            use_set="train",
        )
        # Convert to serializable format for checkpoint
        ckpt["fp_results"] = {
            cls: fps for cls, fps in fp_results.items()
        }
        save_checkpoint(ckpt_path, ckpt)
    else:
        fp_results = ckpt.get("fp_results")
        if fp_results is None and _should_run_phase("finetune", args.phase):
            logger.error(
                "Phase 3 (fp_test) must run first. Use --phase fp_test"
            )
            return

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Fine-tuning
    # ══════════════════════════════════════════════════════════════════════
    if _should_run_phase("finetune", args.phase):
        if fp_results is None:
            fp_results = {}

        classes_with_fps = [
            cls for cls in target_classes if cls in fp_results and fp_results[cls]
        ]
        logger.info(
            f"Phase 4: Fine-tuning {len(classes_with_fps)} classes with FPs..."
        )

        if classes_with_fps:
            generalizer = SmirksGeneralizer(
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                use_cache=args.use_cache,
                timeout=args.timeout,
            )

        finetune_results = ckpt.get("finetune_results", {})
        for cls in tqdm(classes_with_fps, desc="Phase 4: Fine-tuning"):
            if cls in finetune_results:
                continue

            result = train_results.get(cls, {})
            current_smirks = result.get("smirks", [])
            if not current_smirks:
                continue

            # Get training examples for TP validation
            train_idx = get_class_split(split, cls)["train"]
            train_rxns = [
                df.at[i, "SANITIZED_REACTION"]
                for i in train_idx
                if not pd.isna(df.at[i, "SANITIZED_REACTION"])
            ]
            # Use a sample for speed (max 50)
            if len(train_rxns) > 50:
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(len(train_rxns), 50, replace=False)
                train_sample = [train_rxns[i] for i in sample_idx]
            else:
                train_sample = train_rxns

            reaction_class_name = result.get(
                "reaction_class", cls
            )

            refined = refine_class_smirks(
                cls=cls,
                generalizer=generalizer,
                current_smirks=current_smirks,
                fps=fp_results[cls],
                train_examples=train_sample,
                reaction_class_name=reaction_class_name,
                named_dict=named_dict,
                max_rounds=args.max_finetune_rounds,
                tp_drop_limit=args.tp_drop_limit,
            )

            finetune_results[cls] = refined
            ckpt["finetune_results"] = finetune_results
            save_checkpoint(ckpt_path, ckpt)

        if classes_with_fps:
            usage = generalizer.token_usage
            logger.info(
                f"Phase 4 token usage: {usage['total_calls']} calls, "
                f"{usage['prompt_tokens']:,} prompt + "
                f"{usage['completion_tokens']:,} completion = "
                f"{usage['total_tokens']:,} total tokens"
            )
            ckpt["token_usage_phase4"] = usage
            save_checkpoint(ckpt_path, ckpt)

        # Optional: re-run FP test after fine-tuning
        if args.fp_retest and classes_with_fps:
            logger.info("Phase 4: Re-running global FP test after fine-tuning...")
            # Build updated train_results with refined SMIRKS
            updated_results = copy.deepcopy(train_results)
            for cls, refined in finetune_results.items():
                if cls in updated_results:
                    updated_results[cls]["smirks"] = refined["smirks_refined"]

            fp_results_v2 = run_fp_test_global(
                df=df,
                split=split,
                train_results=updated_results,
                n_jobs=args.fp_n_jobs,
                use_set="train",
            )
            ckpt["fp_results_v2"] = fp_results_v2
            save_checkpoint(ckpt_path, ckpt)

            total_before = sum(
                len(fps) for fps in fp_results.values()
            )
            total_after = sum(
                len(fps) for fps in fp_results_v2.values()
            )
            logger.info(
                f"  FP retest: {total_before} -> {total_after} "
                f"total false positives"
            )
    else:
        finetune_results = ckpt.get("finetune_results", {})

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Held-out Evaluation
    # ══════════════════════════════════════════════════════════════════════
    if _should_run_phase("eval", args.phase):
        logger.info("Phase 5: Held-out evaluation...")

        # Build final SMIRKS dict (use refined where available)
        final_smirks: dict[str, list[str]] = {}
        for cls in target_classes:
            if cls in finetune_results:
                final_smirks[cls] = finetune_results[cls]["smirks_refined"]
            elif cls in train_results and train_results[cls].get("smirks"):
                final_smirks[cls] = train_results[cls]["smirks"]

        eval_results = evaluate_held_out(
            df=df,
            split=split,
            final_smirks=final_smirks,
            n_jobs=args.fp_n_jobs,
        )
        ckpt["eval_results"] = eval_results
        save_checkpoint(ckpt_path, ckpt)
    else:
        eval_results = ckpt.get("eval_results", {})

    # ══════════════════════════════════════════════════════════════════════
    # Build final output
    # ══════════════════════════════════════════════════════════════════════
    if args.phase in ("all", "eval"):
        logger.info("Building final output...")

        # Load existing output file if it exists (merge mode)
        existing_classes: dict[str, Any] = {}
        if os.path.exists(args.output):
            try:
                with open(args.output, "r", encoding="utf-8") as f:
                    existing_output = json.load(f)
                existing_classes = existing_output.get("classes", {})
                logger.info(
                    f"Merging into existing output ({len(existing_classes)} "
                    f"classes already present)"
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"Could not read existing output file, overwriting: {exc}"
                )

        output: dict[str, Any] = {
            "metadata": {
                "database": str(args.database),
                "n_reactions": len(df),
                "n_classes": len(target_classes),
                "split_column": args.split_column,
                "class_column": args.class_column,
                "test_fraction": args.test_fraction,
                "min_class_size": args.min_class_size,
                "seed": args.seed,
                "model": args.model,
            },
            "classes": dict(existing_classes),  # start from existing
        }

        for cls in target_classes:
            tr = train_results.get(cls, {})
            ft = finetune_results.get(cls, {})
            ev = eval_results.get(cls, {})

            initial_smirks = tr.get("smirks", [])
            final_smirks_cls = (
                ft.get("smirks_refined", initial_smirks) if ft else initial_smirks
            )

            # --- Train FP hierarchy (initial = before fine-tuning) ---
            cls_fps_initial = fp_results.get(cls, []) if fp_results else []
            n_fp_initial = len(cls_fps_initial)
            fp_hier_initial = categorize_fps(cls_fps_initial, cls)

            n_fp_final = ft.get("n_fp_after", n_fp_initial)

            # --- Test FP hierarchy ---
            test_fp_hier = ev.get("test_fp_hierarchy", {})

            test_recall = ev.get("test_recall", 0.0)

            # New class data overwrites existing entry for same key
            output["classes"][cls] = {
                "reaction_class": tr.get("reaction_class", cls),
                "smirks_initial": initial_smirks,
                "smirks_final": final_smirks_cls,
                "n_train": len(get_class_split(split, cls)["train"]) if _class_in_split(cls, split) else 0,
                "n_test": len(get_class_split(split, cls)["test"]) if _class_in_split(cls, split) else 0,
                "train_tp_initial": tr.get("combined_coverage", 0.0),
                "train_tp_final": ft.get(
                    "tp_after", tr.get("combined_coverage", 0.0)
                ),
                "n_fp_initial": n_fp_initial,
                "n_fp_final": n_fp_final,
                "fp_hierarchy_train": fp_hier_initial,
                "finetune_rounds": ft.get("finetune_rounds", 0),
                "rollback": ft.get("rollback", False),
                "test_recall": test_recall,
                "test_fp": ev.get("test_fp", 0),
                "test_fp_unique": ev.get("test_fp_unique", ev.get("test_fp", 0)),
                "test_fp_hierarchy": test_fp_hier,
                "reasoning": tr.get("reasoning", ""),
            }

        # Recompute summary from ALL classes in the merged output
        all_classes = output["classes"]
        recalls = [c.get("test_recall", 0.0) for c in all_classes.values()]
        total_fp_before = sum(c.get("n_fp_initial", 0) for c in all_classes.values())
        total_fp_after = sum(c.get("n_fp_final", 0) for c in all_classes.values())

        train_hier_totals: Counter = Counter()
        test_hier_totals: Counter = Counter()
        for c in all_classes.values():
            for k, v in c.get("fp_hierarchy_train", {}).items():
                if k.startswith("n_") and isinstance(v, int):
                    train_hier_totals[k] += v
            for k, v in c.get("test_fp_hierarchy", {}).items():
                if k.startswith("n_") and isinstance(v, int):
                    test_hier_totals[k] += v

        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        n_finetuned = sum(
            1 for c in all_classes.values() if c.get("finetune_rounds", 0) > 0
        )
        n_rollbacks = sum(
            1 for c in all_classes.values() if c.get("rollback", False)
        )

        output["metadata"]["n_classes"] = len(all_classes)
        output["summary"] = {
            "n_classes_total": len(all_classes),
            "n_classes_with_smirks": sum(
                1
                for c in all_classes.values()
                if c.get("smirks_final")
            ),
            "mean_test_recall": mean_recall,
            "total_fp_before": total_fp_before,
            "total_fp_after": total_fp_after,
            "train_fp_hierarchy": dict(train_hier_totals),
            "test_fp_hierarchy": dict(test_hier_totals),
            "n_classes_finetuned": n_finetuned,
            "n_rollbacks": n_rollbacks,
            "token_usage": {
                "phase2": ckpt.get("token_usage_phase2", {}),
                "phase4": ckpt.get("token_usage_phase4", {}),
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        n_new = len(target_classes)
        n_existing = len(existing_classes)
        n_total = len(all_classes)
        if n_existing > 0:
            logger.info(
                f"Results saved to {args.output} "
                f"({n_new} new/updated + {n_total - n_new} existing = "
                f"{n_total} total classes)"
            )
        else:
            logger.info(f"Results saved to {args.output}")

        # Summary log — dynamically list hierarchy levels
        th = output["summary"]["train_fp_hierarchy"]
        teh = output["summary"]["test_fp_hierarchy"]
        hier_keys = sorted(
            {k for k in list(th) + list(teh) if k != "n_total"},
            key=lambda k: (0 if "cross" in k else 1, k),
        )

        def _fmt_hier(d: dict, keys: list[str]) -> str:
            lines = []
            for k in keys:
                label = k.replace("n_", "").replace("_", " ")
                lines.append(f"    {label}: {d.get(k, 0)}")
            return "\n".join(lines)

        logger.info(
            f"Summary:\n"
            f"  Classes: {output['summary']['n_classes_total']}\n"
            f"  With SMIRKS: {output['summary']['n_classes_with_smirks']}\n"
            f"  Mean test recall: {mean_recall:.1%}\n"
            f"  Train FPs (total): {total_fp_before} -> {total_fp_after}\n"
            f"  Train FPs by hierarchy:\n"
            f"{_fmt_hier(th, hier_keys)}\n"
            f"  Test FPs by hierarchy:\n"
            f"{_fmt_hier(teh, hier_keys)}\n"
            f"  Fine-tuned: {output['summary']['n_classes_finetuned']}\n"
            f"  Rollbacks: {output['summary']['n_rollbacks']}"
        )

        # ------------------------------------------------------------------
        # Export 1: Rxn-INSIGHT-compatible SMIRKS database (JSONL)
        # ------------------------------------------------------------------
        out_stem = Path(args.output).stem
        out_dir = Path(args.output).parent

        smirks_db_path = out_dir / f"{out_stem}_smirks_db.json"
        with open(smirks_db_path, "w", encoding="utf-8") as f:
            for cls, cdata in sorted(all_classes.items()):
                name = cdata.get("reaction_class", cls)
                for s in cdata.get("smirks_final", []):
                    line = json.dumps(
                        {"name": name, "smirks": s}, ensure_ascii=False
                    )
                    f.write(line + "\n")
        logger.info(f"Rxn-INSIGHT SMIRKS database saved to {smirks_db_path}")

        # ------------------------------------------------------------------
        # Export 2: Per-class metrics CSV
        # ------------------------------------------------------------------
        metrics_path = out_dir / f"{out_stem}_metrics.csv"

        # Total test reactions across all classes (for FP rate denominator)
        total_test_rxns = sum(
            c.get("n_test", 0) for c in all_classes.values()
        )

        rows = []
        for cls in sorted(all_classes):
            c = all_classes[cls]
            n_test_cls = c.get("n_test", 0)
            test_recall = c.get("test_recall", 0.0)
            test_fp = c.get("test_fp", 0)
            n_smirks = len(c.get("smirks_final", []))

            # TP = correctly matched own-class test reactions
            tp = round(test_recall * n_test_cls) if n_test_cls > 0 else 0
            # FN = own-class test reactions NOT matched
            fn = n_test_cls - tp
            # FP = unique other-class test reactions incorrectly matched
            # test_fp counts (SMIRKS, reaction) pairs; test_fp_unique counts
            # unique reactions (what matters for TN computation)
            fp = c.get("test_fp_unique", test_fp)
            # TN = other-class test reactions NOT matched
            n_other_test = total_test_rxns - n_test_cls
            tn = n_other_test - fp

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # F1 = 2 * precision * recall / (precision + recall)
            f1 = (
                2 * precision * test_recall / (precision + test_recall)
                if (precision + test_recall) > 0
                else 0.0
            )
            # Specificity = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # FP rate = FP / (FP + TN) = 1 - specificity
            fp_rate = 1.0 - specificity
            # Balanced accuracy = (recall + specificity) / 2
            balanced_acc = (test_recall + specificity) / 2

            rows.append(
                {
                    "class": cls,
                    "reaction_class": c.get("reaction_class", cls),
                    "n_smirks": n_smirks,
                    "n_train": c.get("n_train", 0),
                    "n_test": n_test_cls,
                    "train_recall_initial": c.get("train_tp_initial", 0.0),
                    "train_recall_final": c.get("train_tp_final", 0.0),
                    "test_recall": test_recall,
                    "test_fp_pairs": test_fp,
                    "test_fp_unique": fp,
                    "test_tp": tp,
                    "test_fn": fn,
                    "test_tn": tn,
                    "precision": precision,
                    "f1_score": f1,
                    "specificity": specificity,
                    "fp_rate": fp_rate,
                    "balanced_accuracy": balanced_acc,
                    "n_fp_train_initial": c.get("n_fp_initial", 0),
                    "n_fp_train_final": c.get("n_fp_final", 0),
                    "finetune_rounds": c.get("finetune_rounds", 0),
                    "rollback": c.get("rollback", False),
                    "smirks_final": " | ".join(
                        c.get("smirks_final", [])
                    ),
                }
            )

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Per-class metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
