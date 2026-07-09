"""Stand-alone Gemini-powered SMIRKS generalization.

No dependency on gen-rxn-insight.  Only requires:
    pip install rdkit google-genai pydantic pandas tqdm

Usage (CLI)
-----------
python generalize_smirks_standalone.py \
    --database classification_database.parquet \
    --mapping structured_mapping.json \
    --output generalized_smirks.json \
    --api-key $GEMINI_API_KEY

Or import in a notebook::

    gen = SmirksGeneralizer(api_key="...")
    data = screen_tier3("7.1.1", df, named_dict)
    result = gen.generalize(
        reaction_class=data["reaction_class"],
        templates=data["top_n_templates"],
        examples=data["top_n_examples"],
    )
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── test_smirks (inlined from gen_rxn_insight.naming) ────────────────────────

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
    out: dict[str, Any] = {"applicable": False, "correct": False, "products": set()}
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


# ── screen_tier3 (inlined from screen_tier_3.py) ─────────────────────────────

def get_example_reaction(df: pd.DataFrame, template: str) -> str:
    """Get the shortest reaction example for a given template."""
    dfc = df[df["TEMPLATE_rr0rp1_ring0"] == template].copy()
    series = dfc["SANITIZED_REACTION"].dropna()
    return series.loc[series.str.len().idxmin()]


def screen_tier3(
    llm_class: str,
    df: pd.DataFrame,
    named_dict: dict,
    top_n: int = 10,
) -> dict[str, Any]:
    """Screen a tier-3 class: get top-N templates, examples, and coverage.

    Args:
        llm_class:  Tier-3 class code (e.g. ``"7.1.1"``).
        df:         Database DataFrame with ``tier_2``, ``tier_3``,
                    ``TEMPLATE_rr0rp1_ring0``, ``SANITIZED_REACTION``.
        named_dict: Mapping of class codes to human-readable names
                    (from ``structured_mapping.json``).
        top_n:      Number of most-common templates to include.

    Returns:
        Dict with ``reaction_class``, ``top_n_templates``, ``top_n_examples``,
        and ``coverage``.
    """
    dfc = df[df["tier_3"] == llm_class].copy()
    idx = dfc.index[0]
    tier_2 = dfc["tier_2"][idx]
    top_n_templates = Counter(dfc["TEMPLATE_rr0rp1_ring0"]).most_common(top_n)
    occs = sum(t[1] for t in top_n_templates)
    coverage = occs / len(dfc.index)
    top_n_examples = [get_example_reaction(dfc, t[0]) for t in top_n_templates]
    name = f"{llm_class} - {named_dict[tier_2]}: {named_dict[llm_class]}"
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
) -> dict[str, Any]:
    """Screen a tier-3 class, auto-increasing top_n until coverage >= min_coverage.

    Starts at ``start_n`` templates. If coverage is below ``min_coverage``,
    doubles ``top_n`` (capped at ``max_n``) and retries.

    Args:
        llm_class:    Tier-3 class code.
        df:           Database DataFrame.
        named_dict:   Class code -> name mapping.
        min_coverage: Target template coverage (default 0.9 = 90%).
        start_n:      Initial top-N value.
        max_n:        Hard ceiling on top-N.

    Returns:
        Same dict as ``screen_tier3``, with ``top_n`` added.
    """
    top_n = start_n
    while top_n <= max_n:
        data = screen_tier3(llm_class, df, named_dict, top_n=top_n)
        if data["coverage"] >= min_coverage or top_n >= max_n:
            data["top_n"] = top_n
            return data
        # Double, but don't exceed max_n
        top_n = min(top_n * 2, max_n)

    data["top_n"] = top_n
    return data


# ── Pydantic response model ──────────────────────────────────────────────────

class SmirksGeneralization(BaseModel):
    """Structured output from Gemini for SMIRKS generalization."""

    reasoning: str = Field(
        description="Step-by-step reasoning following steps A-F."
    )
    smirks: list[str] = Field(
        description="One or more generalized SMIRKS strings."
    )


# ── System prompt ─────────────────────────────────────────────────────────────

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


# ── User prompt template ──────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    combined_coverage = len(combined_correct) / len(examples) if examples else 0.0

    return {
        "valid": combined_coverage >= threshold,
        "combined_coverage": combined_coverage,
        "per_smirks": per_smirks,
        "failures": failures,
    }


# ── Main class ────────────────────────────────────────────────────────────────

class SmirksGeneralizer:
    """Generalizes reaction templates into broad SMIRKS via Gemini.

    Args:
        api_key:       Gemini API key (falls back to GEMINI_API_KEY env var).
        model:         Gemini model ID.
        temperature:   Sampling temperature.
        use_cache:     Enable Gemini context caching for the system prompt.

    Example::

        gen = SmirksGeneralizer(api_key="...")
        result = gen.generalize(
            reaction_class="7.1.1 - ...",
            templates=["[C;H2;D2;+0:1]-...>>..."],
            examples=["CCO>>CC=O"],
        )
        print(result["smirks"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.3,
        use_cache: bool = False,
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

    def _call(
        self,
        user_prompt: str,
        retries: int = 3,
    ) -> SmirksGeneralization:
        """Make one Gemini call with retries. Returns parsed Pydantic object."""
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": SmirksGeneralization,
            "temperature": self.temperature,
        }
        if self._cache_name:
            config_kwargs["cached_content"] = self._cache_name
        else:
            config_kwargs["system_instruction"] = _SYSTEM_PROMPT

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    config=self._gtypes.GenerateContentConfig(**config_kwargs),
                    contents=[user_prompt],
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

        Args:
            reaction_class:       Full class name (e.g. "7.1.1 - ...").
            templates:            List of specific SMIRKS templates.
            examples:             List of unmapped reaction SMILES.
            max_retries:          Max LLM retries with error feedback on failure.
            validation_threshold: Min combined coverage to accept.

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

    def generalize_batch(
        self,
        class_data: list[dict[str, Any]],
        max_retries: int = 3,
        validation_threshold: float = 0.5,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Generalize SMIRKS for multiple reaction classes.

        Args:
            class_data: List of dicts from ``screen_tier3()``, each with
                        ``reaction_class``, ``top_n_templates``,
                        ``top_n_examples``.
            max_retries:          Max LLM retries per class.
            validation_threshold: Min combined coverage to accept.
            progress:             Show tqdm progress bar.

        Returns:
            List of result dicts (one per class).
        """
        results = []
        for item in tqdm(class_data, desc="Generalizing SMIRKS",
                         disable=not progress):
            result = self.generalize(
                reaction_class=item["reaction_class"],
                templates=item["top_n_templates"],
                examples=item["top_n_examples"],
                max_retries=max_retries,
                validation_threshold=validation_threshold,
            )
            results.append(result)
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stand-alone SMIRKS generalization via Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--database", required=True,
        help="Parquet file with columns: tier_2, tier_3, "
             "TEMPLATE_rr0rp1_ring0, SANITIZED_REACTION.",
    )
    p.add_argument(
        "--mapping", required=True,
        help="Path to structured_mapping.json.",
    )
    p.add_argument(
        "--output", required=True,
        help="Output JSON file path for generalized SMIRKS.",
    )
    p.add_argument(
        "--classes", nargs="*", default=None,
        help="Specific tier-3 class codes to process (e.g. 7.1.1 1.1.1). "
             "Default: all unique tier-3 classes.",
    )
    p.add_argument("--api-key", default=None,
                   help="Gemini API key (or set GEMINI_API_KEY env var).")
    p.add_argument("--model", default="gemini-3-flash-preview")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--use-cache", action="store_true",
                   help="Enable Gemini context caching for the system prompt.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max LLM retries with error feedback per class.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Min combined coverage to accept SMIRKS.")
    p.add_argument("--min-template-coverage", type=float, default=0.9,
                   help="Min template coverage for screen_tier3_auto (0.9 = 90%%).")
    p.add_argument("--max-templates", type=int, default=50,
                   help="Max templates per class for auto top-N search.")
    p.add_argument("--gold-standard", default=None,
                   help="Path to gold_standard.json. If provided, update it "
                        "with validated SMIRKS.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load data
    logger.info("Loading database...")
    df = pd.read_parquet(args.database)
    logger.info(f"  Reactions: {len(df):,}")

    logger.info("Loading structured mapping...")
    with open(args.mapping, encoding="utf-8") as f:
        named_dict = json.load(f)

    if args.classes:
        classes = args.classes
    else:
        classes = sorted(df["tier_3"].dropna().unique().tolist())
    logger.info(f"  Classes to process: {len(classes)}")

    # Screen each class with auto top-N
    logger.info(
        f"Screening classes (target coverage >= "
        f"{args.min_template_coverage:.0%}, max templates = "
        f"{args.max_templates})..."
    )
    class_data = []
    skipped = 0
    for cls in tqdm(classes, desc="Screening"):
        try:
            data = screen_tier3_auto(
                cls, df, named_dict,
                min_coverage=args.min_template_coverage,
                max_n=args.max_templates,
            )
            class_data.append(data)
            logger.debug(
                f"  {cls}: top_n={data['top_n']}, "
                f"coverage={data['coverage']:.0%}"
            )
        except Exception as exc:
            logger.warning(f"Skipping {cls}: {exc}")
            skipped += 1
    logger.info(f"  Screened: {len(class_data)}, skipped: {skipped}")

    # Initialize generalizer
    gen = SmirksGeneralizer(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        use_cache=args.use_cache,
    )

    # Run generalization
    results = gen.generalize_batch(
        class_data=class_data,
        max_retries=args.max_retries,
        validation_threshold=args.threshold,
    )

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output}")

    # Optionally update gold standard
    if args.gold_standard:
        gold_path = Path(args.gold_standard)
        if gold_path.exists():
            with open(gold_path, encoding="utf-8") as f:
                gold = json.load(f)
        else:
            gold = {}

        updated = 0
        for r in results:
            if r["valid"] and r["smirks"]:
                code = r["reaction_class"].split(" - ")[0].strip()
                gold[code] = r["smirks"]
                updated += 1

        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False)
        logger.info(f"Gold standard updated: {updated} classes -> {gold_path}")

    # Summary
    n_valid = sum(1 for r in results if r["valid"])
    n_total = len(results)
    coverages = [r["combined_coverage"] for r in results]
    avg_cov = sum(coverages) / len(coverages) if coverages else 0
    logger.info(f"Summary: {n_valid}/{n_total} valid, avg coverage={avg_cov:.1%}")


if __name__ == "__main__":
    main()
