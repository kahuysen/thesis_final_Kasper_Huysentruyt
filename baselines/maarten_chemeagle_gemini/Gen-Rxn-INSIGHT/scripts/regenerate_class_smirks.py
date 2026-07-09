"""Regenerate SMIRKS for specific classes with better template sampling.

The default screen_tier3_auto can miss important template subgroups
(e.g., halide vs sulfonate leaving groups) because it picks the top-N
most frequent templates, which may be biased toward one variant.

This script:
1. Groups templates by their leaving group / RC variant
2. Samples examples from EACH group to ensure diversity
3. Calls Gemini to generalize, producing SMIRKS that cover all variants
4. Validates and patches the checkpoint / ordered SMIRKS DB

Usage
-----
    python regenerate_class_smirks.py \
        --database reaction_db.parquet \
        --mapping structured_mapping.json \
        --checkpoint gemini_smirks_checkpoint.json \
        --classes 1.7.1 1.3.5 \
        --api-key $GEMINI_API_KEY \
        [--output patched_smirks_db.json] \
        [--model gemini-3-flash-preview] \
        [--top-n 20] \
        [--max-retries 5]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)


# ── Template column helpers ──────────────────────────────────────────────────


def _template_col(rr: int, rp: int) -> str:
    return f"TEMPLATE_rr{rr}rp{rp}_ring0"


def extract_templates_for_class(
    df_train: pd.DataFrame,
    col: str,
    rr: int,
    rp: int,
    relax_context: bool,
    n_jobs: int,
) -> pd.DataFrame:
    """Extract templates at a custom radius and add the column to *df_train*."""
    if col in df_train.columns:
        return df_train

    from gen_rxn_insight.database import extract_templates_batch

    mapped = df_train["MAPPED_REACTION"].dropna()
    print(f"  Extracting templates at radius ({rr},{rp}) for {len(mapped):,} "
          f"reactions (relax_context={relax_context})...")

    df_tmpl = extract_templates_batch(
        mapped_reactions=mapped,
        radii=[(rr, rp)],
        include_ring_info=[False],
        n_jobs=n_jobs,
        relax_context=relax_context,
    )

    # Merge on MAPPED_REACTION to align correctly
    df_train = df_train.copy()
    tmpl_map = dict(zip(df_tmpl["MAPPED_REACTION"], df_tmpl[col]))
    df_train[col] = df_train["MAPPED_REACTION"].map(tmpl_map)
    return df_train


# ── SMIRKS testing (from generalize_smirks_validated.py) ─────────────────────


def test_smirks(rxn: str, smirks: str) -> dict:
    out = {"applicable": False, "correct": False, "products": set()}
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


def validate_smirks(smirks_list, examples, threshold=0.5):
    combined_correct = set()
    for smirks in smirks_list:
        for i, rxn in enumerate(examples):
            try:
                r = test_smirks(rxn, smirks)
            except Exception:
                r = {"correct": False}
            if r["correct"]:
                combined_correct.add(i)
    coverage = len(combined_correct) / len(examples) if examples else 0.0
    return {"valid": coverage >= threshold, "combined_coverage": coverage}


# ── Diverse template screening ───────────────────────────────────────────────


def get_example_reaction(
    df: pd.DataFrame, template: str, template_col: str = "TEMPLATE_rr0rp1_ring0",
) -> str:
    dfc = df[df[template_col] == template].copy()
    series = dfc["SANITIZED_REACTION"].dropna()
    return series.loc[series.str.len().idxmin()]


def screen_diverse(
    cls: str,
    df: pd.DataFrame,
    named_dict: dict,
    top_n: int = 20,
    class_col: str = "tier_3",
    template_col: str = "TEMPLATE_rr0rp1_ring0",
) -> dict:
    """Screen a class with diverse template sampling.

    Instead of just taking the top-N most frequent templates, this groups
    templates by their leaving group pattern and samples from each group
    to ensure all variants are represented.
    """
    dfc = df[df[class_col] == cls].copy()

    # Build class name
    name = f"{cls} - {named_dict.get(cls, cls)}"

    # All templates with counts
    template_counts = Counter(dfc[template_col].dropna())
    total = sum(template_counts.values())

    if not template_counts:
        return {"reaction_class": name, "top_n_templates": [], "top_n_examples": [], "coverage": 0.0}

    # Group templates by leaving group type
    groups: dict[str, list[tuple[str, int]]] = {}
    for tmpl, cnt in template_counts.items():
        # Detect leaving group pattern
        if "F,Cl,Br,I" in tmpl or "Cl,Br,I" in tmpl:
            lg = "merged_halide"
        elif "[Cl;H0" in tmpl or "-[Cl]" in tmpl:
            lg = "Cl"
        elif "[Br;H0" in tmpl or "-[Br]" in tmpl:
            lg = "Br"
        elif "[I;H0" in tmpl or "-[I]" in tmpl:
            lg = "I"
        elif "[F;H0" in tmpl or "-[F]" in tmpl:
            lg = "F"
        elif "[S;H0" in tmpl and "=[O" in tmpl:
            lg = "sulfonate"
        elif "[O;H0;+0]-[S;H0" in tmpl:
            lg = "sulfonate"
        elif "[B;" in tmpl or "[B(" in tmpl:
            lg = "boronate"
        else:
            lg = "other"
        groups.setdefault(lg, []).append((tmpl, cnt))

    # Sort each group by count
    for lg in groups:
        groups[lg].sort(key=lambda x: -x[1])

    # Allocate slots proportionally to group size, min 1 per group
    group_sizes = {lg: sum(cnt for _, cnt in tmpls) for lg, tmpls in groups.items()}
    total_size = sum(group_sizes.values())

    slots: dict[str, int] = {}
    remaining_slots = top_n
    for lg in sorted(groups.keys(), key=lambda x: -group_sizes[x]):
        if remaining_slots <= 0:
            break
        n = max(1, round(top_n * group_sizes[lg] / total_size))
        n = min(n, remaining_slots, len(groups[lg]))
        slots[lg] = n
        remaining_slots -= n

    # Distribute any remaining slots to the largest groups
    for lg in sorted(groups.keys(), key=lambda x: -group_sizes[x]):
        if remaining_slots <= 0:
            break
        extra = min(remaining_slots, len(groups[lg]) - slots.get(lg, 0))
        if extra > 0:
            slots[lg] = slots.get(lg, 0) + extra
            remaining_slots -= extra

    # Select templates
    selected: list[tuple[str, int]] = []
    for lg, n in slots.items():
        selected.extend(groups[lg][:n])

    # Sort by count descending for the prompt
    selected.sort(key=lambda x: -x[1])

    templates = [t for t, _ in selected]
    examples = [get_example_reaction(dfc, t, template_col=template_col) for t in templates]
    coverage = sum(cnt for _, cnt in selected) / total

    print(f"  Template groups for {cls}:")
    for lg in sorted(groups.keys(), key=lambda x: -group_sizes[x]):
        n_tmpls = len(groups[lg])
        n_rxns = group_sizes[lg]
        n_selected = slots.get(lg, 0)
        print(f"    {lg:>15s}: {n_rxns:>6,} rxns, {n_tmpls:>4} templates, {n_selected:>2} selected")

    return {
        "reaction_class": name,
        "top_n_templates": templates,
        "top_n_examples": examples,
        "coverage": coverage,
    }


# ── Gemini call (reused from generalize_smirks_validated.py) ─────────────────

# Import the system prompt and classes from the validated pipeline
sys.path.insert(0, str(Path(__file__).parent))
from generalize_smirks_validated import (
    SmirksGeneralizer,
    _build_user_prompt,
)


# ── Discrimination refinement ────────────────────────────────────────────────

_DISCRIMINATE_SYSTEM_PROMPT = """\
You are an expert cheminformatician. Your task: refine SMIRKS patterns so they \
ONLY match reactions of one specific class and NOT reactions from competing \
classes that share the same core transformation.

== BACKGROUND ==
These competing classes share the same bond-forming/breaking event (e.g., all \
form a C-O bond via SN2). They differ in the STRUCTURAL CONTEXT around the \
reaction center: which functional group the reacting atom belongs to, what \
ring system it sits in, or what substituent pattern is nearby. Your job is to \
add SMARTS constraints that capture these distinguishing features.

== SMARTS NOTATION REFERENCE ==
- [C] = aliphatic carbon, [c] = aromatic carbon, [#6] = any carbon
- [N] = aliphatic nitrogen, [n] = aromatic nitrogen, [#7] = any nitrogen
- H<n> = total hydrogen count, D<n> = degree (explicit connections)
- +0 = neutral charge, :<n> = atom map number
- [F,Cl,Br,I] = list notation: matches any listed element
- Ring membership: atoms in rings can use ring-bond notation or [R] constraint
- Recursive SMARTS: $([pattern]) matches an atom whose environment matches pattern
- Reaction SMARTS: [atom:n] on reactant side, >> separator, [atom:n] on product side

== RULES ==
1. REACTION CENTER atoms (those that change bonds/H/charge) keep FULL \
   specificity: H, D, charge, map numbers. Do NOT relax RC atoms.
2. CONTEXT atoms adjacent to RC are where you ADD discriminating constraints. \
   These should capture the structural feature that distinguishes this class \
   from competitors.
3. You may ADD new mapped context atoms that were not in the original SMIRKS \
   if they are needed for discrimination (e.g., the carbonyl C=O next to an \
   ester oxygen, or the aromatic ring attached to a phenol).
4. Keep +0 on all mapped atoms. Every mapped atom in products MUST appear \
   in reactants.
5. Produce MULTIPLE SMIRKS if the class has multiple structural variants \
   (e.g., methyl ether + benzyl ether + silyl ether for alcohol deprotection).
6. Leaving groups (unmapped atoms) can stay general: [F,Cl,Br,I;H0;+0] etc.
7. The refined SMIRKS must still match the training examples for this class. \
   If adding context would drop coverage too much, find a balance.

== STRATEGY ==
A. IDENTIFY DISCRIMINATING FEATURES: Compare the training examples of THIS \
   class vs the competing classes. What structural motif is present in THIS \
   class but absent in competitors (or vice versa)?
B. TRANSLATE TO SMARTS: Express that motif as SMARTS constraints on context \
   atoms. For example:
   - Phenol deprotection: require [c]-[O] (aromatic carbon attached to oxygen)
   - Ester hydrolysis: require [C](=O)-[O] (carbonyl adjacent to oxygen)
   - Pyrrole N-alkylation: require [n] (aromatic nitrogen in 5-membered ring)
C. VERIFY: Check that the added constraints still match the training examples \
   but exclude the competing examples.
"""

_DISCRIMINATE_USER_PROMPT_TEMPLATE = """\
Refine the SMIRKS for this class so they discriminate from competing classes.

== TARGET CLASS ==
{target_class}

Current SMIRKS (too general — also match competing classes):
{current_smirks}

Training examples for THIS class (must still be matched):
{target_examples}

== COMPETING CLASSES ==
{competing_sections}

== TASK ==
Add structural context to the SMIRKS so they ONLY match the target class. \
Focus on the distinguishing feature: {discrimination_hint}

If the class has multiple structural variants, produce one SMIRKS per variant. \
Every SMIRKS must include enough context to exclude the competing classes."""


def _get_discrimination_hint(cls: str, competitors: list[str], named_dict: dict) -> str:
    """Generate a hint about what distinguishes this class from competitors."""
    name = named_dict.get(cls, cls)
    comp_names = [named_dict.get(c, c) for c in competitors]
    return (
        f"What structural feature distinguishes '{name}' from "
        + ", ".join(f"'{n}'" for n in comp_names)
        + "?"
    )


def _build_discriminate_prompt(
    cls: str,
    smirks: list[str],
    target_examples: list[str],
    competing_info: list[dict],
    named_dict: dict,
) -> str:
    """Build a discrimination refinement prompt.

    Args:
        cls: Target class code.
        smirks: Current (too general) SMIRKS for target class.
        target_examples: Example reactions for target class.
        competing_info: List of dicts with keys: cls, name, smirks, examples.
        named_dict: Class code -> name mapping.
    """
    current_smirks = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(smirks))
    target_examples_str = "\n".join(
        f"  {i+1}. {e}" for i, e in enumerate(target_examples[:15])
    )

    sections = []
    for comp in competing_info:
        lines = [f"Class: {comp['cls']} - {comp['name']}"]
        if comp.get("smirks"):
            lines.append("  SMIRKS:")
            for i, s in enumerate(comp["smirks"], 1):
                lines.append(f"    {i}. {s}")
        lines.append("  Example reactions:")
        for i, e in enumerate(comp["examples"][:10], 1):
            lines.append(f"    {i}. {e}")
        sections.append("\n".join(lines))

    competitors = [c["cls"] for c in competing_info]
    hint = _get_discrimination_hint(cls, competitors, named_dict)

    return _DISCRIMINATE_USER_PROMPT_TEMPLATE.format(
        target_class=f"{cls} - {named_dict.get(cls, cls)}",
        current_smirks=current_smirks,
        target_examples=target_examples_str,
        competing_sections="\n\n".join(sections),
        discrimination_hint=hint,
    )


def parse_competing_groups(competing_arg: list[str] | None) -> dict[str, list[str]]:
    """Parse --competing argument into {class: [competitors]} mapping.

    Format: "1.6.1,1.6.5,1.6.8 1.7.1,1.7.5" → two groups.
    Each class in a group competes with all others in the same group.
    """
    if not competing_arg:
        return {}
    result: dict[str, list[str]] = {}
    for group_str in competing_arg:
        members = [c.strip() for c in group_str.split(",")]
        for cls in members:
            result[cls] = [c for c in members if c != cls]
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate SMIRKS for specific classes with diverse template sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database", required=True, help="Reaction DB (parquet).")
    p.add_argument("--mapping", required=True, help="structured_mapping.json")
    p.add_argument("--checkpoint", required=True, help="Checkpoint JSON to patch.")
    p.add_argument("--classes", nargs="+", required=True, help="Class codes to regenerate.")
    p.add_argument("--api-key", default=None, help="Gemini API key.")
    p.add_argument("--output", default=None, help="Output JSONL SMIRKS DB (optional, patches checkpoint if omitted).")
    p.add_argument("--model", default="gemini-3-flash-preview")
    p.add_argument("--top-n", type=int, default=40, help="Templates per class.")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip classes that already have SMIRKS in the checkpoint.")
    p.add_argument("--radius-reactant", type=int, default=0,
                   help="Reactant-side template radius.")
    p.add_argument("--radius-product", type=int, default=1,
                   help="Product-side template radius.")
    rc_group = p.add_mutually_exclusive_group()
    rc_group.add_argument("--relax-context", action="store_true", default=None,
                          help="Use minimal SMARTS for context atoms (auto-enabled for non-default radii).")
    rc_group.add_argument("--no-relax-context", action="store_false", dest="relax_context",
                          help="Keep full SMARTS for context atoms.")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel workers for template extraction.")
    p.add_argument("--competing", nargs="+", default=None,
                   help="Competing class groups, comma-separated. "
                        "E.g.: '1.6.1,1.6.5,1.6.8' '1.7.1,1.7.5'")
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading database: {args.database}")
    df = pd.read_parquet(args.database)
    print(f"  {len(df):,} reactions")

    print(f"Loading mapping: {args.mapping}")
    with open(args.mapping, encoding="utf-8") as f:
        named_dict = json.load(f)

    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, encoding="utf-8") as f:
        ckpt = json.load(f)

    train_results = ckpt.get("train_results", {})
    split = ckpt.get("split", {})

    # ── Template radius configuration ────────────────────────────────
    rr = args.radius_reactant
    rp = args.radius_product
    template_col = _template_col(rr, rp)
    is_default_radius = (rr == 0 and rp == 1)

    if args.relax_context is None:
        relax_context = not is_default_radius
    else:
        relax_context = args.relax_context

    if not is_default_radius:
        print(f"Template radius: ({rr}, {rp}) [will extract on-the-fly]")
        print(f"  relax_context={relax_context}, n_jobs={args.n_jobs}")
    else:
        print(f"Template radius: (0, 1) [using pre-computed column]")

    # ── Initialize Gemini ────────────────────────────────────────────────
    generalizer = SmirksGeneralizer(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    # ── Process each class ───────────────────────────────────────────────
    results: dict[str, dict] = {}

    for cls in args.classes:
        print(f"\n{'='*60}")
        print(f"Processing {cls} ({named_dict.get(cls, '?')})")
        print(f"{'='*60}")

        # Skip if already has SMIRKS
        if args.skip_existing and cls in train_results and train_results[cls].get("smirks"):
            print(f"  SKIPPED (already has {len(train_results[cls]['smirks'])} SMIRKS)")
            continue

        # Determine tier column
        depth = cls.replace("CONFLICT:", "").count(".") + 1
        class_col = f"tier_{depth}"

        # Use training data: aggregate this class AND all its descendants
        train_idx = []
        for k, v in split.items():
            if k == cls or k.startswith(cls + "."):
                train_idx.extend(v["train"])
        if train_idx:
            df_train = df.loc[train_idx]
        else:
            # No split info, use all data for this class
            df_train = df[df[class_col] == cls]
            print(f"  WARNING: no split found, using all {len(df_train)} reactions")

        print(f"  Training reactions: {len(df_train):,}")

        # Extract templates at custom radius if needed
        if not is_default_radius:
            df_train = extract_templates_for_class(
                df_train, template_col, rr, rp, relax_context, args.n_jobs,
            )

        # Show old SMIRKS
        old = train_results.get(cls, {})
        if old.get("smirks"):
            print(f"  Old SMIRKS ({len(old['smirks'])}):")
            for s in old["smirks"]:
                print(f"    {s}")
            print(f"  Old coverage: {old.get('combined_coverage', '?')}")
            print(f"  Old valid: {old.get('valid', '?')}")

        # Screen with diverse sampling
        data = screen_diverse(
            cls, df_train, named_dict, top_n=args.top_n, class_col=class_col,
            template_col=template_col,
        )

        if not data["top_n_templates"]:
            print(f"  No templates found, skipping")
            continue

        print(f"  Template coverage: {data['coverage']:.1%}")
        print(f"  Templates: {len(data['top_n_templates'])}")

        # Call Gemini
        result = generalizer.generalize(
            reaction_class=data["reaction_class"],
            templates=data["top_n_templates"],
            examples=data["top_n_examples"],
            max_retries=args.max_retries,
            validation_threshold=args.threshold,
        )

        print(f"\n  Result: valid={result['valid']}, "
              f"coverage={result['combined_coverage']:.1%}, "
              f"attempts={result['attempts']}")
        for i, s in enumerate(result.get("smirks", []), 1):
            print(f"  SMIRKS {i}: {s}")

        # Validate on a broader sample (not just top-N examples)
        all_rxns = df_train["SANITIZED_REACTION"].dropna().tolist()
        if len(all_rxns) > 200:
            import numpy as np
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(all_rxns), 200, replace=False)
            sample_rxns = [all_rxns[i] for i in sample_idx]
        else:
            sample_rxns = all_rxns

        broad_val = validate_smirks(result.get("smirks", []), sample_rxns, threshold=0.0)
        print(f"  Broad validation (on {len(sample_rxns)} random reactions): "
              f"{broad_val['combined_coverage']:.1%}")

        result["broad_coverage"] = broad_val["combined_coverage"]
        results[cls] = result

        # Show reasoning
        if result.get("reasoning"):
            print(f"\n  Reasoning (truncated):")
            print(f"  {result['reasoning'][:500]}")

    # ── Discrimination refinement ────────────────────────────────────────
    competing_map = parse_competing_groups(args.competing)
    if competing_map:
        print(f"\n{'='*60}")
        print("DISCRIMINATION REFINEMENT")
        print(f"{'='*60}")

        for cls in args.classes:
            competitors = competing_map.get(cls, [])
            if not competitors:
                continue

            current_smirks = results.get(cls, {}).get("smirks", [])
            if not current_smirks:
                # Fall back to checkpoint
                current_smirks = train_results.get(cls, {}).get("smirks", [])
            if not current_smirks:
                print(f"\n  {cls}: no SMIRKS to refine, skipping")
                continue

            print(f"\n  Refining {cls} ({named_dict.get(cls, '?')}) "
                  f"against {competitors}")

            # Gather target examples
            depth = cls.replace("CONFLICT:", "").count(".") + 1
            class_col = f"tier_{depth}"
            train_idx = []
            for k, v in split.items():
                if k == cls or k.startswith(cls + "."):
                    train_idx.extend(v["train"])
            if train_idx:
                df_cls = df.loc[train_idx]
            else:
                df_cls = df[df[class_col] == cls]
            target_examples = df_cls["SANITIZED_REACTION"].dropna().tolist()
            if len(target_examples) > 15:
                import numpy as np
                rng = np.random.RandomState(42)
                idx = rng.choice(len(target_examples), 15, replace=False)
                target_examples = [target_examples[i] for i in idx]

            # Gather competing class info
            competing_info = []
            for comp_cls in competitors:
                comp_depth = comp_cls.replace("CONFLICT:", "").count(".") + 1
                comp_col = f"tier_{comp_depth}"
                comp_idx = []
                for k, v in split.items():
                    if k == comp_cls or k.startswith(comp_cls + "."):
                        comp_idx.extend(v["train"])
                if comp_idx:
                    df_comp = df.loc[comp_idx]
                else:
                    df_comp = df[df[comp_col] == comp_cls]
                comp_examples = df_comp["SANITIZED_REACTION"].dropna().tolist()
                if len(comp_examples) > 10:
                    import numpy as np
                    rng = np.random.RandomState(123)
                    idx = rng.choice(len(comp_examples), 10, replace=False)
                    comp_examples = [comp_examples[i] for i in idx]

                # Get competitor SMIRKS from results or checkpoint
                comp_smirks = (
                    results.get(comp_cls, {}).get("smirks", [])
                    or train_results.get(comp_cls, {}).get("smirks", [])
                )

                competing_info.append({
                    "cls": comp_cls,
                    "name": named_dict.get(comp_cls, comp_cls),
                    "smirks": comp_smirks,
                    "examples": comp_examples,
                })

            # Build prompt and call Gemini
            prompt = _build_discriminate_prompt(
                cls, current_smirks, target_examples, competing_info, named_dict,
            )

            try:
                refined = generalizer._call(
                    prompt,
                    retries=3,
                    system_prompt=_DISCRIMINATE_SYSTEM_PROMPT,
                )
            except Exception as exc:
                print(f"  Discrimination call failed: {exc}")
                continue

            # Validate refined SMIRKS on target examples
            all_rxns = df_cls["SANITIZED_REACTION"].dropna().tolist()
            if len(all_rxns) > 200:
                import numpy as np
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(len(all_rxns), 200, replace=False)
                sample_rxns = [all_rxns[i] for i in sample_idx]
            else:
                sample_rxns = all_rxns

            ref_val = validate_smirks(refined.smirks, sample_rxns, threshold=0.0)

            print(f"  Refined SMIRKS ({len(refined.smirks)}):")
            for i, s in enumerate(refined.smirks, 1):
                print(f"    {i}. {s}")
            print(f"  Coverage on target: {ref_val['combined_coverage']:.1%}")

            # Check FP on competing examples
            for comp in competing_info:
                fp_count = 0
                for rxn in comp["examples"]:
                    for s in refined.smirks:
                        r = test_smirks(rxn, s)
                        if r["correct"]:
                            fp_count += 1
                            break
                print(f"  FP on {comp['cls']}: {fp_count}/{len(comp['examples'])}")

            # Accept if coverage stays reasonable
            old_cov = results.get(cls, {}).get("broad_coverage", 0)
            if ref_val["combined_coverage"] >= old_cov * 0.8:
                print(f"  ACCEPTED (coverage {ref_val['combined_coverage']:.1%} "
                      f">= 80% of original {old_cov:.1%})")
                if cls not in results:
                    results[cls] = dict(train_results.get(cls, {}))
                results[cls]["smirks"] = refined.smirks
                results[cls]["combined_coverage"] = ref_val["combined_coverage"]
                results[cls]["broad_coverage"] = ref_val["combined_coverage"]
                results[cls]["discrimination_reasoning"] = refined.reasoning
            else:
                print(f"  REJECTED (coverage {ref_val['combined_coverage']:.1%} "
                      f"< 80% of original {old_cov:.1%})")

            if refined.reasoning:
                print(f"\n  Reasoning (truncated):")
                print(f"  {refined.reasoning[:500]}")

    # ── Patch checkpoint ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Patching checkpoint...")
    for cls, result in results.items():
        if result.get("smirks"):
            train_results[cls] = result
            print(f"  {cls}: {len(result['smirks'])} SMIRKS, "
                  f"coverage={result['combined_coverage']:.1%}")

    ckpt["train_results"] = train_results
    # Atomic write
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=Path(args.checkpoint).parent, suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, args.checkpoint)
        print(f"  Checkpoint saved: {args.checkpoint}")
    except Exception:
        os.unlink(tmp_path)
        raise

    # ── Optionally write SMIRKS DB ───────────────────────────────────────
    if args.output:
        entries = []
        for cls, result in results.items():
            name = result.get("reaction_class", cls)
            for s in result.get("smirks", []):
                entries.append({"name": name, "smirks": s, "class": cls})
        with open(args.output, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  SMIRKS DB written: {args.output}")

    # Token usage
    usage = generalizer.token_usage
    print(f"\nToken usage: {usage['total_calls']} calls, "
          f"{usage['total_tokens']:,} tokens")


if __name__ == "__main__":
    main()
