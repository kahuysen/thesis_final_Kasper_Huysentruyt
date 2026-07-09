"""Evaluate the ordered SMIRKS DB against the full reaction database.

Runs first-match naming on all reactions and compares the predicted class
against the ground-truth tier columns.  Standalone — only requires:
    pip install rdkit pandas joblib tqdm

Usage
-----
    python evaluate_naming.py \
        --smirks-db ordered_smirks_db.json \
        --reactions reaction_db.parquet \
        --output naming_eval.csv \
        [--n-jobs 8] \
        [--sample 10000]       # quick test on a random subset
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.CRITICAL)

_ATOM_MAP_RE = re.compile(r":\d+")


# ── Class merges (structurally indistinguishable pairs) ───────────────────────

# Classes that differ only by mechanism/conditions, not by SMIRKS pattern.
# Each group maps all member prefixes to a shared label.
_CLASS_MERGES = {
    # N-arylation: Buchwald / SNAr / heteroaryl amination / aryl amination
    "1.3.1": "1.3.1/1.3.5/1.3.6/1.3.8",
    "1.3.5": "1.3.1/1.3.5/1.3.6/1.3.8",
    "1.3.6": "1.3.1/1.3.5/1.3.6/1.3.8",
    "1.3.8": "1.3.1/1.3.5/1.3.6/1.3.8",
    # Nitro reductions: all NO₂ → NH₂ with identical core transformation
    "6.1.1": "6.1.1/6.1.11/6.1.14/6.1.16",
    "6.1.11": "6.1.1/6.1.11/6.1.14/6.1.16",
    "6.1.14": "6.1.1/6.1.11/6.1.14/6.1.16",
    "6.1.16": "6.1.1/6.1.11/6.1.14/6.1.16",
    # Ketone reduction: aryl ketone is a subset of general hydride reduction
    "6.5.1": "6.5.1/6.5.5",
    "6.5.5": "6.5.1/6.5.5",
    # Halogenation: heteroarene vs arene — molecules often have both ring types
    "9.1.4": "9.1.4/9.1.5",
    "9.1.5": "9.1.4/9.1.5",
    # Ester cleavage: deprotection vs FGI — same R-COO-alkyl → R-COOH transformation
    "5.1.4.1": "5.1.4.1/8.7.2.1",
    "8.7.2.1": "5.1.4.1/8.7.2.1",
    # Aminolysis of esters: NH3 + ester → primary amide
    "2.1.4.4": "2.1.4.4/8.7.2.2",
    "8.7.2.2": "2.1.4.4/8.7.2.2",
    # Benzylic halogenation: benzylic C-H → benzylic halide
    "9.1.6": "9.1.6/7.6.4",
    "7.6.4": "9.1.6/7.6.4",
    # Fused imidazole synthesis: alpha-haloketone + 2-amino-N-heterocycle
    "4.1.3.4.3": "4.1.3.4.3/3.6.4.1",
    "3.6.4.1": "4.1.3.4.3/3.6.4.1",
    # Knoevenagel: aldehyde + oxindole → arylideneoxindole
    "3.11.5.4.6": "3.11.5.4.6/6.2.4",
    "6.2.4": "3.11.5.4.6/6.2.4",
    # Heteroaryl chlorination: heteroaryl-OH + POCl3 → heteroaryl-Cl
    "8.1.5": "8.1.5/1.2.12",
    "1.2.12": "8.1.5/1.2.12",
    # Pd-catalyzed borylation: ArX + B2pin2 → ArBpin
    "3.5.8": "3.5.8/8.7.7.2.1",
    "8.7.7.2.1": "3.5.8/8.7.7.2.1",
    # Amidoxime formation: nitrile + hydroxylamine → amidoxime
    "8.7.2.6": "8.7.2.6/2.5.1.3",
    "2.5.1.3": "8.7.2.6/2.5.1.3",
}


def _merge_class(cls: str) -> str:
    """Map a class code to its merged label (if any)."""
    for prefix, merged in _CLASS_MERGES.items():
        if cls == prefix or cls.startswith(prefix + "."):
            return merged
    return cls


# ── Compiled SMIRKS cache (per-process) ──────────────────────────────────────

_RXN_CACHE: dict[str, AllChem.ChemicalReaction] = {}


def _get_compiled_rxn(smirks: str):
    """Get or compile a SMIRKS reaction object (cached per process)."""
    if smirks not in _RXN_CACHE:
        try:
            _RXN_CACHE[smirks] = AllChem.ReactionFromSmarts(smirks)
        except Exception:
            _RXN_CACHE[smirks] = None
    return _RXN_CACHE[smirks]


# ── Core naming (returns class code + name) ──────────────────────────────────


def name_reaction(
    rxn: str,
    smirks_records: list[tuple[str, str, str, int]],
) -> tuple[str, str]:
    """Name a reaction using first-match against an ordered SMIRKS list.

    Args:
        rxn: Sanitized reaction SMILES (unmapped, ``reactants>>products``).
        smirks_records: List of ``(class_code, name, smirks, nreact)`` tuples,
            in the order they should be tried.

    Returns:
        ``(class_code, name)`` of the first matching SMIRKS, or
        ``("", "OtherReaction")`` if nothing matches.
    """
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    parts = rxn.split(">>")
    if len(parts) != 2:
        return ("", "OtherReaction")

    reactants_smiles, products_smiles = parts
    reactants = reactants_smiles.split(".")
    products = products_smiles.split(".")

    if len(reactants) > 4 or len(products) > 4:
        return ("", "OtherReaction")

    # Canonicalize expected products
    new_products: list[str] = []
    for product in products:
        try:
            new_products.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(product), isomericSmiles=False)
            )
        except Exception:
            new_products.append(product)

    react_mols = [Chem.MolFromSmiles(r) for r in reactants]
    if any(m is None for m in react_mols):
        return ("", "OtherReaction")

    num_reactants = len(react_mols)

    for cls_code, cls_name, smirks, nreact in smirks_records:
        if nreact > num_reactants:
            continue

        rxn_obj = _get_compiled_rxn(smirks)
        if rxn_obj is None:
            continue

        # Build reactant tuples (subsets × permutations)
        if nreact == num_reactants:
            if num_reactants == 1:
                all_tuples = [tuple(react_mols)]
            else:
                all_tuples = list(itertools.permutations(react_mols))
        else:
            all_tuples = []
            for subset in itertools.combinations(react_mols, nreact):
                if nreact == 1:
                    all_tuples.append(subset)
                else:
                    all_tuples.extend(itertools.permutations(subset))

        for tup in all_tuples:
            try:
                outcomes = rxn_obj.RunReactants(tup)
            except Exception:
                continue
            for prods in outcomes:
                try:
                    prod_smi = Chem.MolToSmiles(prods[0], isomericSmiles=False)
                except Exception:
                    continue
                if prod_smi in new_products:
                    return (cls_code, cls_name)

    return ("", "OtherReaction")


# ── Chunk worker ─────────────────────────────────────────────────────────────


def _name_chunk(
    chunk: list[str],
    smirks_records: list[tuple[str, str, str, int]],
) -> list[tuple[str, str]]:
    """Worker: name a chunk of reactions."""
    return [name_reaction(rxn, smirks_records) for rxn in chunk]


# ── Matching logic ───────────────────────────────────────────────────────────


def _tier_depth(cls: str) -> int:
    """Number of dot-separated segments in a class code."""
    return cls.replace("CONFLICT:", "").count(".") + 1


def check_match(predicted_cls: str, row: pd.Series) -> str:
    """Check if predicted class matches any tier column.

    Returns:
        "exact" if predicted == the tier column at that depth,
        "ancestor" if predicted is a prefix of a finer tier,
        "descendant" if a coarser tier is a prefix of predicted,
        "mismatch" if no match,
        "unmatched" if predicted is empty (OtherReaction).
    """
    if not predicted_cls:
        return "unmatched"

    depth = _tier_depth(predicted_cls)
    tier_col = f"tier_{depth}"

    # Exact match at the class's own tier depth
    if tier_col in row.index:
        actual = row.get(tier_col, "")
        if pd.notna(actual) and actual == predicted_cls:
            return "exact"

    # Check if predicted is an ancestor of a finer tier
    for d in range(depth + 1, 10):
        col = f"tier_{d}"
        if col not in row.index:
            break
        val = row.get(col, "")
        if pd.notna(val) and str(val).startswith(predicted_cls + "."):
            return "ancestor"

    # Check if predicted is a descendant of a coarser tier
    for d in range(1, depth):
        col = f"tier_{d}"
        if col not in row.index:
            continue
        val = row.get(col, "")
        if pd.notna(val) and predicted_cls.startswith(str(val) + "."):
            # Predicted is more specific than the tier column, but
            # the coarser tier matches — need to check finer tiers too
            pass

    return "mismatch"


# ── Main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ordered SMIRKS DB naming on a reaction database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--smirks-db",
        required=True,
        help="Ordered SMIRKS DB (JSONL with name, smirks, class fields).",
    )
    p.add_argument(
        "--reactions",
        required=True,
        help="Reaction database (parquet).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV with per-reaction results.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers (default: half of CPUs).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random sample size for quick testing.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n_jobs = args.n_jobs or max(1, mp.cpu_count() // 2)

    # ── Load SMIRKS DB ───────────────────────────────────────────────────
    print(f"Loading SMIRKS DB: {args.smirks_db}")
    records: list[tuple[str, str, str, int]] = []
    with open(args.smirks_db, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            smirks = entry["smirks"]
            nreact = len(smirks.split(">>")[0].split("."))
            records.append((
                entry.get("class", ""),
                entry["name"],
                smirks,
                nreact,
            ))
    print(f"  {len(records)} SMIRKS patterns loaded")

    # Build set of class codes in the DB (for coverage analysis)
    db_classes: set[str] = {r[0] for r in records if r[0]}
    print(f"  {len(db_classes)} unique classes in DB")

    # ── Load reactions ───────────────────────────────────────────────────
    print(f"Loading reactions: {args.reactions}")
    df = pd.read_parquet(args.reactions)
    print(f"  {len(df):,} reactions")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
        print(f"  Sampled {len(df):,} reactions")
        df = df.reset_index(drop=True)

    # Get sanitized reactions
    reactions = df["SANITIZED_REACTION"].fillna("").tolist()

    # ── Run naming ───────────────────────────────────────────────────────
    print(f"Naming {len(reactions):,} reactions with {n_jobs} workers...")
    t0 = time.time()

    if n_jobs == 1:
        results = [
            name_reaction(rxn, records)
            for rxn in tqdm(reactions, desc="Naming")
        ]
    else:
        n = len(reactions)
        chunk_size = max(50, n // (n_jobs * 8))
        chunks = [reactions[i : i + chunk_size] for i in range(0, n, chunk_size)]

        raw_chunks = Parallel(n_jobs=n_jobs)(
            delayed(_name_chunk)(chunk, records)
            for chunk in tqdm(chunks, desc="Naming")
        )
        results = [r for chunk_res in raw_chunks for r in chunk_res]

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({len(reactions) / elapsed:.0f} rxn/s)")

    # ── Evaluate ─────────────────────────────────────────────────────────
    pred_classes = [r[0] for r in results]
    pred_names = [r[1] for r in results]

    df["pred_class"] = pred_classes
    df["pred_name"] = pred_names

    # Check match against tier columns
    tier_cols = sorted(
        [c for c in df.columns if c.startswith("tier_") and c[5:].isdigit()],
        key=lambda c: int(c.split("_")[1]),
    )
    print(f"  Tier columns: {tier_cols}")

    match_types: list[str] = []
    for idx in tqdm(range(len(df)), desc="Evaluating", disable=len(df) < 1000):
        pc = pred_classes[idx]
        if not pc:
            match_types.append("unmatched")
            continue

        depth = _tier_depth(pc)
        tier_col = f"tier_{depth}"

        if tier_col in df.columns:
            actual = df.iloc[idx][tier_col]
            if pd.notna(actual) and str(actual) == pc:
                match_types.append("exact")
                continue
            # Check merged classes: if pred and actual map to the same group
            if pd.notna(actual):
                merged_pred = _merge_class(pc)
                merged_actual = _merge_class(str(actual))
                if merged_pred == merged_actual and merged_pred != pc:
                    match_types.append("exact")
                    continue

        # Check ancestor: predicted is prefix of a finer tier
        found_ancestor = False
        for col in tier_cols:
            col_depth = int(col.split("_")[1])
            if col_depth <= depth:
                continue
            val = df.iloc[idx][col]
            if pd.notna(val) and str(val).startswith(pc + "."):
                match_types.append("ancestor")
                found_ancestor = True
                break
            # Also check merged: pred 1.3.5.x is ancestor of true 1.3.6.x.y
            if pd.notna(val):
                merged_pred = _merge_class(pc)
                merged_val = _merge_class(str(val))
                if merged_pred == merged_val and merged_pred != pc:
                    match_types.append("ancestor")
                    found_ancestor = True
                    break

        if found_ancestor:
            continue

        # Check descendant: predicted is a child of a coarser tier value
        # that is NOT itself in the DB (only its children are).
        # E.g., true tier_4 = "2.6.2", DB has "2.6.2.1" and "2.6.2.2",
        # prediction = "2.6.2.1" → ancestor match (correct but more specific).
        found_descendant = False
        for col in tier_cols:
            col_depth = int(col.split("_")[1])
            if col_depth >= depth:
                continue
            val = df.iloc[idx][col]
            if pd.notna(val):
                val_s = str(val)
                if pc.startswith(val_s + ".") and val_s not in db_classes:
                    match_types.append("ancestor")
                    found_descendant = True
                    break

        if found_descendant:
            continue

        match_types.append("mismatch")

    df["match_type"] = match_types

    # ── Determine if each reaction's true class is in the SMIRKS DB ─────
    # A reaction's "best" true class is the finest tier that appears in db_classes,
    # or failing that, the finest tier whose ancestor appears.
    true_class_in_db: list[bool] = []
    true_class_best: list[str] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        found = False
        best = ""
        # Check tiers from finest to coarsest
        for col in reversed(tier_cols):
            val = row.get(col, "")
            if pd.notna(val) and str(val):
                val_s = str(val)
                if val_s in db_classes:
                    found = True
                    best = val_s
                    break
                # Check if any DB class is a descendant (DB has finer class)
                if any(dc.startswith(val_s + ".") for dc in db_classes):
                    found = True
                    best = val_s
                    break
        true_class_in_db.append(found)
        true_class_best.append(best)

    df["true_class_in_db"] = true_class_in_db
    df["true_class_best"] = true_class_best

    n_in_db = sum(true_class_in_db)
    n_not_in_db = len(df) - n_in_db

    # ── Summary statistics ───────────────────────────────────────────────
    n_total = len(df)
    n_exact = match_types.count("exact")
    n_ancestor = match_types.count("ancestor")
    n_mismatch = match_types.count("mismatch")
    n_unmatched = match_types.count("unmatched")
    n_correct = n_exact + n_ancestor

    print(f"\n{'='*60}")
    print(f"NAMING EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total reactions:      {n_total:>10,}")
    print(f"  Class in DB:        {n_in_db:>10,}  ({n_in_db/n_total*100:5.1f}%)")
    print(f"  Class NOT in DB:    {n_not_in_db:>10,}  ({n_not_in_db/n_total*100:5.1f}%)")
    print(f"")
    print(f"Overall:")
    print(f"  Exact match:        {n_exact:>10,}  ({n_exact/n_total*100:5.1f}%)")
    print(f"  Ancestor match:     {n_ancestor:>10,}  ({n_ancestor/n_total*100:5.1f}%)")
    print(f"  Correct (ex+anc):   {n_correct:>10,}  ({n_correct/n_total*100:5.1f}%)")
    print(f"  Mismatch (wrong):   {n_mismatch:>10,}  ({n_mismatch/n_total*100:5.1f}%)")
    print(f"  Unmatched (Other):  {n_unmatched:>10,}  ({n_unmatched/n_total*100:5.1f}%)")

    # Split by coverage
    if n_in_db > 0:
        in_db_mask = df["true_class_in_db"]
        mt_arr = df["match_type"]
        n_ex_db = int(((mt_arr == "exact") & in_db_mask).sum())
        n_anc_db = int(((mt_arr == "ancestor") & in_db_mask).sum())
        n_cor_db = n_ex_db + n_anc_db
        n_mis_db = int(((mt_arr == "mismatch") & in_db_mask).sum())
        n_unm_db = int(((mt_arr == "unmatched") & in_db_mask).sum())
        print(f"")
        print(f"Among reactions whose class IS in the DB ({n_in_db:,}):")
        print(f"  Exact match:        {n_ex_db:>10,}  ({n_ex_db/n_in_db*100:5.1f}%)")
        print(f"  Ancestor match:     {n_anc_db:>10,}  ({n_anc_db/n_in_db*100:5.1f}%)")
        print(f"  Correct (ex+anc):   {n_cor_db:>10,}  ({n_cor_db/n_in_db*100:5.1f}%)")
        print(f"  Mismatch (wrong):   {n_mis_db:>10,}  ({n_mis_db/n_in_db*100:5.1f}%)")
        print(f"  Unmatched (Other):  {n_unm_db:>10,}  ({n_unm_db/n_in_db*100:5.1f}%)")

    print(f"{'='*60}")

    # ── Mismatch breakdown ───────────────────────────────────────────────
    if n_mismatch > 0:
        df_mis = df[df["match_type"] == "mismatch"]

        # How many mismatches are sibling classes (same parent)?
        def _parent(cls: str) -> str:
            parts = cls.replace("CONFLICT:", "").split(".")
            return ".".join(parts[:-1]) if len(parts) > 1 else ""

        n_sibling = 0
        n_same_t1 = 0
        for _, row in df_mis.iterrows():
            pc = row["pred_class"]
            depth = _tier_depth(pc)
            tier_col = f"tier_{depth}"
            actual = str(row.get(tier_col, "")) if tier_col in row.index else ""
            if actual and _parent(pc) == _parent(actual):
                n_sibling += 1
            elif actual and pc.split(".")[0] == actual.split(".")[0]:
                n_same_t1 += 1

        n_cross = n_mismatch - n_sibling - n_same_t1
        print(f"\nMismatch breakdown:")
        print(f"  Sibling classes:    {n_sibling:>10,}  ({n_sibling/n_total*100:5.1f}%)")
        print(f"  Same superclass:    {n_same_t1:>10,}  ({n_same_t1/n_total*100:5.1f}%)")
        print(f"  Cross-superclass:   {n_cross:>10,}  ({n_cross/n_total*100:5.1f}%)")

    # ── Save results ─────────────────────────────────────────────────────
    out_cols = ["SANITIZED_REACTION"] + tier_cols + [
        "true_class_in_db", "true_class_best",
        "pred_class", "pred_name", "match_type",
    ]
    # Only include columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # ── Save summary JSON ────────────────────────────────────────────────
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary = {
        "n_total": n_total,
        "n_in_db": n_in_db,
        "n_not_in_db": n_not_in_db,
        "n_exact": n_exact,
        "n_ancestor": n_ancestor,
        "n_correct": n_correct,
        "n_mismatch": n_mismatch,
        "n_unmatched": n_unmatched,
        "pct_exact": round(n_exact / n_total * 100, 2),
        "pct_correct": round(n_correct / n_total * 100, 2),
        "pct_mismatch": round(n_mismatch / n_total * 100, 2),
        "pct_unmatched": round(n_unmatched / n_total * 100, 2),
        "in_db_pct_correct": round(n_cor_db / n_in_db * 100, 2) if n_in_db > 0 else 0.0,
        "in_db_pct_mismatch": round(n_mis_db / n_in_db * 100, 2) if n_in_db > 0 else 0.0,
        "in_db_pct_unmatched": round(n_unm_db / n_in_db * 100, 2) if n_in_db > 0 else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "rxn_per_second": round(len(reactions) / elapsed, 1),
        "smirks_db": args.smirks_db,
        "reactions": args.reactions,
        "n_smirks": len(records),
        "n_db_classes": len(db_classes),
        "n_jobs": n_jobs,
        "sample": args.sample,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
