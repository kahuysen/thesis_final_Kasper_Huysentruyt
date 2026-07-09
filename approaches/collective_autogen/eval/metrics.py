"""Metrics for evaluating ReactionRecord extractions.

Three categories:
- Reference-free: schema_conformance, smiles_validity_rate, role_enum_compliance,
  reaction_count. Work on a single record.
- Set-based reference: reactant_iou, product_iou, condition_coverage. Compare
  the *union* of all reactant / product SMILES across the whole record.
- Reaction-level reference (ChemEagle-style): soft_match_f1, hard_match_f1.
  Match each predicted reaction to a gold reaction by content; report
  precision / recall / F1 across the reaction set.

All SMILES comparisons go through RDKit canonicalisation so equivalent SMILES
written differently (e.g. `c1ccccc1C=O` vs `O=Cc1ccccc1`) are treated as equal.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from pydantic import ValidationError
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFMCS
from scipy.optimize import linear_sum_assignment

from schema import CONDITION_ROLES, ReactionRecord

RDLogger.DisableLog("rdApp.*")


# ---------- helpers ----------

def _canonical(smiles: str | None) -> str | None:
    """Canonical SMILES via RDKit, or None if unparseable.

    R-group wildcards are normalised: `[1*]`, `[2*]`, `[N*]` and bare `*` all
    canonicalise to bare `*`. This is intentional for set-IoU comparison —
    different sources label R-groups differently (the gold for the review zip
    uses `[1*]` / `[2*]` numbering; agents typically emit bare `*`), but the
    underlying molecular structure is the same. Comparing without normalising
    gives spurious 0.0 IoU on every wildcard-bearing template.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        # Atomic number 0 == wildcard / dummy atom (`*` and `[N*]` variants).
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() != 0:
            atom.SetIsotope(0)
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _canonical_no_stereo(smiles: str | None) -> str | None:
    """Like _canonical but with all stereochemistry stripped. Useful for the
    constitution-match metric: tells us whether two molecules differ only in
    stereo (a partial extraction failure) vs. in connectivity (a worse failure).
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() != 0:
            atom.SetIsotope(0)
    try:
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


_NUM_NORM = re.compile(r"\s+")


def _normalise_condition_text(text: str) -> str:
    """Normalise condition text for set comparison: lower-case, strip, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return _NUM_NORM.sub(" ", text).strip().lower()


def _all_smiles(record: dict, role: str) -> list[str | None]:
    """Pull every SMILES from `reactants` (role='reactant') or `products` (role='product')."""
    field = "reactants" if role == "reactant" else "products"
    out: list[str | None] = []
    for r in record.get("reactions", []):
        for c in r.get(field, []) or []:
            out.append(c.get("smiles"))
    return out


def _condition_set(record: dict) -> set[tuple[str, str]]:
    """Set of (role, normalised_text) tuples across all reactions."""
    out: set[tuple[str, str]] = set()
    for r in record.get("reactions", []):
        for c in r.get("conditions", []) or []:
            out.add((c.get("role", ""), _normalise_condition_text(c.get("text", ""))))
    return out


# ---------- reference-free metrics ----------

def schema_conformance(record: dict) -> dict:
    """Does the record validate against the canonical pydantic schema?"""
    try:
        ReactionRecord.model_validate(record)
        return {"valid": True, "errors": []}
    except ValidationError as e:
        return {"valid": False, "errors": [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]}


def smiles_validity_rate(record: dict) -> dict:
    """Fraction of non-null SMILES (across reactants, products, conditions) that
    RDKit can parse."""
    smis: list[str] = []
    for r in record.get("reactions", []):
        for c in (r.get("reactants") or []) + (r.get("products") or []):
            s = c.get("smiles")
            if s:
                smis.append(s)
        for cond in r.get("conditions") or []:
            s = cond.get("smiles")
            if s:
                smis.append(s)
    total = len(smis)
    valid = sum(1 for s in smis if _canonical(s) is not None)
    return {"valid": valid, "total": total, "rate": (valid / total) if total else 1.0}


def role_enum_compliance(record: dict) -> dict:
    """Fraction of condition roles that are within the schema's allowed enum."""
    legal = set(CONDITION_ROLES)
    roles = [c.get("role", "") for r in record.get("reactions", []) for c in r.get("conditions") or []]
    illegal = [r for r in roles if r not in legal]
    total = len(roles)
    return {
        "total": total,
        "legal": total - len(illegal),
        "illegal": len(illegal),
        "illegal_values": sorted(set(illegal)),
        "rate": ((total - len(illegal)) / total) if total else 1.0,
    }


def reaction_count(record: dict) -> dict:
    return {"count": len(record.get("reactions", []))}


# ---------- reference-based metrics ----------

def _smiles_iou(pred: Iterable[str | None], gold: Iterable[str | None]) -> dict:
    p = {c for c in (_canonical(s) for s in pred) if c}
    g = {c for c in (_canonical(s) for s in gold) if c}
    inter = p & g
    union = p | g
    return {
        "pred_size": len(p),
        "gold_size": len(g),
        "intersection": len(inter),
        "union": len(union),
        "iou": (len(inter) / len(union)) if union else 1.0,
        "missing": sorted(g - p),
        "extra": sorted(p - g),
    }


def reactant_iou(record: dict, gold: dict) -> dict:
    return _smiles_iou(_all_smiles(record, "reactant"), _all_smiles(gold, "reactant"))


def product_iou(record: dict, gold: dict) -> dict:
    return _smiles_iou(_all_smiles(record, "product"), _all_smiles(gold, "product"))


def condition_coverage(record: dict, gold: dict) -> dict:
    """Recall of (role, normalised_text) condition pairs from gold over predicted."""
    pred = _condition_set(record)
    g = _condition_set(gold)
    matched = pred & g
    missing = g - pred
    extra = pred - g
    return {
        "gold_size": len(g),
        "pred_size": len(pred),
        "matched": len(matched),
        "recall": (len(matched) / len(g)) if g else 1.0,
        "precision": (len(matched) / len(pred)) if pred else 1.0,
        "missing": sorted(missing),
        "extra": sorted(extra),
    }


# ---------- ChemEagle-style per-reaction match metrics ----------

def _soft_fingerprint(reaction: dict) -> tuple:
    """Reaction content tuple used for ChemEagle's soft-match.

    `(frozenset(canonical reactant SMILES), frozenset(canonical product SMILES))`.
    Null and unparseable SMILES are dropped — they can't carry any information
    about whether the reaction is correct. Labels and the reaction_id are
    deliberately ignored: ChemEagle matches by chemistry, not by index.
    """
    reactants = frozenset(
        c for c in (_canonical(x.get("smiles")) for x in (reaction.get("reactants") or [])) if c
    )
    products = frozenset(
        c for c in (_canonical(x.get("smiles")) for x in (reaction.get("products") or [])) if c
    )
    return (reactants, products)


def _hard_fingerprint(reaction: dict) -> tuple:
    """Soft fingerprint + a frozenset of condition signatures.

    A condition signature is `(role, normalised_text, canonical_smiles)`.
    For conditions without a SMILES (temperature, time, …) the third slot is
    None, so two such conditions match iff their role + text agree. For
    conditions with a SMILES (catalyst, named reagent), the SMILES must also
    match — that's the part hard-match adds over soft-match.
    """
    soft = _soft_fingerprint(reaction)
    conditions = frozenset(
        (
            (c.get("role") or "").strip().lower(),
            _normalise_condition_text(c.get("text", "")),
            _canonical(c.get("smiles")) if c.get("smiles") else None,
        )
        for c in (reaction.get("conditions") or [])
    )
    return soft + (conditions,)


def _match_f1(pred_record: dict, gold_record: dict, *, hard: bool,
              skip_empty_gold_conditions: bool = True) -> dict:
    """Per-reaction precision / recall / F1.

    Matching is multiset-equality on the chosen fingerprint:
        matched = sum((Counter(pred_fps) & Counter(gold_fps)).values())
    Equivalent to optimal bipartite matching with a 0/1 cost matrix.

    `skip_empty_gold_conditions` (hard mode only, default True): if a gold
    reaction has zero conditions, drop the condition-set requirement for
    that reaction — i.e. hard-match collapses to soft-match for it. This
    matches ChemEagle's convention: their reported 81% hard-F1 on the
    OpenChemIE subset (whose gold has no conditions) only makes sense
    under this convention.
    """
    pred = pred_record.get("reactions") or []
    gold = gold_record.get("reactions") or []
    n_pred = len(pred)
    n_gold = len(gold)
    if n_pred == 0 and n_gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "matched": 0, "pred_count": 0, "gold_count": 0,
                "skipped_cond_check": 0}

    # Build fingerprints per reaction. In hard mode, gold reactions with empty
    # conditions get a "soft-equivalent" fingerprint (no condition slot), and
    # we use the same soft-equivalent fingerprint on the pred side to allow
    # the match. We tag those gold reactions and compute the same fingerprint
    # variant for the corresponding (any) pred reaction at match time.
    if not hard:
        pred_fps = [_soft_fingerprint(r) for r in pred]
        gold_fps = [_soft_fingerprint(r) for r in gold]
        skipped = 0
    else:
        gold_fps: list[tuple] = []
        gold_skips: list[bool] = []
        for r in gold:
            empty_conds = not (r.get("conditions") or [])
            if empty_conds and skip_empty_gold_conditions:
                gold_fps.append(_soft_fingerprint(r))
                gold_skips.append(True)
            else:
                gold_fps.append(_hard_fingerprint(r))
                gold_skips.append(False)
        # Pred fingerprints: if ANY gold reaction triggered the skip, we
        # want its soft variant available for matching, AND the regular hard
        # variant available for matching against gold reactions that DO
        # require conditions. So we keep both possible fingerprints per pred
        # reaction and try multiset matching twice.
        pred_soft_fps = [_soft_fingerprint(r) for r in pred]
        pred_hard_fps = [_hard_fingerprint(r) for r in pred]

        # Multiset match: pair each pred (in soft *or* hard form) to each
        # gold according to gold's required form. Two-stage to keep counting
        # correct: first match the strict (hard) gold reactions to pred
        # using hard fingerprints, then match the lenient (skipped) gold to
        # any leftover pred using soft fingerprints.
        gold_strict = [fp for fp, sk in zip(gold_fps, gold_skips) if not sk]
        gold_lenient = [fp for fp, sk in zip(gold_fps, gold_skips) if sk]
        pred_hard_counter = Counter(pred_hard_fps)
        gold_strict_counter = Counter(gold_strict)
        strict_matches = pred_hard_counter & gold_strict_counter
        # Track which pred indices were consumed by strict matches by
        # decrementing counters; then build a "remaining" pred-soft list.
        remaining_hard = pred_hard_counter - strict_matches
        # Map remaining_hard back to which pred indices remain
        consumed_idx: set[int] = set()
        # Greedy: walk pred in order, consume from strict_matches as we go
        sm = Counter(strict_matches)
        for i, fp in enumerate(pred_hard_fps):
            if sm[fp] > 0:
                consumed_idx.add(i)
                sm[fp] -= 1
        remaining_soft_fps = [pred_soft_fps[i] for i in range(len(pred)) if i not in consumed_idx]
        gold_lenient_counter = Counter(gold_lenient)
        remaining_soft_counter = Counter(remaining_soft_fps)
        lenient_matches = remaining_soft_counter & gold_lenient_counter

        matched_strict = sum(strict_matches.values())
        matched_lenient = sum(lenient_matches.values())
        matched_total = matched_strict + matched_lenient
        precision = (matched_total / n_pred) if n_pred else 0.0
        recall = (matched_total / n_gold) if n_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": matched_total,
            "pred_count": n_pred,
            "gold_count": n_gold,
            "skipped_cond_check": sum(gold_skips),
            "matched_strict": matched_strict,
            "matched_lenient": matched_lenient,
        }

    # Soft mode (default branch).
    matched = sum((Counter(pred_fps) & Counter(gold_fps)).values())
    precision = (matched / n_pred) if n_pred else 0.0
    recall = (matched / n_gold) if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "pred_count": n_pred,
        "gold_count": n_gold,
        "skipped_cond_check": 0,
    }


def soft_match_f1(record: dict, gold: dict) -> dict:
    """ChemEagle soft-match: per-reaction binary equality on
    `(reactants, products)` frozensets. Returns precision / recall / F1."""
    return _match_f1(record, gold, hard=False)


def hard_match_f1(record: dict, gold: dict, *,
                  skip_empty_gold_conditions: bool = True) -> dict:
    """ChemEagle hard-match: soft-match plus condition signatures.

    `skip_empty_gold_conditions=True` (the ChemEagle-compatible default)
    collapses hard match to soft match for any gold reaction that has zero
    conditions. The OpenChemIE subset's gold has no conditions; without
    this convention, hard-F1 on that subset would always be 0.
    """
    return _match_f1(record, gold, hard=True,
                     skip_empty_gold_conditions=skip_empty_gold_conditions)


# ---------- Partial-credit auxiliary metrics ----------
#
# These complement the strict ChemEagle soft/hard-F1 above (which are
# all-or-nothing per the paper). They give partial credit so we can tell
# "very close, one reactant missing" from "completely off". Diagnostic only —
# don't use them as the headline number.

def _stereo_stripped_fingerprint(reaction: dict) -> tuple:
    """Like `_soft_fingerprint` but with stereochemistry removed from every SMILES."""
    reactants = frozenset(
        c for c in (_canonical_no_stereo(x.get("smiles")) for x in (reaction.get("reactants") or [])) if c
    )
    products = frozenset(
        c for c in (_canonical_no_stereo(x.get("smiles")) for x in (reaction.get("products") or [])) if c
    )
    return (reactants, products)


def constitution_match_f1(record: dict, gold: dict) -> dict:
    """Soft-match with stereochemistry stripped from all SMILES.

    Use to diagnose: when this is significantly higher than soft_match_f1,
    extractions are getting molecular constitution right but missing stereo.
    """
    pred = record.get("reactions") or []
    gold_rxns = gold.get("reactions") or []
    n_pred, n_gold = len(pred), len(gold_rxns)
    if n_pred == 0 and n_gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0,
                "pred_count": 0, "gold_count": 0}
    pred_fps = [_stereo_stripped_fingerprint(r) for r in pred]
    gold_fps = [_stereo_stripped_fingerprint(r) for r in gold_rxns]
    matched = sum((Counter(pred_fps) & Counter(gold_fps)).values())
    precision = (matched / n_pred) if n_pred else 0.0
    recall = (matched / n_gold) if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": matched, "pred_count": n_pred, "gold_count": n_gold}


def _jaccard_reaction(pred_fp: tuple, gold_fp: tuple) -> float:
    """Jaccard overlap on (reactants ∪ products) sets, treated as one bag of SMILES."""
    p_all = pred_fp[0] | pred_fp[1]
    g_all = gold_fp[0] | gold_fp[1]
    if not p_all and not g_all:
        return 1.0
    inter = len(p_all & g_all)
    union = len(p_all | g_all)
    return inter / union if union else 0.0


def partial_match_f1(record: dict, gold: dict, *, threshold: float = 0.5,
                     use_stereo: bool = False) -> dict:
    """Hungarian-aligned per-reaction Jaccard match.

    For each pred-gold reaction pair, compute Jaccard overlap on the union of
    reactant + product canonical SMILES. A pair counts as a match when Jaccard
    >= `threshold`. Final F1 uses optimal assignment (so each reaction matches
    at most once on each side), unlike `_match_f1` which uses multiset equality.

    `use_stereo=False` strips stereochemistry before canonicalising — most
    "very close" misses are stereo-only, so this gives the most informative
    partial-credit signal.

    Returns the F1 plus the mean Jaccard across the optimal assignment, which
    is itself a continuous similarity score (0=disjoint, 1=identical).
    """
    pred = record.get("reactions") or []
    gold_rxns = gold.get("reactions") or []
    n_pred, n_gold = len(pred), len(gold_rxns)
    if n_pred == 0 and n_gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0,
                "pred_count": 0, "gold_count": 0, "mean_jaccard": 1.0,
                "threshold": threshold}
    fp_fn = _stereo_stripped_fingerprint if not use_stereo else _soft_fingerprint
    pred_fps = [fp_fn(r) for r in pred]
    gold_fps = [fp_fn(r) for r in gold_rxns]
    n = max(n_pred, n_gold)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n_pred):
        for j in range(n_gold):
            sim[i, j] = _jaccard_reaction(pred_fps[i], gold_fps[j])
    # Hungarian minimises cost; convert similarity to cost.
    row_ind, col_ind = linear_sum_assignment(-sim)
    matched_jaccards: list[float] = []
    matched = 0
    for i, j in zip(row_ind, col_ind):
        if i < n_pred and j < n_gold:
            matched_jaccards.append(float(sim[i, j]))
            if sim[i, j] >= threshold:
                matched += 1
    mean_jaccard = (sum(matched_jaccards) / len(matched_jaccards)) if matched_jaccards else 0.0
    precision = (matched / n_pred) if n_pred else 0.0
    recall = (matched / n_gold) if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched,
            "pred_count": n_pred, "gold_count": n_gold,
            "mean_jaccard": mean_jaccard, "threshold": threshold}


# ---------- Graph Edit Distance (ChemEagle-style) ----------

def _mol_size(mol) -> int:
    """Topological size of a molecule = #atoms + #bonds. Used as the
    'unmatched penalty' per the ChemEagle paper."""
    if mol is None:
        return 0
    try:
        return int(mol.GetNumAtoms() + mol.GetNumBonds())
    except Exception:
        return 0


def _ged_smiles_pair(s1: str | None, s2: str | None, *, mcs_timeout: int = 5) -> int:
    """MCS-derived GED between two SMILES strings.

    Approximated as: |G_pred ∪ G_gold| − |MCS|, where the size operator is
    atoms + bonds. This is the number of node/edge insertions+deletions
    required to transform one molecule into the other when bond/atom
    substitutions are not modelled separately.

    Returns the topological size of the surviving molecule when one input
    is missing (None or unparseable) — that's the "unmatched penalty"
    convention from the paper.
    """
    m1 = Chem.MolFromSmiles(s1) if isinstance(s1, str) and s1 else None
    m2 = Chem.MolFromSmiles(s2) if isinstance(s2, str) and s2 else None
    if m1 is None and m2 is None:
        return 0
    if m1 is None:
        return _mol_size(m2)
    if m2 is None:
        return _mol_size(m1)
    if Chem.MolToSmiles(m1, canonical=True) == Chem.MolToSmiles(m2, canonical=True):
        return 0
    try:
        mcs = rdFMCS.FindMCS([m1, m2], timeout=mcs_timeout)
    except Exception:
        return _mol_size(m1) + _mol_size(m2)
    if mcs.canceled or mcs.numAtoms == 0:
        return _mol_size(m1) + _mol_size(m2)
    common = mcs.numAtoms + mcs.numBonds
    return (_mol_size(m1) - common) + (_mol_size(m2) - common)


def _reaction_molecule_smiles(reaction: dict) -> list[str]:
    """All non-null SMILES from reactants + products of a reaction."""
    out = []
    for c in (reaction.get("reactants") or []) + (reaction.get("products") or []):
        s = c.get("smiles") if isinstance(c, dict) else None
        if s:
            out.append(s)
    return out


def _bipartite_min_cost(cost: np.ndarray) -> int:
    """Hungarian-aligned minimum total cost. Matrix must be square (pad with
    unmatched-penalty entries before calling)."""
    if cost.size == 0:
        return 0
    row, col = linear_sum_assignment(cost)
    return int(cost[row, col].sum())


def _reaction_pair_ged(pred_rxn: dict, gold_rxn: dict) -> int:
    """Total GED between the molecule sets of two reactions, after Hungarian
    alignment of pred molecules to gold molecules. Unmatched molecules
    (size mismatch) are penalised by their full topological size."""
    p_mols = _reaction_molecule_smiles(pred_rxn)
    g_mols = _reaction_molecule_smiles(gold_rxn)
    n_p, n_g = len(p_mols), len(g_mols)
    if n_p == 0 and n_g == 0:
        return 0
    p_sizes = [_mol_size(Chem.MolFromSmiles(s)) for s in p_mols]
    g_sizes = [_mol_size(Chem.MolFromSmiles(s)) for s in g_mols]
    n = max(n_p, n_g)
    cost = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i < n_p and j < n_g:
                cost[i, j] = _ged_smiles_pair(p_mols[i], g_mols[j])
            elif i < n_p:                # j is padding — pred unmatched
                cost[i, j] = p_sizes[i]
            elif j < n_g:                # i is padding — gold unmatched
                cost[i, j] = g_sizes[j]
    return _bipartite_min_cost(cost)


def graph_edit_distance(record: dict, gold: dict) -> dict:
    """ChemEagle's average per-reaction Graph Edit Distance.

    Two-level Hungarian alignment:
      1. For each (pred_reaction, gold_reaction) candidate pair, align their
         molecule sets via Hungarian on a pairwise GED matrix; this gives the
         per-reaction-pair GED.
      2. Build a reaction-level pairwise GED matrix; align reactions via
         Hungarian; sum the costs.

    Unmatched reactions (when |pred| ≠ |gold|) cost the sum of their
    constituent molecule sizes — same penalty convention as molecules.

    Lower is better. Reported as `avg_ged` (total / max(n_pred, n_gold)).
    """
    pred = record.get("reactions") or []
    gold_rxns = gold.get("reactions") or []
    n_p, n_g = len(pred), len(gold_rxns)
    if n_p == 0 and n_g == 0:
        return {"avg_ged": 0.0, "total_ged": 0, "count": 0}

    def _rxn_size(r: dict) -> int:
        return sum(_mol_size(Chem.MolFromSmiles(s)) for s in _reaction_molecule_smiles(r))

    p_sizes = [_rxn_size(r) for r in pred]
    g_sizes = [_rxn_size(r) for r in gold_rxns]
    n = max(n_p, n_g)
    cost = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i < n_p and j < n_g:
                cost[i, j] = _reaction_pair_ged(pred[i], gold_rxns[j])
            elif i < n_p:
                cost[i, j] = p_sizes[i]
            elif j < n_g:
                cost[i, j] = g_sizes[j]
    total = _bipartite_min_cost(cost)
    return {"avg_ged": total / n, "total_ged": total, "count": n}


# ---------- top-level ----------

def evaluate(record: dict, gold: dict | None = None) -> dict:
    """Run every metric. Reference-based metrics omitted if gold is None."""
    out: dict = {
        "schema_conformance": schema_conformance(record),
        "smiles_validity_rate": smiles_validity_rate(record),
        "role_enum_compliance": role_enum_compliance(record),
        "reaction_count": reaction_count(record),
    }
    if gold is not None:
        out["reactant_iou"] = reactant_iou(record, gold)
        out["product_iou"] = product_iou(record, gold)
        out["condition_coverage"] = condition_coverage(record, gold)
        out["soft_match"] = soft_match_f1(record, gold)
        out["hard_match"] = hard_match_f1(record, gold)
        out["constitution_match"] = constitution_match_f1(record, gold)
        out["partial_match"] = partial_match_f1(record, gold, threshold=0.5, use_stereo=False)
        out["graph_edit_distance"] = graph_edit_distance(record, gold)
        out["reaction_count_gold"] = reaction_count(gold)
    return out


def summarise(metrics: dict) -> str:
    """Pretty one-screen summary table."""
    lines = []
    sc = metrics["schema_conformance"]
    sc_status = "PASS" if sc["valid"] else f"FAIL ({len(sc['errors'])} errors)"
    lines.append(f"  schema_conformance         : {sc_status}")
    sv = metrics["smiles_validity_rate"]
    lines.append(f"  smiles_validity_rate       : {sv['valid']}/{sv['total']} ({sv['rate']*100:5.1f}%)")
    re_ = metrics["role_enum_compliance"]
    lines.append(f"  role_enum_compliance       : {re_['legal']}/{re_['total']} ({re_['rate']*100:5.1f}%)")
    if re_["illegal_values"]:
        lines.append(f"    illegal_values           : {re_['illegal_values']}")
    rc = metrics["reaction_count"]
    suffix = ""
    if "reaction_count_gold" in metrics:
        suffix = f"  (gold: {metrics['reaction_count_gold']['count']})"
    lines.append(f"  reaction_count             : {rc['count']}{suffix}")
    if "reactant_iou" in metrics:
        ri = metrics["reactant_iou"]
        lines.append(f"  reactant_iou               : {ri['iou']:.3f}  ({ri['intersection']}/{ri['union']})")
        if ri["missing"]:
            lines.append(f"    missing                  : {ri['missing']}")
        if ri["extra"]:
            lines.append(f"    extra                    : {ri['extra']}")
    if "product_iou" in metrics:
        pi = metrics["product_iou"]
        lines.append(f"  product_iou                : {pi['iou']:.3f}  ({pi['intersection']}/{pi['union']})")
        if pi["missing"]:
            lines.append(f"    missing                  : {pi['missing']}")
        if pi["extra"]:
            lines.append(f"    extra                    : {pi['extra']}")
    if "condition_coverage" in metrics:
        cc = metrics["condition_coverage"]
        lines.append(f"  condition_coverage         : recall={cc['recall']:.3f} precision={cc['precision']:.3f}  ({cc['matched']}/{cc['gold_size']} gold)")
        if cc["missing"]:
            lines.append(f"    missing                  : {cc['missing'][:5]}{' ...' if len(cc['missing'])>5 else ''}")
    if "soft_match" in metrics:
        sm = metrics["soft_match"]
        lines.append(f"  soft_match (ChemEagle)     : F1={sm['f1']:.3f}  P={sm['precision']:.3f}  R={sm['recall']:.3f}  ({sm['matched']}/{sm['gold_count']} gold reactions)")
    if "hard_match" in metrics:
        hm = metrics["hard_match"]
        lines.append(f"  hard_match (ChemEagle)     : F1={hm['f1']:.3f}  P={hm['precision']:.3f}  R={hm['recall']:.3f}  ({hm['matched']}/{hm['gold_count']} gold reactions)")
    if "constitution_match" in metrics:
        cm = metrics["constitution_match"]
        lines.append(f"  constitution_match (no stereo): F1={cm['f1']:.3f}  ({cm['matched']}/{cm['gold_count']} gold reactions)")
    if "partial_match" in metrics:
        pm = metrics["partial_match"]
        lines.append(f"  partial_match (Jaccard>={pm['threshold']:.1f}) : F1={pm['f1']:.3f}  mean_jaccard={pm['mean_jaccard']:.3f}  ({pm['matched']}/{pm['gold_count']} gold reactions)")
    if "graph_edit_distance" in metrics:
        ged = metrics["graph_edit_distance"]
        lines.append(f"  GED (avg per reaction)     : {ged['avg_ged']:.2f}  (total={ged['total_ged']}, n={ged['count']})")
    return "\n".join(lines)
