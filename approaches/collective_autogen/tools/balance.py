"""Atom-balance sanity check for a reaction (heavy-atom counts only)."""
from __future__ import annotations

from collections import Counter

from rdkit import Chem

from .trace import trace


def _atom_counts(smiles: str) -> Counter:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Counter()
    counts: Counter = Counter()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "*":
            continue
        if sym == "H":
            continue
        counts[sym] += 1
    return counts


@trace()
def atom_balance_check(reactants_smiles: list[str], products_smiles: list[str]) -> dict:
    """Check whether reactant and product heavy-atom counts match.

    Ignores hydrogens (often not made explicit in SMILES) and `*` wildcards
    (R-group placeholders). This is a sanity check, not a definitive balance —
    catalysts, solvents, and small leaving groups will trip it.

    Returns:
        {balanced: bool, reactants: dict, products: dict, missing_atoms: dict,
         error: str|None}.
    """
    if not isinstance(reactants_smiles, list) or not isinstance(products_smiles, list):
        return {"balanced": False, "reactants": {}, "products": {}, "missing_atoms": {}, "error": "inputs must be lists"}
    r_total: Counter = Counter()
    for s in reactants_smiles:
        r_total += _atom_counts(s)
    p_total: Counter = Counter()
    for s in products_smiles:
        p_total += _atom_counts(s)
    diff = Counter()
    for el in set(r_total) | set(p_total):
        delta = p_total.get(el, 0) - r_total.get(el, 0)
        if delta != 0:
            diff[el] = delta
    return {
        "balanced": len(diff) == 0,
        "reactants": dict(r_total),
        "products": dict(p_total),
        "missing_atoms": dict(diff),
        "error": None,
    }
