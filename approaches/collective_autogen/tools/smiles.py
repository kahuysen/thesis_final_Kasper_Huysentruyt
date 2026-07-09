"""SMILES canonicalisation and validation via RDKit."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

from .trace import trace

# Silence RDKit's noisy parser warnings; we surface them in the return dict.
RDLogger.DisableLog("rdApp.*")


@trace()
def canonicalize_smiles(smiles: str) -> dict:
    """Canonicalise a SMILES string and report validity + molecular formula.

    Args:
        smiles: SMILES string. May contain '*' wildcards for R-groups.

    Returns:
        dict with keys: canonical (str|None), valid (bool), formula (str|None),
        error (str|None).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {"canonical": None, "valid": False, "formula": None, "error": "empty input"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"canonical": None, "valid": False, "formula": None, "error": "RDKit could not parse SMILES"}
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return {"canonical": None, "valid": False, "formula": None, "error": f"canonicalisation failed: {e}"}
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = None
    return {"canonical": canonical, "valid": True, "formula": formula, "error": None}


@trace()
def validate_smiles(smiles: str) -> dict:
    """Quick yes/no validity check (lighter than canonicalize_smiles).

    Returns:
        dict with keys: valid (bool), error (str|None).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {"valid": False, "error": "empty input"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": "RDKit could not parse SMILES"}
    return {"valid": True, "error": None}
