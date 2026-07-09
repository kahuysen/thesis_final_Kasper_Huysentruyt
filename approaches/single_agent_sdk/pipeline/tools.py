"""RDKit-backed validation tools the agent can call mid-loop.

Each function returns a JSON-serializable dict. Errors are caught and
returned as `{"ok": False, "error": ...}` so the agent sees them and can
self-correct rather than the loop crashing.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def validate_smiles(smiles: str) -> dict:
    """Parse a SMILES with RDKit; return canonical form and basic descriptors."""
    if not smiles:
        return {"ok": False, "error": "empty SMILES string"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "error": f"RDKit could not parse SMILES: {smiles!r}"}
    return {
        "ok": True,
        "input": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


# Tool schemas exposed to the model. Hand-written rather than auto-derived
# so the descriptions are tuned for chemistry context.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "validate_smiles",
        "description": (
            "Parse a SMILES string with RDKit. Returns canonical SMILES, "
            "molecular formula, and exact (monoisotopic) mass. Use this on "
            "every structure you extract before submitting, to catch typos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string to validate."}
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "submit_extraction",
        "description": (
            "Call this exactly ONCE, after you have validated every structure. "
            "The argument is the final structured extraction. After calling "
            "this, stop — do not produce more text or tool calls."
        ),
        # The schema is patched in at runtime from the Pydantic model so it
        # stays in sync with schema.py.
        "input_schema": {},
    },
]


TOOL_DISPATCH = {
    "validate_smiles": validate_smiles,
}
