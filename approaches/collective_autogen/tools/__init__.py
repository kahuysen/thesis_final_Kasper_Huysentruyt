"""Tools exposed to agents.

Each tool is a typed callable returning a dict. They are wrapped with `trace`
so every invocation is appended to the active tool trace file (set per-run by
main.py via `trace.set_trace_path`).
"""
from .balance import atom_balance_check
from .coref_tool import detect_label_coref
from .derive_rgroups import apply_substituents, derive_substituents
from .image import crop_image_region
from .moldetect_tool import detect_molecules
from .molnextr_tool import predict_molecule_smiles
from .ocr import ocr_image
from .pdf_extract import extract_figures_from_pdf
from .pubchem import lookup_compound
from .render import render_smiles_grid
from .rgroup import enumerate_rgroups
from .schema_validate import validate_reaction_json
from .scripted_molrec import ScriptedMolRecAgent
from .smiles import canonicalize_smiles, validate_smiles
from .structural_verifier import StructuralVerifierAgent
from .trace import set_trace_path

__all__ = [
    "ScriptedMolRecAgent",
    "StructuralVerifierAgent",
    "apply_substituents",
    "atom_balance_check",
    "canonicalize_smiles",
    "crop_image_region",
    "derive_substituents",
    "detect_label_coref",
    "detect_molecules",
    "enumerate_rgroups",
    "extract_figures_from_pdf",
    "lookup_compound",
    "ocr_image",
    "predict_molecule_smiles",
    "render_smiles_grid",
    "set_trace_path",
    "validate_reaction_json",
    "validate_smiles",
]
