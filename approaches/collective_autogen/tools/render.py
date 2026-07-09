"""Render SMILES strings to a labeled grid PNG with RDKit.

Used by the structural_verifier agent to compare predicted product SMILES
visually against the original reaction image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import Draw

from .trace import trace


@trace("render_smiles_grid")
def render_smiles_grid(
    items: Sequence[dict],
    out_path: str,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (300, 300),
) -> dict:
    """Render `[{smiles, label}, ...]` to a labeled grid PNG.

    SMILES that fail RDKit parsing are dropped and reported under "skipped";
    the call still succeeds as long as at least one entry rendered.
    """
    rendered: list[dict] = []
    skipped: list[dict] = []
    mols: list = []
    legends: list[str] = []
    for it in items:
        smi = (it.get("smiles") or "").strip()
        label = str(it.get("label") or "")
        if not smi:
            skipped.append({"label": label, "reason": "empty smiles"})
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped.append({"label": label, "smiles": smi, "reason": "rdkit parse failed"})
            continue
        mols.append(mol)
        legends.append(label or smi[:24])
        rendered.append({"label": label, "smiles": smi})

    if not mols:
        return {
            "image_path": None,
            "rendered": [],
            "skipped": skipped,
            "error": "no parseable SMILES to render",
        }

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=False,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out), format="PNG")
    return {"image_path": str(out), "rendered": rendered, "skipped": skipped}
