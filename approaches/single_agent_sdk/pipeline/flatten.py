"""Flatten FigureExtraction objects into a CSV for downstream analysis.

One row per (figure, reaction, product). Reactants and reagents are
concatenated into compact summary columns so the CSV stays tabular.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .schema import FigureExtraction

COLUMNS = [
    "source_image",
    "figure_caption",
    "entry_id",
    "reaction_title",
    "reactants",          # "label: SMILES; ..."
    "reagents",           # "label (loading, role): SMILES; ..."
    "solvent",
    "temperature",
    "time",
    "atmosphere",
    "product_label",
    "product_name",
    "product_smiles",
    "yield_pct",
    "yield_note",
    "notes",
]


def _summarize_species_list(items, kind: str) -> str:
    bits = []
    for s in items:
        head = s.label or s.name or "?"
        meta = []
        loading = getattr(s, "loading", None) or getattr(s, "equiv", None)
        role = getattr(s, "role", None)
        if loading:
            meta.append(loading)
        if role:
            meta.append(role)
        if meta:
            head += f" ({', '.join(meta)})"
        smi = s.smiles or ""
        bits.append(f"{head}: {smi}" if smi else head)
    return "; ".join(bits)


def figure_to_rows(fx: FigureExtraction, source_image: str) -> list[dict]:
    rows: list[dict] = []
    for rxn in fx.reactions:
        reactants_s = _summarize_species_list(rxn.reactants, "reactant")
        reagents_s = _summarize_species_list(rxn.reagents, "reagent")
        if not rxn.products:
            products_iter = [None]
        else:
            products_iter = rxn.products
        for prod in products_iter:
            rows.append(
                {
                    "source_image": source_image,
                    "figure_caption": fx.figure_caption or "",
                    "entry_id": rxn.entry_id,
                    "reaction_title": rxn.title or "",
                    "reactants": reactants_s,
                    "reagents": reagents_s,
                    "solvent": rxn.conditions.solvent or "",
                    "temperature": rxn.conditions.temperature or "",
                    "time": rxn.conditions.time or "",
                    "atmosphere": rxn.conditions.atmosphere or "",
                    "product_label": (prod.label if prod else "") or "",
                    "product_name": (prod.name if prod else "") or "",
                    "product_smiles": (prod.smiles if prod else "") or "",
                    "yield_pct": "" if (prod is None or prod.yield_pct is None) else prod.yield_pct,
                    "yield_note": (prod.yield_note if prod else "") or "",
                    "notes": rxn.notes or "",
                }
            )
    return rows


def write_csv(rows: Iterable[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path
