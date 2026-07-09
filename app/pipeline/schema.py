"""Schema for extracted reactions. The agent's `submit_extraction` tool
takes one of these as its argument, which guarantees structured output."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Species(BaseModel):
    label: Optional[str] = Field(
        None, description="Compound label as drawn (e.g. '1a', '8b'). Null if none."
    )
    name: Optional[str] = Field(
        None, description="Common or IUPAC name if obvious. Optional."
    )
    smiles: str = Field(description="SMILES string. Use '' if a generic R-group placeholder.")
    role_note: Optional[str] = Field(
        None,
        description="Optional clarifier (e.g. 'TBS-protected variant', 'tautomer 8b').",
    )


class Reactant(Species):
    equiv: Optional[str] = Field(None, description="Equivalents/loading as written in figure.")


class Product(Species):
    yield_pct: Optional[float] = Field(None, description="Isolated yield in percent. Null if N/A or trace.")
    yield_note: Optional[str] = Field(
        None, description="Free-text yield qualifier ('trace', 'NR', 'ND', '65% on 5 mmol scale')."
    )


class Reagent(Species):
    loading: Optional[str] = Field(None, description="Loading as written ('20 mol%', '2 equiv').")
    role: Optional[str] = Field(None, description="catalyst | reductant | additive | solvent | trap | oxidant | base | other")


class Conditions(BaseModel):
    temperature: Optional[str] = Field(None, description="As written (e.g. '75 °C', 'rt').")
    time: Optional[str] = Field(None, description="As written (e.g. '1 h', '24 h').")
    atmosphere: Optional[str] = Field(None, description="'air' | 'Ar' | 'N2' | etc., if specified.")
    solvent: Optional[str] = Field(None, description="Reaction solvent and amount/concentration.")
    other: Optional[str] = Field(None, description="Anything else (irradiation, additive timing, etc.).")


class Reaction(BaseModel):
    entry_id: str = Field(description="Identifier for this reaction within the figure (e.g. 'I', '1', '8b').")
    title: Optional[str] = Field(None, description="Short caption for this row/equation.")
    reactants: list[Reactant]
    reagents: list[Reagent] = Field(default_factory=list)
    conditions: Conditions
    products: list[Product]
    notes: Optional[str] = Field(None, description="Mechanistic remark, control-experiment purpose, etc.")


class FigureExtraction(BaseModel):
    figure_caption: Optional[str] = Field(None, description="Figure caption text if visible.")
    reactions: list[Reaction]
    intermediates: list[Species] = Field(
        default_factory=list,
        description="Bracketed/intermediate compounds drawn outside the main scheme.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Anything the model wants to flag for the human reviewer (low-confidence call, ambiguous subscript, etc.).",
    )
