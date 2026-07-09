"""Canonical schema for the chemistry-extraction pipeline.

Single source of truth: imported by tools, prompts (via JSON-schema dump),
the validation/retry loop in main.py, and the evaluation harness.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _reject_attachment_markers(v: str | None) -> str | None:
    """Forbid `[*:N]` (atom-map-numbered wildcard) in final-output SMILES.

    `[*:N]` is RDKit's marker for an attachment point on an R-group fragment
    returned by `derive_substituents`. Such fragments must always be passed
    through `apply_substituents` (Chem.molzip) to be fused with a template
    before they appear in the final ReactionRecord. If we see one in the
    output it means the agent skipped that step and dumped the raw fragment
    as a final SMILES — that's wrong.
    """
    if v and isinstance(v, str) and "[*:" in v:
        raise ValueError(
            f"SMILES contains an attachment-point marker `[*:N]` ({v!r}). "
            "This looks like a raw R-fragment from derive_substituents — "
            "you forgot to feed it through apply_substituents(template, r_assignments) "
            "to fuse it into the reactant or product template before emitting JSON."
        )
    return v


ConditionRole = Literal[
    "reagent",
    "solvent",
    "catalyst",
    "temperature",
    "time",
    "yield",
    "atmosphere",
    "loading",
    "ee",
    "dr",
    "pressure",
]

CONDITION_ROLES: tuple[str, ...] = (
    "reagent",
    "solvent",
    "catalyst",
    "temperature",
    "time",
    "yield",
    "atmosphere",
    "loading",
    "ee",
    "dr",
    "pressure",
)


BBox = tuple[int, int, int, int]
"""Pixel-coordinate bounding box on the source image, in xyxy order
(x1, y1, x2, y2). Origin = top-left. Optional; populated by upstream
agents when they used MolDetect / OCR regions to locate this entity.
The web UI uses these to draw overlays over the source image."""


class Compound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: str | None = None
    label: str | None = None
    uncertain: bool = False
    bbox: BBox | None = None

    _check_smiles = field_validator("smiles")(_reject_attachment_markers)


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ConditionRole
    text: str
    smiles: str | None = None
    bbox: BBox | None = None

    _check_smiles = field_validator("smiles")(_reject_attachment_markers)


class Reaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reaction_id: str = Field(pattern=r"^\d+_\d+$")
    reactants: list[Compound]
    products: list[Compound]
    conditions: list[Condition] = Field(default_factory=list)
    additional_info: list[str] = Field(default_factory=list)


class ReactionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    reactions: list[Reaction]


def schema_json(indent: int = 2) -> str:
    """JSON-Schema string for prompt injection."""
    return json.dumps(ReactionRecord.model_json_schema(), indent=indent)


def condition_role_enum() -> str:
    """Comma-separated condition role enum for prompt injection."""
    return ", ".join(CONDITION_ROLES)
