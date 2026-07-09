"""Schema validation tool — wraps the canonical pydantic schema."""
from __future__ import annotations

from pydantic import ValidationError

from schema import ReactionRecord

from .trace import trace


@trace()
def validate_reaction_json(obj: dict) -> dict:
    """Validate a candidate ReactionRecord against the canonical schema.

    Args:
        obj: the JSON-decoded record (already a dict, not a string).

    Returns:
        {valid: bool, errors: list[{loc, msg}]}.
    """
    if not isinstance(obj, dict):
        return {"valid": False, "errors": [{"loc": [], "msg": f"expected dict, got {type(obj).__name__}"}]}
    try:
        ReactionRecord.model_validate(obj)
        return {"valid": True, "errors": []}
    except ValidationError as e:
        errs = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
        return {"valid": False, "errors": errs}
