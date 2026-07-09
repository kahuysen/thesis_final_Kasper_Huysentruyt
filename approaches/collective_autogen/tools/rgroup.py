"""Programmatic R-group enumeration."""
from __future__ import annotations

import re

from rdkit import Chem

from .trace import trace

_R_PATTERN = re.compile(r"\*")
_NAMED_R = re.compile(r"\[R(\d+)\]|R(\d+)")


def _apply_substitution(template: str, subs: dict[str, str]) -> str:
    """Replace `*` (or `[R1]`, `R1`) tokens in a template SMILES with provided fragments.

    Strategy:
    - If the template contains named R-groups (`[R1]`, `R2`), replace each by name
      using the fragment SMILES from `subs` (key is "R1" / "R2" / ...).
    - Else, replace successive `*` tokens with subs["*1"], subs["*2"], ... (or
      a single `*` with subs["R"] / subs["*"] for convenience).
    """
    out = template
    if _NAMED_R.search(out):
        # Replace [R1] / R1 forms.
        def repl(m: re.Match) -> str:
            idx = m.group(1) or m.group(2)
            key = f"R{idx}"
            if key in subs:
                return subs[key]
            return m.group(0)

        return _NAMED_R.sub(repl, out)

    star_keys_in_order = sorted(
        [k for k in subs if k.startswith("*") or k.upper().startswith("R") or k == "*"],
        key=lambda k: (0 if k == "*" else int("".join(c for c in k if c.isdigit()) or 0)),
    )
    star_count = out.count("*")
    if star_count == 1 and len(subs) == 1:
        # Single wildcard, single substitution — accept any key.
        only = next(iter(subs.values()))
        return out.replace("*", only, 1)
    # Multiple wildcards: replace left-to-right using *1, *2, ... or R1, R2, ...
    result = out
    for i in range(1, star_count + 1):
        key_candidates = [f"*{i}", f"R{i}"]
        replacement = next((subs[k] for k in key_candidates if k in subs), None)
        if replacement is None:
            break
        result = result.replace("*", replacement, 1)
    return result


@trace()
def enumerate_rgroups(template_smiles: str, substitutions: list[dict]) -> list[dict]:
    """Enumerate all R-group variants by substituting wildcards in a template SMILES.

    Args:
        template_smiles: SMILES with `*` wildcards (or `[R1]`, `R1` forms).
        substitutions: list of dicts. Each dict represents one variant. It must
            contain a "label" key plus one substitution per R-group, e.g.
            {"label": "3a", "R1": "C", "R2": "OC"}. For a single `*`, you may
            also use {"label": "3a", "R": "C"}.

    Returns:
        list of dicts: {label, r_assignments, smiles, canonical_smiles, valid, error}.
    """
    out: list[dict] = []
    for entry in substitutions:
        if not isinstance(entry, dict):
            out.append({"label": None, "r_assignments": {}, "smiles": None, "canonical_smiles": None, "valid": False, "error": "entry is not a dict"})
            continue
        label = entry.get("label")
        r_assignments = {k: v for k, v in entry.items() if k != "label"}
        try:
            substituted = _apply_substitution(template_smiles, r_assignments)
        except Exception as e:
            out.append({"label": label, "r_assignments": r_assignments, "smiles": None, "canonical_smiles": None, "valid": False, "error": f"substitution failed: {e}"})
            continue
        mol = Chem.MolFromSmiles(substituted)
        if mol is None:
            out.append({"label": label, "r_assignments": r_assignments, "smiles": substituted, "canonical_smiles": None, "valid": False, "error": "RDKit could not parse substituted SMILES"})
            continue
        out.append({
            "label": label,
            "r_assignments": r_assignments,
            "smiles": substituted,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "valid": True,
            "error": None,
        })
    return out
