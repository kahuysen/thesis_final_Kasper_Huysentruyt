"""Deterministic R-group derivation and template-substitution via RDKit.

Two complementary functions:

- `derive_substituents(template, [concrete_smiles, ...])` — given a wildcard
  template (e.g. `*C(=O)n1nc(C(F)(F)F)nc1N`) and a list of concrete molecules
  that share that template, runs `RGroupDecompose` and returns the R-group
  fragment(s) per variant. Fragments are returned with their attachment-point
  markers `[*:1]` etc. preserved so they can be re-attached losslessly.

- `apply_substituents(template, r_assignments)` — given a template with
  `*` / `[*:N]` wildcards and an `{R1: <fragment>, R2: <fragment>}` dict from
  `derive_substituents`, fuses them via `Chem.molzip` and returns the
  canonical SMILES of the substituted molecule.

Round-tripping a concrete molecule through derive→apply with the same
template should be lossless. Substituting one variant's R into a *different*
template (e.g. the reactant template) gives the corresponding specific
reactant SMILES — this is the key trick for substrate-scope figures where
only products are drawn.
"""
from __future__ import annotations

import re

from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition

from .trace import trace


_MAP_RE = re.compile(r"\*\:?(\d+)")


def _ensure_marker(template_smiles: str) -> str:
    """If the template uses bare `*` instead of `[*:N]`, label them
    sequentially `[*:1]`, `[*:2]`, … so molzip can match them. If the
    template already has labelled wildcards, return it unchanged."""
    if not isinstance(template_smiles, str):
        return template_smiles
    if "[*" in template_smiles or "*:" in template_smiles:
        return template_smiles
    out = []
    counter = 1
    i = 0
    while i < len(template_smiles):
        ch = template_smiles[i]
        if ch == "*":
            out.append(f"[*:{counter}]")
            counter += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@trace()
def derive_substituents(template_smiles: str, concrete_smiles: list[str]) -> dict:
    """Decompose each concrete molecule against a wildcard template.

    Args:
        template_smiles: SMILES with `*` wildcards (or `[*:N]` labelled form).
        concrete_smiles: list of full SMILES that should share the template.

    Returns:
        dict with keys:
            results — list[dict] one per concrete entry:
                {smiles, matched, r_assignments: {R1: '<frag-with-[*:N]>', ...},
                 core, error}.
                The R fragments KEEP their `[*:N]` markers — pass them
                straight to `apply_substituents`.
            unmatched_indices, template, error.
    """
    if not isinstance(template_smiles, str) or not template_smiles.strip():
        return {"results": [], "unmatched_indices": [], "template": None,
                "error": "empty template"}
    if not isinstance(concrete_smiles, list) or not concrete_smiles:
        return {"results": [], "unmatched_indices": [], "template": None,
                "error": "empty concrete_smiles list"}

    core = Chem.MolFromSmiles(template_smiles)
    if core is None:
        return {"results": [], "unmatched_indices": [], "template": None,
                "error": f"template did not parse: {template_smiles!r}"}
    template_canonical = Chem.MolToSmiles(core, canonical=True)

    mols = []
    for s in concrete_smiles:
        mols.append(Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    valid_mols = [m for m in mols if m is not None]
    if not valid_mols:
        return {"results": [{"smiles": s, "matched": False, "r_assignments": {},
                              "core": None, "error": "did not parse"} for s in concrete_smiles],
                "unmatched_indices": list(range(len(concrete_smiles))),
                "template": template_canonical, "error": None}

    try:
        decomp, unmatched = rdRGroupDecomposition.RGroupDecompose(
            [core], valid_mols, asSmiles=True
        )
    except Exception as e:
        return {"results": [], "unmatched_indices": [], "template": template_canonical,
                "error": f"RGroupDecompose failed: {type(e).__name__}: {e}"}

    unmatched_set = set(unmatched)
    results = []
    valid_idx = 0
    decomp_iter = iter(decomp)
    for s, m in zip(concrete_smiles, mols):
        if m is None:
            results.append({"smiles": s, "matched": False, "r_assignments": {},
                            "core": None, "error": "did not parse"})
            continue
        if valid_idx in unmatched_set:
            results.append({"smiles": s, "matched": False, "r_assignments": {},
                            "core": None, "error": "did not match template core"})
            valid_idx += 1
            continue
        d = next(decomp_iter, None)
        if d is None:
            results.append({"smiles": s, "matched": False, "r_assignments": {},
                            "core": None, "error": "missing decomposition entry"})
            valid_idx += 1
            continue
        r_assignments = {k: v for k, v in d.items() if k != "Core" and isinstance(v, str)}
        results.append({
            "smiles": s,
            "matched": True,
            "r_assignments": r_assignments,
            "core": d.get("Core", ""),
            "error": None,
        })
        valid_idx += 1

    return {
        "results": results,
        "unmatched_indices": [i for i, r in enumerate(results) if not r["matched"]],
        "template": template_canonical,
        "error": None,
    }


@trace()
def apply_substituents(template_smiles: str, r_assignments: dict) -> dict:
    """Fuse a template with R-group fragments via `Chem.molzip`.

    Args:
        template_smiles: SMILES with wildcards (`*` or `[*:N]`).
        r_assignments: dict like `{R1: '<frag-with-[*:1]>', R2: '<frag-with-[*:2]>'}`
            as returned by `derive_substituents`. Map numbers MUST match the
            template's wildcards.

    Returns:
        {smiles: <canonical SMILES of fused molecule>, valid: bool, error: str|None}.
    """
    if not isinstance(template_smiles, str) or not template_smiles.strip():
        return {"smiles": None, "valid": False, "error": "empty template"}
    template_marked = _ensure_marker(template_smiles)
    template_mol = Chem.MolFromSmiles(template_marked)
    if template_mol is None:
        return {"smiles": None, "valid": False, "error": f"template did not parse: {template_smiles!r}"}
    if not isinstance(r_assignments, dict) or not r_assignments:
        return {"smiles": None, "valid": False, "error": "empty r_assignments"}

    fragments = []
    for key, frag_smiles in r_assignments.items():
        if not isinstance(frag_smiles, str):
            continue
        m = Chem.MolFromSmiles(frag_smiles)
        if m is None:
            return {"smiles": None, "valid": False, "error": f"R-fragment did not parse: {key}={frag_smiles!r}"}
        fragments.append(m)

    if not fragments:
        return {"smiles": None, "valid": False, "error": "no parseable R fragments"}

    combo = template_mol
    for frag in fragments:
        combo = Chem.CombineMols(combo, frag)
    try:
        zipped = Chem.molzip(combo)
    except Exception as e:
        return {"smiles": None, "valid": False, "error": f"molzip failed: {type(e).__name__}: {e}"}
    try:
        out = Chem.MolToSmiles(zipped, canonical=True)
    except Exception as e:
        return {"smiles": None, "valid": False, "error": f"canonicalisation failed: {type(e).__name__}: {e}"}
    return {"smiles": out, "valid": True, "error": None}
