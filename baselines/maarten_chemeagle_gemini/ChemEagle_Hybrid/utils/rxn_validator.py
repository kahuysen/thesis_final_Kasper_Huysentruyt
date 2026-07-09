"""Reaction-level SMILES validation + LLM-driven correction.

Flash often drops or garbles atoms inside SMILES. We catch those cases by:

  1. **Parse check** — every SMILES must be parseable by RDKit.
  2. **Atom conservation** — over parseable species, the multiset of heavy
     atoms on the product side must be a subset of (reactants ∪ reagents)
     ignoring leaving groups. We only flag gross imbalances (a heavy atom
     appears on the product side that isn't on the reactant side at all).

When issues are found, the reaction is sent to Gemini with the broken SMILES
plus the text context (labels, conditions, original molnextr symbols) and
asked for a corrected JSON. R-group placeholders ``[*]`` / ``[1*]`` / ``[2*]``
are preserved across the round-trip.

Rxn-INSIGHT is used (when importable) to tag the reaction class so the fixer
prompt can offer the LLM a concrete chemistry hint rather than free-form
guessing.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Optional

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False

_RXN_INSIGHT_READY = False
# Optional: set RXN_INSIGHT_PATH to the Gen-Rxn-INSIGHT/src checkout to enable
# reaction-class tagging in the validator. Unset/missing => validator runs
# without it; _try_import_rxn_insight returns False and callers skip that step.
_RXN_INSIGHT_PATH = os.environ.get("RXN_INSIGHT_PATH", "")


def _try_import_rxn_insight():
    global _RXN_INSIGHT_READY
    if _RXN_INSIGHT_READY:
        return True
    if not _RXN_INSIGHT_PATH:
        return False
    if _RXN_INSIGHT_PATH not in sys.path:
        sys.path.insert(0, _RXN_INSIGHT_PATH)
    try:
        from gen_rxn_insight.reaction import Reaction  # noqa: F401
        _RXN_INSIGHT_READY = True
        return True
    except Exception:
        return False


def _parse_ok(smi: Optional[str]) -> bool:
    if not _RDKIT or not smi or not isinstance(smi, str):
        return False
    return Chem.MolFromSmiles(smi) is not None


def _heavy_atom_counts(smi: str) -> Counter:
    """Counter of heavy-atom symbols ignoring wildcards ([*], [N*])."""
    c = Counter()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return c
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        c[atom.GetSymbol()] += 1
    return c


def _rxn_has_rgroup_placeholder(rxn) -> bool:
    for bucket in ("reactants", "products"):
        for mol in rxn.get(bucket) or []:
            smi = mol.get("smiles") if isinstance(mol, dict) else None
            if not smi:
                continue
            # `*` is always a wildcard atom in SMILES, so any occurrence —
            # bare (`*[C@H]...`), bracketed (`[*]`, `[1*]`), or ChemDraw-style
            # (`[R]`, `[R1]`, `[Ar]`) — means this reaction carries an
            # R-group placeholder and atom-conservation is not meaningful.
            if "*" in smi or "[R" in smi or "[Ar" in smi:
                return True
    return False


def reaction_issues(rxn: dict) -> dict:
    """Report parseability and atom-conservation issues. No mutation."""
    issues = {"unparseable": [], "imbalance": {}, "has_placeholder": False}
    if not isinstance(rxn, dict):
        return issues

    issues["has_placeholder"] = _rxn_has_rgroup_placeholder(rxn)

    for side in ("reactants", "products"):
        for i, mol in enumerate(rxn.get(side) or []):
            if not isinstance(mol, dict):
                continue
            smi = mol.get("smiles")
            orig = mol.get("smiles_original")
            if smi is None and orig:
                issues["unparseable"].append({"side": side, "index": i, "smiles": None, "smiles_original": orig})
            elif smi is None and (mol.get("symbols") or mol.get("label") or mol.get("category") == "[Mol]"):
                # Molecule slot but SMILES missing — still worth a repair attempt.
                issues["unparseable"].append({"side": side, "index": i, "smiles": None, "symbols": mol.get("symbols")})
            elif smi and not _parse_ok(smi):
                issues["unparseable"].append({"side": side, "index": i, "smiles": smi})

    # Atom conservation only meaningful without R-group placeholders.
    if not issues["has_placeholder"] and _RDKIT:
        r_atoms = Counter()
        for mol in rxn.get("reactants") or []:
            smi = mol.get("smiles") if isinstance(mol, dict) else None
            if smi and _parse_ok(smi):
                r_atoms += _heavy_atom_counts(smi)
        for cond in rxn.get("conditions") or []:
            smi = cond.get("smiles") if isinstance(cond, dict) else None
            if smi and _parse_ok(smi):
                r_atoms += _heavy_atom_counts(smi)
        p_atoms = Counter()
        for mol in rxn.get("products") or []:
            smi = mol.get("smiles") if isinstance(mol, dict) else None
            if smi and _parse_ok(smi):
                p_atoms += _heavy_atom_counts(smi)
        for atom, n_p in p_atoms.items():
            if n_p > r_atoms.get(atom, 0):
                issues["imbalance"][atom] = {"reactants": r_atoms.get(atom, 0), "products": n_p}
    return issues


def classify_reaction(rxn: dict) -> dict:
    """Rxn-INSIGHT class/name/by-products. Empty dict when unavailable."""
    if not _try_import_rxn_insight():
        return {}
    if _rxn_has_rgroup_placeholder(rxn):
        return {}  # Rxn-INSIGHT can't map placeholder-bearing SMILES.
    try:
        from gen_rxn_insight.reaction import Reaction
        reactant_smis = [m.get("smiles") for m in (rxn.get("reactants") or []) if isinstance(m, dict) and m.get("smiles")]
        product_smis = [m.get("smiles") for m in (rxn.get("products") or []) if isinstance(m, dict) and m.get("smiles")]
        if not reactant_smis or not product_smis:
            return {}
        smi = ".".join(reactant_smis) + ">>" + ".".join(product_smis)
        info = Reaction(smi).get_reaction_info()
        return {
            "class": info.get("CLASS"),
            "name": info.get("NAME"),
            "by_products": info.get("BY-PRODUCTS"),
        }
    except Exception:
        return {}


_FIX_PROMPT = (
    "You are a chemistry SMILES corrector. A reaction has been extracted from a chemical "
    "diagram, but one or more SMILES are invalid (won't parse in RDKit), null (graph-to-smiles "
    "failed), or atom-imbalanced (heavy atoms on the product side not accounted for on the "
    "reactant side).\n\n"
    "For each molecule you are given:\n"
    "  - `smiles` — the current value (may be null if the backend failed);\n"
    "  - `smiles_original_broken` — a previous broken attempt, if any;\n"
    "  - `symbols` — the ATOM LIST that molnextr detected in the image. These atoms are the\n"
    "    ground truth for what the molecule contains. Treat tokens like `[BocHN]`, `[CO2H]`,\n"
    "    `[CO2Me]`, `[Ph]`, `[Bn]`, `[Ac]`, `[Ts]` as standard chemical abbreviations and\n"
    "    expand them into the SMILES (BocHN = NC(=O)OC(C)(C)C; CO2H = C(=O)O; Ph = c1ccccc1;\n"
    "    Bn = Cc1ccccc1; Ac = C(=O)C; Ts = S(=O)(=O)c1ccc(C)cc1).\n\n"
    "Fix the SMILES so that:\n"
    "  - every SMILES parses in RDKit,\n"
    "  - atoms are conserved across the transformation,\n"
    "  - stereochemistry is preserved when present ([C@H], [C@@H], /, \\),\n"
    "  - R-group placeholders (`*`, `[1*]`, `[2*]`) are preserved when present,\n"
    "  - the reaction makes chemical sense given the conditions and class hint (if given).\n\n"
    "Return ONLY a JSON object of the shape:\n"
    "{\n"
    '  "reactants": [{"smiles": "..."}, ...],\n'
    '  "products":  [{"smiles": "..."}, ...]\n'
    "}\n"
    "Do not change the number of reactants or products. Preserve order."
)


def _build_fix_messages(rxn: dict, issues: dict, rxn_class: dict) -> list:
    context = {
        "reactants": [
            {
                "smiles": m.get("smiles"),
                "smiles_original_broken": m.get("smiles_original"),
                "symbols": m.get("symbols"),
                "label": m.get("label"),
            }
            for m in (rxn.get("reactants") or []) if isinstance(m, dict)
        ],
        "products": [
            {
                "smiles": m.get("smiles"),
                "smiles_original_broken": m.get("smiles_original"),
                "symbols": m.get("symbols"),
                "label": m.get("label"),
            }
            for m in (rxn.get("products") or []) if isinstance(m, dict)
        ],
        "conditions_text": [
            (c.get("text") or c.get("role")) for c in (rxn.get("conditions") or []) if isinstance(c, dict)
        ],
        "reported_issues": issues,
        "reaction_class_hint": rxn_class,
    }
    user_text = (
        "Reaction data:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\nReturn the corrected JSON only."
    )
    return [
        {"role": "system", "content": _FIX_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _apply_fix(rxn: dict, fix: dict) -> dict:
    """Overwrite SMILES fields in rxn from `fix` where shapes align."""
    if not isinstance(fix, dict):
        return rxn
    for side in ("reactants", "products"):
        orig = rxn.get(side) or []
        new = fix.get(side) or []
        if len(new) != len(orig):
            continue
        for i, entry in enumerate(new):
            if not isinstance(entry, dict):
                continue
            new_smi = entry.get("smiles")
            if new_smi and isinstance(orig[i], dict):
                orig[i]["smiles"] = new_smi
    return rxn


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Look for a fenced ```json block.
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    # Fallback: first {...} span.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def validate_and_fix(result: dict, client=None, model: Optional[str] = None, max_calls: int = 40) -> dict:
    """Run validator over every reaction; invoke LLM correction where needed.
    Mutates `result` in place and also returns it. When `client` is None we
    just report issues without calling the LLM."""
    if not isinstance(result, dict):
        return result
    reactions = result.get("reactions") or []
    calls = 0
    for rxn in reactions:
        if not isinstance(rxn, dict):
            continue
        issues = reaction_issues(rxn)
        needs_fix = bool(issues["unparseable"]) or bool(issues["imbalance"])
        if not needs_fix:
            continue
        if client is None or calls >= max_calls:
            rxn["_validation_issues"] = issues
            continue
        rxn_class = classify_reaction(rxn)
        messages = _build_fix_messages(rxn, issues, rxn_class)
        try:
            resp = client.chat.completions.create(
                model=model or os.environ.get("CHEMEAGLE_MODEL", "gemini-3-flash-preview"),
                messages=messages,
                temperature=0,
                extra_body={"think": False},
                response_format={"type": "json_object"},
            )
            calls += 1
            content = resp.choices[0].message.content
            fix = _extract_json(content)
            if fix:
                _apply_fix(rxn, fix)
                rxn["_validation_applied"] = {
                    "issues": issues,
                    "class": rxn_class,
                }
            else:
                rxn["_validation_issues"] = issues
        except Exception as e:
            rxn["_validation_error"] = f"{type(e).__name__}: {e}"
            rxn["_validation_issues"] = issues
    return result
