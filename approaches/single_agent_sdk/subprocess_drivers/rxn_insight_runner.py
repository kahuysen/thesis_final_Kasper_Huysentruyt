"""Rxn-INSIGHT subprocess runner.

Runs inside `.venv-rxn-insight/` (Python 3.12 + numpy 1.x + scipy 1.14).
Reads JSON on stdin, writes JSON on stdout. Never imported from the main venv.

Input shape (single line of JSON):
    {"reactions": [
      {"entry_id": "1", "rxn_smiles": "A.B>>C",
       "solvent": "...", "reagent": "...", "catalyst": "...",
       "yield": 99, "ref": "..."}
    ]}

Output shape:
    {"results": [
      {"entry_id": "1", "ok": true,  "row": {<all Section 5.9 columns>}},
      {"entry_id": "2", "ok": false, "error": "..."}
    ]}

We invoke `Database.create_database_from_df` per reaction so the
entry_id mapping is unambiguous even when Rxn-INSIGHT skips a reaction
(its analyzer drops malformed entries silently).
"""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

# Some Rxn-INSIGHT submodules (notably .database) print install hints to
# stdout at import time. Reroute stdout to stderr while importing.
_real_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    import pandas as pd
    # Force the legacy object dtype — pandas' new pyarrow-backed string
    # columns reject the int writes Rxn-INSIGHT does mid-flight.
    pd.options.future.infer_string = False
    from rxn_insight.database import Database
finally:
    sys.stdout = _real_stdout


# Columns produced by create_database_from_df (Section 5.9, Table 5.3),
# plus the input passthroughs (SOLVENT/REAGENT/CATALYST/YIELD/REF).
EXPECTED_COLUMNS = [
    "REACTION", "REACTANTS", "PRODUCTS", "SANITIZED_REACTION", "MAPPED_REACTION",
    "N_REACTANTS", "N_PRODUCTS",
    "CLASS", "NAME",
    "FG_REACTANTS", "FG_PRODUCTS",
    "PARTICIPATING_RINGS_REACTANTS", "PARTICIPATING_RINGS_PRODUCTS",
    "ALL_RINGS_PRODUCTS", "BY-PRODUCTS",
    "TEMPLATE", "TEMPLATE_NR",
    "TAG", "TAG2", "SCAFFOLD",
    "rxn_str_patt_fp", "rxn_dif_patt_fp",
    "rxn_str_morgan_fp", "rxn_dif_morgan_fp",
    "SOLVENT", "REAGENT", "CATALYST", "YIELD", "REF",
]


def _to_jsonable(v):
    """Convert pandas/numpy scalars and array-likes to JSON-friendly values."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "tolist"):  # numpy scalar / ndarray
        return v.tolist()
    if isinstance(v, (list, tuple, set)):
        return [_to_jsonable(x) for x in v]
    # pandas NA, Timestamp, etc. fall through to str()
    return str(v)


def _row_to_dict(row) -> dict:
    """Pull only the documented output columns out of a result row."""
    out: dict = {}
    for col in EXPECTED_COLUMNS:
        if col in row.index:
            out[col] = _to_jsonable(row[col])
        else:
            out[col] = None
    return out


def analyze_one(item: dict) -> dict:
    """Run Rxn-INSIGHT on a single reaction. Always returns a JSON-safe dict."""
    entry_id = item.get("entry_id")
    smi = (item.get("rxn_smiles") or "").strip()
    if not smi or ">>" not in smi:
        return {"entry_id": entry_id, "ok": False, "error": "empty or malformed rxn_smiles"}

    df = pd.DataFrame({
        "REACTION": [smi],
        "SOLVENT":  [item.get("solvent")  or "not-reported"],
        "REAGENT":  [item.get("reagent")  or "not-reported"],
        "CATALYST": [item.get("catalyst") or "not-reported"],
        "YIELD":    [item.get("yield") if item.get("yield") is not None else "not-reported"],
        "REF":      [item.get("ref")      or "not-reported"],
    }).astype(object)

    try:
        db = Database()
        out = db.create_database_from_df(
            df,
            reaction_column="REACTION",
            classify=True,
            add_fp=True,
            n_jobs=1,
        )
    except Exception as exc:
        return {"entry_id": entry_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if out is None or len(out) == 0:
        return {"entry_id": entry_id, "ok": False, "error": "Rxn-INSIGHT skipped this reaction"}

    return {"entry_id": entry_id, "ok": True, "row": _row_to_dict(out.iloc[0])}


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        json.dump({"results": []}, sys.stdout)
        return
    payload = json.loads(raw)
    results = [analyze_one(item) for item in payload.get("reactions", [])]
    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()
