"""Wrapper around the Rxn-INSIGHT subprocess driver.

The actual analysis runs in `.venv-rxn-insight/` (Python 3.12, numpy 1.x,
scipy 1.14) — that env is intentionally separate from the main pipeline
so Rxn-INSIGHT's pinned deps don't conflict with our numpy 2 / Python 3.14.

The subprocess uses `Database.create_database_from_df()`, which produces
the full Section 5.9 column set (REACTANTS, PRODUCTS, MAPPED_REACTION,
TAG, TAG2, TEMPLATE, TEMPLATE_NR, the four fingerprints, etc.) — not just
the slim `Reaction.get_reaction_info()` payload.

Public API:
    build_rxn_smiles(reaction)          -> str | None
    analyze_extraction(extraction, ...) -> list[dict]
    extraction_to_insight_rows(...)     -> list[dict]   (CSV-flat)
    write_insight_csv(rows, path)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from rdkit import Chem
from rdkit import RDLogger

from .schema import FigureExtraction, Reaction, Species

RDLogger.DisableLog("rdApp.error")

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv-rxn-insight" / "bin" / "python3"
RUNNER_SCRIPT = ROOT / "subprocess_drivers" / "rxn_insight_runner.py"

# Database.create_database_from_df calls into rdchiral templates for every
# reaction; budget ~5 s/reaction worst-case for big tables.
TIMEOUT_S = 180


# ---------- SMILES helpers ----------

def _is_parseable(smi: str) -> bool:
    if not smi:
        return False
    if "*" in smi:
        # Rxn-INSIGHT chokes on wildcards — bond-electron matrices need
        # real atoms. Generic R-group placeholders are skipped.
        return False
    return Chem.MolFromSmiles(smi) is not None


def build_rxn_smiles(rxn: Reaction) -> Optional[str]:
    """Build a `reactants>>products` Reaction SMILES from one Reaction.

    Returns None if either side has no parseable, non-wildcard SMILES.
    """
    reactants = [r.smiles for r in rxn.reactants if _is_parseable(r.smiles)]
    products = [p.smiles for p in rxn.products if _is_parseable(p.smiles)]
    if not reactants or not products:
        return None
    return f"{'.'.join(reactants)}>>{'.'.join(products)}"


def _summarize_reagents(rxn: Reaction, role_keyword: str) -> str:
    """Dot-join SMILES of reagents whose declared role matches `role_keyword`."""
    bits = []
    for r in rxn.reagents:
        r_role = (r.role or "").lower()
        if role_keyword in r_role and r.smiles:
            bits.append(r.smiles)
    return ".".join(bits)


# ---------- Subprocess driver ----------

def analyze_extraction(
    extraction: FigureExtraction,
    *,
    venv_python: Path = VENV_PYTHON,
    runner: Path = RUNNER_SCRIPT,
    timeout: float = TIMEOUT_S,
    source_image: str = "",
) -> list[dict]:
    """Run Rxn-INSIGHT on every reaction in `extraction`.

    Returns a list (parallel to `extraction.reactions`) where each entry is:
        {
          "entry_id":   str,
          "rxn_smiles": str | None,
          "ok":         bool,
          "row":        dict | None,    # all Section 5.9 columns when ok=True
          "error":      str | None,
        }

    Reactions whose SMILES are wildcard / empty are reported with
    `ok=False, error="generic or empty SMILES, skipped"`.
    """
    if not venv_python.exists():
        raise RuntimeError(
            f"Rxn-INSIGHT venv not found at {venv_python}. "
            "Set it up: see README or pipeline/rxn_insight.py docstring."
        )

    items_to_run: list[dict] = []
    out_skeleton: list[dict] = []
    for rxn in extraction.reactions:
        rxn_smiles = build_rxn_smiles(rxn)
        out_skeleton.append(
            {
                "entry_id": rxn.entry_id,
                "rxn_smiles": rxn_smiles,
                "ok": False,
                "row": None,
                "error": None,
            }
        )
        if rxn_smiles is None:
            out_skeleton[-1]["error"] = "generic or empty SMILES, skipped"
            continue
        product_yield = next(
            (p.yield_pct for p in rxn.products if p.yield_pct is not None), None
        )
        items_to_run.append(
            {
                "entry_id": rxn.entry_id,
                "rxn_smiles": rxn_smiles,
                "solvent":  _summarize_reagents(rxn, "solvent")
                            or (rxn.conditions.solvent or ""),
                "reagent":  _summarize_reagents(rxn, "reagent")
                            or _summarize_reagents(rxn, "additive"),
                "catalyst": _summarize_reagents(rxn, "catalyst"),
                "yield":    product_yield,
                "ref":      source_image,
            }
        )

    if not items_to_run:
        return out_skeleton

    payload = json.dumps({"reactions": items_to_run})
    try:
        result = subprocess.run(
            [str(venv_python), str(runner)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired:
        for o in out_skeleton:
            if o["error"] is None:
                o["error"] = f"Rxn-INSIGHT subprocess exceeded {timeout}s"
        return out_skeleton
    except subprocess.CalledProcessError as exc:
        for o in out_skeleton:
            if o["error"] is None:
                o["error"] = f"Rxn-INSIGHT subprocess failed: {exc.stderr[:200]}"
        return out_skeleton

    sub_results = json.loads(result.stdout).get("results", [])
    by_id = {str(r["entry_id"]): r for r in sub_results}

    for o in out_skeleton:
        sub = by_id.get(str(o["entry_id"]))
        if sub is None:
            continue
        o["ok"] = bool(sub.get("ok"))
        o["row"] = sub.get("row")
        if not o["ok"]:
            o["error"] = sub.get("error") or "unknown error"

    return out_skeleton


# ---------- CSV flattener (full Section 5.9 columns) ----------

# Order matches Rxn-INSIGHT's documentation Table 5.3, with our own
# bookkeeping columns (ENTRY_ID, STATUS, ERROR) bracketing the payload.
INSIGHT_CSV_COLUMNS = [
    "ENTRY_ID",
    # Section 5.9 — Database Output Columns (Table 5.3)
    "REACTION",
    "REACTANTS",
    "PRODUCTS",
    "SANITIZED_REACTION",
    "MAPPED_REACTION",
    "N_REACTANTS",
    "N_PRODUCTS",
    "CLASS",
    "NAME",
    "FG_REACTANTS",
    "FG_PRODUCTS",
    "PARTICIPATING_RINGS_REACTANTS",
    "PARTICIPATING_RINGS_PRODUCTS",
    "ALL_RINGS_PRODUCTS",
    "BY-PRODUCTS",
    "TEMPLATE",
    "TEMPLATE_NR",
    "TAG",
    "TAG2",
    "SCAFFOLD",
    "rxn_str_patt_fp",
    "rxn_dif_patt_fp",
    "rxn_str_morgan_fp",
    "rxn_dif_morgan_fp",
    # Input passthroughs
    "SOLVENT",
    "REAGENT",
    "CATALYST",
    "YIELD",
    "REF",
    # Bookkeeping
    "STATUS",
    "ERROR",
]


def extraction_to_insight_rows(
    extraction: FigureExtraction,
    analyses: Iterable[dict],
    *,
    source_image: str = "",
) -> list[dict]:
    """Build CSV rows containing every Section 5.9 column. One row per reaction."""
    rows: list[dict] = []
    for rxn, an in zip(extraction.reactions, analyses):
        row = an.get("row") or {}
        product_yield = next(
            (p.yield_pct for p in rxn.products if p.yield_pct is not None), None
        )
        flat = {col: "" for col in INSIGHT_CSV_COLUMNS}
        flat["ENTRY_ID"] = rxn.entry_id
        flat["STATUS"] = "ok" if an.get("ok") else "skipped"
        flat["ERROR"] = an.get("error") or ""

        # Default REF / YIELD from our extraction; let Rxn-INSIGHT row override.
        flat["REF"] = source_image
        flat["YIELD"] = product_yield if product_yield is not None else ""
        flat["SOLVENT"] = _summarize_reagents(rxn, "solvent") or (rxn.conditions.solvent or "")
        flat["REAGENT"] = _summarize_reagents(rxn, "reagent") or _summarize_reagents(rxn, "additive")
        flat["CATALYST"] = _summarize_reagents(rxn, "catalyst")
        flat["REACTION"] = an.get("rxn_smiles") or ""

        # Pull the full Section 5.9 column set when present.
        for col in INSIGHT_CSV_COLUMNS:
            if col in row and row[col] not in (None, ""):
                flat[col] = row[col]

        rows.append(flat)
    return rows


def write_insight_csv(rows: Iterable[dict], out_path: Path) -> Path:
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INSIGHT_CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path
