"""Convert pipeline result.json files into a Rxn-INSIGHT-compatible CSV.

Usage:
    python to_rxn_insight.py runs/20260424_150150/result.json -o bank.csv
    python to_rxn_insight.py runs/ -o bank.csv --molecules-csv mols.csv
    python to_rxn_insight.py runs/ -o bank.csv --resolve      # name->SMILES via tools.pubchem

Then in the rxn-insight env:
    from rxn_insight.database import Database
    db = Database()
    db.create_database_from_csv("bank.csv", reaction_column="REACTION")
    db.save_to_parquet("bank")
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

WILDCARD = re.compile(r"\*|\[\*[^\]]*\]")
YIELD_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _has_wildcard(smiles: str) -> bool:
    return bool(WILDCARD.search(smiles or ""))


def _parse_yield(text: str) -> float | None:
    if not text:
        return None
    m = YIELD_PCT.search(text)
    return float(m.group(1)) if m else None


def _resolver():
    """Return a name->SMILES resolver using tools.pubchem (opt-in, slow)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.pubchem import lookup_compound

    cache: dict[str, str | None] = {}

    def resolve(text: str) -> str | None:
        if not text:
            return None
        if text in cache:
            return cache[text]
        try:
            r = lookup_compound(text)
            smi = r.get("smiles")
        except Exception:
            smi = None
        cache[text] = smi
        return smi

    return resolve


def _join_role(conditions: list[dict], role: str, resolver=None) -> str:
    """Join all condition entries of the given role into a dot-separated string.

    Prefers c['smiles'] when present; otherwise falls back to c['text']
    (resolved via OPSIN/PubChem/CIR if a resolver is supplied).
    """
    parts: list[str] = []
    for c in conditions:
        if c.get("role") != role:
            continue
        smi = c.get("smiles")
        if not smi and resolver is not None:
            smi = resolver(c.get("text", ""))
        parts.append(smi or c.get("text", "") or "")
    parts = [p for p in parts if p]
    return ".".join(parts)


def _notes(conditions: list[dict], additional_info: list[str]) -> str:
    bits: list[str] = []
    for role in ("temperature", "time", "atmosphere"):
        vals = [c.get("text", "") for c in conditions if c.get("role") == role]
        if vals:
            bits.append(f"{role}={'/'.join(v for v in vals if v)}")
    if additional_info:
        bits.extend(str(x) for x in additional_info if x)
    return " | ".join(bits)


def _smiles_list(items: list[dict]) -> tuple[list[str], bool]:
    """Return (smiles, any_uncertain). Drops empty/wildcard entries."""
    out: list[str] = []
    uncertain = False
    for it in items:
        smi = (it.get("smiles") or "").strip()
        if not smi or _has_wildcard(smi):
            return [], False  # whole reaction is unusable
        if "(uncertain)" in (it.get("label") or "").lower():
            uncertain = True
        out.append(smi)
    return out, uncertain


def convert_reaction(rxn: dict, file_name: str, resolver=None) -> dict | None:
    reactants, unc_r = _smiles_list(rxn.get("reactants", []))
    products, unc_p = _smiles_list(rxn.get("products", []))
    if not reactants or not products:
        return None  # figure-of-structures or wildcard-only — drop

    conditions = rxn.get("conditions", []) or []
    yield_text = next(
        (c.get("text", "") for c in conditions if c.get("role") == "yield"), ""
    )

    return {
        "REACTION": ".".join(reactants) + ">>" + ".".join(products),
        "SOLVENT": _join_role(conditions, "solvent", resolver),
        "REAGENT": _join_role(conditions, "reagent", resolver),
        "CATALYST": _join_role(conditions, "catalyst", resolver),
        "YIELD": _parse_yield(yield_text),
        "REF": f"{file_name}#{rxn.get('reaction_id', '')}",
        "NOTES": _notes(conditions, rxn.get("additional_info", []) or []),
        "UNCERTAIN": int(unc_r or unc_p),
    }


def extract_molecules(rxn: dict, file_name: str) -> list[dict]:
    """Yield one row per standalone molecule from a figure-of-structures entry.

    Triggered when a reaction has no reactants — the products are just labelled
    structures, not transformations. Skips wildcard SMILES.
    """
    if rxn.get("reactants"):
        return []
    info = " | ".join(str(x) for x in (rxn.get("additional_info") or []) if x)
    out: list[dict] = []
    for p in rxn.get("products", []) or []:
        smi = (p.get("smiles") or "").strip()
        if not smi or _has_wildcard(smi):
            continue
        label = p.get("label") or ""
        out.append({
            "SMILES": smi,
            "LABEL": label.replace(" (uncertain)", ""),
            "REF": f"{file_name}#{rxn.get('reaction_id', '')}",
            "NOTES": info,
            "UNCERTAIN": int("(uncertain)" in label.lower()),
        })
    return out


def iter_result_files(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("result.json")):
        yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="result.json file or directory containing runs/")
    ap.add_argument("-o", "--output", type=Path, default=Path("rxn_insight_input.csv"))
    ap.add_argument("--molecules-csv", type=Path, default=None,
                    help="Also write standalone molecules (figure-of-structures rows) to this CSV")
    ap.add_argument("--resolve", action="store_true",
                    help="Resolve condition names to SMILES via tools.pubchem (slow, network)")
    args = ap.parse_args()

    resolver = _resolver() if args.resolve else None
    rxn_rows: list[dict] = []
    mol_rows: list[dict] = []
    skipped = 0

    for jf in iter_result_files(args.input):
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"[skip] {jf}: {e}", file=sys.stderr)
            continue
        file_name = data.get("file_name") or jf.parent.name
        for rxn in data.get("reactions", []) or []:
            row = convert_reaction(rxn, file_name, resolver)
            if row is not None:
                rxn_rows.append(row)
                continue
            mols = extract_molecules(rxn, file_name)
            if mols:
                mol_rows.extend(mols)
            else:
                skipped += 1

    if not rxn_rows and not mol_rows:
        print("No usable reactions or molecules found.", file=sys.stderr)
        return 1

    if rxn_rows:
        fields = ["REACTION", "SOLVENT", "REAGENT", "CATALYST", "YIELD", "REF", "NOTES", "UNCERTAIN"]
        with args.output.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rxn_rows)
        print(f"Wrote {len(rxn_rows)} reactions to {args.output}.")

    if args.molecules_csv and mol_rows:
        mol_fields = ["SMILES", "LABEL", "REF", "NOTES", "UNCERTAIN"]
        with args.molecules_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=mol_fields)
            w.writeheader()
            w.writerows(mol_rows)
        print(f"Wrote {len(mol_rows)} molecules to {args.molecules_csv}.")
    elif mol_rows:
        print(f"({len(mol_rows)} standalone molecules ignored — pass --molecules-csv to keep them.)")

    print(f"Skipped {skipped} entries with empty/wildcard SMILES.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
