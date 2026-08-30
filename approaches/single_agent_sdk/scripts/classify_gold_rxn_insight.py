"""Classify every gold reaction of the full benchmark with Rxn-INSIGHT.

Runs under .venv-rxn-insight (the only env with rxn_insight installed).
Writes one JSON with per-reaction {stem, slice, rid, cls, name} to
benchmark_runs/gold_rxn_insight.json. Wildcard (R-group template)
reactions are bucketed as "R-group template" without classification:
they are reaction schemes, not concrete reactions, and atom mapping on
`*` atoms is meaningless. Classifier errors become "Unclassified".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT.parent.parent / "data" / "benchmark_full" / "ground_truth"
OUT = ROOT / "benchmark_runs" / "gold_rxn_insight.json"

from rxn_insight.reaction import Reaction  # noqa: E402

rows = []
files = sorted(GT.glob("*.json"))
for i, f in enumerate(files):
    g = json.loads(f.read_text())
    for r in g["reactions"]:
        rs = [x["smiles"] for x in r.get("reactants", []) if x.get("smiles")]
        ps = [x["smiles"] for x in r.get("products", []) if x.get("smiles")]
        row = {"stem": f.stem, "slice": g["slice"], "rid": r.get("reaction_id")}
        rxn = ".".join(rs) + ">>" + ".".join(ps)
        if not rs or not ps:
            row.update(cls="Unclassified", name=None)
        elif "*" in rxn:
            row.update(cls="R-group template", name=None)
        else:
            try:
                info = Reaction(rxn).get_reaction_info()
                row.update(cls=info.get("CLASS") or "Unclassified",
                           name=info.get("NAME"))
            except Exception:
                row.update(cls="Unclassified", name=None)
        rows.append(row)
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(files)} images, {len(rows)} reactions", flush=True)

OUT.write_text(json.dumps(rows, indent=1))
print("wrote", OUT, len(rows), "reactions")
