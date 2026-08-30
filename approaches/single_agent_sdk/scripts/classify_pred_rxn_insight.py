"""Classify every predicted reaction of all five evaluated systems with
Rxn-INSIGHT, mirroring classify_gold_rxn_insight.py (same bucketing:
wildcards as "R-group template", errors as "Unclassified"). Also counts
ring systems in predicted products. Runs under .venv-rxn-insight.
Writes benchmark_runs/pred_rxn_insight.json.
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEMS = {
    "opus5": ROOT / "benchmark_runs" / "full_opus5",
    "g37f": ROOT / "benchmark_runs" / "full_gemini37flash",
    "g3f": ROOT / "benchmark_runs" / "full_gemini3flash",
    "gpt54": ROOT / "benchmark_runs" / "full_gpt54",
    "chemeagle": ROOT.parent.parent / "baselines" / "chemeagle" / "runs" / "full",
}

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from rxn_insight.reaction import Reaction  # noqa: E402
from rxn_insight.utils import get_ring_systems  # noqa: E402

out = {"reactions": [], "rings": {}}
for sysname, d in SYSTEMS.items():
    rings = collections.Counter()
    files = [f for f in sorted(d.glob("*.json"))
             if not (f.name.endswith(".meta.json") or f.name.endswith(".usage.json")
                     or f.name.startswith("_") or f.name == "full_eval.json")]
    for i, f in enumerate(files):
        try:
            g = json.loads(f.read_text())
        except Exception:
            continue
        for r in g.get("reactions", []):
            rs = [x.get("smiles") for x in r.get("reactants", []) if x.get("smiles")]
            ps = [x.get("smiles") for x in r.get("products", []) if x.get("smiles")]
            row = {"system": sysname, "stem": f.stem}
            rxn = ".".join(rs) + ">>" + ".".join(ps)
            if not rs or not ps:
                row.update(cls="Unclassified")
            elif "*" in rxn:
                row.update(cls="R-group template")
            else:
                try:
                    info = Reaction(rxn).get_reaction_info()
                    row.update(cls=info.get("CLASS") or "Unclassified")
                except Exception:
                    row.update(cls="Unclassified")
            out["reactions"].append(row)
            for smi in ps:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                for atoms in get_ring_systems(m):
                    frag = Chem.MolFragmentToSmiles(m, atoms, canonical=True)
                    fm = Chem.MolFromSmiles(frag)
                    rings[Chem.MolToSmiles(fm) if fm else frag] += 1
        if (i + 1) % 50 == 0:
            print(f"{sysname}: {i+1}/{len(files)} images", flush=True)
    out["rings"][sysname] = dict(rings.most_common())
    print(f"{sysname} done: {sum(1 for r in out['reactions'] if r['system'] == sysname)} reactions", flush=True)

(ROOT / "benchmark_runs" / "pred_rxn_insight.json").write_text(json.dumps(out))
print("wrote pred_rxn_insight.json")
