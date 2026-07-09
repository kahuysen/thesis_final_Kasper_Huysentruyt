"""Offline smoke test: skip the LLM, hand-build FigureExtraction objects
from our earlier extractions, then run them through render + flatten + CSV.
Proves the post-extraction pipeline end-to-end."""
from pathlib import Path

from pipeline.flatten import figure_to_rows, write_csv
from pipeline.renderer import render_figure
from pipeline.schema import (
    Conditions,
    FigureExtraction,
    Product,
    Reactant,
    Reaction,
    Reagent,
    Species,
)

OUT = Path("results")

# ------------------------------------------------------------------
# Figure 1: control experiments (subset — eqs I, IV, VI, VIII)
# ------------------------------------------------------------------
fe_cat = Reagent(label="Fe(acac)$_3$", smiles="CC(=O)C=C(C)O[Fe]", loading="20 mol%", role="catalyst")
phsih3 = Reagent(label="PhSiH$_3$", smiles="[SiH3]c1ccccc1", loading="2 equiv", role="reductant")
etoh_1ml = Conditions(temperature="75 °C", time="1 h", solvent="EtOH (1 mL)", atmosphere="air")

fig1 = FigureExtraction(
    figure_caption="Figure 3. Control experiments.",
    reactions=[
        Reaction(
            entry_id="I",
            title="Standard conditions",
            reactants=[
                Reactant(label="1a", smiles="O=C1c2ccccc2C(=O)C=C1N=P(c1ccccc1)(c1ccccc1)c1ccccc1"),
                Reactant(label="2a", smiles="CC(=C)CO"),
            ],
            reagents=[fe_cat, phsih3],
            conditions=etoh_1ml,
            products=[
                Product(label="3a", smiles="O=C1c2ccccc2C(=O)C2=C1OCC2(C)C", yield_pct=80),
                Product(label="4 (PPh$_3$)", smiles="P(c1ccccc1)(c1ccccc1)c1ccccc1", yield_pct=75),
            ],
            notes="Standard reaction conditions",
        ),
        Reaction(
            entry_id="IV",
            title="TBS-protected substrate intercepts intermediates",
            reactants=[
                Reactant(label="1a", smiles="O=C1c2ccccc2C(=O)C=C1N=P(c1ccccc1)(c1ccccc1)c1ccccc1"),
                Reactant(label="2a (OTBS)", smiles="CC(=C)CO[Si](C)(C)C(C)(C)C"),
            ],
            reagents=[fe_cat, phsih3],
            conditions=etoh_1ml,
            products=[
                Product(label="5", smiles="O=C1c2ccccc2C(=O)C(C(C)(C)CO[Si](C)(C)C(C)(C)C)=C1N=P(c1ccccc1)(c1ccccc1)c1ccccc1", yield_pct=23),
                Product(label="6", smiles="NC1=C(C(C)(C)CO[Si](C)(C)C(C)(C)C)C(=O)c2ccccc2C1=O", yield_pct=40),
            ],
        ),
        Reaction(
            entry_id="VI",
            title="Bi(OTf)$_3$ desilylation of 5 → 7 → 3a",
            reactants=[Reactant(label="5", smiles="O=C1c2ccccc2C(=O)C(C(C)(C)CO[Si](C)(C)C(C)(C)C)=C1N=P(c1ccccc1)(c1ccccc1)c1ccccc1")],
            reagents=[
                Reagent(label="Bi(OTf)$_3$", smiles="", loading="30 mol%", role="catalyst"),
                fe_cat, phsih3,
            ],
            conditions=Conditions(temperature="75 °C", time="1 h", solvent="EtOH (0.2 M)"),
            products=[Product(label="3a", smiles="O=C1c2ccccc2C(=O)C2=C1OCC2(C)C", yield_pct=39)],
            hrms_observed="[M+H]+ 506.1884",
            notes="Intermediate 7 confirmed by HRMS",
        ),
        Reaction(
            entry_id="VIII",
            title="Free aminoquinone is unreactive",
            reactants=[
                Reactant(label="9", smiles="NC1=CC(=O)c2ccccc2C1=O"),
                Reactant(label="2a", smiles="CC(=C)CO"),
            ],
            reagents=[fe_cat, phsih3],
            conditions=etoh_1ml,
            products=[Product(label="3a", smiles="O=C1c2ccccc2C(=O)C2=C1OCC2(C)C", yield_pct=None, yield_note="NR")],
            notes="N=PPh3 group is required",
        ),
    ],
    intermediates=[
        Species(label="7", smiles="O=C1c2ccccc2C(=O)C(C(C)(C)CO)=C1N=P(c1ccccc1)(c1ccccc1)c1ccccc1",
                role_note="Bracketed intermediate; matches HRMS [M+H]+ 506.1884"),
        Species(label="8a", smiles="NC1=C(C(C)(C)CO)C(=O)c2ccccc2C1=O",
                role_note="Bracketed intermediate; matches HRMS [M+Na]+ 268.0947"),
        Species(label="8b", smiles="O=C1C(=N)C(C(C)(C)CO)C(=O)c2ccccc21",
                role_note="Imine tautomer of 8a"),
    ],
)

# ------------------------------------------------------------------
# Figure 2: Table 3 transamidation scope (all 9 entries)
# ------------------------------------------------------------------
borinic = Reagent(label="cat. 1", smiles="OB(c1ccccc1Cl)c1ccccc1Cl",
                  loading="2.5 mol%", role="catalyst")
acoh = Reagent(label="AcOH", smiles="CC(=O)O", loading="10 mol%", role="additive")
hconh2 = Reactant(label="HCONH$_2$", smiles="NC=O", equiv="5 equiv")
table3_conds = Conditions(temperature="45 °C", time="24 h")

ENTRIES = [
    ("8a", "glycine",                     "COC(=O)CN",                                   "COC(=O)CNC=O",                                   99, None),
    ("8b", "L-alanine",                   "COC(=O)[C@@H](C)N",                           "COC(=O)[C@@H](C)NC=O",                           99, None),
    ("8c", "L-phenylalanine",             "COC(=O)[C@@H](Cc1ccccc1)N",                   "COC(=O)[C@@H](Cc1ccccc1)NC=O",                   98, None),
    ("8d", "L-leucine",                   "COC(=O)[C@@H](CC(C)C)N",                      "COC(=O)[C@@H](CC(C)C)NC=O",                      97, None),
    ("8e", "L-valine",                    "COC(=O)[C@@H](C(C)C)N",                       "COC(=O)[C@@H](C(C)C)NC=O",                       96, "97% on 5 mmol"),
    ("8f", "L-methionine",                "COC(=O)[C@@H](CCSC)N",                        "COC(=O)[C@@H](CCSC)NC=O",                        91, None),
    ("8g", "L-proline",                   "COC(=O)[C@@H]1CCCN1",                         "COC(=O)[C@@H]1CCCN1C=O",                         99, None),
    ("8h", "Nδ-Boc-L-ornithine",          "COC(=O)[C@@H](CCCNC(=O)OC(C)(C)C)N",          "COC(=O)[C@@H](CCCNC(=O)OC(C)(C)C)NC=O",          99, None),
    ("8i", "L-glutamic acid dimethyl",    "COC(=O)[C@@H](CCC(=O)OC)N",                   "COC(=O)[C@@H](CCC(=O)OC)NC=O",                   57, None),
]

fig2 = FigureExtraction(
    figure_caption="Table 3. Scope of the transamidation of HCONH2 with α-aminoesters.",
    reactions=[
        Reaction(
            entry_id=str(i + 1),
            title=f"{label} from {origin} (product {label})",
            reactants=[
                Reactant(label=f"7{chr(ord('a')+i)}", smiles=react_smi),
                hconh2,
            ],
            reagents=[borinic, acoh],
            conditions=table3_conds,
            products=[Product(label=label, smiles=prod_smi, yield_pct=ypct, yield_note=note)],
        )
        for i, (label, origin, react_smi, prod_smi, ypct, note) in enumerate(ENTRIES)
    ],
)


def main():
    OUT.mkdir(exist_ok=True)
    all_rows = []
    for stem, fx in [("fig3_controls", fig1), ("table3_transamidation", fig2)]:
        json_path = OUT / f"{stem}.json"
        json_path.write_text(fx.model_dump_json(indent=2))
        cards = render_figure(fx, OUT / f"{stem}_cards")
        rows = figure_to_rows(fx, source_image=stem)
        all_rows.extend(rows)
        print(f"{stem}: wrote {json_path.name}, {len(cards)} cards, {len(rows)} CSV rows")
    csv_path = write_csv(all_rows, OUT / "reactions.csv")
    print(f"\nWrote {len(all_rows)} total rows → {csv_path}")


if __name__ == "__main__":
    main()
