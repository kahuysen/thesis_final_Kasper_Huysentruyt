"""Generate one reaction card per entry of Table 3 (transamidation scope)."""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

OUT = Path("cards")
OUT.mkdir(exist_ok=True)

# Shared conditions for every entry in Table 3
CATALYST = {
    "name": "bis(2-chlorophenyl)borinic acid (1)",
    "smiles": "OB(c1ccccc1Cl)c1ccccc1Cl",
    "loading": "2.5 mol%",
}
COREAGENT = {
    "name": "formamide (HCONH2)",
    "smiles": "NC=O",
    "loading": "5 equiv",
}
ADDITIVE = {
    "name": "AcOH",
    "smiles": "CC(=O)O",
    "loading": "10 mol%",
}
CONDITIONS_TEXT = (
    "cat. 1 (2.5 mol%)\n"
    "AcOH (10 mol%)\n"
    "HCONH$_2$ (5 equiv)\n"
    "45 °C, 24 h"
)

ENTRIES = [
    dict(entry=1, prod="8a", aa="glycine",
         reactant_smiles="COC(=O)CN",
         product_smiles="COC(=O)CNC=O", yield_pct=99),
    dict(entry=2, prod="8b", aa="L-alanine",
         reactant_smiles="COC(=O)[C@@H](C)N",
         product_smiles="COC(=O)[C@@H](C)NC=O", yield_pct=99),
    dict(entry=3, prod="8c", aa="L-phenylalanine",
         reactant_smiles="COC(=O)[C@@H](Cc1ccccc1)N",
         product_smiles="COC(=O)[C@@H](Cc1ccccc1)NC=O", yield_pct=98),
    dict(entry=4, prod="8d", aa="L-leucine",
         reactant_smiles="COC(=O)[C@@H](CC(C)C)N",
         product_smiles="COC(=O)[C@@H](CC(C)C)NC=O", yield_pct=97),
    dict(entry=5, prod="8e", aa="L-valine",
         reactant_smiles="COC(=O)[C@@H](C(C)C)N",
         product_smiles="COC(=O)[C@@H](C(C)C)NC=O",
         yield_pct=96, extra="(97% on 5 mmol scale)"),
    dict(entry=6, prod="8f", aa="L-methionine",
         reactant_smiles="COC(=O)[C@@H](CCSC)N",
         product_smiles="COC(=O)[C@@H](CCSC)NC=O", yield_pct=91),
    dict(entry=7, prod="8g", aa="L-proline",
         reactant_smiles="COC(=O)[C@@H]1CCCN1",
         product_smiles="COC(=O)[C@@H]1CCCN1C=O", yield_pct=99),
    dict(entry=8, prod="8h", aa="N$_\\delta$-Boc-L-ornithine",
         reactant_smiles="COC(=O)[C@@H](CCCNC(=O)OC(C)(C)C)N",
         product_smiles="COC(=O)[C@@H](CCCNC(=O)OC(C)(C)C)NC=O", yield_pct=99),
    dict(entry=9, prod="8i", aa="L-glutamic acid dimethyl ester",
         reactant_smiles="COC(=O)[C@@H](CCC(=O)OC)N",
         product_smiles="COC(=O)[C@@H](CCC(=O)OC)NC=O", yield_pct=57),
]


def mol_image(smiles: str, size=(420, 260)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def panel(ax, smiles: str, label: str, sublabel: str = ""):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    img = mol_image(smiles)
    ax.imshow(img)
    ax.set_title(label, fontsize=12, fontweight="bold", pad=4)
    if sublabel:
        ax.text(0.5, -0.05, sublabel, ha="center", va="top",
                transform=ax.transAxes, fontsize=10, color="#444")


def arrow(ax, text: str):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(0.5, 0.78, text, ha="center", va="bottom", fontsize=9,
            family="DejaVu Sans")


def make_card(e: dict):
    fig = plt.figure(figsize=(13, 4.4), facecolor="white")
    fig.suptitle(
        f"Entry {e['entry']}  ·  Product {e['prod']}  ·  from {e['aa']}",
        fontsize=14, fontweight="bold", y=0.97,
    )

    gs = fig.add_gridspec(
        1, 5, width_ratios=[3.2, 1.4, 2.2, 1.6, 3.2],
        left=0.02, right=0.98, top=0.86, bottom=0.06, wspace=0.15,
    )

    ax_r = fig.add_subplot(gs[0, 0])
    panel(ax_r, e["reactant_smiles"], "Reactant 7", e["aa"] + " methyl ester")

    ax_plus1 = fig.add_subplot(gs[0, 1])
    ax_plus1.set_xticks([]); ax_plus1.set_yticks([])
    for s in ax_plus1.spines.values():
        s.set_visible(False)
    ax_plus1.text(0.5, 0.5, "+", ha="center", va="center", fontsize=28)

    ax_co = fig.add_subplot(gs[0, 2])
    panel(ax_co, COREAGENT["smiles"], "HCONH$_2$", "formamide (5 equiv)")

    ax_arrow = fig.add_subplot(gs[0, 3])
    arrow(ax_arrow, CONDITIONS_TEXT)

    ax_p = fig.add_subplot(gs[0, 4])
    sub = f"yield: {e['yield_pct']}%"
    if e.get("extra"):
        sub += "  " + e["extra"]
    panel(ax_p, e["product_smiles"], f"Product {e['prod']}", sub)

    out = OUT / f"entry{e['entry']:02d}_{e['prod']}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def make_catalyst_legend():
    fig, ax = plt.subplots(figsize=(6, 3), facecolor="white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    img = mol_image(CATALYST["smiles"], size=(500, 320))
    ax.imshow(img)
    ax.set_title("Catalyst 1: bis(2-chlorophenyl)borinic acid",
                 fontsize=12, fontweight="bold")
    out = OUT / "catalyst_1.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    paths = [make_catalyst_legend()]
    for e in ENTRIES:
        paths.append(make_card(e))
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
