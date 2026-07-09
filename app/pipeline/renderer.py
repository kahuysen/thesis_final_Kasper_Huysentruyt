"""Render a FigureExtraction to one PNG card per Reaction.

Each card shows: reactants + reagents on the left, conditions over an arrow,
products on the right, with the entry id and yields labelled.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # thread-safe; required when rendering from worker threads
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from .schema import FigureExtraction, Product, Reactant, Reagent, Reaction, Species

PANEL_SIZE = (380, 240)


def _mol_image(smiles: str, size: tuple[int, int] = PANEL_SIZE) -> Image.Image | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _label(s: Species) -> str:
    bits = []
    if s.label:
        bits.append(s.label)
    if s.name and not s.label:
        bits.append(s.name[:30])
    return "  ".join(bits) or "?"


def _sublabel(s: Species) -> str:
    parts = []
    if isinstance(s, Product):
        pct = f"{s.yield_pct:g}%" if s.yield_pct is not None else None
        note = s.yield_note or None
        # Avoid "96% 96% (97% on 5 mmol)" — if note already starts with the pct, keep only note.
        if pct and note and note.lstrip().startswith(pct):
            parts.append(note)
        else:
            if pct:
                parts.append(pct)
            if note:
                parts.append(note)
    if isinstance(s, Reactant) and s.equiv:
        parts.append(s.equiv)
    if isinstance(s, Reagent) and s.loading:
        parts.append(s.loading)
    return "  ".join(parts)


def _reagent_display(r: Reagent) -> str:
    """Best human-readable name for a reagent in the conditions text."""
    if r.label:
        return r.label
    if r.name:
        # Trim long IUPAC names to first ~28 chars for the conditions block.
        return r.name if len(r.name) <= 28 else r.name[:25] + "..."
    if r.smiles and len(r.smiles) <= 12:
        return r.smiles
    return "(unnamed)"


def _conditions_text(rxn: Reaction) -> str:
    lines: list[str] = []
    catalysts = [r for r in rxn.reagents if (r.role or "").lower() == "catalyst"]
    others = [r for r in rxn.reagents if (r.role or "").lower() != "catalyst"]

    # Skip a reagent if it's also listed as the solvent (avoid duplicate lines
    # like "HCONH2 (5 equiv)" appearing twice).
    solvent_text = (rxn.conditions.solvent or "").lower()
    def _shown_as_solvent(r: Reagent) -> bool:
        if not solvent_text:
            return False
        for tag in (r.label, r.name, r.smiles):
            if tag and tag.lower() in solvent_text:
                return True
        return False

    for r in catalysts + others:
        if _shown_as_solvent(r):
            continue
        bit = _reagent_display(r)
        if r.loading:
            bit += f" ({r.loading})"
        lines.append(bit)
    c = rxn.conditions
    cond_bits = [b for b in (c.solvent, c.temperature, c.time, c.atmosphere, c.other) if b]
    if cond_bits:
        lines.append(", ".join(cond_bits))
    return "\n".join(lines) or "—"


def _draw_panel(ax, sp: Species):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    img = _mol_image(sp.smiles)
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, sp.smiles or "(generic)", ha="center", va="center",
                fontsize=10, transform=ax.transAxes, wrap=True)
    ax.set_title(_label(sp), fontsize=10, fontweight="bold", pad=2)
    sub = _sublabel(sp)
    if sub:
        ax.text(0.5, -0.04, sub, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, color="#555")


def _draw_arrow(ax, text: str):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(0.5, 0.78, text, ha="center", va="bottom", fontsize=8)


def _draw_op(ax, op: str):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, 0.5, op, ha="center", va="center", fontsize=22)


def render_reaction(rxn: Reaction, out_path: Path) -> Path:
    n_react = max(len(rxn.reactants), 1)
    n_prod = max(len(rxn.products), 1)

    # Width ratios: reactant panels (with + between) | arrow | product panels (with + between)
    react_widths = []
    for i in range(n_react):
        if i:
            react_widths.append(0.4)  # plus
        react_widths.append(2.6)
    prod_widths = []
    for i in range(n_prod):
        if i:
            prod_widths.append(0.4)
        prod_widths.append(2.6)
    arrow_width = [1.6]
    widths = react_widths + arrow_width + prod_widths
    n_cols = len(widths)

    fig_w = max(10, sum(widths) * 1.0)
    fig = plt.figure(figsize=(fig_w, 4.0), facecolor="white")
    title = f"Entry {rxn.entry_id}"
    if rxn.title:
        title += f" — {rxn.title}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    gs = fig.add_gridspec(
        1, n_cols, width_ratios=widths,
        left=0.02, right=0.98, top=0.86, bottom=0.10, wspace=0.12,
    )

    col = 0
    for i, r in enumerate(rxn.reactants or [Reactant(smiles="", label="?")]):
        if i:
            _draw_op(fig.add_subplot(gs[0, col]), "+"); col += 1
        _draw_panel(fig.add_subplot(gs[0, col]), r); col += 1

    _draw_arrow(fig.add_subplot(gs[0, col]), _conditions_text(rxn)); col += 1

    for i, p in enumerate(rxn.products or [Product(smiles="", label="?")]):
        if i:
            _draw_op(fig.add_subplot(gs[0, col]), "+"); col += 1
        _draw_panel(fig.add_subplot(gs[0, col]), p); col += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def render_figure(fx: FigureExtraction, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, rxn in enumerate(fx.reactions, start=1):
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in rxn.entry_id)
        out = out_dir / f"rxn_{i:02d}_{safe_id}.png"
        paths.append(render_reaction(rxn, out))
    return paths
