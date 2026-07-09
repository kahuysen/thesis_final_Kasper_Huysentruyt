"""
Grouped bar chart comparing three reaction-extraction pipelines across
seven overlap / F1 / condition metrics.

Styled after the ChemEagle vs ChemEagle-Gemini comparison figure:
clean grouped bars, value labels above each bar, horizontal gridlines,
schema-validity reported as an annotation rather than a bar.

Produces two figures with IDENTICAL geometry/axes/legend:
  * pipeline_comparison.*            -> all three pipelines drawn
  * pipeline_comparison_no_single.*  -> Opus single-agent slot reserved but
                                        bars not drawn, so the single-agent
                                        bars can be added/animated on a slide.

Run with the molnextr venv that already has matplotlib:
    Collective_autogen/.venv-molnextr/bin/python outputs/pipeline_comparison_plot.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- data ---------------------------------------------------------------
metrics = [
    "Product IoU",
    "Reactant IoU",
    "Soft F1",
    "Partial F1",
    "Constituent F1",
    "Cond. Recall",
    "Cond. Precision",
]

series = {
    "GPT-5.4 multi-agent":  [0.55, 0.39, 0.22, 0.37, 0.22, 0.77, 0.71],
    "Opus 4.7 multi-agent": [0.80, 0.67, 0.64, 0.78, 0.68, 0.68, 0.49],
    "Opus 4.7 single-agent":[0.76, 0.68, 0.60, 0.85, 0.71, 0.78, 0.69],
}

# Orange = GPT family, blues = Opus variants (dark = multi, light = single)
colors = {
    "GPT-5.4 multi-agent":   "#E18A3B",
    "Opus 4.7 multi-agent":  "#1F3B66",
    "Opus 4.7 single-agent": "#6FA8DC",
}

# schema-validity annotation (top-right), mirroring the inspiration figure
schema_txt = (
    "n = 16 images   ·   schema valid (of 16):   "
    "GPT-5.4 MA 15   ·   Opus MA 16   ·   Opus SA 16"
)


def make_figure(hidden=()):
    """Draw the chart. Series named in `hidden` keep their slot/legend entry
    but are not rendered, so they can be added on a PowerPoint slide."""
    x = np.arange(len(metrics))
    n = len(series)            # geometry always assumes all three -> stable layout
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)

    for i, (name, vals) in enumerate(series.items()):
        if name in hidden:
            continue           # slot reserved by the offset math below, just not drawn
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=colors[name],
                      edgecolor="white", linewidth=0.4, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color="#333333")

    # axes / grid
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # legend: built manually so it is identical across both figures
    handles = [mpatches.Patch(color=colors[name], label=name) for name in series]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
              edgecolor="#dddddd", fontsize=9)

    ax.text(0.995, 1.02, schema_txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f2f2f2",
                      edgecolor="#dddddd"))

    fig.tight_layout()
    return fig


# full chart
fig = make_figure()
fig.savefig("outputs/pipeline_comparison.png", bbox_inches="tight")
fig.savefig("outputs/pipeline_comparison.pdf", bbox_inches="tight")

# build version: single-agent slot reserved, bars not drawn
fig2 = make_figure(hidden=("Opus 4.7 single-agent",))
fig2.savefig("outputs/pipeline_comparison_no_single.png", bbox_inches="tight")
fig2.savefig("outputs/pipeline_comparison_no_single.pdf", bbox_inches="tight")

print("wrote outputs/pipeline_comparison{,_no_single}.{png,pdf}")
