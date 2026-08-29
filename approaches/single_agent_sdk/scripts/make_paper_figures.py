"""Generate the arXiv paper's figures from the canonical full_eval.json files.

Reads benchmark_runs/full_*/full_eval.json (written by eval_full_benchmark.py)
and emits PDF+PNG into paper/Arxive_paper/figures/.

Colors are fixed per model entity (never re-assigned by rank) and the
palette was validated for CVD separation; identity is always doubled by a
direct label, never carried by color alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent.parent / "paper" / "Arxive_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# entity → (run dir, display name, color slot, $/successful image)
# Costs: sidecar token counts × OpenRouter rates, reconciled vs billing
# (see FULL_RUN_PLAN.md and the paper's cost section).
MODELS = {
    # costs: billed all-in $/successful image from the two OpenRouter
    # activity exports (evaluation/results/billing_summary.json)
    "opus5":   ("full_opus5",         "Claude Opus 5",    "#2a78d6", 0.838),
    "g37f":    ("full_gemini37flash", "Gemini 3.7 Flash", "#eb6834", 0.101),
    "g3f":     ("full_gemini3flash",  "Gemini 3 Flash",   "#1baf7a", 0.049),
    "gpt54":   ("full_gpt54",         "GPT-5.4",          "#eda100", 0.089),
    # multi-agent baseline; run dir lives under baselines/chemeagle
    "chemeagle": ("../../../baselines/chemeagle/runs/full", "ChemEAGLE (7 agents)", "#e87ba4", 0.055),
}

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def load():
    data = {}
    for key, (run, name, color, cost) in MODELS.items():
        run_dir = (ROOT / "benchmark_runs" / run) if "/" not in run else (ROOT / "benchmark_runs" / run).resolve()
        rep = json.loads((run_dir / "full_eval.json").read_text())
        g = rep["groups"]
        data[key] = {
            "name": name, "color": color, "cost": cost,
            "held": g["heldout"]["partial_match_f1"],
            "ci": g["heldout"]["ci95"]["partial_match_f1"],
            "dev16": g["dev16"]["partial_match_f1"],
            "slices": {s: g[s]["partial_match_f1"] for s in ("gt1", "gt2", "gt3", "gt4")},
            "missing": g["heldout"]["n_missing_predictions"],
        }
    return data


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_heldout(d):
    order = sorted(d, key=lambda k: d[k]["held"])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, k in enumerate(order):
        m = d[k]
        lo, hi = m["ci"]
        ax.plot([lo, hi], [i, i], color=m["color"], linewidth=2, solid_capstyle="round")
        ax.plot(m["held"], i, "o", color=m["color"], markersize=8, zorder=3)
        ax.plot(m["dev16"], i, "o", markerfacecolor="white",
                markeredgecolor=m["color"], markersize=7, markeredgewidth=1.6, zorder=3)
        ax.annotate(f'{m["held"]:.3f}', (m["held"], i), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=TEXT)
        ax.annotate(m["name"], (min(lo, m["dev16"]) - 0.015, i), ha="right",
                    va="center", fontsize=9, color=TEXT)
    ax.plot([], [], "o", color=MUTED, markersize=8, label="held-out 305 (95% CI)")
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=MUTED,
            markersize=7, markeredgewidth=1.6, label="dev16")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("partial-match F1", fontsize=9, color=TEXT)
    style_ax(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_heldout.{ext}", dpi=300)
    plt.close(fig)


def fig_cost_quality(d):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    offs = {"opus5": (-8, -12), "g37f": (8, 4), "g3f": (8, -10), "gpt54": (8, 0)}
    for k, m in d.items():
        lo, hi = m["ci"]
        ax.plot([m["cost"], m["cost"]], [lo, hi], color=m["color"], linewidth=1.6)
        ax.plot(m["cost"], m["held"], "o", color=m["color"], markersize=9, zorder=3)
        dx, dy = offs.get(k, (8, 0))
        ax.annotate(m["name"], (m["cost"], m["held"]), textcoords="offset points",
                    xytext=(dx, dy), ha="left" if dx > 0 else "right",
                    fontsize=8.5, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 1.2)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("cost per successful image (USD, log scale)", fontsize=9, color=TEXT)
    ax.set_ylabel("held-out partial-match F1", fontsize=9, color=TEXT)
    style_ax(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cost_quality.{ext}", dpi=300)
    plt.close(fig)


def fig_subsets(d):
    slices = [("gt1", "GT1\nfull records"), ("gt2", "GT2\nR-group tables"),
              ("gt3", "GT3\nReaxys-derived"), ("gt4", "GT4\nrich tables")]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for si, (s, label) in enumerate(slices):
        for k, m in d.items():
            ax.plot(m["slices"][s], si, "o", color=m["color"], markersize=8,
                    zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    for k, m in d.items():
        xs = [m["slices"][s] for s, _ in slices]
        ax.plot(xs, range(len(slices)), color=m["color"], linewidth=1,
                alpha=0.45, zorder=2)
    for k, m in d.items():  # direct labels at the GT1 row end
        ax.annotate(m["name"], (m["slices"]["gt1"], 0), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, color=TEXT,
                    rotation=0) if False else None
    handles = [plt.Line2D([], [], marker="o", linestyle="-", linewidth=1,
                          color=m["color"], markersize=7, label=m["name"])
               for m in d.values()]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    ax.set_yticks(range(len(slices)),
                  [l for _, l in slices], fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("partial-match F1 (all scored images of subset)", fontsize=9, color=TEXT)
    style_ax(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_subsets.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    fig_heldout(d)
    fig_cost_quality(d)
    fig_subsets(d)
    print("wrote figures to", OUT)
