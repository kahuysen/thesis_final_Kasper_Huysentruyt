"""Generate the arXiv paper's figures from the canonical full_eval.json files.

Reads benchmark_runs/full_*/full_eval.json (written by eval_full_benchmark.py)
and emits PDF+PNG into paper/Arxive_paper/figures/.

Colors are fixed per model entity (never re-assigned by rank) and the
palette was validated for CVD separation; identity is always doubled by a
direct label, never carried by color alone.
"""
from __future__ import annotations

import csv
import json
import statistics
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
    # activity exports (evaluation/results/billing_summary.json).
    # Palette: warm scheme derived from the author's chosen swatches,
    # deepened to print weight; passes CVD-separation and legibility
    # checks in display order (validated 2026-08-29).
    "opus5":   ("full_opus5",         "Claude Opus 5",    "#96551F", 0.838),
    "g37f":    ("full_gemini37flash", "Gemini 3.7 Flash", "#EE7419", 0.101),
    "g3f":     ("full_gemini3flash",  "Gemini 3 Flash",   "#2E8FBF", 0.049),
    "gpt54":   ("full_gpt54",         "GPT-5.4",          "#8A5BB8", 0.089),
    # multi-agent baseline; run dir lives under baselines/chemeagle
    "chemeagle": ("../../../baselines/chemeagle/runs/full", "ChemEAGLE (7 agents)", "#6F9A34", 0.055),
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
        data[key].update(run_stats(key, run_dir))
    return data


def run_stats(key, run_dir):
    """Median tokens and wall time per successful image, from the sidecars.

    Single-agent runs: one *.meta.json per completed image. ChemEAGLE:
    *.usage.json for tokens; wall time comes from _batch_summary.json, which
    only covers the images attempted in the final driver invocation (120 of
    the successes) — the median over that subset is what we report.
    """
    if key == "chemeagle":
        toks = []
        for f in run_dir.glob("*.usage.json"):
            u = json.loads(f.read_text())
            if u.get("status") == "ok":
                toks.append(u["prompt_tokens"] + u["completion_tokens"])
        rows = json.loads((run_dir / "_batch_summary.json").read_text())
        times = [r["elapsed_s"] for r in rows if r["status"] == "ok"]
    else:
        toks, times = [], []
        for f in run_dir.glob("*.meta.json"):
            m = json.loads(f.read_text())
            toks.append(m.get("input_tokens", 0) + m.get("output_tokens", 0))
            times.append(m.get("elapsed_s", 0.0))
    return {"med_tokens": statistics.median(toks),
            "med_time": statistics.median(times)}


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_heldout(d, show_dev16=False, fname="fig_heldout"):
    order = sorted(d, key=lambda k: d[k]["held"])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, k in enumerate(order):
        m = d[k]
        lo, hi = m["ci"]
        ax.plot([lo, hi], [i, i], color=m["color"], linewidth=2, solid_capstyle="round")
        ax.plot(m["held"], i, "o", color=m["color"], markersize=8, zorder=3)
        if show_dev16:
            ax.plot(m["dev16"], i, "o", markerfacecolor="white",
                    markeredgecolor=m["color"], markersize=7, markeredgewidth=1.6, zorder=3)
        ax.annotate(f'{m["held"]:.3f}', (m["held"], i), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=TEXT)
        left = min(lo, m["dev16"]) if show_dev16 else lo
        ax.annotate(m["name"], (left - 0.015, i), ha="right",
                    va="center", fontsize=9, color=TEXT)
    ax.plot([], [], "o", color=MUTED, markersize=8, label="held-out 305 (95% CI)")
    if show_dev16:
        ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=MUTED,
                markersize=7, markeredgewidth=1.6, label="development subset")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("partial-match F1", fontsize=9, color=TEXT)
    style_ax(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=300)
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


def fig_cost_tokens_time(d):
    """Two panels: (a) cost per image vs tokens per image, (b) wall time vs
    cost per image. Tokens/time are medians over successful images; cost is
    the billed all-in figure (failures included), same as everywhere else."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    offs1 = {"opus5": (-10, -3), "g37f": (0, 10), "g3f": (8, -3),
             "gpt54": (8, -3), "chemeagle": (0, -16)}
    for k, m in d.items():
        ax1.plot(m["med_tokens"] / 1000, m["cost"], "o", color=m["color"],
                 markersize=9, zorder=3)
        dx, dy = offs1.get(k, (8, 0))
        ax1.annotate(m["name"], (m["med_tokens"] / 1000, m["cost"]),
                     textcoords="offset points", xytext=(dx, dy),
                     ha="left" if dx > 0 else ("center" if dx == 0 else "right"),
                     fontsize=8, color=TEXT)
    ax1.set_yscale("log")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0.02, 1.5)
    ax1.set_xlabel("tokens per image (thousands, median)", fontsize=9, color=TEXT)
    ax1.set_ylabel("cost per successful image (USD)", fontsize=9, color=TEXT)

    offs2 = {"opus5": (0, 10), "g37f": (8, 4), "g3f": (0, -16),
             "gpt54": (8, -3), "chemeagle": (-10, -3)}
    for k, m in d.items():
        ax2.plot(m["med_time"], m["cost"], "o", color=m["color"],
                 markersize=9, zorder=3)
        dx, dy = offs2.get(k, (8, 0))
        ax2.annotate(m["name"], (m["med_time"], m["cost"]),
                     textcoords="offset points", xytext=(dx, dy),
                     ha="left" if dx > 0 else ("center" if dx == 0 else "right"),
                     fontsize=8, color=TEXT)
    ax2.set_yscale("log")
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0.02, 1.5)
    ax2.set_xlabel("wall time per image (s, median)", fontsize=9, color=TEXT)
    ax2.set_ylabel("cost per successful image (USD)", fontsize=9, color=TEXT)

    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        style_ax(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(tag, loc="left", fontsize=9, color=TEXT)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cost_tokens_time.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Literature-trend figure (WoS papers vs DataCite datasets, 1990-2025),
# restyled from the poster original into the paper's palette. Source counts
# are versioned under figures/data/ (WoS export ';'-free, DataCite ';').
# ---------------------------------------------------------------------------

TREND_CATS = [
    ("Flow chemistry", "#EE7419"),
    ("Organic chemistry", "#2E8FBF"),
    ("Solvent & process sustainability", "#8A5BB8"),
    ("Sustainable chemistry", "#6F9A34"),
]

_TREND_PREFIX = {  # main-category prefix in the raw column headers
    "Flow chemistry": "Flow chemistry",
    "Organic chemistry": "Organic synthesis",
    "Solvent & process sustainability": "Solvent & process sustainability",
    "Sustainable chemistry": "Sustainable chemistry",
}


def _load_trend_csv(path, sep):
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh, delimiter=sep))
    header, body = rows[0], rows[1:]
    out = {}  # year -> {cat: count}
    for row in body:
        if not row or not row[0].strip():
            continue
        year = int(row[0])
        if not (1990 <= year < 2026):
            continue
        agg = {cat: 0 for cat, _ in TREND_CATS}
        for col, val in zip(header[1:], row[1:]):
            for cat, _ in TREND_CATS:
                if col.startswith(_TREND_PREFIX[cat]):
                    agg[cat] += int(val or 0)
        out[year] = agg
    return out


def _k_fmt(x, _pos):
    if x >= 10000:
        return f"{x/1000:.0f}k"
    if x >= 1000:
        return f"{x/1000:.1f}k"
    return f"{int(x)}"


def fig_literature_trend():
    from matplotlib.ticker import FuncFormatter

    data_dir = OUT / "data"
    papers = _load_trend_csv(data_dir / "wos_together_wide.csv", ",")
    datasets = _load_trend_csv(data_dir / "datacite_wide.csv", ";")
    years = sorted(papers)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12})

    for ax, table, ylabel in ((ax1, papers, "WoS papers"),
                              (ax2, datasets, "DataCite datasets")):
        bottom = [0.0] * len(years)
        for cat, color in TREND_CATS:
            vals = [table.get(y, {}).get(cat, 0) for y in years]
            ax.bar(years, vals, bottom=bottom, width=0.85, color=color,
                   edgecolor="white", linewidth=0.3, label=cat)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
        ax.yaxis.set_major_formatter(FuncFormatter(_k_fmt))
        style_ax(ax)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)

    ax1.legend(frameon=False, loc="upper left", fontsize=8, handlelength=1.2)
    ax2.set_xticks(years[::3])
    ax2.set_xlim(min(years) - 0.8, max(years) + 0.8)
    ax2.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_literature_trend.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    fig_heldout(d)
    fig_heldout(d, show_dev16=True, fname="fig_heldout_dev")
    fig_cost_quality(d)
    fig_subsets(d)
    fig_cost_tokens_time(d)
    fig_literature_trend()
    print("wrote figures to", OUT)
