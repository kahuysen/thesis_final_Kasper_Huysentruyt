"""Build the resolution-ablation figure in the same visual style as
`figures/fig2_cost_quality_with_qwen_and_gemini35.png`.

Two-panel figure:
  (a) Reference-based quality (Partial F1, Product IoU, Constituent F1,
      Soft F1) vs. linear scale factor (1, 1/2, 1/4, 1/10).
  (b) Projected USD cost and summed wall-clock seconds vs. the same
      scale axis.

Output:
  benchmark_runs/resolution_report/fig_resolution_ablation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "fig_resolution_ablation.png"

# UGent + reference-figure palette.
SDK_BLUE  = "#1E64C8"   # primary (Partial F1)
DARK_BLUE = "#0E3A78"   # secondary (Product IoU)
WARM      = "#FB8500"   # tertiary (Constituent F1)
GREY      = "#9A9A9A"   # auxiliary (Soft F1)
INK       = "#1A1814"

# Resolution conditions: linear scale factor (each side).
SCALES   = [1.0,    0.50,  0.25,  0.125, 0.10]
# Mean source DPI = 124.8, computed from the 5/16 source PNGs that carry
# DPI metadata (96 dpi for 2 images, 144 dpi for 3); effective DPI at each
# scale = mean_source_DPI × scale_factor, rounded to nearest integer.
MEAN_SRC_DPI = 124.8
DPI_AT_SCALE = [round(MEAN_SRC_DPI * s) for s in SCALES]
LABELS   = [f"{name}\n≈{d} dpi" for name, d in zip(
    ["Full", "1/2", "1/4", "1/8", "1/10"], DPI_AT_SCALE)]
N_IMAGES = 16

# eval_via_collective.py results (macro-averages across the 16 images).
PARTIAL_F1  = [0.79, 0.78, 0.78, 0.25, 0.06]
PRODUCT_IOU = [0.79, 0.75, 0.67, 0.27, 0.07]
CONSTIT_F1  = [0.58, 0.59, 0.63, 0.13, 0.00]
SOFT_F1     = [0.46, 0.51, 0.45, 0.13, 0.00]

# Token / time / cost aggregates.
INPUT_TOK   = [416000, 300262, 306392, 302812, 390311]   # full-res from xlsx Fig3 row
OUTPUT_TOK  = [ 40000,  41201,  44251,  44908,  47515]
TOTAL_TOK   = [i+o for i, o in zip(INPUT_TOK, OUTPUT_TOK)]
WALL_S      = [855,    784,    762,    1019,   1086]
COST_USD    = [0.23,   0.19,   0.20,   0.20,   0.24]    # @ $0.30/M in + $2.50/M out


def add_value_labels(ax, xs, ys, fmt="{:.2f}", dy=10, color=INK, fontsize=9):
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=fontsize, color=color)


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="both", color="#888", alpha=0.25, linewidth=0.5, linestyle="--")
    ax.tick_params(colors=INK, labelsize=10)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"  # Calibri-like fallback present everywhere

    fig, (axL, axR, axT) = plt.subplots(1, 3, figsize=(18.5, 5.6), facecolor="white")

    # ───────────── panel (a) — quality vs scale ─────────────
    axL.set_facecolor("white")
    series = [
        ("Partial F1 (Jaccard ≥ 0.5)", PARTIAL_F1,  SDK_BLUE,  "o"),
        ("Product IoU",                PRODUCT_IOU, DARK_BLUE, "s"),
        ("Constituent F1",             CONSTIT_F1,  WARM,      "^"),
        ("Soft F1",                    SOFT_F1,     GREY,      "D"),
    ]
    for label, ys, color, marker in series:
        axL.plot(SCALES, ys, color=color, linewidth=2.0, alpha=0.95, zorder=2)
        axL.scatter(SCALES, ys, s=110, c=color, marker=marker,
                    edgecolors="white", linewidths=1.2, zorder=3, label=label)

    # Annotate the Partial F1 cliff explicitly.
    add_value_labels(axL, SCALES, PARTIAL_F1, fmt="{:.2f}", dy=8,
                     color=SDK_BLUE, fontsize=9)

    axL.set_xscale("log")
    axL.set_xticks(SCALES)
    axL.set_xticklabels(LABELS,
                        fontsize=10, color=INK)
    axL.invert_xaxis()  # full on the left, tenth on the right (decreasing resolution →)
    axL.set_xlabel("Input-image scale factor (each side, log axis)",
                   fontsize=11, color=INK)
    axL.set_ylabel("Reference-based metric (macro-average, n=16)",
                   fontsize=11, color=INK)
    axL.set_ylim(-0.03, 1.0)
    axL.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axL.set_title("(a)  Quality vs input resolution",
                  fontsize=12, fontweight="bold", color=INK, pad=10)
    style_axes(axL)
    axL.legend(loc="lower left", frameon=False, fontsize=9.5,
               handletextpad=0.6, labelspacing=0.4)

    # ───────────── panel (b) — cost & wall time vs scale ─────────────
    axR.set_facecolor("white")

    axR.plot(SCALES, COST_USD, color=SDK_BLUE, linewidth=2.0, zorder=2)
    axR.scatter(SCALES, COST_USD, s=120, c=SDK_BLUE, marker="o",
                edgecolors="white", linewidths=1.2, zorder=3,
                label="Cost")
    add_value_labels(axR, SCALES, COST_USD, fmt="${:.2f}", dy=8,
                     color=SDK_BLUE, fontsize=9)

    axR.set_xscale("log")
    axR.set_xticks(SCALES)
    axR.set_xticklabels(LABELS,
                        fontsize=10, color=INK)
    axR.invert_xaxis()
    axR.set_xlabel("Input-image scale factor (each side, log axis)",
                   fontsize=11, color=INK)
    axR.set_ylabel("Projected cost (USD, 16 images)", fontsize=11, color=SDK_BLUE)
    axR.set_ylim(0.0, 0.40)
    axR.set_title("(b)  Cost and wall time vs input resolution",
                  fontsize=12, fontweight="bold", color=INK, pad=10)
    style_axes(axR)

    # Twin axis: wall-clock seconds.
    axR2 = axR.twinx()
    axR2.plot(SCALES, WALL_S, color=WARM, linewidth=2.0, zorder=2,
              linestyle="--")
    axR2.scatter(SCALES, WALL_S, s=120, c=WARM, marker="s",
                 edgecolors="white", linewidths=1.2, zorder=3,
                 label="Wall-clock seconds (16-image sum)")
    for x, y in zip(SCALES, WALL_S):
        axR2.annotate(f"{y:d} s", (x, y), xytext=(0, -16),
                      textcoords="offset points", ha="center", va="top",
                      fontsize=9, color=WARM)
    axR2.set_ylabel("Wall-clock time, sum across 16 images (s)",
                    fontsize=11, color=WARM)
    axR2.set_ylim(600, 1150)
    axR2.tick_params(axis="y", colors=WARM, labelsize=10)
    for spine in ("top",):
        axR2.spines[spine].set_visible(False)

    # Combined legend across both y-axes.
    lines_a, labels_a   = axR.get_legend_handles_labels()
    lines_b, labels_b   = axR2.get_legend_handles_labels()
    axR.legend(lines_a + lines_b, labels_a + labels_b,
               loc="upper left", frameon=False, fontsize=9.5,
               handletextpad=0.6, labelspacing=0.4)

    # ───────────── panel (c) — input / output tokens vs scale ─────────────
    axT.set_facecolor("white")

    axT.plot(SCALES, INPUT_TOK, color=SDK_BLUE, linewidth=2.0, zorder=2)
    axT.scatter(SCALES, INPUT_TOK, s=120, c=SDK_BLUE, marker="o",
                edgecolors="white", linewidths=1.2, zorder=3,
                label="Input tokens (16-image total)")
    for x, y in zip(SCALES, INPUT_TOK):
        axT.annotate(f"{y/1000:.0f}k", (x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=9, color=SDK_BLUE)

    axT.set_xscale("log")
    axT.set_xticks(SCALES)
    axT.set_xticklabels(LABELS,
                        fontsize=10, color=INK)
    axT.invert_xaxis()
    axT.set_xlabel("Input-image scale factor (each side, log axis)",
                   fontsize=11, color=INK)
    axT.set_ylabel("Input tokens, sum across 16 images", fontsize=11, color=SDK_BLUE)
    axT.set_ylim(0, 500_000)
    axT.set_yticks([0, 100_000, 200_000, 300_000, 400_000, 500_000])
    axT.set_yticklabels(["0", "100k", "200k", "300k", "400k", "500k"])
    axT.set_title("(c)  Token usage vs input resolution",
                  fontsize=12, fontweight="bold", color=INK, pad=10)
    style_axes(axT)

    # Twin axis: output tokens (different magnitude).
    axT2 = axT.twinx()
    axT2.plot(SCALES, OUTPUT_TOK, color=WARM, linewidth=2.0, zorder=2,
              linestyle="--")
    axT2.scatter(SCALES, OUTPUT_TOK, s=120, c=WARM, marker="s",
                 edgecolors="white", linewidths=1.2, zorder=3,
                 label="Output tokens (16-image total)")
    for x, y in zip(SCALES, OUTPUT_TOK):
        axT2.annotate(f"{y/1000:.1f}k", (x, y), xytext=(0, -16),
                      textcoords="offset points", ha="center", va="top",
                      fontsize=9, color=WARM)
    axT2.set_ylabel("Output tokens, sum across 16 images",
                    fontsize=11, color=WARM)
    axT2.set_ylim(30_000, 55_000)
    axT2.set_yticks([30_000, 35_000, 40_000, 45_000, 50_000, 55_000])
    axT2.set_yticklabels(["30k", "35k", "40k", "45k", "50k", "55k"])
    axT2.tick_params(axis="y", colors=WARM, labelsize=10)
    for spine in ("top",):
        axT2.spines[spine].set_visible(False)

    lines_c, labels_c = axT.get_legend_handles_labels()
    lines_d, labels_d = axT2.get_legend_handles_labels()
    axT.legend(lines_c + lines_d, labels_c + labels_d,
               loc="upper left", frameon=False, fontsize=9.5,
               handletextpad=0.6, labelspacing=0.4)

    # ───────────── overall title and footer ─────────────
    fig.suptitle("Gemini 3 Flash Preview — resolution ablation on the GT3_Maarten 16-image benchmark",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.00)
    fig.text(0.5, -0.04,
             "Single-SDK agent, gemini-3-flash-preview, identical agent loop and system prompt across all five conditions.   "
             "Scoring: scripts/eval_via_collective.py.   Cost projected from Google Standard-tier sticker; per-image .meta.json totals used for half / quarter / eighth / tenth.\n"
             "Effective DPI per condition = mean source-image DPI (124.8, from the 5/16 source PNGs carrying DPI metadata) × linear scale factor.",
             ha="center", va="top", fontsize=8.5, color="#555")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    print(f"Saved: {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
