#!/usr/bin/env python3
"""Make a self-contained A5-v2 figure from frozen exact-evaluation results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "a5_memory_v2" / "comparison.json"
OUT = HERE / "results" / "a5_memory_v2"

INK = "#17243A"
MUTED = "#56657B"
GRID = "#DCE4EC"
BG = "#F7F9FC"
PURPLE = "#7254A3"
TEAL = "#00877F"
GREEN = "#3F8065"
AMBER = "#A66B1B"
RED = "#B55054"
GRAY = "#526477"


def rounded(ax, xy, wh, face, edge="none", radius=0.04, linewidth=1):
    patch = FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.012,rounding_size={}".format(radius),
                           facecolor=face, edgecolor=edge, linewidth=linewidth,
                           transform=ax.transAxes, clip_on=False)
    ax.add_patch(patch)
    return patch


def arrow(ax, a, b, color=MUTED, width=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=width, color=color, transform=ax.transAxes))


def add_top_diagram(ax):
    ax.set_axis_off()
    ax.text(0, 1.02, "A  What the carrier knows before deciding", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=INK, va="bottom")
    rounded(ax, (0.015, 0.13), (0.43, 0.75), "#E9F4F2", edge="#A6D3CA")
    rounded(ax, (0.555, 0.13), (0.43, 0.75), "#FFF4E4", edge="#E8C88D")
    ax.text(0.23, 0.78, "REGION 1", ha="center", va="center", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=TEAL)
    ax.text(0.77, 0.78, "REGION 2", ha="center", va="center", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=AMBER)
    ax.text(0.23, 0.60, "Scout measured geometry + traction", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color=INK)
    ax.text(0.23, 0.43, "A4 sent; both readings arrived", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color=INK)
    ax.text(0.23, 0.25, "Carrier ledger:  G₁ ✓   T₁ ✓", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEAL)
    ax.text(0.77, 0.60, "Geometry and traction unresolved", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color=INK)
    ax.text(0.77, 0.43, "Shortcut value depends on both", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color=INK)
    ax.text(0.77, 0.25, "Carrier ledger:  G₂ ?   T₂ ?", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color=AMBER)
    arrow(ax, (0.46, 0.50), (0.54, 0.50), color=INK)


def add_canonical(ax):
    ax.set_axis_off()
    ax.text(0, 1.02, "B  Same history, different acquisition", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=INK, va="bottom")
    rounded(ax, (0.01, 0.52), (0.98, 0.37), "#F1EAF8", edge="#D5C5E8")
    rounded(ax, (0.01, 0.08), (0.98, 0.37), "#E5F5F2", edge="#A9DAD1")
    ax.text(0.05, 0.80, "Frozen one-fork A3", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=PURPLE)
    ax.text(0.05, 0.65, "Re-measure  G₁ + T₁", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0.63, 0.65, "value  −0.08", transform=ax.transAxes,
            fontsize=12, color=PURPLE, fontweight="bold")
    ax.text(0.05, 0.36, "Memory-aware A5-v2", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=TEAL)
    ax.text(0.05, 0.20, "Inspect  G₂ + T₂", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0.63, 0.20, "value  +0.42", transform=ax.transAxes,
            fontsize=12, color=TEAL, fontweight="bold")
    ax.text(0.02, -0.035,
            "All-good world; both packets delivered (seed 1701). Branch cost: 4.2088 → 2.2088.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")


def metric_panel(ax, title, metric, methods, data, xlim, formatter, oracle_line=None):
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_facecolor("white")
    n = len(methods)
    ax.set_ylim(-0.55, n - 0.45)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_yticks(range(n))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if oracle_line is not None:
        ax.axvline(oracle_line, ls="--", lw=1, color=GREEN, alpha=0.55)
    for y, (name, label, color) in enumerate(methods):
        values = data[name][metric]["by_seed"]
        mean = data[name][metric]["mean"]
        if name == "independent" and metric == "duplicate_measurement_rate":
            ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), y, "n/a¹", va="center",
                    color=MUTED, fontsize=9)
            continue
        ax.scatter(values, [y] * len(values), s=18, color=color, alpha=0.30, zorder=3)
        ax.scatter([mean], [y], s=76, color=color, edgecolor="white", lw=1.1, zorder=4)
        ax.annotate(formatter(mean), (mean, y), xytext=(7, -1), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center", fontweight="bold")
    return ax


def main():
    report = json.loads(DATA.read_text())
    agg = report["aggregate"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(15.6, 8.3), facecolor="white")
    top = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.13,
                           left=0.055, right=0.975, top=0.82, bottom=0.53)
    add_top_diagram(fig.add_subplot(top[0]))
    add_canonical(fig.add_subplot(top[1]))

    fig.text(0.055, 0.955, "Cooperative memory changes what the carrier chooses to learn",
             fontsize=21, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.910,
             "A5-v2 finite two-region task  •  A4 sharing frozen  •  exact world and delivery enumeration",
             fontsize=10.8, color=MUTED, va="top")

    bottom = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.19,
                              left=0.17, right=0.975, top=0.44, bottom=0.14)
    methods = [
        ("independent", "No sharing", GRAY),
        ("shared_reset", "Shared, reset", AMBER),
        ("persistent", "Frozen A3 + memory", PURPLE),
        ("additive", "A5-v2 memory critic", TEAL),
        ("oracle_memory_v2", "Exact reference", GREEN),
        ("shuffled", "Wrong-region memory", RED),
    ]
    a = fig.add_subplot(bottom[0])
    b = fig.add_subplot(bottom[1], sharey=a)
    c = fig.add_subplot(bottom[2], sharey=a)
    metric_panel(a, "C  Expected team cost  ↓", "expected_total_team_cost",
                 methods, agg, (5.10, 6.27), lambda x: "{:.3f}".format(x),
                 oracle_line=agg["oracle_memory_v2"]["expected_total_team_cost"]["mean"])
    metric_panel(b, "Repeated delivered evidence  ↓", "duplicate_measurement_rate",
                 methods, agg, (-0.02, 1.02), lambda x: "{:.1f}%".format(100*x))
    metric_panel(c, "Useful Region-2 acquisition  ↑", "region2_useful_acquisition_rate",
                 methods, agg, (-0.02, 1.13), lambda x: "{:.1f}%".format(100*x))
    a.set_yticklabels([label for _, label, _ in methods], fontsize=9.5, color=INK)
    b.tick_params(axis="y", labelleft=False)
    c.tick_params(axis="y", labelleft=False)
    fig.text(0.055, 0.066,
             "Dots: three paired checkpoints; large dot: mean.  ¹No-sharing agents have no delivered evidence to duplicate.  "
             "Exact reference optimizes the restricted carrier acquisition menu under frozen A4 and the fixed route executor.",
             fontsize=8.8, color=MUTED)
    fig.text(0.055, 0.037,
             "Scope: noiseless finite information task; this figure does not evaluate two-region VMAS motion or port-Hamiltonian control.",
             fontsize=8.8, color=MUTED)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("a5_story." + extension), dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
