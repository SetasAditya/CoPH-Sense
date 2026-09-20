"""Render a compact, scope-aware CoPH-Sense research status graphic."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
NAVY = "#17324d"
TEAL = "#117e73"
BLUE = "#3566b7"
AMBER = "#ba6b19"
PALE = "#f3f7fb"


def box(ax, xy, wh, title, body, edge, fill):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.014,rounding_size=0.025",
                 linewidth=1.5, edgecolor=edge, facecolor=fill))
    ax.text(x + .025, y + h - .045, title, ha="left", va="top",
            fontsize=11.8, fontweight="bold", color=edge)
    ax.text(x + .025, y + h - .112, body, ha="left", va="top",
            fontsize=9.5, color=NAVY, linespacing=1.3)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=18, linewidth=1.6, color="#6d8196"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.04, .95, "CoPH-Sense: demonstrated mechanism, open learning gate",
            fontsize=21, fontweight="bold", color=NAVY, va="top")
    ax.text(.04, .895, "Status for Prof. Bajaj  |  18 September 2026",
            fontsize=11, color="#5a6f82", va="top")

    box(ax, (.04, .525), (.27, .305), "1  Physical mechanism",
        "Scout measures R1 and sends evidence.\n"
        "Carrier avoids a repeat R1 probe\n"
        "and measures unresolved R2.\n"
        "Canonical team cost: 2.116 vs 3.089\n"
        "with reset memory.", TEAL, "#e8f6f1")
    arrow(ax, .32, .355, .68)
    box(ax, (.365, .525), (.27, .305), "2  pH integration",
        "Fixed material-aware pH execution\n"
        "uses the updated terrain belief.\n"
        "NEED-conditioned A4 learns selective\n"
        "compact sharing in a controlled\n"
        "bidirectional task.", BLUE, "#edf3fc")
    arrow(ax, .645, .68, .68)
    box(ax, (.69, .525), (.27, .305), "3  Learning gate open",
        "Validation: 0 pair-optimal states;\n"
        "0 memory-switch states.\n"
        "Full 32-state training fit: 25/32\n"
        "exact actions after 3,000 steps.\n"
        "Final evaluation has not launched.", AMBER, "#fff4e7")

    ax.add_patch(FancyBboxPatch((.04, .29), .92, .18,
                 boxstyle="round,pad=0.014,rounding_size=0.025",
                 linewidth=1, edgecolor="#ccd8e2", facecolor=PALE))
    ax.text(.065, .435, "What the latest diagnostic isolates", fontsize=13,
            fontweight="bold", color=NAVY, va="top")
    ax.text(.065, .382,
            "The same critic fits the two-state R1→R2 memory switch exactly, but does not fit the full 32-state training set. "
            "The held-out challenge also lacks the decisive mechanism cases.",
            fontsize=11.1, color=NAVY, va="top", wrap=True)

    ax.text(.04, .235, "Next gate", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(.16, .235,
            "audit belief-conditional teacher targets → certify pair/memory states under frozen pH → one A3/A5 fit → paired evaluation",
            fontsize=11.2, color=NAVY, va="center")
    ax.text(.04, .11,
            "Scope: the R1→R2 replay is a restricted scripted-VMAS example; the NEED result is a controlled communication task. "
            "Neither is a held-out E2 result or proof of structural pH superiority.",
            fontsize=10, color="#5a6f82", va="center")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"status_figure.{ext}", dpi=180,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
