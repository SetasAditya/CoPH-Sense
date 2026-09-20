"""Render reproducible procedural terrain layouts and hidden risk realizations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from coph_terrain.generator import FAMILIES, generate_map, realize_map


ROOT = Path(__file__).resolve().parent
SEEDS = (600000, 600001, 600002, 600003, 600004)


def main():
    fig, axes = plt.subplots(len(FAMILIES), 2, figsize=(11.6, 19.5),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("CoPH-Terrain: procedural layouts and material-risk realizations",
                 fontsize=17, weight="bold")
    obstacle_colors = ListedColormap(["#f7f7f2", "#30343b"])
    risk_cmap = plt.colormaps["YlOrRd"].copy()
    risk_cmap.set_bad("#30343b")
    image = None
    for row, (family, seed) in enumerate(zip(FAMILIES, SEEDS)):
        terrain = realize_map(generate_map(family, seed), realization_seed=0)
        occupancy = terrain.occupancy.T
        risk = np.ma.masked_where(occupancy, terrain.risk.T)
        left, right = axes[row]
        left.imshow(occupancy, origin="lower", extent=(-6, 6, -6, 6),
                    interpolation="nearest", cmap=obstacle_colors, vmin=0, vmax=1)
        image = right.imshow(risk, origin="lower", extent=(-6, 6, -6, 6),
                             interpolation="bilinear", cmap=risk_cmap,
                             vmin=0, vmax=1)
        label = family.replace("_", " ").title()
        if family in ("deep_labyrinth", "irregular"):
            label += " (OOD family)"
        left.set_ylabel(f"{label}\nseed {seed}\n\ny (m)", fontsize=11)
        for axis in (left, right):
            axis.set_aspect("equal")
            axis.set_xlim(-6, 6)
            axis.set_ylim(-6, 6)
            axis.set_xticks((-6, -3, 0, 3, 6))
            axis.set_yticks((-6, -3, 0, 3, 6))
            axis.scatter([-5, -5], [.25, -.25], s=80,
                         c=["#35c970", "#4cb8f8"], edgecolors="#1f2937",
                         linewidths=.8, zorder=5)
            axis.scatter([5], [0], s=150, marker="*", c="#b643ff",
                         edgecolors="#1f2937", linewidths=.7, zorder=5)
        left.text(-5.45, .62, "scout", fontsize=8)
        left.text(-5.45, -.73, "carrier", fontsize=8)
        left.text(4.55, .5, "goal", fontsize=8)
    axes[0, 0].set_title("Obstacle layout (dark = blocked)", fontsize=13)
    axes[0, 1].set_title("True material risk (blocked cells dark)", fontsize=13)
    for axis in axes[-1]:
        axis.set_xlabel("x (m)")
    colorbar = fig.colorbar(image, ax=axes[:, 1], shrink=.62, pad=.025,
                            label=r"$r_{\mathrm{true}}=\mathrm{clip}((0.55-\mu)/0.40,0,1)$")
    colorbar.set_ticks((0, .25, .5, .75, 1))
    ROOT.mkdir(parents=True, exist_ok=True)
    png = ROOT / "terrain_layouts_and_risk.png"
    pdf = ROOT / "terrain_layouts_and_risk.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
