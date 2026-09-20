"""Rebuild the progress report's numerical plots and maze risk maps.

All numerical panels read frozen result JSON. The maze panels are re-rendered
from the current CoPH-Terrain generator rather than cropped from old images.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coph_terrain.generator import generate_map, realize_map


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORK = ROOT / "coph_fork" / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def load(path):
    return json.loads(path.read_text())


def maze_figure():
    cases = (("Shallow labyrinth", "labyrinth", 600002),
             ("Deep labyrinth (OOD family)", "deep_labyrinth", 600003))
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 8.6),
                             sharex=True, sharey=True,
                             layout="constrained")
    risk_map = plt.colormaps["YlOrRd"].copy()
    risk_map.set_bad("#2e3440")
    image = None
    for row, (name, family, seed) in enumerate(cases):
        terrain = realize_map(generate_map(family, seed), 0)
        occupancy = terrain.occupancy.T
        risk = np.ma.masked_where(occupancy, terrain.risk.T)
        a, b = axes[row]
        a.imshow(occupancy, origin="lower", extent=(-6, 6, -6, 6),
                 interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        image = b.imshow(risk, origin="lower", extent=(-6, 6, -6, 6),
                         interpolation="bilinear", cmap=risk_map,
                         vmin=0, vmax=1)
        for axis in (a, b):
            axis.scatter([-5, -5], [.25, -.25], s=50,
                         c=["#1d986f", "#3776c8"], edgecolors="white",
                         linewidths=.6, zorder=4)
            axis.scatter([5], [0], s=110, marker="*", c="#9b42c9",
                         edgecolors="white", linewidths=.5, zorder=4)
            axis.set_aspect("equal")
            axis.set_xlim(-6, 6)
            axis.set_ylim(-6, 6)
            axis.set_xticks((-6, -3, 0, 3, 6))
            axis.set_yticks((-6, -3, 0, 3, 6))
        a.set_ylabel(f"{name}\nseed {seed}\ny (m)")
    axes[0, 0].set_title("Public obstacle layout")
    axes[0, 1].set_title("Hidden realized material risk")
    for axis in axes[-1]:
        axis.set_xlabel("x (m)")
    fig.colorbar(image, ax=axes[:, 1], shrink=.78, pad=.025,
                 label=r"$r=0.4g+0.4t+0.2gt$ (clipped)")
    fig.savefig(OUT / "maze_layouts_risk.pdf", bbox_inches="tight")
    fig.savefig(OUT / "maze_layouts_risk.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def results_figure():
    a3 = load(FORK / "a3" / "summary.json")["aggregate"]
    a4 = load(FORK / "a4_v2" / "summary.json")["aggregate"]
    a5 = load(FORK / "a5_memory_v2" / "comparison.json")["aggregate"]
    ph = load(FORK / "ph_factorial" / "summary.json")["cells"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), layout="constrained")
    colors = ("#9aabc0", "#62a6a0", "#327d72", "#1c6b62", "#af79ba")

    labels = ("Additive", "DeepSets", "Pairwise")
    keys = ("additive", "deepsets", "pairwise")
    values = [a3[k]["mean_acquisition_regret"]["mean"] for k in keys]
    errors = [a3[k]["mean_acquisition_regret"]["sample_std"] for k in keys]
    ax = axes[0, 0]
    ax.bar(labels, values, color=colors[:3], yerr=errors, capsize=4)
    ax.set_title("A3: finite acquisition regret")
    ax.set_ylabel("Mean regret (cost units)")
    ax.grid(axis="y", alpha=.2)

    labels = ("Additive", "Weighted", "Pairwise")
    keys = ("additive_unweighted", "additive_weighted", "pairwise_weighted")
    values = [a4[k]["standard_regret"]["mean"] for k in keys]
    errors = [a4[k]["standard_regret"]["sample_std"] for k in keys]
    ax = axes[0, 1]
    ax.bar(labels, values, color=colors[:3], yerr=errors, capsize=4)
    ax.set_title("A4: post-reading sharing regret")
    ax.set_ylabel("Mean regret (cost units)")
    ax.grid(axis="y", alpha=.2)

    labels = ("Independent", "Shared reset", "Old memory", "A5-v2", "Oracle")
    keys = ("independent", "shared_reset", "persistent", "additive",
            "oracle_memory_v2")
    values = [a5[k]["expected_total_team_cost"]["mean"] for k in keys]
    ax = axes[1, 0]
    ax.bar(range(len(keys)), values, color=colors)
    ax.set_xticks(range(len(keys)), labels, rotation=23, ha="right")
    ax.set_ylim(4.9, 6.1)
    ax.set_title("A5: finite memory changes later sensing")
    ax.set_ylabel("Expected total team cost")
    ax.grid(axis="y", alpha=.2)

    labels = ("Ordinary\ninfo", "CoPH\ninfo")
    direct = (ph["ordinary_direct"]["mean_team_cost"],
              ph["coph_direct"]["mean_team_cost"])
    split = (ph["ordinary_ph"]["mean_team_cost"],
             ph["coph_ph"]["mean_team_cost"])
    ax = axes[1, 1]
    x = np.arange(2)
    ax.bar(x-.18, direct, width=.36, label="direct integrator",
           color="#638abd")
    ax.bar(x+.18, split, width=.36, label="split pH integrator",
           color="#51957d")
    ax.set_xticks(x, labels)
    ax.set_ylim(.62, .73)
    ax.set_title("pH integration: 16 paired scripted worlds")
    ax.set_ylabel("Mean team cost")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "progress_results.pdf", bbox_inches="tight")
    fig.savefig(OUT / "progress_results.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    maze_figure()
    results_figure()
    print(OUT)
