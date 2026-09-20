#!/usr/bin/env python3
"""Plot exact physical marginal sensing value before and after scout memory."""

import json

import matplotlib.pyplot as plt
import numpy as np

from coph_fork.search_two_fork_physical_family import OUT


def main():
    frozen = json.loads((OUT / "frozen_physical_layout_v1.json").read_text())
    labels = ("R1 G", "R1 T", "R1 G+T", "R2 G", "R2 T", "R2 G+T")
    names = ("R1_G", "R1_T", "R1", "R2_G", "R2_T", "R2")
    colors = ("#4C78A8",) * 3 + ("#59A14F",) * 3
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, memory, title in zip(axes, ("M0", "M1"),
                                 ("No memory: choose Region 1",
                                  "Valid R1 memory: choose Region 2")):
        q = frozen["q"][memory]
        values = np.asarray([q["skip"] - q[name] for name in names])
        bars = ax.bar(np.arange(6), values, color=colors)
        for index in (2, 5):
            bars[index].set_edgecolor("#222222")
            bars[index].set_linewidth(1.3)
        ax.axhline(0, color="#333333", linewidth=.9)
        ax.axhline(.05, color="#888888", linewidth=.8,
                   linestyle="--", label="acceptance margin")
        ax.set_xticks(np.arange(6), labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Marginal value vs skip (team-cost units)")
    fig.suptitle("Team memory reallocates physically valuable sensing")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"frozen_two_fork_marginal_value.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
