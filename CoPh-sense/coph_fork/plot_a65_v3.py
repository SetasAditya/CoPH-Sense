#!/usr/bin/env python3
"""Plot paired physical acquisition values from saved A6.5-v3 audits."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coph_fork.run_moving_bridge_a65 import OUT


def main():
    deterministic = json.loads((OUT / "v3_oracle_summary.json").read_text())
    communication = json.loads((OUT / "v3_comm_summary.json").read_text())
    labels = ["low value", "useful", "expensive probe", "7 s delay",
              "20% delivery", "short range"]
    items = deterministic + communication
    inspect = [item["mean_decision_cost"]["always" if i < 4 else "inspect"]
               for i, item in enumerate(items)]
    skip = [item["mean_decision_cost"]["never" if i < 4 else "skip"]
            for i, item in enumerate(items)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.bar(x - 0.18, skip, width=0.36, color="#4c78a8", label="Skip")
    ax.bar(x + 0.18, inspect, width=0.36, color="#f58518", label="Inspect")
    for i, (a, b) in enumerate(zip(skip, inspect)):
        winner = "INSPECT" if b < a else "SKIP"
        ax.text(i, max(a, b) + 0.04, winner, ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(skip + inspect) + 0.35)
    ax.set_ylabel("Expected post-decision VMAS team cost ↓")
    ax.set_title("A6.5-v3: physical value of scout inspection")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "v3_oracle_decisions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
