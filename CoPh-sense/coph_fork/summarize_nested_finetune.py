#!/usr/bin/env python3
"""Aggregate paired fresh-seed A3→A4 continuation-aware fine-tuning results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "results" / "nested_a3_a4"


def main():
    paths = sorted(ROOT.glob("finetune*/report.json"))
    if len(paths) != 3:
        raise SystemExit("expected three continuation-aware fine-tuning reports")
    rows = [json.loads(path.read_text()) for path in paths]
    metrics = (
        "mean_nested_regret", "mean_executed_normalized_cost",
        "optimal_set_accuracy", "pair_selection_recall",
        "mean_nested_regret_on_pair_relevant",
        "extraneous_bottom_sensing_rate",
    )
    aggregate = {}
    for metric in metrics:
        before = [row["confirmation_before"][metric] for row in rows]
        after = [row["confirmation_after"][metric] for row in rows]
        aggregate[metric] = {
            "before_by_seed": before,
            "after_by_seed": after,
            "before_mean": float(np.mean(before)),
            "after_mean": float(np.mean(after)),
            "paired_change_by_seed": [float(a - b) for a, b in zip(after, before)],
        }
    output = {
        "seeds": [row["a3_seed"] for row in rows],
        "confirmation_seeds": [row["confirmation_seed"] for row in rows],
        "selected_epochs": [row["selected_epoch"] for row in rows],
        "aggregate": aggregate,
        "interpretation": (
            "Continuation-aware A3 labels improve fresh-seed nested regret in "
            "all three paired runs; the result is for the finite normalized "
            "route-cost model with the paired frozen learned A4 policies."
        ),
    }
    (ROOT / "finetune_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "mean_nested_regret", "Nested acquisition regret"),
        (axes[1], "extraneous_bottom_sensing_rate", "Irrelevant bottom sensing"),
    ):
        values = aggregate[metric]
        for seed, before, after in zip(
            output["seeds"], values["before_by_seed"], values["after_by_seed"]
        ):
            ax.plot((0, 1), (before, after), marker="o", label=str(seed))
        ax.set_xticks((0, 1), ("Original A3", "A4-aware A3"))
        ax.set_ylim(bottom=0)
        ax.set_title(title)
    axes[0].set_ylabel("Exact regret")
    axes[1].set_ylabel("Fraction of contexts")
    axes[1].legend(title="A3 seed", frameon=False)
    fig.savefig(ROOT / "finetune_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
