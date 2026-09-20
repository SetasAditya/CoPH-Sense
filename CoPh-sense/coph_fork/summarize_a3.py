#!/usr/bin/env python3
"""Aggregate three frozen A3 seeds and audit the canonical A2 state."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.critic import SUBSETS, SetCostCritic  # noqa: E402
from coph_fork.train_critic import canonical_a2_state  # noqa: E402


def main():
    root = HERE / "results" / "a3"
    paths = [root / "report.json", root / "seed1702" / "report.json", root / "seed1703" / "report.json"]
    reports = [json.loads(path.read_text()) for path in paths]
    state, candidates, labels, oracle_index = canonical_a2_state()
    canonical = {}
    for path, report in zip(paths, reports):
        for kind in ("additive", "deepsets", "pairwise"):
            model = SetCostCritic(len(state), candidates.shape[-1], kind)
            model.load_state_dict(torch.load(path.parent / (kind + ".pth"), map_location="cpu"))
            model.eval()
            with torch.no_grad():
                prediction = model(
                    torch.tensor(state[None, :]),
                    torch.tensor(candidates[None, :, :]),
                ).sum(-1)[0].numpy()
            index = int(prediction.argmin())
            canonical.setdefault(kind, []).append(
                {
                    "seed": report["setup"]["seed"],
                    "selected_subset": list(SUBSETS[index]),
                    "exact_regret": float(labels[index].sum() - labels[oracle_index].sum()),
                    "predicted_q": {
                        str(subset): float(prediction[SUBSETS.index(subset)])
                        for subset in (
                            (), ("top_geometry",), ("top_traction",),
                            ("top_geometry", "top_traction"),
                        )
                    },
                }
            )

    fields = (
        "mean_acquisition_regret", "top1_accuracy", "pair_selection_recall",
        "harmful_sensing_rate", "value_rmse", "ranking_accuracy",
        "complementarity_sign_accuracy",
    )
    aggregate = {}
    for kind in ("additive", "deepsets", "pairwise"):
        aggregate[kind] = {}
        for field in fields:
            values = [report["models"][kind]["test"][field] for report in reports]
            aggregate[kind][field] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "per_seed": values,
            }
    result = {
        "seeds": [report["setup"]["seed"] for report in reports],
        "aggregate": aggregate,
        "canonical_a2_audit": canonical,
        "true_canonical_q": {
            str(subset): float(labels[SUBSETS.index(subset)].sum())
            for subset in (
                (), ("top_geometry",), ("top_traction",),
                ("top_geometry", "top_traction"),
            )
        },
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    kinds = ("additive", "deepsets", "pairwise")
    colors = ("#999999", "#4c78a8", "#54a24b")
    for index, kind in enumerate(kinds):
        regrets = aggregate[kind]["mean_acquisition_regret"]["per_seed"]
        recall = aggregate[kind]["pair_selection_recall"]["per_seed"]
        axes[0].scatter([index] * len(regrets), regrets, color=colors[index], alpha=0.65)
        axes[0].plot(index, np.mean(regrets), "_", color="black", markersize=20)
        axes[1].scatter([index] * len(recall), recall, color=colors[index], alpha=0.65)
        axes[1].plot(index, np.mean(recall), "_", color="black", markersize=20)
    for axis in axes:
        axis.set_xticks(range(len(kinds)))
        axis.set_xticklabels(kinds)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set(ylabel="Exact acquisition regret", title="Lower is better")
    axes[1].set(ylabel="Pair-selection recall", ylim=(0, 1.05), title="Complementary states")
    figure.tight_layout()
    figure.savefig(root / "a3_three_seed_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
