#!/usr/bin/env python3
"""Compare frozen A3 and memory-aware A5-v2 under one corrected A4 protocol."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.memory_a5 import exact_evaluation  # noqa: E402
from coph_fork.memory_critic_a5 import (  # noqa: E402
    MemorySetCritic, delivery_posteriors, model_action,
)
from coph_fork.oracle import ExactOracleConfig  # noqa: E402
from coph_fork.run_memory_a5 import load_models  # noqa: E402
from coph_fork.train_memory_a5 import ledger_from_pattern  # noqa: E402


OUTPUT = HERE / "results" / "a5_memory_v2"
KINDS = ("additive", "deepsets", "pairwise")


def main():
    torch.set_num_threads(2)
    training = json.loads((OUTPUT / "summary.json").read_text())
    by_seed = {}
    for seed in (1701, 1702, 1703):
        a3, a4 = load_models(seed, seed + 100)
        config = ExactOracleConfig()
        posteriors, _, mass = delivery_posteriors(a4, config)
        seed_result = {"delivery_history_probability": {
            str(key): value for key, value in mass.items()
        }}
        for condition in ("independent", "shared_reset", "persistent", "shuffled",
                          "oracle_memory_v2"):
            result = exact_evaluation(condition, a3, a4, config,
                                      a5_posteriors=posteriors)
            result.pop("branches")
            seed_result[condition] = result
        seed_result["models"] = {}
        canonical = ledger_from_pattern((1, 1, -1, -1))
        key = (("geometry", True), ("traction", True))
        for kind in KINDS:
            model = MemorySetCritic(kind)
            model.load_state_dict(torch.load(
                OUTPUT / ("{}_{}.pth".format(kind, seed)),
                map_location="cpu", weights_only=True
            ))
            model.eval()
            result = exact_evaluation("learned_memory_v2", a3, a4, config,
                                      model, posteriors)
            result.pop("branches")
            baseline = seed_result["independent"]["expected_total_team_cost"]
            oracle = seed_result["oracle_memory_v2"]["expected_total_team_cost"]
            result["team_cost_gap_closure"] = (
                (baseline - result["expected_total_team_cost"]) / (baseline - oracle)
            )
            result["canonical_action"] = [list(atom) for atom in model_action(
                model, canonical, config, posterior=posteriors[key]
            )]
            result["heldout_regret"] = training["models"][str(seed)][kind]["heldout"]["regret"]
            seed_result["models"][kind] = result
        by_seed[str(seed)] = seed_result
    methods = ("independent", "shared_reset", "persistent", *KINDS,
               "oracle_memory_v2", "shuffled")
    metrics = ("expected_total_team_cost", "duplicate_measurement_rate",
               "region2_useful_acquisition_rate", "expected_nested_acquisition_regret")
    aggregate = {}
    for method in methods:
        rows = [by_seed[str(seed)]["models"][method] if method in KINDS
                else by_seed[str(seed)][method] for seed in (1701, 1702, 1703)]
        aggregate[method] = {metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "by_seed": [row[metric] for row in rows],
        } for metric in metrics}
    report = {
        "scope": "exact two-region finite A5-v2; frozen A4-v2, fixed route executor",
        "posterior_semantics": "carrier conditions on delivered content and silence under known A4 policy and packet channel",
        "route_semantics": "fixed executor uses explicit delivered/acquired evidence; posterior does not itself authorize top route",
        "by_seed": by_seed,
        "aggregate": aggregate,
        "sufficiency_audit": training["sufficiency_audit"],
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    plotted = ("independent", "shared_reset", "persistent", *KINDS,
               "oracle_memory_v2", "shuffled")
    for ax, metric, title in (
        (axes[0], "expected_total_team_cost", "Expected team cost"),
        (axes[1], "duplicate_measurement_rate", "Duplicate delivered measurements"),
        (axes[2], "region2_useful_acquisition_rate", "Useful Region-2 sensing"),
    ):
        values = [aggregate[name][metric]["mean"] for name in plotted]
        ax.bar(range(len(plotted)), values)
        ax.set_xticks(range(len(plotted)), [name.replace("_", "\n") for name in plotted],
                      rotation=25, ha="right")
        ax.set_ylim(bottom=0)
        ax.set_title(title)
    fig.savefig(OUTPUT / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({name: {metric: round(aggregate[name][metric]["mean"], 5)
                             for metric in metrics} for name in methods}, indent=2))


if __name__ == "__main__":
    main()
