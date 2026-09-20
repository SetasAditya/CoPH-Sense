#!/usr/bin/env python3
"""Evaluate frozen A3/A4 on the exact two-region cooperative-memory task."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.critic import SetCostCritic  # noqa: E402
from coph_fork.memory_a5 import exact_evaluation  # noqa: E402
from coph_fork.oracle import ExactOracleConfig  # noqa: E402


OUTPUT = HERE / "results" / "a5_memory"
CONDITIONS = (
    "independent", "shared_reset", "persistent", "oracle_memory", "shuffled"
)


def load_models(a3_seed, a4_seed):
    a3 = SetCostCritic(12, 8, "pairwise")
    a3.load_state_dict(torch.load(
        HERE / "results" / "nested_a3_a4" / ("finetune" + str(a3_seed)) /
        "pairwise_nested.pth", map_location="cpu", weights_only=True
    ))
    a4 = SetCostCritic(32, 9, "pairwise")
    a4.load_state_dict(torch.load(
        HERE / "results" / "a4_v2" / ("seed" + str(a4_seed)) /
        "pairwise_weighted.pth", map_location="cpu", weights_only=True
    ))
    a3.eval()
    a4.eval()
    return a3, a4


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = ExactOracleConfig()
    by_seed = {}
    all_branches = {}
    for a3_seed, a4_seed in ((1701, 1801), (1702, 1802), (1703, 1803)):
        a3, a4 = load_models(a3_seed, a4_seed)
        key = str(a3_seed)
        by_seed[key] = {}
        all_branches[key] = {}
        for condition in CONDITIONS:
            result = exact_evaluation(condition, a3, a4, config)
            all_branches[key][condition] = result.pop("branches")
            by_seed[key][condition] = result
    metrics = (
        "expected_total_team_cost", "expected_scout_sensing_cost",
        "expected_communication_cost", "expected_carrier_sensing_cost",
        "expected_route_cost", "redundant_batch_rate",
        "duplicate_measurement_rate", "region2_useful_acquisition_rate",
        "unsafe_execution_rate", "expected_nested_acquisition_regret",
    )
    aggregate = {
        condition: {
            metric: {
                "mean": float(np.mean(values)),
                "by_seed": values,
            }
            for metric, values in {
                metric: [by_seed[str(seed)][condition][metric] for seed in (1701, 1702, 1703)]
                for metric in metrics
            }.items()
        }
        for condition in CONDITIONS
    }
    canonical = {}
    for seed, condition_rows in all_branches.items():
        canonical[seed] = {}
        for condition, rows in condition_rows.items():
            matches = [row for row in rows if row["world"] == [True] * 4
                       and (condition == "independent" or len(row["delivered_ids"]) == 2)]
            if matches:
                row = matches[0]
                canonical[seed][condition] = {
                    "carrier_action": row["carrier_action"],
                    "exact_acquisition_value": row["exact_acquisition_value"],
                    "total_team_cost": row["total_team_cost"],
                }
    report = {
        "scope": "exact finite two-region route-cost diagnostic; not VMAS or pH motion",
        "worlds": 16,
        "scout_action": "acquire Region-1 geometry and traction in every episode",
        "carrier_action_menu": "none, either singleton or pair in one region; at most two measurements",
        "channel": "independent evidence delivery with probability 0.9; successful delivery triggers a charged ACK",
        "memory_semantics": "received evidence persists only in persistent and oracle_memory; shared_reset clears only the acquisition actor's memory; shuffled relabels received evidence to the wrong region",
        "policy_scope": "frozen one-region continuation-aware A3 applied per region with posterior-prior adapter; frozen A4-v2 shares scout readings",
        "config": {
            "sensing_cost": config.sensing_cost,
            "packet_cost": config.packet_cost,
            "ack_payload_bytes": config.ack_payload_bytes,
            "top_safe_cost": config.top_safe_cost,
            "bottom_safe_cost": config.bottom_safe_cost,
            "unsafe_route_cost": config.unsafe_route_cost,
        },
        "by_seed": by_seed,
        "aggregate": aggregate,
        "canonical_all_good_trace": canonical,
        "status": "A5 not passed: frozen A3 still repeats delivered Region-1 evidence",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "branches.json").write_text(json.dumps(all_branches, indent=2) + "\n")
    labels = [name.replace("_", "\n") for name in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "expected_total_team_cost", "Team cost"),
        (axes[1], "duplicate_measurement_rate", "Repeated delivered evidence"),
        (axes[2], "region2_useful_acquisition_rate", "Useful Region-2 sensing"),
    ):
        values = [aggregate[name][metric]["mean"] for name in CONDITIONS]
        ax.bar(range(len(CONDITIONS)), values)
        ax.set_xticks(range(len(CONDITIONS)), labels)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    fig.savefig(OUTPUT / "a5_memory_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({name: {
        metric: round(aggregate[name][metric]["mean"], 4)
        for metric in ("expected_total_team_cost", "redundant_batch_rate",
                       "duplicate_measurement_rate", "region2_useful_acquisition_rate",
                       "expected_nested_acquisition_regret")
    } for name in CONDITIONS}, indent=2))


if __name__ == "__main__":
    main()
