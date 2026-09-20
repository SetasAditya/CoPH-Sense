#!/usr/bin/env python3
"""Read-only audit of frozen A4 checkpoints and their input representation."""

import json
from pathlib import Path

import numpy as np
import torch

from coph_fork.critic import SUBSETS, SetCostCritic
from coph_fork.oracle import ExactOracleConfig
from coph_fork.share_oracle import ExactShareOracle, ShareState
from coph_fork.train_share_critic import (
    baseline_metrics, build_features, generate, model_metrics,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "a4"


def warning_mask(data):
    state, _, labels, _, optimal, _ = data
    x = state.numpy()
    q = labels.sum(-1).numpy()
    acquired = x[:, 9:17:2] > 0.5
    values = x[:, 10:17:2] > 0.5
    ack_flags = x[:, 17:25:2] > 0.5
    ack_values = x[:, 18:25:2]
    receiver_p_g = np.where(ack_flags[:, 0], ack_values[:, 0], x[:, 0])
    receiver_p_t = np.where(ack_flags[:, 1], ack_values[:, 1], x[:, 1])
    p_top_safe = receiver_p_g * receiver_p_t
    carrier_prefers_top = (
        (x[:, 8] > 0.5)
        & (p_top_safe * x[:, 5] + (1 - p_top_safe) * x[:, 7] < x[:, 6])
    )
    bad_top = (acquired[:, :2] & ~values[:, :2]).any(axis=1)
    advantage = q[:, 0] - q[np.arange(len(x)), optimal]
    selected_bad = np.asarray([
        any(
            name in SUBSETS[int(chosen)] and acquired[j, i] and not values[j, i]
            for i, name in enumerate(("top_geometry", "top_traction"))
        )
        for j, chosen in enumerate(optimal)
    ])
    return carrier_prefers_top & bad_top & selected_bad & (advantage > 1e-7), advantage


def collision_witness():
    config = ExactOracleConfig(decision_mode="expected_cost")
    state = ShareState(acquired=(("top_geometry", False),), timely_probability=1.0)
    low_joint = {
        (True, True, True, True): 0.5,
        (True, False, True, True): 0.25,
        (False, True, True, True): 0.25,
    }
    high_joint = {
        (True, True, True, True): 0.75,
        (False, False, True, True): 0.25,
    }
    x_low, candidates_low = build_features(config, low_joint, state)
    x_high, candidates_high = build_features(config, high_joint, state)
    oracle_low = ExactShareOracle(state, config, low_joint).solve()[0]
    oracle_high = ExactShareOracle(state, config, high_joint).solve()[0]
    return {
        "same_old_state_input": bool(np.array_equal(x_low, x_high)),
        "same_old_candidate_inputs": bool(np.array_equal(candidates_low, candidates_high)),
        "low_joint_optimum": list(oracle_low.message_set),
        "high_joint_optimum": list(oracle_high.message_set),
        "low_joint_top_good_probability": 0.5,
        "high_joint_top_good_probability": 0.75,
        "explanation": "Old A4 features expose only prior marginals, although expected-cost route choice depends on their joint probability.",
    }


def main():
    torch.set_num_threads(2)
    rows = {}
    for seed in (1801, 1802, 1803):
        folder = RESULTS / ("seed" + str(seed))
        report = json.loads((folder / "report.json").read_text())
        train = generate(600, seed)
        test = generate(200, seed + 2)
        train_warning, train_advantage = warning_mask(train)
        test_warning, test_advantage = warning_mask(test)
        report["baselines"] = baseline_metrics(test)
        model_audit = {}
        for kind in ("additive", "deepsets", "pairwise"):
            model = SetCostCritic(train[0].shape[-1], train[1].shape[-1], kind)
            model.load_state_dict(torch.load(
                folder / (kind + ".pth"), map_location="cpu", weights_only=True
            ))
            model.eval()
            metrics = model_metrics(model, test)
            with torch.no_grad():
                chosen = model(test[0], test[1]).sum(-1).masked_fill(
                    ~test[3], float("inf")
                ).argmin(-1).numpy()
            warning_indices = np.flatnonzero(test_warning)
            warning_recall = float(np.mean([
                set(SUBSETS[int(test[4][j])]).issubset(SUBSETS[int(chosen[j])])
                for j in warning_indices
            ])) if len(warning_indices) else None
            report["models"][kind]["test"] = metrics
            model_audit[kind] = {
                "test_missed_useful_value": metrics["missed_useful_value"],
                "critical_recall_at_0_5": metrics["critical_recall_at_0_5"],
                "standard_test_bad_warning_recall": warning_recall,
            }
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        rows[str(seed)] = {
            "train_warning_count": int(train_warning.sum()),
            "test_warning_count": int(test_warning.sum()),
            "mean_train_warning_advantage": float(train_advantage[train_warning].mean()),
            "mean_test_warning_advantage": float(test_advantage[test_warning].mean()),
            "models": model_audit,
        }
    result = {"frozen_checkpoint_audit": rows, "input_collision": collision_witness()}
    path = RESULTS / "input_and_warning_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
