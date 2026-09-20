#!/usr/bin/env python3
"""Aggregate A4-v2 tests and canonical interventions across frozen seeds."""

import json
from pathlib import Path

import numpy as np
import torch

from coph_fork.critic import SUBSETS, SetCostCritic
from coph_fork.share_oracle import ExactShareOracle
from coph_fork.summarize_a4 import cases
from coph_fork.train_share_critic import build_features_v2


ROOT = Path(__file__).resolve().parent / "results" / "a4_v2"


def selected(model, state, config, prior):
    x, e = build_features_v2(config, prior, state)
    feasible = set(ExactShareOracle(state, config, prior).candidate_sets())
    with torch.no_grad():
        costs = model(torch.tensor(x)[None], torch.tensor(e)[None])[0].sum(-1).numpy()
    return min(
        (subset for subset in SUBSETS if subset in feasible),
        key=lambda subset: (costs[SUBSETS.index(subset)], len(subset), subset),
    )


def main():
    paths = sorted(ROOT.glob("seed*/report.json"))
    if len(paths) != 3:
        raise SystemExit("expected three A4-v2 seed reports")
    reports = [json.loads(path.read_text()) for path in paths]
    names = list(reports[0]["models"])
    aggregate = {}
    for name in names:
        entries = [report["models"][name] for report in reports]
        aggregate[name] = {
            metric: {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
            for metric, values in {
                "standard_regret": [entry["standard_test"]["mean_regret"] for entry in entries],
                "standard_wasteful_send_rate": [entry["standard_test"]["wasteful_send_rate"] for entry in entries],
                "standard_missed_useful_value": [entry["standard_test"]["missed_useful_value"] for entry in entries],
                "standard_critical_recall": [entry["standard_test"]["critical_recall_at_0_5"] for entry in entries],
                "standard_bytes": [entry["standard_test"]["mean_bytes"] for entry in entries],
                "held_out_warning_recall": [entry["warning_test"]["recall"] for entry in entries],
                "held_out_warning_regret": [entry["warning_test"]["mean_regret"] for entry in entries],
            }.items()
        }
    interventions = {}
    for case_name, (state, config, prior) in cases().items():
        oracle = ExactShareOracle(state, config, prior)
        optimum = oracle.solve()[0]
        interventions[case_name] = {"oracle": list(optimum.message_set), "models": {}}
        for name in names:
            selections = []
            for path, report in zip(paths, reports):
                kind = report["models"][name]["architecture"]
                x, e = build_features_v2(config, prior, state)
                model = SetCostCritic(len(x), e.shape[-1], kind)
                model.load_state_dict(torch.load(
                    path.parent / (name + ".pth"), map_location="cpu", weights_only=True
                ))
                model.eval()
                selections.append(list(selected(model, state, config, prior)))
            interventions[case_name]["models"][name] = {
                "selected_by_seed": selections,
                "correct_count": sum(
                    selection == list(optimum.message_set) for selection in selections
                ),
            }
    output = {
        "seeds": [report["setup"]["seed"] for report in reports],
        "aggregate": aggregate,
        "canonical_interventions": interventions,
    }
    path = ROOT / "summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
