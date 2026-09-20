#!/usr/bin/env python3
"""Aggregate frozen A4 runs and audit hand-specified sharing interventions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from coph_fork.critic import SUBSETS, SetCostCritic
from coph_fork.oracle import ExactOracleConfig, VARIABLES, default_prior
from coph_fork.share_oracle import ExactShareOracle, ShareState
from coph_fork.train_share_critic import build_features


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "a4"
G, T, B = "top_geometry", "top_traction", "bottom_geometry"


def cases():
    base = ExactOracleConfig()
    return {
        "joint_good": (ShareState(acquired=((G, True), (T, True))), base, default_prior()),
        "ack_completes_pair": (
            ShareState(acquired=((T, True),), acknowledged=((G, True),)), base, default_prior()
        ),
        "duplicate": (
            ShareState(acquired=((G, True),), acknowledged=((G, True),)), base, default_prior()
        ),
        "irrelevant": (ShareState(acquired=((B, True),)), base, default_prior()),
        "stale": (
            ShareState(acquired=((G, True), (T, True)), timely_probability=0.0),
            base, default_prior(),
        ),
        "costly_low_link": (
            ShareState(acquired=((G, True), (T, True)), timely_probability=0.2),
            ExactOracleConfig(communication_attempt_cost=0.1), default_prior(),
        ),
        "bad_warning": (
            ShareState(acquired=((G, False),), timely_probability=1.0),
            ExactOracleConfig(decision_mode="expected_cost"),
            {
                (geometry, traction, True, True):
                    (0.9 if geometry else 0.1) * (0.9 if traction else 0.1)
                for geometry in (False, True) for traction in (False, True)
            },
        ),
    }


def predict(model, state, config, prior):
    features, candidates = build_features(config, prior, state)
    feasible = set(ExactShareOracle(state, config, prior).candidate_sets())
    with torch.no_grad():
        scores = model(
            torch.tensor(features)[None], torch.tensor(candidates)[None]
        )[0].sum(-1).numpy()
    return min(
        (subset for subset in SUBSETS if subset in feasible),
        key=lambda subset: (scores[SUBSETS.index(subset)], len(subset), subset),
    )


def main():
    seed_dirs = sorted(RESULTS.glob("seed*/report.json"))
    if not seed_dirs:
        raise SystemExit("no frozen A4 reports found")
    reports = [json.loads(path.read_text()) for path in seed_dirs]
    method_names = list(reports[0]["baselines"]) + list(reports[0]["models"])
    aggregate = {}
    for name in method_names:
        entries = [
            report["models"][name]["test"] if name in report["models"]
            else report["baselines"][name]
            for report in reports
        ]
        aggregate[name] = {
            metric: {
                "mean": float(np.mean([entry[metric] for entry in entries])),
                "sample_std": float(np.std([entry[metric] for entry in entries], ddof=1)),
            }
            for metric in ("mean_regret", "top1_accuracy", "wasteful_send_rate",
                           "missed_useful_rate", "missed_useful_value",
                           "critical_recall_at_0_5", "mean_bytes")
        }
    audit = {}
    for case_name, (state, config, prior) in cases().items():
        oracle = ExactShareOracle(state, config, prior)
        optimum, _ = oracle.solve()
        case_result = {"oracle": list(optimum.message_set), "models": {}}
        for name in reports[0]["models"]:
            selections = []
            regrets = []
            for path in seed_dirs:
                features, candidates = build_features(config, prior, state)
                model = SetCostCritic(len(features), candidates.shape[-1], name)
                model.load_state_dict(torch.load(
                    path.parent / (name + ".pth"), map_location="cpu", weights_only=True
                ))
                model.eval()
                chosen = predict(model, state, config, prior)
                selections.append(list(chosen))
                regrets.append(oracle.evaluate(chosen).total_cost - optimum.total_cost)
            case_result["models"][name] = {
                "chosen_by_seed": selections,
                "correct_count": sum(selection == list(optimum.message_set) for selection in selections),
                "mean_regret": float(np.mean(regrets)),
            }
        audit[case_name] = case_result
    result = {
        "seeds": [report["setup"]["seed"] for report in reports],
        "test_states_per_seed": reports[0]["setup"]["test_states"],
        "aggregate": aggregate,
        "canonical_interventions": audit,
        "interpretation": (
            "Exact A4 oracle is verified, but the pairwise learned model is not a "
            "consistent winner across seeds; inspect canonical interventions before "
            "declaring the learned sharing checkpoint passed."
        ),
    }
    destination = RESULTS / "summary.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for ax, names, title in (
        (axes[0], list(reports[0]["baselines"]), "Fixed sharing rules"),
        (axes[1], list(reports[0]["models"]), "Learned set critics"),
    ):
        means = [aggregate[name]["mean_regret"]["mean"] for name in names]
        spreads = [aggregate[name]["mean_regret"]["sample_std"] for name in names]
        ax.bar(range(len(names)), means, yerr=spreads, capsize=3)
        ax.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names])
        ax.set_title(title)
        ax.set_ylabel("Exact sharing regret")
        ax.set_ylim(bottom=0)
    fig.savefig(RESULTS / "a4_three_seed_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
