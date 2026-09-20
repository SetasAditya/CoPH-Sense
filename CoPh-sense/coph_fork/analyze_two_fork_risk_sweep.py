#!/usr/bin/env python3
"""Exact repricing of fixed VMAS trajectories over traction-risk coefficients."""

import json

import numpy as np

from coph_fork.run_two_fork_memory_bridge import OUT


def grouped_values(path):
    data = json.loads(path.read_text())
    groups = {}
    for row in data["rows"]:
        key = json.dumps(row["memory_signature"])
        group = groups.setdefault(key, {})
        group.setdefault(row["action_index"], []).append(row)
    return {
        key: {action: (
            float(np.mean([row["downstream_cost"] for row in rows])),
            float(np.mean([row["downstream_risk_cost"] for row in rows])))
              for action, rows in by_action.items()}
        for key, by_action in groups.items()
    }


def reprice(pair, coefficient, reference=.12):
    cost, risk = pair
    return cost + (coefficient / reference - 1.) * risk


def main():
    files = {
        "independent": OUT / "physical_oracle_all_groups_risk_components_independent_v2.json",
        "persistent": OUT / "physical_oracle_all_groups_risk_components_persistent_v2.json",
    }
    coefficients = (.12, 1., 4., 8., 16., 32.)
    report = {"scope": "development-only exact cost repricing; fixed VMAS actions and trajectories",
              "coefficients": list(coefficients), "groups": {}}
    for condition, path in files.items():
        report["groups"][condition] = {}
        for signature, values in grouped_values(path).items():
            rows = []
            for coefficient in coefficients:
                q = {str(action): reprice(pair, coefficient)
                     for action, pair in values.items()}
                rows.append({"risk_coefficient": coefficient, "q": q,
                             "best_action_index": int(min(values, key=lambda a: q[str(a)]))})
            report["groups"][condition][signature] = {
                "reference_cost_and_risk": {str(k): list(v) for k, v in values.items()},
                "repriced": rows,
            }
    output = OUT / "risk_sweep_symmetric_v2.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for condition, groups in report["groups"].items():
        for signature, value in groups.items():
            print(condition, signature,
                  [(row["risk_coefficient"], row["best_action_index"])
                   for row in value["repriced"]])


if __name__ == "__main__":
    main()
