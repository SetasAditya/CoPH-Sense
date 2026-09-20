#!/usr/bin/env python3
"""Compute and visualize exact A2 sensing and sharing values."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.oracle import ExactAcquisitionOracle, ExactOracleConfig  # noqa: E402


def label(subset):
    aliases = {
        "top_geometry": "Gᵀ",
        "top_traction": "Tᵀ",
        "bottom_geometry": "Gᴮ",
        "bottom_traction": "Tᴮ",
    }
    return "∅" if not subset else "+".join(aliases[name] for name in subset)


def main():
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    oracle = ExactAcquisitionOracle()
    report_path = output / "exact_oracle.json"
    oracle.save_report(report_path)
    report = json.loads(report_path.read_text())
    optimum = report["optimum"]
    expected_optimum = ExactAcquisitionOracle(
        ExactOracleConfig(decision_mode="expected_cost")
    ).solve()[0]

    rows = sorted(report["q_sense_star"], key=lambda row: (len(row["sense_subset"]), row["sense_subset"]))
    figure, axis = plt.subplots(figsize=(12, 5.5))
    colors = ["tab:green" if row["sense_subset"] == optimum["sense_subset"] else "tab:blue" for row in rows]
    axis.bar(range(len(rows)), [row["total_cost"] for row in rows], color=colors)
    axis.axhline(rows[0]["total_cost"], color="black", linestyle="--", label="no-sensing value")
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([label(row["sense_subset"]) for row in rows], rotation=55, ha="right")
    axis.set_ylabel(r"Exact $Q^*_{\mathrm{sense}}(D)$")
    axis.set_title("CoPH-Fork A2: exact sensing values after optimizing sharing and route")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "exact_oracle_qsense.png", dpi=180)
    plt.close(figure)

    concise = {
        "worlds": report["scope"]["worlds"],
        "sensing_subsets": report["scope"]["sensing_subsets"],
        "sharing_policies": report["scope"]["sharing_policies"],
        "optimal_sensing": optimum["sense_subset"],
        "optimal_sharing_rules": optimum["share_rules"],
        "optimal_total_cost": optimum["total_cost"],
        "no_sensing_cost": next(row["total_cost"] for row in rows if not row["sense_subset"]),
        "receiver_decisions": optimum["receiver_decisions"],
        "expected_cost_comparison": {
            "optimal_sensing": expected_optimum.sense_subset,
            "optimal_sharing_rules": expected_optimum.share_rules,
            "optimal_total_cost": expected_optimum.total_cost,
        },
    }
    (output / "exact_oracle_summary.json").write_text(json.dumps(concise, indent=2) + "\n")
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
