#!/usr/bin/env python3
"""Evaluate both frozen A6.5 gates on the second untouched range test."""

import json

import torch

from coph_fork.run_moving_bridge_a65 import load_actors
from coph_fork.train_vmas_gate_a65 import (
    VMASAdvantageGate, arrays, baselines, evaluate,
)
from coph_fork.train_vmas_gate_structured_a65 import charged
from coph_fork.vmas_gate_data_a65 import DATA


def predict(rows, filename, structured):
    checkpoint = torch.load(DATA / filename, map_location="cpu", weights_only=False)
    model = VMASAdvantageGate()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        value = (model(arrays(rows, checkpoint["mean"], checkpoint["std"]))
                 .numpy() * checkpoint["target_scale"])
    return value - charged(rows) if structured else value


def main():
    torch.set_num_threads(2)
    rows = json.loads((DATA / "range_final_labels.json").read_text())
    _, finite_gate = load_actors()
    methods = {
        "raw_VMAS_advantage_MLP": predict(rows, "vmas_advantage_gate.pth", False),
        "structured_VMAS_advantage": predict(rows, "structured_vmas_gate.pth", True),
        **baselines(rows, finite_gate),
    }
    report = [evaluate(rows, values, name) for name, values in methods.items()]
    (DATA / "range_final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report:
        print(row["method"], "regret", round(row["mean_physical_regret"], 4),
              "accuracy", round(row["decision_accuracy"], 3))


if __name__ == "__main__":
    main()
