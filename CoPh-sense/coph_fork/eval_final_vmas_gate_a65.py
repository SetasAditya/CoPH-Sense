#!/usr/bin/env python3
"""Evaluate the frozen VMAS advantage gate on the untouched final split."""

import json

import numpy as np
import torch

from coph_fork.run_moving_bridge_a65 import load_actors
from coph_fork.train_vmas_gate_a65 import (
    VMASAdvantageGate, arrays, baselines, evaluate,
)
from coph_fork.vmas_gate_data_a65 import DATA


def main():
    torch.set_num_threads(2)
    rows = json.loads((DATA / "final_labels.json").read_text())
    checkpoint = torch.load(DATA / "vmas_advantage_gate.pth",
                            map_location="cpu", weights_only=False)
    model = VMASAdvantageGate()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        values = (model(arrays(rows, checkpoint["mean"], checkpoint["std"]))
                  .numpy() * checkpoint["target_scale"])
    _, finite_gate = load_actors()
    methods = {"VMAS_advantage_MLP": values, **baselines(rows, finite_gate)}
    report = [evaluate(rows, predictions, name)
              for name, predictions in methods.items()]
    (DATA / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report:
        print(row["method"], "regret", round(row["mean_physical_regret"], 4),
              "accuracy", round(row["decision_accuracy"], 3))


if __name__ == "__main__":
    main()
