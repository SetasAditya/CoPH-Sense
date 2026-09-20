#!/usr/bin/env python3
"""Physical-benefit regressor with exact known sensing charges subtracted."""

import json

import numpy as np
import torch
from torch import nn

from coph_fork.run_moving_bridge_a65 import load_actors
from coph_fork.train_vmas_gate_a65 import (
    VMASAdvantageGate, arrays, baselines, canonical_rows, evaluate,
)
from coph_fork.vmas_gate_data_a65 import DATA


def charged(rows):
    return np.asarray([2 * row["context"]["sensing_cost"] for row in rows],
                      dtype=np.float32)


def main():
    torch.set_num_threads(2)
    rows = json.loads((DATA / "labels.json").read_text())
    groups = {split: [row for row in rows if row["context"]["split"] == split]
              for split in ("train", "val", "ood")}
    train, val = groups["train"], groups["val"]
    raw = np.asarray([row["features"] for row in train], dtype=np.float32)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0).clip(min=1e-4)
    y_raw = np.asarray([row["advantage"] for row in train],
                       dtype=np.float32) + charged(train)
    scale = max(.1, float(y_raw.std()))
    x_train = arrays(train, mean, std)
    y_train = torch.from_numpy(y_raw / scale)
    x_val = arrays(val, mean, std)
    y_val = torch.from_numpy((np.asarray([row["advantage"] for row in val],
                                         dtype=np.float32) + charged(val)) / scale)
    torch.manual_seed(1607)
    model = VMASAdvantageGate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best, best_state, wait = float("inf"), None, 0
    for step in range(2500):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.huber_loss(model(x_train), y_train, delta=1.0)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.huber_loss(model(x_val), y_val,
                                                      delta=1.0))
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 250:
            break
    model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "target_scale": scale, "known_cost_coefficient": 2.0},
               DATA / "structured_vmas_gate.pth")
    _, finite_gate = load_actors()
    groups["canonical"] = canonical_rows()
    groups["first_final_development"] = json.loads((DATA / "final_labels.json").read_text())
    report = {"training": {"steps": step + 1, "best_val_huber": best},
              "splits": {}}
    for split, group in groups.items():
        with torch.no_grad():
            prediction = (model(arrays(group, mean, std)).numpy() * scale
                          - charged(group))
        methods = {"structured_VMAS_advantage": prediction,
                   **baselines(group, finite_gate)}
        report["splits"][split] = [evaluate(group, value, name)
                                    for name, value in methods.items()]
        print(split, [(r["method"], round(r["mean_physical_regret"], 4),
                       round(r["decision_accuracy"], 3))
                      for r in report["splits"][split]])
    (DATA / "structured_gate_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
