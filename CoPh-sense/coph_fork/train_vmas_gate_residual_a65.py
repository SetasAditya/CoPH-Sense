#!/usr/bin/env python3
"""Analytic VoI plus a learned, public-trajectory residual for A6.5."""

import json

import numpy as np
import torch
from torch import nn

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import gate_snapshot, load_actors
from coph_fork.train_vmas_gate_a65 import baselines, canonical_rows, evaluate
from coph_fork.vmas_gate_data_a65 import DATA, GateContext
from coph_fork.vmas_gate_trajectory_a65 import reconnect_features


class ResidualGate(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def feature_matrix(rows):
    output = []
    for row in rows:
        context = GateContext(**row["context"])
        world = ForkWorld(top_geometry=True, top_traction=.9,
                          bottom_geometry=True,
                          bottom_traction=context.bottom_traction_mean)
        snapshot = gate_snapshot(world, context.config(), 1701)
        output.append(row["features"] + reconnect_features(snapshot))
    return np.asarray(output, dtype=np.float32)


def fit(train, val, analytic, feature_map, use_trajectory, seed):
    dimension = 29 if use_trajectory else 21
    train_x = feature_map["train"][:, :dimension]
    val_x = feature_map["val"][:, :dimension]
    mean = train_x.mean(0)
    std = train_x.std(0).clip(min=1e-4)
    x_train = torch.from_numpy((train_x - mean) / std)
    x_val = torch.from_numpy((val_x - mean) / std)
    target = np.asarray([r["advantage"] for r in train], dtype=np.float32) - analytic["train"]
    val_target = np.asarray([r["advantage"] for r in val], dtype=np.float32) - analytic["val"]
    scale = max(.05, float(target.std()))
    y_train = torch.from_numpy(target / scale)
    y_val = torch.from_numpy(val_target / scale)
    torch.manual_seed(seed)
    model = ResidualGate(dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, patience = float("inf"), None, 0
    for step in range(1800):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.huber_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(nn.functional.huber_loss(model(x_val), y_val))
        if validation < best - 1e-5:
            best = validation
            best_state = {key: val.detach().clone()
                          for key, val in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 200:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, scale, {"steps": step + 1, "validation_huber": best}


def main():
    torch.set_num_threads(2)
    boundary_file = DATA / "untouched_boundary_labels.json"
    if not boundary_file.exists():
        raise RuntimeError("freeze boundary test before fitting residual")
    rows = json.loads((DATA / "labels.json").read_text())
    groups = {split: [r for r in rows if r["context"]["split"] == split]
              for split in ("train", "val", "ood")}
    groups["canonical"] = canonical_rows()
    groups["untouched_boundary"] = json.loads(boundary_file.read_text())
    _, finite = load_actors()
    analytic = {name: baselines(group, finite)["analytic_VoI"].astype(np.float32)
                for name, group in groups.items()}
    features = {name: feature_matrix(group) for name, group in groups.items()}
    report = {"training": {}, "splits": {}}
    models = {}
    for use_trajectory, label in ((False, "residual_no_trajectory"),
                                  (True, "trajectory_residual")):
        model, mean, std, scale, fit_info = fit(
            groups["train"], groups["val"], analytic, features,
            use_trajectory, 2107)
        report["training"][label] = fit_info
        models[label] = (model, mean, std, scale)
        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "target_scale": scale, "dimension": len(mean),
                    "formula": "analytic_VoI + learned_residual"},
                   DATA / f"{label}.pth")
    for split, group in groups.items():
        method = {}
        for label, (model, mean, std, scale) in models.items():
            with torch.no_grad():
                x = torch.from_numpy((features[split][:, :len(mean)] - mean) / std)
                method[label] = analytic[split] + model(x).numpy() * scale
        method.update(baselines(group, finite))
        report["splits"][split] = [evaluate(group, pred, name)
                                    for name, pred in method.items()]
        print(split, [(r["method"], round(r["mean_physical_regret"], 4),
                       round(r["decision_accuracy"], 3))
                      for r in report["splits"][split]], flush=True)
    (DATA / "residual_gate_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
