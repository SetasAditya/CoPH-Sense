#!/usr/bin/env python3
"""Final A6.5 actionability-aware analytic VoI and small learned residual."""

import json

import numpy as np
import torch
from torch import nn

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import gate_snapshot, load_actors
from coph_fork.train_vmas_gate_a65 import baselines, canonical_rows, evaluate
from coph_fork.train_vmas_gate_residual_a65 import ResidualGate
from coph_fork.vmas_gate_data_a65 import DATA, GateContext
from coph_fork.vmas_gate_trajectory_a65 import actionability_features, reconnect_features


def physical_features(rows):
    x, action = [], []
    for row in rows:
        context = GateContext(**row["context"])
        world = ForkWorld(top_geometry=True, top_traction=.9,
                          bottom_geometry=True,
                          bottom_traction=context.bottom_traction_mean)
        snapshot = gate_snapshot(world, context.config(), 1701)
        reconnect = reconnect_features(snapshot)
        actionable = actionability_features(snapshot, reconnect)
        x.append(row["features"] + reconnect + actionable)
        action.append(actionable[2] > 0)
    return np.asarray(x, dtype=np.float32), np.asarray(action, dtype=bool)


def main():
    torch.set_num_threads(2)
    frozen = DATA / "final_actionability_labels.json"
    if not frozen.exists():
        raise RuntimeError("freeze final actionability test before training")
    rows = json.loads((DATA / "labels.json").read_text())
    groups = {s: [r for r in rows if r["context"]["split"] == s]
              for s in ("train", "val", "ood")}
    groups["canonical"] = canonical_rows()
    groups["final_actionability"] = json.loads(frozen.read_text())
    _, finite = load_actors()
    features, masks, analytic, baseline = {}, {}, {}, {}
    for split, group in groups.items():
        features[split], masks[split] = physical_features(group)
        baseline[split] = baselines(group, finite)
        old = baseline[split]["analytic_VoI"].astype(np.float32)
        # Sensing cannot improve the carrier route after the last feasible
        # switch. In this one-fork bridge the late scout-only benefit is
        # treated as zero; verify that restriction against paired labels.
        charges = np.asarray([2 * r["context"]["sensing_cost"] for r in group],
                             dtype=np.float32)
        analytic[split] = np.where(masks[split], old, -charges)
    train, val = groups["train"], groups["val"]
    raw_train = features["train"]
    mean = raw_train.mean(0)
    std = raw_train.std(0).clip(min=1e-4)
    target = np.asarray([r["advantage"] for r in train], dtype=np.float32) - analytic["train"]
    val_target = np.asarray([r["advantage"] for r in val], dtype=np.float32) - analytic["val"]
    scale = max(.05, float(target[masks["train"]].std()))
    x_train = torch.from_numpy((raw_train[masks["train"]] - mean) / std)
    y_train = torch.from_numpy(target[masks["train"]] / scale)
    x_val = torch.from_numpy((features["val"][masks["val"]] - mean) / std)
    y_val = torch.from_numpy(val_target[masks["val"]] / scale)
    if not len(x_train) or not len(x_val):
        raise RuntimeError("no actionable training or validation contexts")
    torch.manual_seed(2207)
    model = ResidualGate(33)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, wait = float("inf"), None, 0
    for step in range(1800):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.huber_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.huber_loss(model(x_val), y_val))
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 200:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "target_scale": scale, "dimension": 33,
                "formula": "hard public actionability gate + analytic VoI + residual"},
               DATA / "actionability_gate.pth")
    report = {"training": {"steps": step + 1, "best_val_huber": best,
                            "train_actionable": int(masks["train"].sum()),
                            "val_actionable": int(masks["val"].sum())},
              "splits": {}}
    for split, group in groups.items():
        with torch.no_grad():
            normalized = torch.from_numpy((features[split] - mean) / std)
            residual = model(normalized).numpy() * scale
        learned = np.where(masks[split], analytic[split] + residual,
                           analytic[split])
        methods = {"actionability_residual": learned,
                   "actionability_analytic": analytic[split],
                   **baseline[split]}
        report["splits"][split] = [evaluate(group, pred, name)
                                    for name, pred in methods.items()]
        print(split, [(r["method"], round(r["mean_physical_regret"], 4),
                       round(r["decision_accuracy"], 3))
                      for r in report["splits"][split]], flush=True)
    (DATA / "actionability_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
