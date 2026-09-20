#!/usr/bin/env python3
"""Regress VMAS inspect advantage; evaluate physical regret on held-out contexts."""

from dataclasses import asdict
import json

import numpy as np
import torch
from torch import nn

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import OUT, gate_snapshot, load_actors, timing_context
from coph_fork.vmas_gate_data_a65 import DATA, GateContext, public_features


class VMASAdvantageGate(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def canonical_rows():
    physical = json.loads((OUT / "v3_oracle_summary.json").read_text())
    communication = json.loads((OUT / "v3_comm_summary.json").read_text())
    specs = [
        ("low_value", .50, .04, 3, 1., .90),
        ("useful", .15, .04, 3, 1., .90),
        ("expensive_probe", .15, .40, 3, 1., .90),
        ("useful_medium_delay", .15, .04, 140, 1., .90),
        ("weak_link", .15, .04, 3, .20, .90),
        ("short_range", .15, .04, 3, 1., .45),
    ]
    values = physical + communication
    result = []
    for (name, bottom, sensing, delay, p, radio), summary in zip(specs, values):
        if summary.get("layout", summary.get("condition")) != name:
            raise ValueError("canonical summary order changed")
        context = GateContext(
            key=name, split="canonical", bottom_traction_mean=bottom,
            sensing_cost=sensing, delay_steps=delay,
            delivery_probability=p, communication_range=radio,
            safe_top_prior=.25, carrier_start_y=.05, scout_start_y=.40,
            bottom_route_y=-.65, carrier_top_y=.32,
        )
        world = ForkWorld(top_geometry=True, top_traction=.90,
                          bottom_geometry=True, bottom_traction=bottom)
        snapshot = gate_snapshot(world, context.config(), 1701)
        mean = summary["mean_decision_cost"]
        inspect = mean.get("always", mean.get("inspect"))
        skip = mean.get("never", mean.get("skip"))
        result.append({"context": asdict(context),
                       "features": public_features(context, snapshot),
                       "q_inspect": inspect, "q_skip": skip,
                       "advantage": skip - inspect})
    return result


def arrays(rows, mean, std):
    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    return torch.from_numpy((x - mean) / std)


def evaluate(rows, predictions, name):
    truth = np.asarray([row["advantage"] for row in rows])
    choices = predictions > 0
    oracle = truth > 0
    regret = np.where(choices, np.maximum(-truth, 0), np.maximum(truth, 0))
    return {"method": name, "n": len(rows),
            "inspect_count": int(choices.sum()),
            "oracle_inspect_count": int(oracle.sum()),
            "decision_accuracy": float((choices == oracle).mean()),
            "mean_physical_regret": float(regret.mean()),
            "max_physical_regret": float(regret.max()),
            "predictions": [float(value) for value in predictions]}


def baselines(rows, finite_gate):
    n = len(rows)
    deadline = np.asarray([row["features"][-1] > 0 for row in rows], dtype=float)
    finite = []
    heuristic = []
    for row in rows:
        context = GateContext(**row["context"])
        world = ForkWorld(top_geometry=True, top_traction=.90,
                          bottom_geometry=True,
                          bottom_traction=context.bottom_traction_mean)
        snapshot = gate_snapshot(world, context.config(), 1701)
        timing = timing_context(snapshot, rollout_forecast=True)
        finite.append(float(finite_gate.chooses_inspect(timing)))
        backup = .61 + .36 / context.bottom_traction_mean
        probe = 2 * context.sensing_cost + .049
        proposed = (context.safe_top_prior * context.delivery_probability ** 2
                    * (backup - .93) - probe)
        heuristic.append(proposed if timing.timely else -probe)
    return {
        "always_inspect": np.ones(n), "never_inspect": -np.ones(n),
        "deadline_only": deadline * 2 - 1,
        "finite_A6": np.asarray(finite) * 2 - 1,
        "analytic_VoI": np.asarray(heuristic),
        "VMAS_oracle": np.asarray([row["advantage"] for row in rows]),
    }


def main():
    torch.set_num_threads(2)
    rows = json.loads((DATA / "labels.json").read_text())
    grouped = {split: [r for r in rows if r["context"]["split"] == split]
               for split in ("train", "val", "ood")}
    if any(len(grouped[s]) == 0 for s in grouped):
        raise RuntimeError("train, val, and OOD labels are all required")
    train, val = grouped["train"], grouped["val"]
    raw_train = np.asarray([r["features"] for r in train], dtype=np.float32)
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0).clip(min=1e-4)
    target = np.asarray([r["advantage"] for r in train], dtype=np.float32)
    target_scale = max(0.1, float(target.std()))
    x_train = arrays(train, mean, std)
    y_train = torch.from_numpy(target / target_scale)
    x_val = arrays(val, mean, std)
    y_val = torch.tensor([r["advantage"] / target_scale for r in val],
                         dtype=torch.float32)
    torch.manual_seed(1607)
    model = VMASAdvantageGate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best, best_state, wait = float("inf"), None, 0
    for step in range(2500):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = nn.functional.huber_loss(prediction, y_train, delta=1.0)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.huber_loss(model(x_val), y_val,
                                                      delta=1.0))
        if val_loss < best - 1e-5:
            best, best_state, wait = val_loss, {k: v.detach().clone()
                                                 for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= 250:
            break
    model.load_state_dict(best_state)
    artifact = DATA / "vmas_advantage_gate.pth"
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "target_scale": target_scale}, artifact)
    _, finite_gate = load_actors()
    grouped["canonical"] = canonical_rows()
    report = {"training": {"n_train": len(train), "n_val": len(val),
                            "best_val_huber": best, "steps": step + 1},
              "splits": {}}
    for split in ("train", "val", "ood", "canonical"):
        group = grouped[split]
        with torch.no_grad():
            learned = (model(arrays(group, mean, std)).numpy() * target_scale)
        methods = {"VMAS_advantage_MLP": learned, **baselines(group, finite_gate)}
        report["splits"][split] = [evaluate(group, values, name)
                                    for name, values in methods.items()]
        print(split, [(row["method"], round(row["mean_physical_regret"], 4),
                       row["decision_accuracy"]) for row in report["splits"][split]])
    (DATA / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
