#!/usr/bin/env python3
"""Learn the missing local scout inspect/skip gate from exact A6 values."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.run_timing_a6 import (  # noqa: E402
    DELAYS, DELIVERY, OUTPUT, RANGES, SEEDS, SLACK, key, load_evaluator,
)
from coph_fork.timing_a6 import TimingContext  # noqa: E402


PROBABILITIES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def features(context):
    """Only pre-sensing public state; no hidden terrain or oracle cost."""
    return np.asarray([
        abs(context.scout_position - context.carrier_position),
        context.communication_range,
        context.return_distance,
        context.return_duration,
        context.network_delay,
        context.delivery_probability,
        context.decision_slack,
        context.acquisition_duration,
        context.temporal_margin,
        float(context.timely),
        context.return_effort_per_distance,
        context.waiting_cost_per_second,
    ], dtype=np.float32)


class ScoutTimingGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

    def chooses_inspect(self, context):
        with torch.no_grad():
            return bool(self(torch.from_numpy(features(context))[None])[0] < 0)


def contexts(count, seed):
    rng = np.random.RandomState(seed)
    result = []
    for _ in range(count):
        result.append(TimingContext(
            scout_position=float(rng.uniform(0.75, 1.25)),
            carrier_position=0.0,
            communication_range=float(rng.uniform(0.45, 1.45)),
            network_delay=float(rng.uniform(0.0, 0.60)),
            delivery_probability=float(rng.choice(PROBABILITIES)),
            decision_slack=float(rng.uniform(0.12, 0.95)),
            acquisition_duration=0.05,
            return_speed=float(rng.uniform(0.8, 1.3)),
            return_effort_per_distance=float(rng.uniform(0.06, 0.18)),
            waiting_cost_per_second=float(rng.uniform(0.01, 0.06)),
        ))
    return result


def label_contexts(scenarios, evaluators):
    rows = []
    for scenario in scenarios:
        x = features(scenario)
        for seed, evaluator in evaluators.items():
            value = evaluator.gate_oracle(scenario)
            rows.append((x, value["skip"]["expected_team_cost"],
                         value["inspect"]["expected_team_cost"], seed,
                         scenario.__dict__))
    return rows


def arrays(rows):
    x = torch.tensor(np.asarray([row[0] for row in rows]))
    q = torch.tensor(np.asarray([[row[1], row[2]] for row in rows]), dtype=torch.float32)
    return x, q


def metrics(model, rows):
    x, q = arrays(rows)
    with torch.no_grad():
        delta = model(x)
    selected = delta < 0
    oracle = q[:, 1] < q[:, 0] - 1e-7
    regret = torch.where(selected, q[:, 1], q[:, 0]) - q.min(1).values
    late = x[:, 9] == 0
    return {
        "decision_accuracy": float((selected == oracle).float().mean()),
        "mean_regret": float(regret.mean()),
        "fixed_always_inspect_mean_regret": float((q[:, 1] - q[:, 0]).clamp_min(0).mean()),
        "oracle_inspect_fraction": float(oracle.float().mean()),
        "late_inspection_rate": float(selected[late].float().mean()) if late.any() else None,
        "early_inspection_accuracy": float((selected[~late] == oracle[~late]).float().mean())
        if (~late).any() else None,
        "examples": len(rows),
    }


def train(train_rows, validation_rows):
    torch.manual_seed(2601)
    model = ScoutTimingGate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    x, q = arrays(train_rows)
    best = None
    best_regret = float("inf")
    best_epoch = 0
    for epoch in range(1, 301):
        order = torch.randperm(len(x))
        model.train()
        for indices in order.split(64):
            prediction = model(x[indices])
            target = q[indices, 1] - q[indices, 0]
            value_loss = nn.functional.smooth_l1_loss(prediction, target)
            classification = nn.functional.binary_cross_entropy_with_logits(
                -prediction * 8, (target < 0).float()
            )
            loss = value_loss + 0.05 * classification
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        validation = metrics(model, validation_rows)
        if validation["mean_regret"] < best_regret - 1e-7:
            best_regret = validation["mean_regret"]
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
    model.load_state_dict(best)
    model.eval()
    return model, best_epoch


def grid_metrics(model, grid_report):
    decisions = []
    regrets = []
    for seed in SEEDS:
        for comm_range in RANGES:
            for delay in DELAYS:
                for probability in DELIVERY:
                    for slack in SLACK:
                        scenario = TimingContext(
                            communication_range=comm_range, network_delay=delay,
                            delivery_probability=probability, decision_slack=slack,
                        )
                        row = grid_report["by_seed"][str(seed)][key(
                            comm_range, delay, probability, slack
                        )]
                        inspect = model.chooses_inspect(scenario)
                        q = row["inspect"]["expected_team_cost"] if inspect else \
                            row["skip"]["expected_team_cost"]
                        regrets.append(q - row["oracle_cost"])
                        decisions.append(inspect == (row["oracle_choice"] == "inspect"))
    return {"decision_accuracy": float(np.mean(decisions)),
            "mean_regret": float(np.mean(regrets)), "examples": len(decisions)}


def plot_grid(model, grid_report):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.5), constrained_layout=True)
    for col, comm_range in enumerate(RANGES):
        exact = np.asarray([
            [grid_report["aggregate"][key(comm_range, delay, probability, 0.4)]
             ["inspect_fraction"] for delay in DELAYS]
            for probability in DELIVERY
        ])
        learned = np.asarray([
            [float(model.chooses_inspect(TimingContext(
                communication_range=comm_range, network_delay=delay,
                delivery_probability=probability, decision_slack=0.4
            ))) for delay in DELAYS]
            for probability in DELIVERY
        ])
        for row, grid in enumerate((exact, learned)):
            ax = axes[row, col]
            ax.imshow(grid, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
            ax.set_xticks(range(len(DELAYS)), [str(value) for value in DELAYS])
            ax.set_yticks(range(len(DELIVERY)), [str(value) for value in DELIVERY])
            ax.set_title(("Exact gate" if row == 0 else "Learned gate") +
                         ", range {:.1f}".format(comm_range))
            ax.set_xlabel("Network delay (s)")
            if col == 0:
                ax.set_ylabel("Packet success probability")
            for yi in range(len(DELIVERY)):
                for xi in range(len(DELAYS)):
                    value = grid[yi, xi]
                    ax.text(xi, yi, "I" if value >= 0.99 else
                            "S" if value <= 0.01 else "{:.0f}% I".format(100 * value),
                            ha="center", va="center", fontsize=9, fontweight="bold")
    fig.suptitle("A6 inspect/skip decisions under frozen A4/A5 continuation\n"
                 "Slack = 0.4 s; exact row is the fraction across three paired checkpoints")
    fig.savefig(OUTPUT / "gate_vs_exact_phase.png", dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluators = {seed: load_evaluator(seed) for seed in SEEDS}
    train_contexts = contexts(360, 26011)
    validation_contexts = contexts(120, 26012)
    confirmation_contexts = contexts(240, 26013)
    train_rows = label_contexts(train_contexts, evaluators)
    print("labeled train", len(train_rows), flush=True)
    validation_rows = label_contexts(validation_contexts, evaluators)
    print("labeled validation", len(validation_rows), flush=True)
    model, selected_epoch = train(train_rows, validation_rows)
    confirmation_rows = label_contexts(confirmation_contexts, evaluators)
    print("labeled confirmation", len(confirmation_rows), flush=True)
    torch.save(model.state_dict(), OUTPUT / "scout_timing_gate.pth")
    grid = json.loads((OUTPUT / "summary.json").read_text())
    report = {
        "scope": "local scout inspect/skip gate trained on exact values under frozen A4-v2 and A5-v2 continuations",
        "inputs": "public separation, range, return time/cost, network delay, delivery probability, decision slack, acquisition duration, and exact public timely flag",
        "train_context_seed": 26011,
        "validation_context_seed": 26012,
        "confirmation_context_seed": 26013,
        "selected_epoch": selected_epoch,
        "training": metrics(model, train_rows),
        "validation": metrics(model, validation_rows),
        "confirmation": metrics(model, confirmation_rows),
        "previously_inspected_grid": grid_metrics(model, grid),
    }
    (OUTPUT / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_grid(model, grid)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
