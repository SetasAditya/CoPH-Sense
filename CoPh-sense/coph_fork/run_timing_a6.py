#!/usr/bin/env python3
"""Enumerate the A6 timing/range grid with frozen A4-v2 and A5-v2 actors."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.memory_critic_a5 import MemorySetCritic  # noqa: E402
from coph_fork.run_memory_a5 import load_models  # noqa: E402
from coph_fork.timing_a6 import ExactTimingEvaluator, TimingContext  # noqa: E402


OUTPUT = HERE / "results" / "a6_timing"
RANGES = (0.6, 1.0, 1.4)
DELAYS = (0.0, 0.05, 0.1, 0.25, 0.5)
DELIVERY = (1.0, 0.9, 0.7, 0.5, 0.2, 0.1)
SLACK = (0.2, 0.4, 0.8)
SEEDS = (1701, 1702, 1703)


def key(r, delay, probability, slack):
    return "r={:.2f}|tau={:.2f}|p={:.2f}|slack={:.2f}".format(
        r, delay, probability, slack
    )


def load_evaluator(seed):
    _, a4 = load_models(seed, seed + 100)
    a5 = MemorySetCritic("additive")
    a5.load_state_dict(torch.load(
        HERE / "results" / "a5_memory_v2" / ("additive_{}.pth".format(seed)),
        map_location="cpu", weights_only=True
    ))
    a5.eval()
    return ExactTimingEvaluator(a4, a5)


def canonical_trace(evaluator):
    contexts = {
        "early": TimingContext(communication_range=1.0, network_delay=0.1,
                               delivery_probability=1.0, decision_slack=0.4),
        "near": TimingContext(communication_range=1.0, network_delay=0.34,
                              delivery_probability=1.0, decision_slack=0.4),
        "late": TimingContext(communication_range=1.0, network_delay=0.35,
                              delivery_probability=1.0, decision_slack=0.4),
    }
    result = {}
    for label, context in contexts.items():
        evaluated = evaluator.evaluate(context, True, return_branches=True)
        matching = [row for row in evaluated.pop("branches") if row["world"] == [True] * 4]
        if len(matching) != 1:
            raise RuntimeError("p=1 canonical timing branch is ambiguous")
        result[label] = {
            "context": context.__dict__,
            "delivery_time": context.delivery_time,
            "timely": context.timely,
            "gate_oracle_choice": evaluator.gate_oracle(context)["oracle_choice"],
            "illustrative_branch": matching[0],
            "expected_inspect": evaluated,
        }
    return result


def make_plots(by_seed, aggregate):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    for ax, comm_range in zip(axes, RANGES):
        grid = np.asarray([
            [aggregate[key(comm_range, delay, probability, 0.4)]["inspect_fraction"]
             for delay in DELAYS]
            for probability in DELIVERY
        ])
        ax.imshow(grid, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(DELAYS)), [str(x) for x in DELAYS])
        ax.set_yticks(range(len(DELIVERY)), [str(x) for x in DELIVERY])
        ax.set_xlabel("Network delay (s)")
        ax.set_title("Range = {:.1f}".format(comm_range))
        if ax is axes[0]:
            ax.set_ylabel("Packet success probability")
        for row in range(len(DELIVERY)):
            for col in range(len(DELAYS)):
                fraction = grid[row, col]
                ax.text(col, row, "I" if fraction >= 0.99 else
                        "S" if fraction <= 0.01 else "{:.0f}% I".format(100 * fraction),
                        ha="center", va="center", color="#152333", fontweight="bold")
    fig.suptitle("A6 exact gate under frozen A4/A5 continuation (slack = 0.4 s)\n"
                 "I = inspect, S = skip; frozen scout still inspects every case", fontsize=13)
    fig.savefig(OUTPUT / "decision_phase.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    for comm_range in RANGES:
        values = [aggregate[key(comm_range, delay, 0.9, 0.4)]["mean_inspect_minus_skip"]
                  for delay in DELAYS]
        ax.plot(DELAYS, values, marker="o", label="range {:.1f}".format(comm_range))
    ax.axhline(0, color="#273447", linewidth=1)
    ax.set_xlabel("Network delay (s)")
    ax.set_ylabel("Inspect cost − skip cost")
    ax.set_title("Value disappears when delivery misses commitment")
    ax.legend()
    fig.savefig(OUTPUT / "timing_margin.png", dpi=180)
    plt.close(fig)


def make_canonical_timeline(canonical):
    rows = canonical["1701"]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    ax.axvline(0.4, color="#AE4F55", ls="--", lw=2)
    ax.text(0.408, 1.0, "commitment", rotation=90, va="center",
            color="#AE4F55", fontsize=9)
    labels = ("early", "near", "late")
    for yi, name in enumerate(labels):
        item = rows[name]
        arrival = item["delivery_time"]
        timely = item["timely"]
        color = "#00877F" if timely else "#AE4F55"
        ax.plot([0, arrival], [yi, yi], lw=4, color="#D9E2EC")
        ax.scatter([arrival], [yi], s=110, color=color, zorder=3)
        branch = item["illustrative_branch"]
        ax.text(0.55, yi, "{} before commit; route cost {:.0f}; oracle {}".format(
            len(branch["delivered_before_commit"]), branch["route_cost"],
            item["gate_oracle_choice"]), va="center", fontsize=10)
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Seconds from scout acquisition opportunity")
    ax.set_title("Same all-good world and reliable packets; only arrival time changes")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.savefig(OUTPUT / "canonical_deadline.png", dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_seed = {}
    canonical = {}
    for seed in SEEDS:
        evaluator = load_evaluator(seed)
        canonical[str(seed)] = canonical_trace(evaluator)
        rows = {}
        for comm_range in RANGES:
            for delay in DELAYS:
                for probability in DELIVERY:
                    for slack in SLACK:
                        context = TimingContext(
                            communication_range=comm_range, network_delay=delay,
                            delivery_probability=probability, decision_slack=slack,
                        )
                        result = evaluator.gate_oracle(context)
                        rows[key(comm_range, delay, probability, slack)] = {
                            "temporal_margin": context.temporal_margin,
                            "timely": context.timely,
                            **result,
                        }
        by_seed[str(seed)] = rows
        print("finished seed", seed, flush=True)
    aggregate = {}
    for comm_range in RANGES:
        for delay in DELAYS:
            for probability in DELIVERY:
                for slack in SLACK:
                    label = key(comm_range, delay, probability, slack)
                    rows = [by_seed[str(seed)][label] for seed in SEEDS]
                    differences = [row["inspect"]["expected_team_cost"] -
                                   row["skip"]["expected_team_cost"] for row in rows]
                    aggregate[label] = {
                        "inspect_fraction": sum(row["oracle_choice"] == "inspect"
                                                for row in rows) / len(rows),
                        "mean_inspect_minus_skip": float(np.mean(differences)),
                        "mean_frozen_always_inspect_regret": float(np.mean([
                            row["frozen_always_inspect_regret"] for row in rows
                        ])),
                    }
    report = {
        "scope": "exact finite scout inspect/skip gate under frozen A4-v2 and A5-v2; one common route commitment, no VMAS/pH motion",
        "sensing_policy": "A5 protocol fixes scout to inspect both Region-1 modalities; no trained pre-acquisition timing gate exists",
        "oracle": "exact inspect/skip selection under the same frozen A4 sharing and frozen A5 carrier continuation, not a global decentralized optimum",
        "range_model": "one-dimensional separation, optional return into range at fixed speed and effort price",
        "timing_rule": "delivered message usable iff acquisition + return + network delay < decision slack",
        "cost_ledger": "scout sensing, return effort, waiting, sent packets, delivered ACKs, carrier sensing, and both route costs",
        "grid": {"range": RANGES, "delay": DELAYS,
                 "delivery_probability": DELIVERY, "decision_slack": SLACK},
        "canonical": canonical,
        "aggregate": aggregate,
        "by_seed": by_seed,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    make_plots(by_seed, aggregate)
    make_canonical_timeline(canonical)
    print("phase plot:", OUTPUT / "decision_phase.png")


if __name__ == "__main__":
    main()
