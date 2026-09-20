#!/usr/bin/env python3
"""Summarize the exact paired 16-world controller/information factorial."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "ph_factorial"
CELLS = (("ordinary", "direct"), ("ordinary", "ph"),
         ("coph", "direct"), ("coph", "ph"))


def main():
    payload = json.loads((OUT / "paired_factorial.json").read_text())
    rows = payload["rows"]
    if len(rows) != 64:
        raise ValueError("expected 16 paired worlds x four cells")
    blocks = [rows[i:i + 4] for i in range(0, len(rows), 4)]
    for block in blocks:
        if [tuple((r["info_policy"], r["scheme"])) for r in block] != list(CELLS):
            raise ValueError("factorial pairing/order mismatch")
        if len({json.dumps(r["world"], sort_keys=True) for r in block}) != 1:
            raise ValueError("world mismatch within pair")
        if len({r["seed"] for r in block}) != 1:
            raise ValueError("seed mismatch within pair")
    costs = np.asarray([[r["team_cost"] for r in block] for block in blocks])
    effects = np.column_stack((costs[:, 2] - costs[:, 0],
                               costs[:, 1] - costs[:, 0],
                               costs[:, 3] - costs[:, 2] - costs[:, 1] + costs[:, 0]))
    summary = {
        "scope": "16 enumerated terrain worlds on one frozen two-fork layout; scripted high-level routes; direct and split executors share the same continuous-time pH vector field, so B-A measures numerical integration rather than structural pH benefit; no held-out generalization or statistical replication",
        "cells": {f"{a}_{b}": {
            "mean_team_cost": float(costs[:, i].mean()),
            "success_count": sum(block[i]["success"] for block in blocks),
            "collision_cost": float(sum(block[i]["collision_cost"] for block in blocks)),
            "mean_steps": float(np.mean([block[i]["steps"] for block in blocks])),
            "mean_time_cost": float(np.mean([block[i]["ledger"]["time"] for block in blocks])),
            "mean_risk_cost": float(np.mean([block[i]["ledger"]["risk"] for block in blocks])),
            "mean_sensing_cost": float(np.mean([block[i]["ledger"]["sensing"] for block in blocks])),
            "mean_communication_cost": float(np.mean([block[i]["ledger"]["communication"] for block in blocks])),
        } for i, (a, b) in enumerate(CELLS)},
        "paired_effects": {name: {"mean": float(effects[:, j].mean()),
                                   "min": float(effects[:, j].min()),
                                   "max": float(effects[:, j].max()),
                                   "beneficial_worlds": int(np.sum(effects[:, j] < 0))}
                           for j, name in enumerate(("information_C_minus_A",
                                                      "ph_B_minus_A",
                                                      "interaction_D_minus_C_minus_B_plus_A"))},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    axes[0].bar(range(4), costs.mean(axis=0), color=["#888888", "#5985b3", "#7ca469", "#387e5f"])
    axes[0].set_xticks(range(4), ["A direct\nordinary", "B pH\nordinary",
                                      "C direct\nCoPH", "D pH\nCoPH"])
    axes[0].set_ylabel("Mean charged team cost")
    axes[0].set_title("Paired two-fork factorial")
    for j, (label, color) in enumerate(zip(("Information C−A", "pH B−A", "Interaction"),
                                           ("#4b8d6b", "#5985b3", "#9a6b56"))):
        x = j + np.linspace(-.15, .15, len(blocks))
        axes[1].scatter(x, effects[:, j], color=color, alpha=.65, s=16)
        axes[1].plot([j - .22, j + .22], [effects[:, j].mean()] * 2,
                     color="black", linewidth=2)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(range(3), ["Information\nC−A", "pH\nB−A", "Interaction\nD−C−B+A"])
    axes[1].set_ylabel("Paired cost difference (lower is better)")
    axes[1].set_title("Each dot is one terrain world")
    fig.savefig(OUT / "paired_factorial.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
