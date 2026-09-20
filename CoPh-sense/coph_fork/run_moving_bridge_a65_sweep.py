#!/usr/bin/env python3
"""Frozen A4/A6 bridge audit across hidden worlds and message delays."""

import json

import torch

from coph_fork.environment import ForkConfig, ForkWorld
from coph_fork.run_moving_bridge_a65 import OUT, load_actors, run_episode


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    a4, gate = load_actors()
    rows = []
    for geometry in (False, True):
        for traction in (0.30, 0.90):
            world = ForkWorld(top_geometry=geometry, top_traction=traction,
                              bottom_geometry=True, bottom_traction=0.50)
            for delay in (3, 140):
                config = ForkConfig(horizon=900, communication_range=0.9,
                                    communication_delay_steps=delay,
                                    packet_drop_probability=0.0)
                for policy in ("learned", "always", "never"):
                    result = run_episode(world, config, policy, 1701, a4, gate)
                    result.pop("trajectory")
                    rows.append(result)
                    print(geometry, traction, delay, policy,
                          "inspect", result["gate"]["inspect"],
                          "route", result["route_commitment"],
                          "success", result["success"],
                          "cost", round(result["team_cost"], 3), flush=True)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    for delay in (3, 140):
        print("delay", delay)
        for policy in ("learned", "always", "never"):
            subset = [row for row in rows if row["config"]["communication_delay_steps"] == delay
                      and row["policy"] == policy]
            print(policy, "mean_cost", sum(r["team_cost"] for r in subset) / len(subset),
                  "success", sum(r["success"] for r in subset), "/", len(subset),
                  "inspections", sum(r["gate"]["inspect"] for r in subset))


if __name__ == "__main__":
    main()
