#!/usr/bin/env python3
"""Exact two-packet reliability and range slices of the A6.5-v3 VMAS oracle."""

from dataclasses import replace
from itertools import product
import json

import torch

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import (
    OUT, gate_snapshot, load_actors, run_episode, timing_context,
)
from coph_fork.run_moving_oracle_a65_v3 import BASE


def evaluate(name, communication_range, delivery_probability, a4, gate):
    config = replace(BASE, communication_range=communication_range,
                     backup_traction_prior_mean=0.15,
                     packet_drop_probability=1 - delivery_probability,
                     communication_delay_steps=3)
    rows = []
    world_values = []
    for geometry, traction in product((False, True), (0.30, 0.90)):
        world = ForkWorld(top_geometry=geometry, top_traction=traction,
                          bottom_geometry=True, bottom_traction=0.15)
        snapshot = gate_snapshot(world, config, 1701)
        context = timing_context(snapshot, rollout_forecast=True)
        skip = run_episode(world, config, "never", 1701, a4, gate,
                           starting_env=snapshot, rollout_forecast=True)
        skip_cost = skip["team_cost"] - snapshot.ledger.total
        masks = list(product((False, True), repeat=2))
        inspect_cost = 0.0
        for mask in masks:
            weight = product_weight(mask, delivery_probability)
            if weight == 0:
                continue
            result = run_episode(world, config, "always", 1701, a4, gate,
                                 starting_env=snapshot, rollout_forecast=True,
                                 forced_evidence_drops=mask)
            decision_cost = result["team_cost"] - snapshot.ledger.total
            inspect_cost += weight * decision_cost
            rows.append({"condition": name, "world": result["world"],
                         "drop_mask": mask, "probability": weight,
                         "inspect_decision_cost": decision_cost,
                         "delivered_before_commit": result["delivered_before_commit"],
                         "success": result["success"],
                         "collision_cost": result["ledger"]["collision"]})
        world_values.append({"geometry": geometry, "traction": traction,
                             "inspect": inspect_cost, "skip": skip_cost,
                             "frozen_gate_inspects": bool(gate.chooses_inspect(context))})
        print(name, geometry, traction, round(inspect_cost, 4),
              round(skip_cost, 4), flush=True)
    means = {action: sum(item[action] for item in world_values) / 4
             for action in ("inspect", "skip")}
    oracle = min(means, key=means.get)
    choice = "inspect" if world_values[0]["frozen_gate_inspects"] else "skip"
    summary = {"condition": name, "communication_range": communication_range,
               "delivery_probability": delivery_probability,
               "mean_decision_cost": means, "oracle_choice": oracle,
               "frozen_gate_choice": choice,
               "frozen_gate_regret": means[choice] - means[oracle],
               "world_values": world_values}
    print("SUMMARY", summary, flush=True)
    return rows, summary


def product_weight(mask, delivery_probability):
    p_drop = 1 - delivery_probability
    weight = 1.0
    for dropped in mask:
        weight *= p_drop if dropped else delivery_probability
    return weight


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    a4, gate = load_actors()
    rows = []
    summaries = []
    for args in (("weak_link", 0.90, 0.20),
                 ("short_range", 0.45, 1.00)):
        part, summary = evaluate(*args, a4, gate)
        rows.extend(part)
        summaries.append(summary)
        (OUT / "v3_comm_rollouts.json").write_text(json.dumps(rows, indent=2) + "\n")
        (OUT / "v3_comm_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
