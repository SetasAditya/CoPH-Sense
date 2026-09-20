#!/usr/bin/env python3
"""Physical shortcut/backup trade-off under paired VMAS rollouts.

Top clearance and traction have a uniform four-world prior. The lower route
remains traversable; its traction changes actual VMAS progress, not a route
reward. Both inspect and skip start from the same simulated decision state.
"""

from dataclasses import replace
import json

import torch

from coph_fork.environment import ForkConfig, ForkWorld
from coph_fork.run_moving_bridge_a65 import (
    OUT, gate_snapshot, load_actors, run_episode, timing_context,
)


BASE = ForkConfig(
    horizon=1800, scout_start_y=0.40, carrier_start_y=0.05,
    bottom_route_y=-0.65, carrier_top_y=0.32, direct_top_exit=True,
    scout_bottom_y=-0.82, scout_detour_x=-0.92,
    goal_y=0.20, carrier_goal_y=0.05, scout_goal_y=0.35,
    decision_x=-0.38, communication_range=0.90, packet_drop_probability=0.0,
)


def evaluate(name, bottom_traction, sensing_cost, delay, a4, gate):
    config = replace(BASE, sensing_cost=sensing_cost,
                     backup_traction_prior_mean=bottom_traction,
                     communication_delay_steps=delay)
    rows = []
    for geometry in (False, True):
        for traction in (0.30, 0.90):
            world = ForkWorld(top_geometry=geometry, top_traction=traction,
                              bottom_geometry=True, bottom_traction=bottom_traction)
            decision_state = gate_snapshot(world, config, 1701)
            context = timing_context(decision_state, rollout_forecast=True)
            for policy in ("always", "never"):
                result = run_episode(world, config, policy, 1701, a4, gate,
                                     starting_env=decision_state,
                                     rollout_forecast=True)
                result.pop("trajectory")
                result["decision_cost"] = result["team_cost"] - decision_state.ledger.total
                result["layout"] = name
                result["bottom_traction"] = bottom_traction
                result["forecast_gate_inspects"] = bool(gate.chooses_inspect(context))
                result["forecast_slack"] = context.decision_slack
                rows.append(result)
                print(name, delay, geometry, traction, policy,
                      round(result["decision_cost"], 4), result["success"],
                      "collision", result["ledger"]["collision"], flush=True)
    means = {policy: sum(row["decision_cost"] for row in rows
                         if row["policy"] == policy) / 4
             for policy in ("always", "never")}
    oracle = min(means, key=means.get)
    gate_choice = "always" if rows[0]["forecast_gate_inspects"] else "never"
    summary = {
        "layout": name, "bottom_traction": bottom_traction,
        "sensing_cost": sensing_cost, "delay_steps": delay,
        "delay_seconds": delay * config.dt,
        "uniform_four_world_prior": True,
        "forecast_slack": rows[0]["forecast_slack"],
        "mean_decision_cost": means, "oracle_choice": oracle,
        "frozen_gate_choice": gate_choice,
        "frozen_gate_regret": means[gate_choice] - means[oracle],
        "all_success": all(row["success"] for row in rows),
        "any_collision_cost": any(row["ledger"]["collision"] > 0 for row in rows),
    }
    print("SUMMARY", summary, flush=True)
    return rows, summary


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    a4, gate = load_actors()
    conditions = [
        ("low_value", 0.50, 0.04, 3),
        ("useful", 0.15, 0.04, 3),
        ("expensive_probe", 0.15, 0.40, 3),
        ("useful_medium_delay", 0.15, 0.04, 140),
    ]
    rows = []
    summaries = []
    for condition in conditions:
        current_rows, summary = evaluate(*condition, a4, gate)
        rows.extend(current_rows)
        summaries.append(summary)
        # Save incrementally so a long audit can be inspected or resumed.
        (OUT / "v3_world_rollouts.json").write_text(json.dumps(rows, indent=2) + "\n")
        (OUT / "v3_oracle_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
