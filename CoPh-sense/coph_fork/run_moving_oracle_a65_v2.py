#!/usr/bin/env python3
"""Paired VMAS inspect/skip values at a shared physical decision state.

The decision is made before the scout has measured top clearance or traction.
Values are averaged over a declared uniform four-world prior, so the oracle
does not choose a different sensing decision after seeing the hidden world.
"""

import json

import torch

from coph_fork.environment import ForkConfig, ForkWorld
from coph_fork.run_moving_bridge_a65 import (
    OUT, gate_snapshot, load_actors, run_episode, timing_context,
)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    a4, gate = load_actors()
    rows = []
    summaries = []
    for delay in (3, 100, 140):
        config = ForkConfig(horizon=900, communication_range=0.9,
                            communication_delay_steps=delay,
                            packet_drop_probability=0.0)
        pair = []
        for geometry in (False, True):
            for traction in (0.30, 0.90):
                world = ForkWorld(top_geometry=geometry, top_traction=traction,
                                  bottom_geometry=True, bottom_traction=0.50)
                snapshot = gate_snapshot(world, config, 1701)
                old_context = timing_context(snapshot, rollout_forecast=False)
                new_context = timing_context(snapshot, rollout_forecast=True)
                old_choice = bool(gate.chooses_inspect(old_context))
                new_choice = bool(gate.chooses_inspect(new_context))
                for policy in ("always", "never"):
                    result = run_episode(world, config, policy, 1701, a4, gate,
                                         starting_env=snapshot, rollout_forecast=True)
                    result.pop("trajectory")
                    result["decision_cost"] = result["team_cost"] - snapshot.ledger.total
                    result["old_gate_choice"] = old_choice
                    result["forecast_gate_choice"] = new_choice
                    result["old_slack"] = old_context.decision_slack
                    result["rollout_slack"] = new_context.decision_slack
                    rows.append(result)
                    pair.append(result)
                    print(delay, geometry, traction, policy,
                          round(result["decision_cost"], 4),
                          "before", result["delivered_before_commit"], flush=True)
        mean = {policy: sum(row["decision_cost"] for row in pair
                            if row["policy"] == policy) / 4
                for policy in ("always", "never")}
        oracle_choice = min(mean, key=mean.get)
        corrected_choice = "always" if pair[0]["forecast_gate_choice"] else "never"
        old_choice = "always" if pair[0]["old_gate_choice"] else "never"
        summaries.append({
            "delay_steps": delay, "delay_seconds": delay * config.dt,
            "uniform_prior": "top geometry {false,true} x top traction {0.30,0.90}",
            "mean_decision_cost": mean,
            "oracle_choice": oracle_choice,
            "forecast_gate_choice": corrected_choice,
            "old_gate_choice": old_choice,
            "forecast_gate_regret": mean[corrected_choice] - mean[oracle_choice],
            "old_gate_regret_with_corrected_A4": mean[old_choice] - mean[oracle_choice],
            "old_slack": pair[0]["old_slack"],
            "rollout_slack": pair[0]["rollout_slack"],
        })
        print("SUMMARY", summaries[-1], flush=True)
    (OUT / "v2_world_rollouts.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "v2_oracle_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
