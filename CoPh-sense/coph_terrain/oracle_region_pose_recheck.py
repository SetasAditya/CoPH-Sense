"""Evaluator-selected region with public, modality-specific physical poses.

The region is selected using evaluator-only ideal value. This upper-bound
diagnostic never enters a deployed candidate generator or actor input.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv, position_to_cell
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import (_acquisition_pose, dijkstra, observed_costmap, GOAL_CELL)


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _one(job):
    row, agent = job
    env = CoPHTerrainEnv(row["family"], row["seed"], 0,
                         seed=row["seed"] + 9, executor="ph", device="cpu")
    advance_without_acquisition(env, row["step"])
    observation = env.observations()[agent]
    cost = observed_costmap(observation)
    start = position_to_cell(observation["kinematics"][:2])
    cost[start] = min(float(cost[start]), 1.)
    from_start = dijkstra(cost, start)[0]
    from_goal = dijkstra(cost, GOAL_CELL)[0]
    cell = tuple(row["top"]["cell"])
    poses = {modality: _acquisition_pose(
        cell, modality, from_start, from_goal, observation,
        env.config, agent) for modality in ("geometry", "traction")}
    baseline = run_continuation(env, (), (), transmit=False)
    results = {}
    for label, modalities in (("G", ("geometry",)),
                              ("T", ("traction",)),
                              ("GT", ("geometry", "traction"))):
        if any(poses[modality] is None for modality in modalities):
            continue
        choices = tuple(AuditChoice(agent, modality, poses[modality], cell)
                        for modality in modalities)
        results[label] = {}
        for mode, transmit in (("hold", False), ("send", True)):
            result = run_continuation(env, choices, tuple(range(len(choices))),
                                      transmit=transmit)
            results[label][mode] = {
                "value": baseline["score_single_world"] - result["score_single_world"],
                "success": result["success"],
                "target_region_covered": result["target_region_covered"],
                "measurements": result["measurements"],
                "packets": result["packets"],
                "executed_choices": result["executed_choices"]}
    return {"seed": row["seed"], "family": row["family"],
            "step": row["step"], "cell": cell,
            "agent": agent,
            "ideal_value": row["top"]["V_GT"],
            "poses": poses, "baseline_cost": baseline["score_single_world"],
            "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="settled_joint_pose_fresh20_value_funnel_v1.json")
    parser.add_argument("--tag", default="settled_joint_pose_fresh20_oracle_region_v1")
    parser.add_argument("--agent", choices=("scout", "carrier"), default="carrier")
    args = parser.parse_args()
    source = OUT / args.source
    rows = [row for row in json.loads(source.read_text())["records"]
            if row["status"] == "evaluated" and row["top"]["V_GT"] > .05]
    with ProcessPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(_one, ((row, args.agent) for row in rows)))
    path = OUT / f"{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"source": source.name, "records": records},
                               indent=2) + "\n")
    print(len(records), path)


if __name__ == "__main__":
    main()
