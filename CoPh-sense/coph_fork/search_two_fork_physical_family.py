#!/usr/bin/env python3
"""Oracle-only search for a matched physical R1-to-R2 memory contrast.

No learned acquisition or sharing model is loaded in this file. Both memory
conditions begin from the same VMAS snapshot; only carrier evidence is masked.
"""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from coph_fork.run_moving_bridge_a65 import state
from coph_fork.run_two_fork_memory_bridge import step
from coph_fork.two_fork_environment import (
    TwoForkConfig, TwoForkEnv, TwoForkWorld,
)


OUT = Path(__file__).resolve().parent / "results" / "a5_moving_bridge" / "physical_search"
ACTIONS = {
    "skip": (),
    "R1": ((1, "geometry"), (1, "traction")),
    "R2": ((2, "geometry"), (2, "traction")),
}


@dataclass(frozen=True)
class Layout:
    backup_y1: float
    backup_y2: float
    backup_traction_1: float
    backup_traction_2: float
    sensing_cost: float = .04
    safe_prior_1: float = .5
    safe_prior_2: float = .5


def staged_snapshot(bits, layout, seed=1701):
    """Fixed two-message prefix; no acquisition actor influences the snapshot."""
    world = TwoForkWorld(
        geometry_1=bool(bits[0]), traction_1=.9 if bits[1] else .3,
        geometry_2=bool(bits[2]), traction_2=.9 if bits[3] else .3,
        backup_traction_1=layout.backup_traction_1,
        backup_traction_2=layout.backup_traction_2)
    config = TwoForkConfig(
        horizon=2600, backup_y1=layout.backup_y1,
        backup_y2=layout.backup_y2, sensing_cost=layout.sensing_cost)
    env = TwoForkEnv(config, world, seed)
    viewpoint = env.probe_position(1, "top")
    while np.linalg.norm(state(env, "scout")[0] - viewpoint) > .10:
        step(env, "probe_1", "hold")
    for modality in ("geometry", "traction"):
        step(env, "probe_1", "hold", scout_sense=(1, "top", modality))
    ids = [item.observation_id for item in env.evidence["scout"].values()
           if item.region == 1]
    assert len(ids) == 2
    for observation_id in ids:
        step(env, "probe_1", "hold", scout_send=observation_id)
    for _ in range(config.communication_delay_steps + 2):
        step(env, "probe_1", "hold")
    assert len(env.received["carrier"]) == 2
    assert env.route_commitments[1] is None
    return env


def paired_continuation(snapshot, memory, action):
    if memory not in ("M0", "M1"):
        raise ValueError(memory)
    env = copy.deepcopy(snapshot)
    sunk = env.ledger.total
    if memory == "M0":
        # Controlled memory intervention; packet/ACK and physical state stay
        # identical. No later communication occurs in this continuation.
        env.received["carrier"].clear()
    if action:
        region = action[0][0]
        viewpoint = env.probe_position(region, "top")
        while np.linalg.norm(state(env, "carrier")[0] - viewpoint) > .10:
            step(env, "navigate", viewpoint)
            if env.done or env.route_commitments[1] is not None:
                raise RuntimeError("viewpoint unreachable before first commitment")
        for _, modality in action:
            step(env, "navigate", viewpoint,
                 carrier_sense=(region, "top", modality))
    while not env.done:
        step(env)
    return {"downstream_cost": env.ledger.total - sunk,
            "success": bool(env.success),
            "collision_cost": float(env.ledger.collision),
            "steps_after_snapshot": env.step_index - snapshot.step_index,
            "routes": dict(env.route_commitments)}


def evaluate_world(spec):
    bits, layout = spec
    snapshot = staged_snapshot(bits, layout)
    position = tuple(float(v) for v in state(snapshot, "carrier")[0])
    velocity = tuple(float(v) for v in state(snapshot, "carrier")[1])
    rows = []
    for memory in ("M0", "M1"):
        if memory == "M1" and bits[:2] != (True, True):
            continue
        for name, action in ACTIONS.items():
            result = paired_continuation(snapshot, memory, action)
            rows.append({"bits": list(bits), "memory": memory,
                         "action": name, **result})
    return {"bits": list(bits), "snapshot": {
        "step": snapshot.step_index, "carrier_position": position,
        "carrier_velocity": velocity,
        "ledger": asdict(snapshot.ledger),
        "acks": {key: sorted(value) for key, value in snapshot.acknowledged.items()},
        "received_ids": sorted(snapshot.received["carrier"]),
    }, "rows": rows}


def _world_mass(bits, layout):
    result = 1.
    for bit, prior in zip(bits, (layout.safe_prior_1, layout.safe_prior_1,
                                 layout.safe_prior_2, layout.safe_prior_2)):
        result *= prior if bit else 1 - prior
    return result


def summarize(world_results, layout, margin):
    rows = [r for item in world_results for r in item["rows"]]
    q = {}
    for memory in ("M0", "M1"):
        q[memory] = {}
        for name in ACTIONS:
            relevant = [r for r in rows if r["memory"] == memory
                        and r["action"] == name]
            weights = np.asarray([_world_mass(r["bits"], layout)
                                  for r in relevant])
            costs = np.asarray([r["downstream_cost"] for r in relevant])
            q[memory][name] = float(np.dot(weights, costs) / weights.sum())
    contrast = {
        "M0_R1_vs_skip": q["M0"]["skip"] - q["M0"]["R1"],
        "M0_R1_vs_R2": q["M0"]["R2"] - q["M0"]["R1"],
        "M1_R2_vs_skip": q["M1"]["skip"] - q["M1"]["R2"],
        "M1_R2_vs_R1": q["M1"]["R1"] - q["M1"]["R2"],
    }
    valid = (all(value >= margin for value in contrast.values())
             and all(r["success"] and r["collision_cost"] == 0. for r in rows))
    return {"q": q, "contrast": contrast, "margin": margin,
            "passed": valid,
            "all_success": all(r["success"] for r in rows),
            "max_collision_cost": max(r["collision_cost"] for r in rows)}


def evaluate_layout(layout, jobs=8, margin=.05, worlds=None):
    if worlds is None:
        worlds = list(product((False, True), repeat=4))
    with ProcessPoolExecutor(max_workers=jobs,
                             mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_world, (bits, layout)) for bits in worlds]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("physical oracle completed world", result["bits"], flush=True)
    return {"layout": asdict(layout), "summary": summarize(results, layout, margin),
            "worlds": sorted(results, key=lambda item: item["bits"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--margin", type=float, default=.05)
    parser.add_argument("--backup-y1", type=float, default=-.43)
    parser.add_argument("--backup-y2", type=float, default=-.43)
    parser.add_argument("--backup-traction1", type=float, default=.3)
    parser.add_argument("--backup-traction2", type=float, default=.3)
    parser.add_argument("--sensing-cost", type=float, default=.04)
    parser.add_argument("--safe-prior1", type=float, default=.5)
    parser.add_argument("--safe-prior2", type=float, default=.5)
    args = parser.parse_args()
    layout = Layout(args.backup_y1, args.backup_y2,
                    args.backup_traction1, args.backup_traction2,
                    args.sensing_cost, args.safe_prior1, args.safe_prior2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    result = evaluate_layout(layout, args.jobs, args.margin)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
