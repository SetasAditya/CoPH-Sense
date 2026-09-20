#!/usr/bin/env python3
"""Exact restricted acquisition values from paired moving VMAS continuations."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import multiprocessing as mp

import numpy as np
import torch

from coph_fork.memory_a5 import CANDIDATES
from coph_fork.run_memory_a5 import load_models
from coph_fork.run_two_fork_memory_bridge import OUT, run_episode
from coph_fork.two_fork_environment import TwoForkConfig, TwoForkWorld


_A4 = None


def init_worker():
    global _A4
    torch.set_num_threads(1)
    _, _A4 = load_models(1701, 1801)


def evaluate_one(spec):
    bits, action_index, condition = spec
    world = TwoForkWorld(
        geometry_1=bits[0], traction_1=.9 if bits[1] else .3,
        geometry_2=bits[2], traction_2=.9 if bits[3] else .3,
    )
    result = run_episode(world, condition, config=TwoForkConfig(),
                         a4_model=_A4, forced_action=CANDIDATES[action_index])
    decision = next(stage for stage in result["stage_log"]
                    if stage["phase"] == "CARRIER_ACQUISITION_DECISION")
    decision_cost = decision["ledger"]
    sunk = sum(decision_cost.values())
    memory_signature = tuple(sorted((record["region"], record["modality"],
                                     record["value"])
                                    for record in decision["acquisition_memory"]))
    return {"world_bits": list(bits), "action_index": action_index,
            "memory_signature": [list(item) for item in memory_signature],
            "downstream_cost": result["team_cost"] - sunk,
            "downstream_risk_cost": (result["ledger"]["risk"] -
                                     decision_cost["risk"]),
            "reference_risk_coefficient": TwoForkConfig().risk_cost,
            "total_team_cost": result["team_cost"],
            "success": result["success"],
            "collision_cost": result["ledger"]["collision"],
            "chosen": result["chosen_acquisition"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--condition", choices=("persistent", "independent"),
                        default="persistent")
    parser.add_argument("--actions", type=int, nargs="+",
                        default=list(range(len(CANDIDATES))))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    groups = [(True, True)] if args.canonical_only else list(product((False, True), repeat=2))
    tasks = [(tuple(first + second), action)
             for first in groups for second in product((False, True), repeat=2)
             for action in args.actions]
    tasks = [(bits, action, args.condition) for bits, action in tasks]
    stem = "physical_oracle_canonical_group" if args.canonical_only else "physical_oracle_all_groups"
    output = OUT / f"{stem}_{args.tag}.json"
    if output.exists():
        raise FileExistsError("physical oracle result exists; preserve frozen values")
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs,
                             mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        futures = [pool.submit(evaluate_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(row["world_bits"], row["action_index"],
                  round(row["downstream_cost"], 4), flush=True)
    summaries = {}
    signatures = sorted({json.dumps(row["memory_signature"]) for row in rows})
    for key in signatures:
        relevant = [row for row in rows if json.dumps(row["memory_signature"]) == key]
        q = {action: float(np.mean([row["downstream_cost"] for row in relevant
                                    if row["action_index"] == action]))
             for action in args.actions}
        best = min(q, key=q.get)
        summaries[key] = {"q": q, "world_count": len(relevant) // len(args.actions),
                          "best_action_index": best,
                          "best_action": [list(atom) for atom in CANDIDATES[best]],
                          "all_success": all(row["success"] for row in relevant),
                          "max_collision_cost": max(row["collision_cost"] for row in relevant)}
    output.write_text(json.dumps({"scope": "two-fork physical continuation; uniform prior over worlds compatible with actually delivered R1 evidence, fixed scripted motion and A4-v2",
                                  "condition": args.condition,
                                  "rows": rows, "by_memory_signature": summaries}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
