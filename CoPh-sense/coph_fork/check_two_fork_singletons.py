#!/usr/bin/env python3
"""Complete the seven-action restricted physical oracle before map freeze."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from coph_fork.search_two_fork_physical_family import (
    Layout, OUT, _world_mass, paired_continuation, staged_snapshot,
)


SINGLETONS = {
    "R1_G": ((1, "geometry"),),
    "R1_T": ((1, "traction"),),
    "R2_G": ((2, "geometry"),),
    "R2_T": ((2, "traction"),),
}


def one_world(spec):
    bits, layout = spec
    snapshot = staged_snapshot(bits, layout)
    rows = []
    for memory in ("M0", "M1"):
        if memory == "M1" and bits[:2] != (True, True):
            continue
        for name, action in SINGLETONS.items():
            rows.append({"bits": list(bits), "memory": memory,
                         "action": name,
                         **paired_continuation(snapshot, memory, action)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--prior1", type=float, default=.65)
    parser.add_argument("--prior2", type=float, default=.60)
    parser.add_argument("--sensing-cost", type=float, default=.02)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    layout = Layout(**base["layout"])
    assert np.isclose(layout.sensing_cost, args.sensing_cost)
    layout = Layout(layout.backup_y1, layout.backup_y2,
                    layout.backup_traction_1, layout.backup_traction_2,
                    args.sensing_cost, args.prior1, args.prior2)
    worlds = list(product((False, True), repeat=4))
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs,
                             mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(one_world, (bits, layout)) for bits in worlds]
        for future in as_completed(futures):
            subset = future.result()
            rows.extend(subset)
            print("singletons completed world", subset[0]["bits"], flush=True)
    q = {}
    for memory in ("M0", "M1"):
        q[memory] = {}
        for name in SINGLETONS:
            selected = [row for row in rows if row["memory"] == memory
                        and row["action"] == name]
            weights = np.asarray([_world_mass(row["bits"], layout)
                                  for row in selected])
            costs = np.asarray([row["downstream_cost"] for row in selected])
            q[memory][name] = float(np.dot(weights, costs) / weights.sum())
    # The base summary used uniform priors. Reweight its paired action rows.
    paired_q = {}
    for memory in ("M0", "M1"):
        paired_q[memory] = {}
        for name in ("skip", "R1", "R2"):
            selected = [row for item in base["worlds"] for row in item["rows"]
                        if row["memory"] == memory and row["action"] == name]
            weights = np.asarray([_world_mass(row["bits"], layout)
                                  for row in selected])
            costs = np.asarray([row["downstream_cost"] for row in selected])
            paired_q[memory][name] = float(np.dot(weights, costs) / weights.sum())
    full_q = {memory: {**paired_q[memory], **q[memory]}
              for memory in ("M0", "M1")}
    best = {memory: min(values, key=values.get)
            for memory, values in full_q.items()}
    result = {"scope": "seven-action restricted physical oracle, matched VMAS snapshots",
              "layout": vars(layout), "q": full_q, "best": best,
              "all_success": all(row["success"] for row in rows),
              "max_collision_cost": max(row["collision_cost"] for row in rows),
              "singleton_rows": rows,
              "source_paired": str(args.base)}
    output = OUT / "layout_01_full_seven_action_oracle.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"best": best, "q": full_q,
                      "all_success": result["all_success"]}, indent=2))


if __name__ == "__main__":
    main()
