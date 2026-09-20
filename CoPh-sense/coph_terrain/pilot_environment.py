#!/usr/bin/env python3
"""Model-free physical feasibility pilot on seeds excluded from all E2 splits."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv, TerrainAction, TerrainConfig, cell_to_position
from .generator import FAMILIES
from .planning import plan_path


OUT = Path(__file__).resolve().parent / "results" / "pilot"
PILOT_FAMILIES = FAMILIES[:3]


def waypoint(env, name, full_map=False):
    obs = env.observations()[name]
    if full_map:
        obs["known_geometry"] = env._truth.occupancy.astype(np.float32)
        obs["known_surface"] = env._truth.surface.copy()
        obs["known_traction"] = env._truth.traction.copy()
    goal = (55, 31 if name == "scout" else 29)
    path, _ = plan_path(obs, goal=goal)
    if not path:
        return tuple(float(x) for x in env.positions[name])
    if len(path) <= 6:
        # The route cells at (55,31)/(55,29) lie near the edge of the shared
        # 0.4 m goal disc. With finite controller settling error, an agent can
        # wait indefinitely just outside that disc. Aim at separated points
        # well inside it while preserving the route up to the terminal cells.
        return (5., .16 if name == "scout" else -.16)
    return tuple(float(x) for x in cell_to_position(path[min(len(path) - 1, 5)]))


def run_one(family, parent_seed, full_map=False, config=None):
    env = CoPHTerrainEnv(family, parent_seed, realization_seed=0,
                         seed=parent_seed + 9, config=config,
                         executor="ph")
    targets = {name: waypoint(env, name, full_map) for name in ("scout", "carrier")}
    while not env.done:
        if env.step_index % 20 == 0:
            targets = {name: waypoint(env, name, full_map)
                       for name in ("scout", "carrier")}
        env.step({name: TerrainAction(waypoint=targets[name])
                  for name in ("scout", "carrier")})
    return {"family": family, "parent_seed": parent_seed,
            "full_map_planner_pilot_only": full_map,
            "success": env.success, "failure_reason": env.failure_reason,
            "steps": env.step_index, "mission_cost": env.ledger.mission_cost,
            "material_exposure": env.ledger.material_exposure,
            "ledger": asdict(env.ledger)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--full-map-oracle", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    rows = []
    for i in range(args.count):
        family = PILOT_FAMILIES[i % len(PILOT_FAMILIES)]
        row = run_one(family, 600000 + i, args.full_map_oracle)
        rows.append(row)
        print(i, family, row["success"], row["failure_reason"],
              row["steps"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "full_map_pilot" if args.full_map_oracle else "local_pilot"
    path = OUT / f"{tag}_{args.count}{'_' + args.tag if args.tag else ''}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print("success", sum(row["success"] for row in rows), "/", len(rows))


if __name__ == "__main__":
    main()
