"""Paired diagnostic of labyrinth navigation failures on excluded pilot maps.

This module does not alter the benchmark. Full-map inputs are given only to
diagnostic rollouts; no actor or acquisition model is trained from them.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .environment import CoPHTerrainEnv, TerrainAction, cell_to_position, position_to_cell
from .generator import RESOLUTION
from .planning import plan_path


OUT = Path(__file__).resolve().parent / "results" / "pilot"
AGENTS = ("scout", "carrier")
MODES = {
    "A_local": (False, False, 20, "ph"),
    "B_geometry": (True, False, 20, "ph"),
    "C_material": (False, True, 20, "ph"),
    "D_full": (True, True, 20, "ph"),
    "E_fast_replan": (False, False, 1, "ph"),
    # These two are deliberately exploratory VMAS velocity-PD controls. They
    # are not certified perfect trackers or matched non-pH baselines.
    "F_local_tracker": (False, False, 1, "vmas"),
    "G_full_tracker": (True, True, 1, "vmas"),
}


def _diagnostic_knowledge(env, full_geometry, full_material):
    if full_geometry:
        for name in AGENTS:
            env.known_geometry[name][:] = env._truth.occupancy
    if full_material:
        for name in AGENTS:
            env.known_traction[name][:] = env._truth.traction


def _target_and_path(env, name):
    observation = env.observations()[name]
    goal = (55, 31 if name == "scout" else 29)
    path, _ = plan_path(observation, goal=goal)
    if not path:
        return tuple(env.positions[name]), ()
    if len(path) <= 6:
        return (5., .16 if name == "scout" else -.16), path
    return tuple(cell_to_position(path[min(len(path) - 1, 5)])), path


def _tracker_action(env, name, target):
    position = env.positions[name]
    velocity = env.physics.world.agents[AGENTS.index(name)].state.vel[0].detach().cpu().numpy()
    desired = np.asarray(target) - position
    distance = float(np.linalg.norm(desired))
    max_speed = .75 if name == "scout" else .55
    desired_velocity = desired / max(distance, 1e-8) * min(max_speed, 2. * distance)
    command = np.clip(4. * (desired_velocity - velocity), -1., 1.)
    return TerrainAction(motion=tuple(float(v) for v in command))


def _clearance(env):
    # Cell-centre distance to nearest occupied cell minus its half-width and
    # the larger carrier radius plus the planner's small safety allowance.
    distance = distance_transform_edt(~env._truth.occupancy) * RESOLUTION
    return distance - .10 - .10 - .015


def run_one(seed, mode):
    full_geometry, full_material, cadence, executor = MODES[mode]
    env = CoPHTerrainEnv("labyrinth", seed, 0, seed=seed + 9,
                         executor=executor, device="cpu")
    _diagnostic_knowledge(env, full_geometry, full_material)
    initial_obstacle = {name: (env.known_geometry[name] == 1.).copy() for name in AGENTS}
    targets, paths = {}, {}
    records = []
    min_route_clearance = float("inf")
    for name in AGENTS:
        targets[name], paths[name] = _target_and_path(env, name)
    if full_geometry:
        clearance = _clearance(env)
        for path in paths.values():
            if path:
                min_route_clearance = min(min_route_clearance,
                                          float(min(clearance[cell] for cell in path)))
    while not env.done:
        replans = 0
        changes = 0
        if env.step_index % cadence == 0 and env.step_index:
            for name in AGENTS:
                old = targets[name]
                targets[name], paths[name] = _target_and_path(env, name)
                replans += 1
                changes += int(np.linalg.norm(np.asarray(old) - targets[name]) > .05)
        before = env.positions
        event_start = len(env.events)
        if executor == "ph":
            actions = {name: TerrainAction(waypoint=targets[name]) for name in AGENTS}
        else:
            actions = {name: _tracker_action(env, name, targets[name]) for name in AGENTS}
        env.step(actions)
        after = env.positions
        new_obstacle = 0
        invalidations = 0
        nearest_new = float("inf")
        for name in AGENTS:
            observed = env.known_geometry[name] == 1.
            newly = observed & ~initial_obstacle[name]
            new_obstacle += int(newly.sum())
            if paths[name] and any(newly[cell] for cell in paths[name]):
                invalidations += 1
            for cell in np.argwhere(newly):
                nearest_new = min(nearest_new,
                                  float(np.linalg.norm(after[name] - cell_to_position(cell))))
            initial_obstacle[name] = observed.copy()
        shield = sum(event["type"] == "shield_intervention"
                     for event in env.events[event_start:])
        speed = [float(np.linalg.norm(env.physics.world.agents[i].state.vel[0]
                                      .detach().cpu().numpy())) for i in range(2)]
        records.append({"step": env.step_index, "replans": replans,
                        "waypoint_changes": changes, "new_obstacles": new_obstacle,
                        "path_invalidations": invalidations, "shield": shield,
                        "progress": sum(float(after[name][0] - before[name][0])
                                        for name in AGENTS),
                        "travel": sum(float(np.linalg.norm(after[name] - before[name]))
                                      for name in AGENTS),
                        "low_speed_agents": sum(value < .02 for value in speed),
                        "waypoint_distance": max(float(np.linalg.norm(
                            after[name] - targets[name])) for name in AGENTS),
                        "nearest_new_obstacle": nearest_new})
    tail = records[-min(len(records), 200):]
    def aggregate(items):
        return {key: float(sum(record[key] for record in items))
                for key in ("replans", "waypoint_changes", "new_obstacles",
                            "path_invalidations", "shield", "progress", "travel",
                            "low_speed_agents")}
    final_position = env.positions
    result = {"seed": seed, "mode": mode, "success": env.success,
              "failure_reason": env.failure_reason, "steps": env.step_index,
              "total": aggregate(records), "last_10s": aggregate(tail),
              "last_waypoint_distance": tail[-1]["waypoint_distance"],
              "last_nearest_new_obstacle": min(
                  (row["nearest_new_obstacle"] for row in tail
                   if np.isfinite(row["nearest_new_obstacle"])), default=None),
              "final_x": {name: float(final_position[name][0]) for name in AGENTS},
              "min_full_route_clearance": (min_route_clearance if
                                           np.isfinite(min_route_clearance) else None)}
    return result


def _run_seed(seed):
    return [run_one(seed, mode) for mode in MODES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="navigation_root_cause_labyrinth30_v1.json")
    args = parser.parse_args()
    if not 1 <= args.count <= 30:
        raise ValueError("use the 30 existing excluded labyrinth seeds")
    seeds = [820000 + 3 * i + 2 for i in range(args.count)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = []
        for index, group in enumerate(pool.map(_run_seed, seeds)):
            records.extend(group)
            print(index + 1, "/", len(seeds), flush=True)
    summary = {mode: {"success": sum(row["success"] for row in records
                                     if row["mode"] == mode),
                      "total": len(seeds)} for mode in MODES}
    path = OUT / args.output
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"seeds": seeds, "modes": MODES,
                                "summary": summary, "records": records},
                               indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
