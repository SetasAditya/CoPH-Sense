"""Evaluator-only decomposition of terrain evidence value on excluded maps.

No environment parameter, public candidate rule, or trained policy is changed.
Idealized reveals are upper-bound diagnostics and are never actor inputs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv, TerrainAction, cell_to_position, position_to_cell
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .pilot_environment import waypoint
from .planning import candidate_viewpoints, dijkstra, observed_costmap, plan_path
from .generator import GRID_SIZE, RESOLUTION


OUT = Path(__file__).resolve().parent / "results" / "pilot"
FAMILIES = ("open", "bottleneck", "labyrinth")
SEEDS = tuple(range(840000, 840040))  # already-excluded development parents
STEPS = (120, 360)
MODALITIES = ((), ("geometry",), ("traction",), ("geometry", "traction"))


def _path_from_cost(cost, start, goal):
    distances, previous = dijkstra(cost, start)
    if not np.isfinite(distances[goal]):
        return ()
    path = [goal]
    while path[-1] != start:
        index = int(previous[path[-1]])
        if index < 0:
            return ()
        path.append((index // GRID_SIZE, index % GRID_SIZE))
    return tuple(reversed(path))


def public_regions(snapshot, limit=6):
    """Dense route/alternative/divergence sampling without latent terrain."""
    obs = snapshot.observations()["carrier"]
    start = position_to_cell(obs["kinematics"][:2])
    goal = (55, 29)
    base_cost = observed_costmap(obs)
    base = _path_from_cost(base_cost, start, goal)
    if not base:
        return ()
    alternatives = [base]
    for penalty in (1.5, 4., 9.):
        cost = base_cost.copy()
        for cell in base[5:-5]:
            if np.isfinite(cost[cell]):
                cost[cell] += penalty
        path = _path_from_cost(cost, start, goal)
        if path and path not in alternatives:
            alternatives.append(path)
    reachable, _ = dijkstra(base_cost, start)
    proposals = []
    if limit > 6:
        appearance = obs["appearance"]
        base_cells = set(base)
        scored = []
        for path in alternatives:
            for cell in path[5:-5]:
                if not np.isfinite(reachable[cell]):
                    continue
                value = appearance[cell]
                uncertainty = (1. if not np.isfinite(value)
                               else 1. - abs(float(value) - .5) * 2.)
                divergence = float(cell not in base_cells)
                scored.append((uncertainty + .25 * divergence, cell))
        for _, cell in sorted(scored, key=lambda row: (-row[0], row[1])):
            if all(np.linalg.norm(np.asarray(cell) - old) >= 3
                   for old in proposals):
                proposals.append(cell)
            if len(proposals) >= min(4, limit):
                break
    for path_index, path in enumerate(alternatives):
        other = set(base) if path_index else set().union(
            *(set(other_path) for other_path in alternatives[1:]))
        distinctive = [cell for cell in path if cell not in other]
        for route in (distinctive, path):
            if not route:
                continue
            fractions = (tuple(np.linspace(.10, .90, 9)) if limit > 6
                         else (.22, .40, .58, .76))
            for fraction in fractions:
                cell = route[min(len(route) - 1, int(fraction * len(route)))]
                if (np.isfinite(reachable[cell]) and
                    all(np.linalg.norm(np.asarray(cell) - old) >= (3 if limit > 6 else 4)
                        for old in proposals)):
                    proposals.append(cell)
                if len(proposals) >= limit:
                    return tuple(proposals)
    return tuple(proposals)


def _footprint(env, cell, modality, ideal):
    if modality == "traction" and not ideal:
        return (cell,)
    position = cell_to_position(cell)
    return tuple(env._cells_in_range(position, env.config.scan_range_carrier))


def _reveal(env, cell, modalities, ideal):
    for modality in modalities:
        grid = (env.known_surface["carrier"] if modality == "geometry"
                else env.known_traction["carrier"])
        truth = (env._truth.surface if modality == "geometry"
                 else env._truth.traction)
        for location in _footprint(env, cell, modality, ideal):
            grid[location] = truth[location]


def _evaluate_reveal(snapshot, cell, modalities, ideal=False, dwell=False):
    env = snapshot.clone()
    if dwell and modalities:
        wait = sum(env.config.scan_dwell_steps if modality == "geometry"
                   else env.config.probe_dwell_steps for modality in modalities)
        for _ in range(wait):
            if env.done:
                break
            positions = env.positions
            env.step({"scout": TerrainAction(waypoint=waypoint(env, "scout")),
                      "carrier": TerrainAction(waypoint=tuple(positions["carrier"]))})
    if not env.done:
        _reveal(env, cell, modalities, ideal)
    path = plan_path(env.observations()["carrier"], goal=(55, 29))[0]
    route_digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    result = run_continuation(env, (), ()) if not env.done else {
        "score_single_world": env.ledger.mission_cost + env.ledger.material_exposure,
        "success": env.success}
    cost = result["score_single_world"]
    if dwell:
        cost += sum(env.config.scan_cost if modality == "geometry"
                    else env.config.probe_cost for modality in modalities)
    return {"cost": cost, "success": bool(result["success"]),
            "route_digest": route_digest, "route_cells": len(path)}


def _menu_coverage(menu, cell, modality, env=None):
    """Whether an actual reading from the menu covers this exact target cell."""
    for candidate in menu:
        if candidate.modality != modality:
            continue
        if modality == "traction":
            if candidate.viewpoint == tuple(cell):
                return True
        elif env is not None:
            if tuple(cell) in env._cells_in_range(
                    cell_to_position(candidate.viewpoint),
                    env.config.scan_range_carrier):
                return True
        elif (np.linalg.norm(cell_to_position(candidate.viewpoint)
                             - cell_to_position(cell)) <= 1.0):
            return True
    return False


def audit_state(family, seed, step):
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, step)
    if env.done:
        return {"family": family, "seed": seed, "step": step,
                "status": "terminated_prefix", "success": env.success}
    regions = public_regions(env)
    menu = candidate_viewpoints(env.observations()["carrier"], env.config,
                                "carrier", max_candidates=8)
    baseline = _evaluate_reveal(env, regions[0] if regions else (5, 30), ())
    rows = []
    for cell in regions:
        ideal = {"none": baseline}
        for label, modalities in (("G", MODALITIES[1]), ("T", MODALITIES[2]),
                                  ("GT", MODALITIES[3])):
            ideal[label] = _evaluate_reveal(env, cell, modalities, ideal=True)
        q0, qg, qt, qgt = (ideal[key]["cost"] for key in ("none", "G", "T", "GT"))
        rows.append({"cell": cell, "ideal": ideal,
                     "V_G": q0 - qg, "V_T": q0 - qt,
                     "V_GT": q0 - qgt, "S_GT": qg + qt - qgt - q0,
                     "menu_G": _menu_coverage(menu, cell, "geometry", env),
                     "menu_T": _menu_coverage(menu, cell, "traction", env)})
    if not rows:
        return {"family": family, "seed": seed, "step": step,
                "status": "no_public_region", "baseline": baseline}
    top = max(rows, key=lambda row: row["V_GT"])
    cell = tuple(top["cell"])
    top["actual_zero_travel"] = {
        label: _evaluate_reveal(env, cell, modalities)
        for label, modalities in (("G", MODALITIES[1]), ("T", MODALITIES[2]),
                                  ("GT", MODALITIES[3]))}
    top["actual_dwell_no_travel"] = {
        label: _evaluate_reveal(env, cell, modalities, dwell=True)
        for label, modalities in (("G", MODALITIES[1]), ("T", MODALITIES[2]),
                                  ("GT", MODALITIES[3]))}
    physical = {}
    for label, modalities in (("G", MODALITIES[1]), ("T", MODALITIES[2]),
                              ("GT", MODALITIES[3])):
        choices = tuple(AuditChoice("carrier", modality, cell)
                        for modality in modalities)
        result = run_continuation(env, choices, tuple(range(len(choices))))
        physical[label] = {"cost": result["score_single_world"],
                           "success": result["success"],
                           "measurements": result["measurements"],
                           "packets": result["packets"]}
    top["physical"] = physical
    # The same public occupancy supports a comparable distance-to-viewpoint
    # proxy for scout/carrier; no hidden material is used in this comparison.
    access = {}
    for agent in ("scout", "carrier"):
        observation = env.observations()[agent]
        distances, _ = dijkstra(observed_costmap(observation),
                                position_to_cell(observation["kinematics"][:2]))
        speed = .75 if agent == "scout" else .55
        access[agent] = float(distances[cell] / speed)
    return {"family": family, "seed": seed, "step": step,
            "status": "evaluated", "baseline": baseline,
            "public_menu": [asdict(candidate) for candidate in menu],
            "regions": rows, "top_cell": cell,
            "access_time_proxy": access}


def _one(index):
    seed = SEEDS[index]
    family = FAMILIES[seed % len(FAMILIES)]
    return [audit_state(family, seed, step) for step in STEPS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=len(SEEDS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", default="value_decomposition_excluded40_v1.json")
    args = parser.parse_args()
    if not 1 <= args.count <= len(SEEDS):
        raise ValueError("count must select 1 to 40 archived development parents")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = []
        for index, group in enumerate(pool.map(_one, range(args.count))):
            records.extend(group)
            if (index + 1) % 5 == 0:
                print(index + 1, "/", args.count, flush=True)
    path = OUT / args.output
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"seed_range": [SEEDS[0], SEEDS[args.count - 1]],
                                "steps": STEPS, "records": records},
                               indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
