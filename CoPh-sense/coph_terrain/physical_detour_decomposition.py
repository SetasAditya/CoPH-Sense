"""Evaluator-only physical opportunity-cost decomposition at high-value regions.

Synthetic tiers hold the sensor footprint fixed but turn travel, dwell and
declared charges on progressively. They are diagnostic counterfactuals, not
deployable actions. The final tier uses the real sensing/packet protocol.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv, TerrainAction, cell_to_position
from .physical_audit import (AuditChoice, _public_travel_waypoint,
                             advance_without_acquisition, run_continuation)
from .planning import plan_path
from .pilot_environment import waypoint


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _route_signature(env):
    path, _ = plan_path(env.observations()["carrier"])
    crossings = []
    for x in (20, 35, 45):
        section = [cell[1] for cell in path if cell[0] == x]
        crossings.append(section[0] if section else None)
    return crossings


def _local_reading(env, cell, modalities):
    position = env.positions["carrier"]
    measured = 0
    for modality in modalities:
        if modality == "geometry":
            cells = tuple(env._cells_in_range(
                position, env.config.scan_range_carrier))
            grid, truth = env.known_surface["carrier"], env._truth.surface
        else:
            cells = (cell,)
            grid, truth = env.known_traction["carrier"], env._truth.traction
        for measured_cell in cells:
            grid[measured_cell] = truth[measured_cell]
        measured += len(cells)
    return measured


def _travel_prefix(snapshot, cell):
    env = snapshot.clone()
    destination = cell_to_position(cell)
    targets = {name: waypoint(env, name) for name in ("scout", "carrier")}
    arrival = None
    for _ in range(600):
        if env.done:
            break
        if np.linalg.norm(env.positions["carrier"] - destination) <= env.config.probe_range:
            arrival = env.step_index
            break
        if env.step_index % 20 == 0:
            targets["scout"] = waypoint(env, "scout")
            targets["carrier"] = _public_travel_waypoint(env, "carrier", destination)
        env.step({name: TerrainAction(waypoint=targets[name])
                  for name in ("scout", "carrier")})
    return env, arrival


def _dwell(env, modalities):
    for modality in modalities:
        count = (env.config.scan_dwell_steps if modality == "geometry"
                 else env.config.probe_dwell_steps)
        for _ in range(count):
            if env.done:
                return
            env.step({"scout": TerrainAction(waypoint=waypoint(env, "scout")),
                      "carrier": TerrainAction(
                          waypoint=tuple(float(x) for x in env.positions["carrier"]))})


def _stage(snapshot, cell, modalities, mode):
    env, arrival = _travel_prefix(snapshot, cell)
    if arrival is None:
        return {"arrival": None, "success": False}
    if mode != "travel_only":
        _dwell(env, modalities)
    if env.done:
        return {"arrival": arrival, "success": False}
    before = _route_signature(env)
    prior_surface = env.known_surface["carrier"].copy()
    prior_traction = env.known_traction["carrier"].copy()
    measured = _local_reading(env, cell, modalities)
    after = _route_signature(env)
    reveal_step = env.step_index
    if mode in ("fee", "fee_radio_price"):
        env.ledger.sensing += sum(env.config.scan_cost if m == "geometry"
                                  else env.config.probe_cost for m in modalities)
    if mode == "fee_radio_price":
        # Charged price only: no packet delivery, waiting, or rendezvous.
        payload = 16 + 4 * measured
        env.ledger.communication += (
            env.config.communication_attempt_cost + env.config.byte_cost *
            (env.config.header_bytes + payload) +
            env.config.communication_attempt_cost +
            env.config.byte_cost * env.config.ack_bytes)
    no_reveal_env = env.clone()
    # Restore the complete prior grids, retaining identical physical history
    # and explicit charges for the local actionability comparison.
    no_reveal_env.known_surface["carrier"][:] = prior_surface
    no_reveal_env.known_traction["carrier"][:] = prior_traction
    no_reveal = run_continuation(no_reveal_env, (), (), transmit=False)
    result = run_continuation(env, (), (), transmit=False)
    return {"arrival": arrival, "reveal_step": reveal_step,
            "carrier_x_at_reveal": float(env.positions["carrier"][0]),
            "route_before": before, "route_after": after,
            "measured_cells": measured,
            "late_reveal_value": (no_reveal["score_single_world"] -
                                  result["score_single_world"]),
            "result": result}


def _one(row):
    env = CoPHTerrainEnv(row["family"], row["seed"], 0,
                         seed=row["seed"] + 9, executor="ph", device="cpu")
    advance_without_acquisition(env, row["step"])
    cell = tuple(row["top"]["cell"])
    modalities = ("geometry", "traction")
    base = run_continuation(env, (), (), transmit=False)
    stages = {mode: _stage(env, cell, modalities, mode)
              for mode in ("travel_only", "travel_dwell", "fee", "fee_radio_price")}
    choices = tuple(AuditChoice("carrier", modality, cell) for modality in modalities)
    stages["full_no_send"] = {"result": run_continuation(
        env, choices, (0, 1), transmit=False)}
    stages["full"] = {"result": run_continuation(env, choices, (0, 1))}
    return {"family": row["family"], "seed": row["seed"],
            "step": row["step"], "cell": cell, "base": base,
            "ideal_value": row["top"]["V_GT"], "stages": stages}


def main():
    source = OUT / "interface_revision_excluded40_value_funnel_v1.json"
    rows = [row for row in json.loads(source.read_text())["records"]
            if row.get("seed", 0) >= 850012 and row["status"] == "evaluated"
            and row["top"]["V_GT"] > .05]
    with ProcessPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(_one, rows))
    path = OUT / "physical_detour_decomposition_fresh28_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"source": source.name, "records": records},
                               indent=2) + "\n")
    print(len(records), path)


if __name__ == "__main__":
    main()
