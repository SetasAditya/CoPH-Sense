"""Paired physical memory intervention on excluded parents.

The scout physically acquires and delivers geometry. At the delivery snapshot,
the carrier's received cells are retained or restored to their pre-delivery
values while all positions, time, resource costs, and hidden world stay fixed.
Both branches then evaluate skip and a physical repeat measurement.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv, TerrainAction, cell_to_position, position_to_cell
from .physical_audit import (AuditChoice, run_continuation, select_choices,
                             _public_travel_waypoint)
from .planning import candidate_viewpoints
from .pilot_environment import waypoint


FAMILIES = ("open", "bottleneck", "labyrinth")
OUT = Path(__file__).resolve().parent / "results" / "pilot"


def acquire_and_deliver(env, viewpoint, carrier_moves=False):
    target = cell_to_position(viewpoint)
    sense_started = False
    evidence_id = None
    sent = False
    carrier_target = tuple(float(v) for v in env.positions["carrier"])
    scout_target = tuple(float(v) for v in env.positions["scout"])
    scout_destination = None
    for _ in range(500):
        if env.done:
            break
        positions = env.positions
        sense = None
        if (not sense_started and
                np.linalg.norm(positions["scout"] - target) <= env.config.probe_range):
            sense = ("geometry", int(viewpoint[0]), int(viewpoint[1]))
            sense_started = True
        if evidence_id is None and env.owned["scout"]:
            evidence_id = next(iter(env.owned["scout"]))
        send = None
        if (evidence_id and not sent and
                np.linalg.norm(positions["scout"] - positions["carrier"])
                <= env.config.communication_range):
            send = evidence_id
            sent = True
        previous_carrier_grid = env.known_surface["carrier"].copy()
        if carrier_moves and env.step_index % 20 == 0:
            carrier_target = waypoint(env, "carrier")
        destination = positions["carrier"] if evidence_id else target
        destination_key = ("radio", tuple(position_to_cell(destination))) if evidence_id else (
            "sense", tuple(viewpoint))
        if destination_key != scout_destination or env.step_index % 20 == 0:
            scout_target = _public_travel_waypoint(env, "scout", destination)
            scout_destination = destination_key
        env.step({"scout": TerrainAction(
                      waypoint=scout_target,
                      sense=sense, send_evidence_id=send),
                  "carrier": TerrainAction(
                      waypoint=carrier_target if carrier_moves else
                      tuple(float(v) for v in positions["carrier"]))})
        if evidence_id in env.received["carrier"]:
            return env, evidence_id, previous_carrier_grid
    return None, evidence_id, None


def audit_one(index, seed_start, rank, source, carrier_moves):
    seed = seed_start + index
    family = FAMILIES[index % len(FAMILIES)]
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    if source == "scout":
        choices = select_choices(env, (rank, 0))
        scout = next((choice for choice in choices if choice.agent == "scout"
                      and choice.modality == "geometry"), None)
    else:
        candidates = candidate_viewpoints(env.observations()["carrier"],
                                          env.config, "carrier")
        viewpoints = tuple(dict.fromkeys(candidate.viewpoint
                                         for candidate in candidates
                                         if candidate.modality == "geometry"))
        scout = (AuditChoice("scout", "geometry", viewpoints[rank])
                 if len(viewpoints) > rank else None)
    if scout is None:
        return {"family": family, "parent_seed": seed,
                "status": "no_public_candidate"}
    delivered, evidence_id, prior_grid = acquire_and_deliver(
        env, scout.viewpoint, carrier_moves)
    if delivered is None:
        return {"family": family, "parent_seed": seed,
                "status": "no_physical_delivery"}
    retained = delivered.clone()
    reset = delivered.clone()
    evidence = reset.received["carrier"].pop(evidence_id)
    for cell in evidence.cells:
        reset.known_surface["carrier"][cell] = prior_grid[cell]
    # Passive appearance does not reveal the high-fidelity surface field.
    def repeat_available(snapshot, candidate_limit):
        return any(candidate.modality == "geometry" and
                   candidate.viewpoint == scout.viewpoint
                   for candidate in candidate_viewpoints(
                       snapshot.observations()["carrier"], snapshot.config,
                       "carrier", max_candidates=candidate_limit))
    public_repeat = {name: {str(limit): repeat_available(snapshot, limit)
                            for limit in (4, 8, 16)}
                     for name, snapshot in (("retained", retained), ("reset", reset))}
    # Same physical source cell, agent rights, and remaining horizon in each
    # continuation. Only the carrier's delivered evidence is intervened on.
    repeat = AuditChoice("carrier", "geometry", scout.viewpoint)
    results = {}
    for name, snapshot in (("retained", retained), ("reset", reset)):
        results[name] = {"skip": run_continuation(snapshot, (repeat,), ()),
                         "repeat": run_continuation(snapshot, (repeat,), (0,))}
    return {"family": family, "parent_seed": seed, "status": "delivered",
            "viewpoint_source": source,
            "carrier_moves": carrier_moves,
            "delivery_step": delivered.step_index,
            "evidence_cells": len(evidence.cells),
            "new_to_carrier_cells": int(sum(bool(np.isnan(prior_grid[cell]))
                                            for cell in evidence.cells)),
            "public_repeat_available": public_repeat,
            "repeat_choice": asdict(repeat), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=690000)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--source", choices=("scout", "carrier"), default="scout")
    parser.add_argument("--carrier-moves", action="store_true")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if (args.parents < 1 or args.workers < 1 or args.seed_start < 600000
            or args.rank < 0):
        raise ValueError("invalid excluded-parent audit configuration")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_one, index, args.seed_start, args.rank,
                               args.source, args.carrier_moves)
                   for index in range(args.parents)]
        records = []
        for index, future in enumerate(futures):
            record = future.result()
            records.append(record)
            print(index + 1, "/", args.parents, record["status"], flush=True)
    delivered = [row for row in records if row["status"] == "delivered"]
    informative = [row for row in delivered if row["new_to_carrier_cells"] > 0]
    comparable = [row for row in informative if all(
        row["results"][m][a]["success"]
        for m in ("retained", "reset") for a in ("skip", "repeat"))]
    def advantage(row, state):
        values = row["results"][state]
        return (values["skip"]["score_single_world"] -
                values["repeat"]["score_single_world"])
    summary = {"parents": len(records), "physical_deliveries": len(delivered),
               "informative_deliveries": len(informative),
               "all_four_continuations_succeeded": len(comparable),
               "repeat_positive_without_memory": sum(
                   advantage(row, "reset") > 0 for row in comparable),
               "repeat_positive_with_memory": sum(
                   advantage(row, "retained") > 0 for row in comparable),
               "memory_reduces_repeat_advantage": sum(
                   advantage(row, "retained") + 1e-6 < advantage(row, "reset")
                   for row in comparable),
               "public_repeat_available_both_16": sum(
                   row["public_repeat_available"]["retained"]["16"] and
                   row["public_repeat_available"]["reset"]["16"]
                   for row in comparable),
               "public_memory_flip_16": sum(
                   row["public_repeat_available"]["retained"]["16"] and
                   row["public_repeat_available"]["reset"]["16"] and
                   advantage(row, "reset") > 0 and
                   advantage(row, "retained") <= 0
                   for row in comparable),
               "scope": "paired delivered-evidence intervention; one scripted repeat candidate"}
    output = {"schema_version": 1, "rank": args.rank, "source": args.source,
              "carrier_moves": args.carrier_moves,
              "seed_range": [args.seed_start,
              args.seed_start + args.parents - 1],
              "summary": summary, "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"memory_resource_audit_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
