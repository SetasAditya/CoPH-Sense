"""Paired delivered-evidence physical acquisition-menu audit.

Uses the first four parents per topology from each of two excluded seed
blocks, independent of their earlier outcome. The source is in both agents'
public initial menus. The carrier keeps moving while the scout physically
acquires and delivers geometry. At delivery, only
packet-borne carrier knowledge is intervened on; each side receives its own
public candidate menu of eight acquisitions plus skip.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv
from .memory_resource_audit import acquire_and_deliver
from .physical_audit import AuditChoice, run_continuation
from .planning import candidate_viewpoints


OUT = Path(__file__).resolve().parent / "results" / "pilot"
FAMILIES = ("open", "bottleneck", "labyrinth")
SETTINGS = ((860000, 0), (861000, 1))
PER_FAMILY = 4


def audit_one(setting, family_index, offset):
    seed_start, rank = SETTINGS[setting]
    seed = seed_start + family_index + 3 * offset
    family = FAMILIES[family_index]
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    observations = env.observations()
    scout_candidates = candidate_viewpoints(observations["scout"],
                                            env.config, "scout")
    carrier_candidates = candidate_viewpoints(observations["carrier"],
                                              env.config, "carrier")
    scout_viewpoints = {candidate.viewpoint for candidate in scout_candidates
                        if candidate.modality == "geometry"}
    carrier_viewpoints = {candidate.viewpoint for candidate in carrier_candidates
                          if candidate.modality == "geometry"}
    viewpoints = tuple(sorted(scout_viewpoints & carrier_viewpoints,
                              key=lambda cell: (cell[0], abs(cell[1] - 30), cell[1])))
    if len(viewpoints) <= rank:
        return {"family": family, "parent_seed": seed, "rank": rank,
                "status": "no_public_source"}
    delivered, evidence_id, prior_grid = acquire_and_deliver(
        env, viewpoints[rank], carrier_moves=True)
    if delivered is None:
        return {"family": family, "parent_seed": seed, "rank": rank,
                "status": "no_physical_delivery"}
    retained = delivered.clone()
    reset = delivered.clone()
    evidence = reset.received["carrier"].pop(evidence_id)
    for cell in evidence.cells:
        reset.known_surface["carrier"][cell] = prior_grid[cell]
    states = {}
    for label, snapshot in (("reset", reset), ("retained", retained)):
        candidates = candidate_viewpoints(snapshot.observations()["carrier"],
                                          snapshot.config, "carrier",
                                          max_candidates=8)
        choices = tuple(AuditChoice("carrier", candidate.modality,
                                    candidate.viewpoint)
                        for candidate in candidates)
        rows = [run_continuation(snapshot, choices, selected)
                for selected in ((),) + tuple((i,) for i in range(len(choices)))]
        successful = [row for row in rows if row["success"]]
        best = min(successful, key=lambda row: row["score_single_world"]) if successful else None
        states[label] = {
            "choices": [asdict(choice) for choice in choices],
            "overlap_with_scout_evidence": [
                choice.viewpoint in evidence.cells for choice in choices],
            "results": rows,
            "best_set": None if best is None else best["selected"]}
    return {"family": family, "parent_seed": seed, "rank": rank,
            "status": "delivered", "delivery_step": delivered.step_index,
            "new_to_carrier_cells": int(sum(bool(np.isnan(prior_grid[cell]))
                                            for cell in evidence.cells)),
            "states": states}


def main():
    jobs = [(setting, family, offset)
            for setting in range(len(SETTINGS))
            for family in range(len(FAMILIES))
            for offset in range(PER_FAMILY)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda_job, jobs))
    comparable = [row for row in records if row["status"] == "delivered" and
                  all(row["states"][state]["best_set"] is not None
                      for state in ("reset", "retained"))]
    def best_choice(row, state):
        data = row["states"][state]
        selected = data["best_set"]
        return None if not selected else data["choices"][selected[0]]
    def best_overlap(row, state):
        data = row["states"][state]
        selected = data["best_set"]
        return False if not selected else data["overlap_with_scout_evidence"][selected[0]]
    summary = {
        "scheduled": len(records),
        "delivered": sum(row["status"] == "delivered" for row in records),
        "comparable_successful_menus": len(comparable),
        "best_action_changes": sum(best_choice(row, "reset") !=
                                   best_choice(row, "retained")
                                   for row in comparable),
        "reset_best_overlaps_scout_evidence": sum(best_overlap(row, "reset")
                                                 for row in comparable),
        "overlap_to_other_or_skip": sum(best_overlap(row, "reset") and
                                        not best_overlap(row, "retained")
                                        for row in comparable),
        "scope": "single-world, eight public carrier candidates per memory state"
    }
    output = {"schema_version": 1, "settings": SETTINGS,
              "summary": summary, "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "memory_menu_audit_known_topology_fresh24_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


def lambda_job(job):
    return audit_one(*job)


if __name__ == "__main__":
    main()
