"""Evaluate actual modality-specific menu poses at high-ideal-value regions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .environment import CoPHTerrainEnv, cell_to_position
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _covers(env, candidate, cell):
    if candidate.modality == "traction":
        return candidate.viewpoint == cell
    return cell in env._cells_in_range(cell_to_position(candidate.viewpoint),
                                      env.config.scan_range_carrier)


def _one(row):
    env = CoPHTerrainEnv(row["family"], row["seed"], 0,
                         seed=row["seed"] + 9, executor="ph", device="cpu")
    advance_without_acquisition(env, row["step"])
    cell = tuple(row["top"]["cell"])
    menu = candidate_viewpoints(env.observations()["carrier"], env.config,
                                "carrier", max_candidates=8)
    selected = {}
    for modality in ("geometry", "traction"):
        options = [candidate for candidate in menu
                   if candidate.modality == modality and _covers(env, candidate, cell)]
        if options:
            selected[modality] = min(options, key=lambda item: item.travel_distance)
    baseline = run_continuation(env, (), ())
    results = {}
    for label, modalities in (("G", ("geometry",)),
                              ("T", ("traction",)),
                              ("GT", ("geometry", "traction"))):
        if any(modality not in selected for modality in modalities):
            continue
        choices = tuple(AuditChoice("carrier", modality,
                                    selected[modality].viewpoint,
                                    selected[modality].target_region)
                        for modality in modalities)
        result = run_continuation(env, choices, tuple(range(len(choices))))
        results[label] = {
            "value": baseline["score_single_world"] - result["score_single_world"],
            "success": result["success"], "measurements": result["measurements"],
            "packets": result["packets"],
            "target_region_covered": result["target_region_covered"],
            "poses": [choice.viewpoint for choice in choices]}
    return {"seed": row["seed"], "family": row["family"],
            "step": row["step"], "top_cell": cell,
            "ideal_value": row["top"]["V_GT"],
            "baseline_cost": baseline["score_single_world"],
            "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="modality_pose_fresh40_value_funnel_v1.json")
    parser.add_argument("--tag", default="modality_pose_fresh40_top_recheck_v1")
    args = parser.parse_args()
    source = OUT / args.source
    rows = [row for row in json.loads(source.read_text())["records"]
            if row["status"] == "evaluated" and row["top"]["V_GT"] > .05]
    with ProcessPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(_one, rows))
    path = OUT / f"{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"source": source.name, "records": records},
                               indent=2) + "\n")
    print(len(records), path)


if __name__ == "__main__":
    main()
