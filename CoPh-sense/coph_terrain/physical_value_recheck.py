"""Re-evaluate physical tiers after public route-to-viewpoint repair.

Ideal reveal data remain fixed. This file creates a new artifact so the
pre-repair continuation results stay visible as an implementation diagnostic.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _one(row):
    if row["status"] != "evaluated":
        return {key: row[key] for key in ("family", "seed", "step", "status")}
    env = CoPHTerrainEnv(row["family"], row["seed"], 0,
                         seed=row["seed"] + 9, executor="ph")
    advance_without_acquisition(env, row["step"])
    cell = tuple(row["top_cell"])
    results = {}
    for agent in ("carrier", "scout"):
        if agent == "scout" and not (
                max(row["regions"], key=lambda item: item["V_GT"])["V_GT"] > .05
                and row["access_time_proxy"]["scout"] <
                row["access_time_proxy"]["carrier"]):
            continue
        for label, modalities in (("G", ("geometry",)),
                                  ("T", ("traction",)),
                                  ("GT", ("geometry", "traction"))):
            choices = tuple(AuditChoice(agent, modality, cell)
                            for modality in modalities)
            result = run_continuation(env, choices, tuple(range(len(choices))))
            results[f"{agent}_{label}"] = {
                key: result[key] for key in
                ("score_single_world", "success", "failure_reason",
                 "measurements", "packets", "delivered")}
    return {"family": row["family"], "seed": row["seed"],
            "step": row["step"], "status": row["status"],
            "top_cell": cell, "baseline_cost": row["baseline"]["cost"],
            "ideal_V_GT": max(row["regions"], key=lambda item: item["V_GT"])["V_GT"],
            "results": results}


def main():
    source = OUT / "value_decomposition_excluded40_v1.json"
    input_rows = json.loads(source.read_text())["records"]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(_one, input_rows))
    path = OUT / "physical_value_recheck_route_to_viewpoint_v2.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"source": source.name, "records": rows},
                               indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
