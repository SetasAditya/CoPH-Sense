"""Value funnel for fresh excluded maps under the revised public menu.

The ideal 1 m traction reveal is diagnostic; the deployed probe remains
point-local. All cost figures are realized-world scripted continuations.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints
from .value_decomposition_audit import (public_regions, _evaluate_reveal,
                                        _menu_coverage)


OUT = Path(__file__).resolve().parent / "results" / "pilot"
SEEDS = tuple(range(850000, 850040))
FAMILIES = ("open", "bottleneck", "labyrinth")


def _state(family, seed, step):
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, step)
    if env.done:
        return {"family": family, "seed": seed, "step": step,
                "status": "terminated_prefix", "success": env.success}
    cells = public_regions(env, limit=12)
    if not cells:
        return {"family": family, "seed": seed, "step": step,
                "status": "no_public_region"}
    baseline = _evaluate_reveal(env, cells[0], ())
    menu = candidate_viewpoints(env.observations()["carrier"], env.config,
                                "carrier", max_candidates=8)
    regions = []
    for cell in cells:
        qg = _evaluate_reveal(env, cell, ("geometry",), ideal=True)
        qt = _evaluate_reveal(env, cell, ("traction",), ideal=True)
        qgt = _evaluate_reveal(env, cell, ("geometry", "traction"), ideal=True)
        q0 = baseline["cost"]
        regions.append({"cell": cell, "V_G": q0 - qg["cost"],
                        "V_T": q0 - qt["cost"],
                        "V_GT": q0 - qgt["cost"],
                        "S_GT": qg["cost"] + qt["cost"] - qgt["cost"] - q0,
                        "all_success": (baseline["success"] and qg["success"]
                                        and qt["success"] and qgt["success"]),
                        "menu_G": _menu_coverage(menu, cell, "geometry", env),
                        "menu_T": _menu_coverage(menu, cell, "traction", env)})
    top = max(regions, key=lambda row: row["V_GT"])
    tiers = {}
    if top["V_GT"] > .05:
        cell = tuple(top["cell"])
        for label, modalities in (("G", ("geometry",)),
                                  ("T", ("traction",)),
                                  ("GT", ("geometry", "traction"))):
            actual = _evaluate_reveal(env, cell, modalities)
            dwell = _evaluate_reveal(env, cell, modalities, dwell=True)
            choices = tuple(AuditChoice("carrier", modality, cell)
                            for modality in modalities)
            physical = run_continuation(env, choices, tuple(range(len(choices))))
            tiers[label] = {"actual_value": baseline["cost"] - actual["cost"],
                            "dwell_value": baseline["cost"] - dwell["cost"],
                            "physical_value": (baseline["cost"] -
                                               physical["score_single_world"]),
                            "physical_success": physical["success"]}
    return {"family": family, "seed": seed, "step": step,
            "status": "evaluated", "baseline": baseline["cost"],
            "regions": regions, "top": top, "tiers": tiers}


def _one(job):
    index, seed = job
    family = FAMILIES[index % len(FAMILIES)]
    return [_state(family, seed, step) for step in (120, 360)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=SEEDS[0])
    parser.add_argument("--parents", type=int, default=len(SEEDS))
    parser.add_argument("--tag", default="interface_revision_excluded40_value_funnel_v1")
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.parents))
    records = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs = enumerate(seeds)
        for index, group in enumerate(pool.map(_one, jobs)):
            records.extend(group)
            if (index + 1) % 10 == 0:
                print(index + 1, "/", len(seeds), flush=True)
    evaluated = [row for row in records if row["status"] == "evaluated"]
    high = [row for row in evaluated if row["top"]["V_GT"] > .05]
    summary = {"scheduled": len(records), "evaluated": len(evaluated),
               "high_ideal_pair": len(high),
               "high_ideal_G": sum(any(r["V_G"] > .05 for r in row["regions"])
                                   for row in evaluated),
               "high_ideal_T": sum(any(r["V_T"] > .05 for r in row["regions"])
                                   for row in evaluated),
               "strong_interaction": sum(any(r["S_GT"] > .05 and
                                             r["V_GT"] > .05 and r["all_success"]
                                             for r in row["regions"])
                                         for row in evaluated),
               "top_menu_G": sum(row["top"]["menu_G"] for row in high),
               "top_menu_T": sum(row["top"]["menu_T"] for row in high),
               "top_actual_positive": sum(row["tiers"]["GT"]["actual_value"] > .01
                                          for row in high),
               "top_dwell_positive": sum(row["tiers"]["GT"]["dwell_value"] > .01
                                         for row in high),
               "top_physical_positive": sum(row["tiers"]["GT"]["physical_value"] > .01
                                            for row in high)}
    path = OUT / f"{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"seed_range": [seeds[0], seeds[-1]],
                                "summary": summary, "records": records},
                               indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
