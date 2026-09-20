"""One-shot fresh audit governed by MISSION_INTEGRATED_STOP_RULE.md."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition, run_continuation
from .rich_resource_audit import FAMILIES, public_choices, stratified_pairs


OUT = Path(__file__).resolve().parent / "results" / "pilot"
SEED_START = 920000
PARENTS = 20
STEPS = (120, 360)


def audit_state(family, seed, step):
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, step)
    if env.done:
        return {"family": family, "parent_seed": seed, "step": step,
                "status": "terminated_before_decision", "prefix_success": env.success}
    choices = public_choices(env)
    pairs = stratified_pairs(choices)
    sets = ((),) + tuple((i,) for i in range(len(choices))) + pairs
    rows = []
    for selected in sets:
        for transmit in ((True,) if not selected else (True, False)):
            result = run_continuation(env, choices, selected, transmit=transmit)
            result["transmit"] = transmit
            rows.append(result)
    return {"family": family, "parent_seed": seed, "step": step,
            "status": "evaluated", "choices": [asdict(c) for c in choices],
            "sets": [list(s) for s in sets], "results": rows}


def _one(index):
    seed = SEED_START + index
    return [audit_state(FAMILIES[index % len(FAMILIES)], seed, step)
            for step in STEPS]


def summarize(records):
    evaluated = [row for row in records if row["status"] == "evaluated"]
    covered = sum(row["prefix_success"] if row["status"] != "evaluated"
                  else any(item["success"] for item in row["results"])
                  for row in records)
    useful = []
    pair_parents = set()
    classes = {"skip": 0, "single": 0, "pair": 0}
    for row in evaluated:
        baseline = row["results"][0]
        feasible = [item for item in row["results"] if item["success"] and
                    all(item["target_region_covered"])]
        if not feasible:
            continue
        best = min(feasible, key=lambda item: item["score_single_world"])
        classes[("skip", "single", "pair")[len(best["selected"])]] += 1
        if (baseline["success"] and best["selected"] and
                baseline["score_single_world"] - best["score_single_world"] > .01):
            useful.append((row["parent_seed"], row["step"]))
        if len(best["selected"]) == 2 and baseline["success"]:
            best_single = min((item["score_single_world"] for item in feasible
                               if len(item["selected"]) <= 1), default=float("inf"))
            if best_single - best["score_single_world"] > .01:
                pair_parents.add(row["parent_seed"])
    coverage = covered / len(records)
    useful_rate = len(useful) / len(evaluated) if evaluated else 0.
    return {"scheduled": len(records), "evaluated": len(evaluated),
            "covered": covered, "coverage": coverage,
            "useful_states": len(useful), "useful_rate": useful_rate,
            "pair_optimal_parent_maps": sorted(pair_parents),
            "best_set_class": classes,
            "predeclared_gate": {"coverage": .95, "useful_rate": .15,
                                 "pair_optimal_distinct_parents": 2,
                                 "minimum_margin": .01},
            "pass": coverage >= .95 and useful_rate >= .15 and
                    len(pair_parents) >= 2,
            "scope": "restricted fixed-world physical continuation; not belief Q*"}


def main():
    path = OUT / "mission_integrated_fresh20_v1.json"
    if path.exists():
        raise FileExistsError(path)
    with ProcessPoolExecutor(max_workers=10) as pool:
        records = []
        for index, group in enumerate(pool.map(_one, range(PARENTS))):
            records.extend(group)
            print(f"audited {index + 1}/{PARENTS} parents", flush=True)
    summary = summarize(records)
    path.write_text(json.dumps({"schema_version": 1,
                                "seed_range": [SEED_START, SEED_START + PARENTS - 1],
                                "summary": summary, "records": records},
                               indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
