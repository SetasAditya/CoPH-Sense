"""Fresh-parent E2 resource audit at early and mid-route public states.

Predeclared before the 65 s run: 90 new parent maps, steps 120 and 360,
16 public candidates (eight per agent), all singles and up to eight
stratified pairs. Coverage passes only at >=95% overall and >=90% in each
ID family. This is a restricted single-world physical diagnostic, not Q*.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints


FAMILIES = ("open", "bottleneck", "labyrinth")
OUT = Path(__file__).resolve().parent / "results" / "pilot"
SEED_START = 820000
PARENTS = 90
DECISION_STEPS = (120, 360)
PAIR_LIMIT = 8


def public_choices(env):
    obs = env.observations()
    return tuple(AuditChoice(agent, candidate.modality, candidate.viewpoint,
                             candidate.target_region)
                 for agent in ("scout", "carrier")
                 for candidate in candidate_viewpoints(
                     obs[agent], env.config, agent, max_candidates=8))


def stratified_pairs(choices):
    pairs = list(combinations(range(len(choices)), 2))
    def category(pair):
        first, second = (choices[index] for index in pair)
        if (first.agent == second.agent and
                (first.target_region or first.viewpoint) ==
                (second.target_region or second.viewpoint)):
            return 0
        if first.agent != second.agent and first.modality == second.modality:
            return 1
        if first.agent != second.agent:
            return 2
        return 3
    buckets = {kind: [] for kind in range(4)}
    for pair in pairs:
        buckets[category(pair)].append(pair)
    chosen = []
    while len(chosen) < PAIR_LIMIT and any(buckets.values()):
        for kind in range(4):
            if buckets[kind] and len(chosen) < PAIR_LIMIT:
                chosen.append(buckets[kind].pop(0))
    return tuple(chosen)


def audit_state(family, seed, step, include_all_colocated=False):
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, step)
    if env.done:
        return {"family": family, "parent_seed": seed,
                "requested_step": step, "actual_step": env.step_index,
                "status": "terminated_before_decision",
                "prefix_success": env.success,
                "prefix_failure_reason": env.failure_reason}
    choices = public_choices(env)
    pairs = list(stratified_pairs(choices))
    if include_all_colocated:
        for pair in combinations(range(len(choices)), 2):
            first, second = (choices[index] for index in pair)
            if (first.agent == second.agent and
                    (first.target_region or first.viewpoint) ==
                    (second.target_region or second.viewpoint) and
                    pair not in pairs):
                pairs.append(pair)
    sets = ((),) + tuple((index,) for index in range(len(choices))) + tuple(pairs)
    rows = [run_continuation(env, choices, selected) for selected in sets]
    eligible = [row for row in rows if row["success"]]
    best = min(eligible, key=lambda row: row["score_single_world"]) if eligible else None
    return {"family": family, "parent_seed": seed,
            "requested_step": step, "actual_step": env.step_index,
            "status": "evaluated", "choices": [asdict(choice) for choice in choices],
            "sets": [list(selected) for selected in sets], "results": rows,
            "best_set": None if best is None else best["selected"]}


def _one(index):
    seed = SEED_START + index
    family = FAMILIES[index % len(FAMILIES)]
    return [audit_state(family, seed, step) for step in DECISION_STEPS]


def main():
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = []
        for index, group in enumerate(pool.map(_one, range(PARENTS))):
            records.extend(group)
            if (index + 1) % 10 == 0:
                print(index + 1, "/", PARENTS, flush=True)
    evaluated = [row for row in records if row["status"] == "evaluated"]
    def covered(row):
        return (row["best_set"] is not None if row["status"] == "evaluated"
                else bool(row["prefix_success"]))
    class_counts = {name: sum(covered(row) and len(row["best_set"]) == length
                              for row in evaluated)
                    for name, length in (("skip", 0), ("single", 1), ("pair", 2))}
    class_counts["no_success"] = sum(not covered(row) for row in evaluated)
    per_family = {family: {"scheduled": sum(row["family"] == family
                                              for row in records),
                           "covered": sum(row["family"] == family and covered(row)
                                          for row in records)}
                  for family in FAMILIES}
    coverage = sum(covered(row) for row in records) / len(records) if records else 0.
    family_coverage = {family: item["covered"] / item["scheduled"]
                       if item["scheduled"] else 0.
                       for family, item in per_family.items()}
    counts = sorted({len(row["choices"]) for row in evaluated})
    summary = {"parents": PARENTS, "decision_steps": DECISION_STEPS,
               "states": len(records), "evaluated": len(evaluated),
               "terminated_before_decision": len(records) - len(evaluated),
               "early_prefix_success": sum(row["status"] != "evaluated" and
                                           row["prefix_success"] for row in records),
               "early_prefix_failure": sum(row["status"] != "evaluated" and
                                           not row["prefix_success"] for row in records),
               "candidate_count": {
                   str(count): sum(len(row["choices"]) == count for row in evaluated)
                   for count in counts},
               "best_set_class": class_counts,
               "coverage": coverage, "family_coverage": family_coverage,
               "predeclared_guard": {"overall": .95, "each_family": .90},
               "coverage_pass": coverage >= .95 and
                   all(value >= .90 for value in family_coverage.values()),
               "scope": "single-world restricted continuation, not exact belief Q*"}
    output = {"schema_version": 1, "seed_range": [SEED_START,
              SEED_START + PARENTS - 1], "summary": summary, "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rich_resource_audit_fresh90_65s_joint16_v4.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
