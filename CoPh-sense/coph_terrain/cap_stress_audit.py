"""Excluded-parent physical stress test for measurement and packet caps.

The forced ten-query script tests whether the declared resource limits can
constrain a physical rollout. It is not an optimized information policy.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition, run_continuation, select_stress_choices


FAMILIES = ("open", "bottleneck", "labyrinth")
OUT = Path(__file__).resolve().parent / "results" / "pilot"


def audit_one(index, seed_start, decision_step):
    seed = seed_start + index
    family = FAMILIES[index % len(FAMILIES)]
    env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, decision_step)
    choices = select_stress_choices(env, 5)
    selected = tuple(range(len(choices)))
    rows = {}
    for label, measurements, packets in (("declared", 4, 4),
                                         ("relaxed_measurements", 5, 4),
                                         ("relaxed_packets", 4, 10),
                                         ("relaxed_both", 5, 10)):
        clone = env.clone()
        clone.config = replace(clone.config,
                               max_measurements_per_agent=measurements,
                               max_team_packets=packets)
        rows[label] = run_continuation(clone, choices, selected)
    return {"family": family, "parent_seed": seed,
            "decision_step": env.step_index,
            "choice_count": len(choices),
            "choices": [asdict(choice) for choice in choices],
            "results": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=680000)
    parser.add_argument("--decision-step", type=int, default=0)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.parents < 1 or args.workers < 1 or args.seed_start < 600000:
        raise ValueError("invalid excluded-parent audit configuration")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_one, index, args.seed_start,
                               args.decision_step) for index in range(args.parents)]
        records = []
        for index, future in enumerate(futures):
            record = future.result()
            records.append(record)
            print(index + 1, "/", args.parents, record["family"], flush=True)
    summary = {"parents": len(records),
               "ten_choice_menus": sum(row["choice_count"] == 10 for row in records),
               "declared_packet_cap_reached": sum(
                   row["results"]["declared"]["packets"] == 4 for row in records),
               "declared_measurement_cap_reached_both": sum(
                   row["results"]["declared"]["measurements"] == 8 for row in records),
               "extra_packets_when_relaxed": sum(
                   row["results"]["relaxed_packets"]["packets"] >
                   row["results"]["declared"]["packets"] for row in records),
               "extra_measurements_when_relaxed": sum(
                   row["results"]["relaxed_measurements"]["measurements"] >
                   row["results"]["declared"]["measurements"] for row in records),
               "successful_extra_packet_pairs": sum(
                   row["results"]["declared"]["success"] and
                   row["results"]["relaxed_packets"]["success"] and
                   row["results"]["relaxed_packets"]["packets"] >
                   row["results"]["declared"]["packets"] for row in records),
               "extra_packets_lower_cost_on_successful_pair": sum(
                   row["results"]["declared"]["success"] and
                   row["results"]["relaxed_packets"]["success"] and
                   row["results"]["relaxed_packets"]["packets"] >
                   row["results"]["declared"]["packets"] and
                   row["results"]["relaxed_packets"]["score_single_world"] <
                   row["results"]["declared"]["score_single_world"] for row in records),
               "scope": "forced physical script, not optimal policy or positive marginal value"}
    output = {"schema_version": 1, "seed_range": [args.seed_start,
              args.seed_start + args.parents - 1],
              "summary": summary, "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cap_stress_audit_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
