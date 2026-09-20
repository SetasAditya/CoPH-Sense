"""Excluded-parent restricted physical resource audit; no learner or test maps."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from .physical_audit import audit_parent


OUT = Path(__file__).resolve().parent / "results" / "pilot"
FAMILIES = ("open", "bottleneck", "labyrinth")


def analyze(records):
    classes = {"skip": 0, "singleton": 0, "pair": 0,
               "no_success_option": 0, "unavailable": 0}
    pair_complementarity = 0
    one_modality_insufficient = 0
    cap_binding = 0
    completed_rows = 0
    all_rows = 0
    improvements = []
    for record in records:
        rows = {tuple(row["selected"]): row for row in record["results"]}
        if () not in rows:
            classes["unavailable"] += 1
            continue
        for row in rows.values():
            all_rows += 1
            completed_rows += int(row["success"])
            cap_binding += int(row["packets"] >= 4)
        eligible = [row for row in rows.values() if row["success"]]
        if not eligible:
            classes["no_success_option"] += 1
            continue
        best = min(eligible, key=lambda row: row["score_single_world"])
        classes[{0: "skip", 1: "singleton", 2: "pair"}[len(best["selected"])]] += 1
        if rows[()]["success"]:
            improvements.append(rows[()]["score_single_world"] - best["score_single_world"])
        for selected, row in rows.items():
            if len(selected) != 2:
                continue
            first, second = selected
            if ((first,) not in rows or (second,) not in rows
                    or not all(rows[key]["success"] for key in
                               ((), (first,), (second,), selected))):
                continue
            q0 = rows[()]["score_single_world"]
            qa = rows[(first,)]["score_single_world"]
            qb = rows[(second,)]["score_single_world"]
            qab = row["score_single_world"]
            contrast = qa + qb - qab - q0
            pair_complementarity += int(contrast > .05)
            one_modality_insufficient += int(
                qab + .05 < q0 and qa >= q0 - .01 and qb >= q0 - .01)
    return {
        "parents": len(records), "best_set_class": classes,
        "score_improvement_over_skip_mean": float(np.mean(improvements))
        if improvements else None,
        "pair_contrasts_above_0p05": pair_complementarity,
        "strict_pair_only_improvements": one_modality_insufficient,
        "completed_branch_fraction": completed_rows / all_rows if all_rows else None,
        "branches_at_four_packet_cap": cap_binding,
        "limits": [
            "single realized world per parent; score is C0+CR, not conditional expectation plus CVaR",
            "scripted continuation, not optimal decentralized policy",
            "four candidate actions imply at most two sends per branch; this pool cannot test four-packet scarcity",
            "memory-induced redundancy and 4+4 measurement caps require separate interventions",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=670000)
    parser.add_argument("--decision-step", type=int, default=120)
    parser.add_argument("--region-ranks", type=int, nargs=2, default=(1, 3))
    parser.add_argument("--tag", default="calibration")
    args = parser.parse_args()
    if args.parents < 1 or args.workers < 1 or args.seed_start < 600000:
        raise ValueError("invalid excluded-parent audit configuration")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_parent, FAMILIES[i % 3],
                               args.seed_start + i, 0, None, "cpu",
                               args.decision_step, tuple(args.region_ranks))
                   for i in range(args.parents)]
        records = []
        for i, future in enumerate(futures):
            record = future.result()
            records.append(record)
            print(i + 1, "/", args.parents, record["family"],
                  record["best_sets"], flush=True)
    elapsed = time.perf_counter() - started
    output = {"schema_version": 1,
              "parent_range": [args.seed_start, args.seed_start + args.parents - 1],
              "decision_step": args.decision_step,
              "region_ranks": args.region_ranks,
              "wall_seconds": elapsed,
              "summary": analyze(records), "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"resource_audit_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
