"""Fresh excluded-parent physical audit after public candidate revision.

The map generator, observation process, prices, caps, and horizon are fixed.
This is a single-world restricted-continuation diagnostic, not belief Q*.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .rich_resource_audit import audit_state, FAMILIES


OUT = Path(__file__).resolve().parent / "results" / "pilot"
SEEDS = tuple(range(850000, 850040))
STEPS = (120, 360)


def _one(job):
    index, seed, all_colocated = job
    family = FAMILIES[index % len(FAMILIES)]
    return [audit_state(family, seed, step,
                        include_all_colocated=all_colocated) for step in STEPS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=SEEDS[0])
    parser.add_argument("--parents", type=int, default=len(SEEDS))
    parser.add_argument("--tag", default="interface_revision_excluded40_physical_v1")
    parser.add_argument("--all-colocated", action="store_true")
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.parents))
    records = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs = ((index, seed, args.all_colocated)
                for index, seed in enumerate(seeds))
        for index, group in enumerate(pool.map(_one, jobs)):
            records.extend(group)
            if (index + 1) % 10 == 0:
                print(index + 1, "/", len(seeds), flush=True)
    evaluated = [row for row in records if row["status"] == "evaluated"]
    completed = [row for row in evaluated if row["best_set"] is not None]
    covered = (len(completed) + sum(row["status"] != "evaluated" and
                                    row["prefix_success"] for row in records))
    classes = {name: sum(len(row["best_set"]) == size for row in completed)
               for name, size in (("skip", 0), ("single", 1), ("pair", 2))}
    summary = {"scheduled": len(records), "evaluated": len(evaluated),
               "covered": covered, "best_set_class": classes,
               "no_success": len(evaluated) - len(completed),
               "scope": "single-world restricted physical continuation"}
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
