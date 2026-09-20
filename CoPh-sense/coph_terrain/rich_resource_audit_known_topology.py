"""Fresh restricted resource audit after the public-topology revision."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .rich_resource_audit import audit_state


FAMILIES = ("open", "bottleneck", "labyrinth")
SEED_START = 840000
PARENTS = 90
STEPS = (120, 360)
OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _one(index):
    seed = SEED_START + index
    family = FAMILIES[index % len(FAMILIES)]
    return [audit_state(family, seed, step) for step in STEPS]


def main():
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = []
        for index, group in enumerate(pool.map(_one, range(PARENTS))):
            records.extend(group)
            if (index + 1) % 10 == 0:
                print(index + 1, "/", PARENTS, flush=True)
    def covered(row):
        return (row["best_set"] is not None if row["status"] == "evaluated"
                else bool(row["prefix_success"]))
    evaluated = [row for row in records if row["status"] == "evaluated"]
    by_family = {family: {"covered": sum(covered(row) for row in records
                                         if row["family"] == family),
                          "scheduled": sum(row["family"] == family for row in records)}
                 for family in FAMILIES}
    classes = {name: sum(row["best_set"] is not None
                         and len(row["best_set"]) == size for row in evaluated)
               for name, size in (("skip", 0), ("single", 1), ("pair", 2))}
    classes["no_success"] = sum(row["best_set"] is None for row in evaluated)
    overall = sum(covered(row) for row in records) / len(records)
    families = {name: count["covered"] / count["scheduled"]
                for name, count in by_family.items()}
    summary = {"parents": PARENTS, "states": len(records),
               "evaluated": len(evaluated), "coverage": overall,
               "family_coverage": families, "best_set_class": classes,
               "predeclared_guard": {"overall": .95, "each_family": .90},
               "coverage_pass": overall >= .95 and
                   all(value >= .90 for value in families.values()),
               "scope": "single-world restricted continuation; not belief Q*"}
    path = OUT / "rich_resource_audit_known_topology_fresh90_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"seed_range": [SEED_START,
                   SEED_START + PARENTS - 1], "summary": summary,
                   "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
