"""Pre-acquisition known-topology navigation check on fresh excluded parents."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .pilot_environment import run_one


FAMILIES = ("open", "bottleneck", "labyrinth")
SEED_START = 830000
PARENTS = 90
OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _one(index):
    family = FAMILIES[index % len(FAMILIES)]
    return run_one(family, SEED_START + index, full_map=False)


def main():
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(_one, range(PARENTS)))
    per_family = {family: {"success": sum(row["success"] for row in records
                                          if row["family"] == family),
                           "total": sum(row["family"] == family for row in records)}
                  for family in FAMILIES}
    success = sum(row["success"] for row in records)
    summary = {"success": success, "total": len(records),
               "coverage": success / len(records), "family": per_family,
               "predeclared_guard": {"overall": .95, "each_family": .90},
               "coverage_pass": success / len(records) >= .95 and all(
                   item["success"] / item["total"] >= .90
                   for item in per_family.values())}
    path = OUT / "navigation_sufficiency_known_topology_fresh90_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"seed_range": [SEED_START, SEED_START + PARENTS - 1],
                                "summary": summary, "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
