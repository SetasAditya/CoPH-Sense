"""Environment-only E2 deadline calibration on excluded parent maps.

The rule is fixed before observing this sample: ceil to five seconds of the
95th percentile *successful* full-map reference completion time plus 30 s.
The 30 s slack covers one extra map-width scout/carrier detour (12/.55 ≈ 22 s),
up to eight sensing dwells, and packet/route-update time. Full-map feasibility
is evaluator-only and never becomes an actor observation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np

from .pilot_environment import run_one


OUT = Path(__file__).resolve().parent / "results" / "pilot"
FAMILIES = ("open", "bottleneck", "labyrinth")
SEED_START = 700000
PARENTS = 120
QUANTILE = .95
SLACK_SECONDS = 30.
ROUND_SECONDS = 5.


def _one(index):
    return run_one(FAMILIES[index % len(FAMILIES)], SEED_START + index,
                   full_map=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="terminal_v2")
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(_one, range(PARENTS)))
    successful = [row["steps"] * .05 for row in records if row["success"]]
    if not successful:
        raise RuntimeError("no feasible full-map reference")
    q95 = float(np.quantile(successful, QUANTILE, method="higher"))
    horizon = ROUND_SECONDS * math.ceil((q95 + SLACK_SECONDS) / ROUND_SECONDS)
    summary = {"parents": PARENTS,
               "full_map_success": len(successful),
               "full_map_success_by_family": {
                   family: sum(row["family"] == family and row["success"]
                               for row in records) for family in FAMILIES},
               "q95_success_seconds": q95,
               "fixed_slack_seconds": SLACK_SECONDS,
               "round_seconds": ROUND_SECONDS,
               "proposed_horizon_seconds": horizon,
               "proposed_horizon_steps": int(round(horizon / .05)),
               "rule": "ceil_5s(q95(full-map successful times) + 30s)",
               "failure_caveat": "unsuccessful full-map references are listed and cannot justify a horizon"}
    output = {"schema_version": 1, "seed_range": [SEED_START,
              SEED_START + PARENTS - 1], "summary": summary, "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"deadline_calibration_excluded120_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
