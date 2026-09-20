"""Evaluator-only E2-Complementarity source-pool audit.

No policy checkpoint is imported. This script records rejected as well as
accepted parents; its output is development evidence until the generator and
selection rule have been frozen and independently hashed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .complementarity import (ChallengeSpec, ComplementarityEnv,
                              geometry_eligible, structural_acquisition_values)


HERE = Path(__file__).resolve().parent


def audit_source_pool(start, count):
    records = []
    for seed in range(start, start + count):
        env = ComplementarityEnv(
            ChallengeSpec(seed, (False, False), (False, False)))
        record = {"parent_seed": seed, "geometry_eligible": geometry_eligible(env)}
        if record["geometry_eligible"]:
            record.update(structural_acquisition_values(env))
        records.append(record)
        print(seed, "admitted" if record.get("admitted") else "rejected",
              record.get("pair_margin"), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = audit_source_pool(args.start, args.count)
    files = ("complementarity.py", "complementarity_audit.py",
             "environment.py", "generator.py", "planning.py",
             "execution.py", "physical_audit.py")
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
              for name in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "development source-pool audit; restricted pH continuation",
        "start": args.start, "count": args.count,
        "source_sha256": hashes, "records": records,
        "eligible_count": sum(item["geometry_eligible"] for item in records),
        "admitted_count": sum(item.get("admitted", False) for item in records),
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
