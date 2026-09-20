"""Write/verify replacement E2-Natural splits under the frozen generator."""

import argparse
import json
from pathlib import Path

from .generator import generate_map
from .manifests import (ALLOCATIONS, DIFFICULTY_EDGES, _difficulty,
                        build_payload, sha256)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "manifests_v3_natural"
FROZEN_SOURCES = ("generator.py", "scenario.py", "environment.py",
                  "planning.py", "execution.py")


def source_hashes():
    return {name: sha256(HERE / name) for name in FROZEN_SOURCES}


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    for split in ALLOCATIONS:
        path = OUT / f"{split}.json"
        if path.exists():
            raise FileExistsError(path)
        payload = build_payload(split)
        payload.update(schema_version=3, benchmark="E2-Natural",
                       source_sha256=sources)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        (OUT / f"{split}.sha256").write_text(sha256(path) + "\n")


def verify():
    seen_seeds = set()
    seen_maps = set()
    sources = source_hashes()
    for split, families in ALLOCATIONS.items():
        path = OUT / f"{split}.json"
        payload = json.loads(path.read_text())
        if (payload["schema_version"] != 3 or
                payload["benchmark"] != "E2-Natural" or
                payload["source_sha256"] != sources or
                payload["generator_sha256"] != sources["generator.py"] or
                sha256(path) != (OUT / f"{split}.sha256").read_text().strip()):
            raise RuntimeError(f"invalid manifest metadata/hash: {split}")
        if len(payload["groups"]) != sum(families.values()):
            raise RuntimeError(f"wrong group count: {split}")
        for row in payload["groups"]:
            seed = row["parent_seed"]
            if seed in seen_seeds or row["map_sha256"] in seen_maps:
                raise RuntimeError(f"reused map group: {row['group_id']}")
            terrain = generate_map(row["family"], seed)
            difficulty, score = _difficulty(terrain)
            if (terrain.digest() != row["map_sha256"] or
                    difficulty != row["difficulty_bin"] or
                    abs(score - row["shortest_path_risk"]) > 1e-10):
                raise RuntimeError(f"map or difficulty changed: {row['group_id']}")
            seen_seeds.add(seed)
            seen_maps.add(row["map_sha256"])
        if tuple(payload["difficulty_edges"]) != DIFFICULTY_EDGES:
            raise RuntimeError("difficulty edges changed")
    return len(seen_seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("write", "verify"))
    args = parser.parse_args()
    if args.operation == "write":
        write()
    print(f"verified {verify()} disjoint E2-Natural groups")


if __name__ == "__main__":
    main()
