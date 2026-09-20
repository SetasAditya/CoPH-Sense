"""Create and verify immutable map-group manifests before E2 training."""

import argparse
import hashlib
import json
from pathlib import Path

from .generator import GRID_SIZE, generate_map, realize_map
import numpy as np
from collections import deque


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "manifests_v2"
ALLOCATIONS = {
    "train": {"open": 200, "bottleneck": 150, "labyrinth": 150},
    "validation": {"open": 40, "bottleneck": 30, "labyrinth": 30},
    "calibration": {"open": 40, "bottleneck": 30, "labyrinth": 30},
    "test": {"open": 80, "bottleneck": 60, "labyrinth": 60},
    "ood_test": {"deep_labyrinth": 100, "irregular": 100},
}
BASE_SEEDS = {"train": 100000, "validation": 200000,
              "calibration": 300000, "test": 400000,
              "ood_test": 500000}
DIFFICULTY_EDGES = (.10, .30)
DIFFICULTY_NAMES = ("low", "medium", "high")


def _shortest_path(occupancy):
    blocked = occupancy.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            blocked[max(0, dx):GRID_SIZE + min(0, dx),
                    max(0, dy):GRID_SIZE + min(0, dy)] |= occupancy[
                        max(0, -dx):GRID_SIZE - max(0, dx),
                        max(0, -dy):GRID_SIZE - max(0, dy)]
    start, goal = (5, 30), (54, 30)
    queue = deque([start])
    previous = {start: None}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE
                    and not blocked[nxt] and nxt not in previous):
                previous[nxt] = (x, y)
                queue.append(nxt)
    if goal not in previous:
        raise RuntimeError("generated map lacks a disc-clear route")
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = previous[cell]
    return tuple(reversed(path))


def _difficulty(parent):
    route = _shortest_path(parent.occupancy)
    scores = []
    for realization in range(5):
        risk = realize_map(parent, realization).risk
        scores.append(np.mean([risk[cell] for cell in route]))
    value = float(np.mean(scores))
    return DIFFICULTY_NAMES[int(np.searchsorted(DIFFICULTY_EDGES, value))], value


def _quotas(count):
    return {name: count // 3 + int(i < count % 3)
            for i, name in enumerate(DIFFICULTY_NAMES)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_payload(split):
    if split not in ALLOCATIONS:
        raise ValueError(split)
    rows = []
    for family_index, (family, count) in enumerate(ALLOCATIONS[split].items()):
        matched_count = count if split != "ood_test" else count // 2
        remaining = _quotas(matched_count)
        accepted = 0
        candidate_index = 0
        while accepted < count:
            seed = BASE_SEEDS[split] + family_index * 20000 + candidate_index
            candidate_index += 1
            terrain = generate_map(family, seed)
            bin_name, score = _difficulty(terrain)
            matched_slice = accepted < matched_count
            if matched_slice and remaining[bin_name] == 0:
                continue
            if matched_slice:
                remaining[bin_name] -= 1
            rows.append({"group_id": f"{split}/{family}/{accepted:04d}",
                         "family": family, "parent_seed": seed,
                         "map_sha256": terrain.digest(),
                         "difficulty_bin": bin_name,
                         "shortest_path_risk": score,
                         "ood_slice": ("topology" if matched_slice else "compound")
                         if split == "ood_test" else None,
                         "obstacle_count": len(terrain.obstacle_boxes),
                         "patch_count": len(terrain.patch_metadata)})
            accepted += 1
    return {"schema_version": 2, "split": split,
            "generator_sha256": sha256(HERE / "generator.py"),
            "difficulty_edges": DIFFICULTY_EDGES,
            "difficulty_realizations": 5,
            "difficulty_quota_rule": "equal low/medium/high per ID family; "
                                     "half matched and half natural per OOD family",
            "episode_realizations_per_test_group": 5 if split in ("test", "ood_test") else None,
            "groups": rows}


def write_all():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ALLOCATIONS:
        path = OUT / f"{split}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen map manifest: {path}")
        path.write_text(json.dumps(build_payload(split), sort_keys=True, indent=2) + "\n")
        (OUT / f"{split}.sha256").write_text(sha256(path) + "\n")


def verify_all():
    seeds = set()
    digests = set()
    for split in ALLOCATIONS:
        path = OUT / f"{split}.json"
        payload = json.loads(path.read_text())
        if payload["generator_sha256"] != sha256(HERE / "generator.py"):
            raise RuntimeError(f"generator changed after manifest freeze: {split}")
        if sha256(path) != (OUT / f"{split}.sha256").read_text().strip():
            raise RuntimeError(f"manifest hash mismatch: {split}")
        if len(payload["groups"]) != sum(ALLOCATIONS[split].values()):
            raise RuntimeError(f"group count mismatch: {split}")
        if payload["schema_version"] != 2 or tuple(payload["difficulty_edges"]) != DIFFICULTY_EDGES:
            raise RuntimeError(f"difficulty protocol mismatch: {split}")
        for row in payload["groups"]:
            if row["parent_seed"] in seeds or row["map_sha256"] in digests:
                raise RuntimeError(f"reused map group: {row['group_id']}")
            terrain = generate_map(row["family"], row["parent_seed"])
            if terrain.digest() != row["map_sha256"]:
                raise RuntimeError(f"map changed after freeze: {row['group_id']}")
            actual_bin, score = _difficulty(terrain)
            if actual_bin != row["difficulty_bin"] or abs(score - row["shortest_path_risk"]) > 1e-10:
                raise RuntimeError(f"difficulty changed: {row['group_id']}")
            seeds.add(row["parent_seed"])
            digests.add(row["map_sha256"])
        for family, count in ALLOCATIONS[split].items():
            selected = [row for row in payload["groups"] if row["family"] == family]
            if len(selected) != count:
                raise RuntimeError(f"family count mismatch: {split}/{family}")
            matched = ([row for row in selected if row["ood_slice"] == "topology"]
                       if split == "ood_test" else selected)
            if {name: sum(row["difficulty_bin"] == name for row in matched)
                    for name in DIFFICULTY_NAMES} != _quotas(len(matched)):
                raise RuntimeError(f"difficulty quota mismatch: {split}/{family}")
    return len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("write", "verify"))
    args = parser.parse_args()
    if args.operation == "write":
        write_all()
    print(f"verified {verify_all()} disjoint map groups")


if __name__ == "__main__":
    main()
