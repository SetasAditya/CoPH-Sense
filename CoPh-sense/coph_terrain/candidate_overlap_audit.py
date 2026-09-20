"""Public candidate-menu overlap on excluded CoPH-Terrain parent maps."""

import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .planning import candidate_viewpoints


OUT = Path(__file__).resolve().parent / "results" / "pilot"
FAMILIES = ("open", "bottleneck", "labyrinth")
SEED_START = 780000
PARENTS = 120


def main():
    records = []
    for index in range(PARENTS):
        family = FAMILIES[index % len(FAMILIES)]
        seed = SEED_START + index
        env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9, executor="ph")
        observations = env.observations()
        menus = {agent: {candidate.viewpoint for candidate in
                         candidate_viewpoints(observations[agent], env.config,
                                              agent, max_candidates=8)
                         if candidate.modality == "geometry"}
                 for agent in ("scout", "carrier")}
        common = sorted(menus["scout"] & menus["carrier"])
        records.append({"family": family, "parent_seed": seed,
                        "scout_geometry_viewpoints": len(menus["scout"]),
                        "carrier_geometry_viewpoints": len(menus["carrier"]),
                        "common_viewpoints": common})
    summary = {family: {
        "parents": sum(row["family"] == family for row in records),
        "parents_with_shared_viewpoint": sum(
            row["family"] == family and bool(row["common_viewpoints"])
            for row in records),
        "total_shared_viewpoints": sum(
            len(row["common_viewpoints"]) for row in records
            if row["family"] == family)}
        for family in FAMILIES}
    output = {"schema_version": 1, "seed_range": [SEED_START,
              SEED_START + PARENTS - 1], "summary": summary, "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "candidate_overlap_joint16_shared120_v2.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
