"""Broader cost-free region screen on already-excluded development maps."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition
from .value_decomposition_audit import public_regions, _evaluate_reveal


OUT = Path(__file__).resolve().parent / "results" / "pilot"
SEEDS = tuple(range(840000, 840040))


def _one(index):
    seed = SEEDS[index]
    family = ("open", "bottleneck", "labyrinth")[index % 3]
    rows = []
    for step in (120, 360):
        env = CoPHTerrainEnv(family, seed, 0, seed=seed + 9, executor="ph")
        advance_without_acquisition(env, step)
        if env.done:
            rows.append({"seed": seed, "family": family, "step": step,
                         "status": "terminated_prefix", "success": env.success})
            continue
        cells = public_regions(env, limit=12)
        baseline = _evaluate_reveal(env, cells[0] if cells else (5, 30), ())
        regions = []
        for cell in cells:
            qg = _evaluate_reveal(env, cell, ("geometry",), ideal=True)
            qt = _evaluate_reveal(env, cell, ("traction",), ideal=True)
            qgt = _evaluate_reveal(env, cell, ("geometry", "traction"), ideal=True)
            regions.append({"cell": cell,
                            "V_G": baseline["cost"] - qg["cost"],
                            "V_T": baseline["cost"] - qt["cost"],
                            "V_GT": baseline["cost"] - qgt["cost"],
                            "S_GT": qg["cost"] + qt["cost"]
                                    - qgt["cost"] - baseline["cost"],
                            "all_success": (baseline["success"] and qg["success"]
                                            and qt["success"] and qgt["success"])})
        rows.append({"seed": seed, "family": family, "step": step,
                     "status": "evaluated", "regions": regions})
    return rows


def main():
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = []
        for index, group in enumerate(pool.map(_one, range(len(SEEDS)))):
            records.extend(group)
            if (index + 1) % 10 == 0:
                print(index + 1, "/", len(SEEDS), flush=True)
    path = OUT / "dense_information_screen_excluded40_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"seed_range": [SEEDS[0], SEEDS[-1]],
                                "records": records}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
