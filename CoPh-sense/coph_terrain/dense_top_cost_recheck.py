"""Cost stages for high-value regions found by the dense oracle screen."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints
from .value_decomposition_audit import _evaluate_reveal, _menu_coverage


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _one(row):
    top = max(row["regions"], key=lambda item: item["V_GT"])
    cell = tuple(top["cell"])
    env = CoPHTerrainEnv(row["family"], row["seed"], 0,
                         seed=row["seed"] + 9, executor="ph")
    advance_without_acquisition(env, row["step"])
    baseline = _evaluate_reveal(env, cell, ())
    menu = candidate_viewpoints(env.observations()["carrier"], env.config,
                                "carrier", max_candidates=8)
    tiers = {}
    for label, modalities in (("G", ("geometry",)),
                              ("T", ("traction",)),
                              ("GT", ("geometry", "traction"))):
        zero = _evaluate_reveal(env, cell, modalities)
        dwell = _evaluate_reveal(env, cell, modalities, dwell=True)
        choices = tuple(AuditChoice("carrier", modality, cell)
                        for modality in modalities)
        physical = run_continuation(env, choices, tuple(range(len(choices))))
        tiers[label] = {"zero_travel": zero, "dwell_no_viewpoint_travel": dwell,
                        "physical": {key: physical[key] for key in
                                     ("score_single_world", "success",
                                      "measurements", "packets", "delivered")}}
    return {"seed": row["seed"], "family": row["family"],
            "step": row["step"], "cell": cell, "baseline": baseline,
            "ideal_V_GT": top["V_GT"], "ideal_S_GT": top["S_GT"],
            "menu_G": _menu_coverage(menu, cell, "geometry", env),
            "menu_T": _menu_coverage(menu, cell, "traction", env),
            "tiers": tiers}


def main():
    source = OUT / "dense_information_screen_excluded40_v1.json"
    rows = [row for row in json.loads(source.read_text())["records"]
            if row["status"] == "evaluated" and row["regions"] and
            max(region["V_GT"] for region in row["regions"]) > .05]
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(_one, rows))
    path = OUT / "dense_top_cost_recheck_excluded40_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"source": source.name, "records": records},
                               indent=2) + "\n")
    print(len(records), path)


if __name__ == "__main__":
    main()
