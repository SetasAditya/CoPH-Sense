"""Paired train-dev acquisition rivals with the same A4/A5 continuation."""

import json

import torch

from .e2_dev_stack import (OUT, TRAIN, acquisition_menu, execute_snapshot)
from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition
from .value_model import VariableSetValueNet


def main():
    torch.set_num_threads(1)
    groups = json.loads(TRAIN.read_text())["groups"]
    acq = VariableSetValueNet()
    share = VariableSetValueNet()
    acq.load_state_dict(torch.load(OUT / "stack_refresh_v1_a3_a5_disposable.pt",
                                   map_location="cpu"))
    share.load_state_dict(torch.load(OUT / "stack_refresh_v1_a4_disposable.pt",
                                     map_location="cpu"))
    results = []
    for index in (6, 206, 356):
        group = groups[index]
        seed = group["parent_seed"]
        env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                             executor="ph", device="cpu")
        advance_without_acquisition(env, 120)
        candidates, _, sets = acquisition_menu(env, "scout")
        singleton = list(range(1, 1 + len(candidates)))
        selections = {"no_sensing": 0,
                      "fixed_first": singleton[0],
                      "max_uncertainty": max(singleton,
                          key=lambda i: candidates[sets[i][0]].unknown_fraction),
                      "least_public_cost": min(singleton,
                          key=lambda i: candidates[sets[i][0]].travel_time +
                                        candidates[sets[i][0]].dwell_time +
                                        candidates[sets[i][0]].sensing_cost)}
        row = {"group": group["group_id"], "methods": {}}
        for name, selected in selections.items():
            row["methods"][name] = execute_snapshot(env, acq, share,
                                                       selected_index=selected)
        row["methods"]["learned"] = execute_snapshot(env, acq, share)
        results.append(row)
        print(group["group_id"], {name: round(result.get("cost", float("nan")), 4)
                                   for name, result in row["methods"].items()},
              flush=True)
    report = {"status": "engineering_comparison_only",
              "matched_continuation": "same fitted A4 and A5 for every acquisition method",
              "groups": results,
              "limitations": ["three current-generator train seed IDs only",
                              "not the complete final rival set",
                              "no independent fitted seeds or tail inference"]}
    path = OUT / "rivals_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
