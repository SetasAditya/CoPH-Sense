"""Freshly execute the disposable A3 smoke checkpoint on a train-dev map."""

import json

import torch

from .e2_dev_smoke import HERE, OUT, TRAIN, local_features
from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints
from .value_model import VariableSetValueNet, decision_score


def main():
    torch.set_num_threads(1)
    group = json.loads(TRAIN.read_text())["groups"][3]
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, 120)
    observation = env.observations()["carrier"]
    candidates = candidate_viewpoints(observation, env.config, "carrier",
                                      max_candidates=4)
    sets = [(-1, -1)] + [(i, -1) for i in range(len(candidates))]
    sets += [(i, j) for i in range(len(candidates))
             for j in range(i + 1, len(candidates))
             if candidates[i].target_region == candidates[j].target_region]
    model = VariableSetValueNet()
    model.load_state_dict(torch.load(OUT / "smoke_v1_disposable.pt",
                                     map_location="cpu"))
    model.eval()
    features = torch.tensor([item.features() for item in candidates],
                            dtype=torch.float32)[None]
    with torch.no_grad():
        mission, quantiles = model(*local_features(observation, "carrier"),
                                   features, torch.tensor(sets, dtype=torch.long))
        chosen = int(torch.argmin(decision_score(mission, quantiles)))
    choices = tuple(AuditChoice("carrier", item.modality, item.viewpoint,
                                item.target_region) for item in candidates)
    selected = tuple(index for index in sets[chosen] if index >= 0)
    outcome = run_continuation(env, choices, selected)
    report = {"status": "engineering_smoke_only", "group": group["group_id"],
              "selected_set": list(sets[chosen]),
              "executed_cost": outcome["score_single_world"],
              "executed_success": outcome["success"],
              "measurements": outcome["measurements"],
              "packets": outcome["packets"]}
    path = OUT / "smoke_v1_fresh_replay.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
