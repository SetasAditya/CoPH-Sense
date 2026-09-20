"""Disposable E2-dev teacher -> A3 critic -> executed-policy plumbing check.

This deliberately does not claim an A4/A5 learning result. Evidence is sent
with the existing scripted protocol and retained in local observations. The
training labels are single-world outcomes, unsuitable for confirmatory CVaR.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints
from .value_model import (VariableSetValueNet, decision_score,
                          observation_map_tensor)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e2_dev"
TRAIN = HERE / "results" / "manifests_v2" / "train.json"


def local_features(observation, agent):
    kin = np.asarray(observation["kinematics"], dtype=np.float32)
    state = np.concatenate((kin, np.asarray([
        observation["remaining_steps"] / 1300.,
        observation["measurements_remaining"] / 4.,
        observation["team_packets_remaining"] / 4.,
        float(agent == "carrier")], dtype=np.float32)))
    # Evidence values are already incorporated into the known-map channels;
    # this compact memory vector records the local receipt/ownership counts.
    memory = np.zeros((1, 12), dtype=np.float32)
    memory[0, 0] = len(observation["owned_evidence_ids"]) / 4.
    memory[0, 1] = len(observation["received_evidence_ids"]) / 4.
    return (observation_map_tensor(observation)[None],
            torch.from_numpy(state[None]),
            torch.from_numpy(memory[None]),
            torch.ones((1, 1), dtype=torch.bool))


def one_state(group, decision_step=120):
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, decision_step)
    if env.done:
        return None
    agent = "carrier"
    observation = env.observations()[agent]
    candidates = candidate_viewpoints(observation, env.config, agent,
                                      max_candidates=4)
    if len(candidates) < 2:
        return None
    choices = tuple(AuditChoice(agent, item.modality, item.viewpoint,
                                item.target_region) for item in candidates)
    sets = [(-1, -1)] + [(i, -1) for i in range(len(choices))]
    sets += [(i, j) for i in range(len(choices))
             for j in range(i + 1, len(choices))
             if choices[i].target_region == choices[j].target_region]
    rows = [run_continuation(env, choices,
                             tuple(index for index in pair if index >= 0))
            for pair in sets]
    return {"group": group["group_id"], "seed": seed,
            "inputs": local_features(observation, agent),
            "candidates": torch.tensor(np.asarray(
                [item.features() for item in candidates], dtype=np.float32))[None],
            "sets": torch.tensor(sets, dtype=torch.long), "rows": rows}


def predict(model, item):
    map_tensor, state, memory, mask = item["inputs"]
    mission, quantiles = model(map_tensor, state, memory, mask,
                               item["candidates"], item["sets"])
    return mission[0], quantiles[0]


def main():
    torch.manual_seed(1701)
    torch.set_num_threads(1)
    groups = json.loads(TRAIN.read_text())["groups"]
    # These are train-manifest maps only; no validation/test/pilot labels enter.
    examples = [one_state(group) for group in groups[:3]]
    examples = [item for item in examples if item is not None]
    if not examples:
        raise RuntimeError("no E2-dev states were available")
    model = VariableSetValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(40):
        optimizer.zero_grad()
        losses = []
        for item in examples:
            mission, quantiles = predict(model, item)
            cost = torch.tensor([row["mission_cost"] for row in item["rows"]])
            risk = torch.tensor([row["material_exposure"] for row in item["rows"]])
            losses.append((mission - cost).square().mean() +
                          (quantiles[:, 18] - risk).abs().mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    # Executed local policy on a separate train-manifest development map.
    heldout = one_state(groups[3])
    if heldout is None:
        raise RuntimeError("development rollout state unavailable")
    with torch.no_grad():
        mission, quantiles = predict(model, heldout)
        chosen = int(torch.argmin(decision_score(mission, quantiles)))
    group = groups[3]
    replay = CoPHTerrainEnv(group["family"], group["parent_seed"], 0,
                            seed=group["parent_seed"] + 9,
                            executor="ph", device="cpu")
    advance_without_acquisition(replay, 120)
    candidate_menu = candidate_viewpoints(replay.observations()["carrier"],
                                          replay.config, "carrier",
                                          max_candidates=4)
    replay_choices = tuple(AuditChoice("carrier", item.modality,
                                       item.viewpoint, item.target_region)
                           for item in candidate_menu)
    executed = run_continuation(replay, replay_choices,
                                tuple(index for index in
                                      heldout["sets"][chosen].tolist()
                                      if index >= 0))
    teacher = min(range(len(heldout["rows"])),
                  key=lambda index: heldout["rows"][index]["score_single_world"])
    report = {"status": "engineering_smoke_only", "train_groups":
              [item["group"] for item in examples],
              "development_group": heldout["group"],
              "train_loss": float(loss.detach()),
              "selected_set": heldout["sets"][chosen].tolist(),
              "selected_realized_cost": executed["score_single_world"],
              "selected_success": executed["success"],
              "restricted_teacher_set": heldout["sets"][teacher].tolist(),
              "restricted_teacher_cost":
                  heldout["rows"][teacher]["score_single_world"],
              "limitations": ["A3 local critic smoke, not a risk-calibrated model",
                              "A4/A5 learning and recurrent decisions not fitted",
                              "single-world teacher is not belief Q or CVaR"]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "smoke_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), OUT / "smoke_v1_disposable.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
