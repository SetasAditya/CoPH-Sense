"""Disposable A3/A4/A5 supervised plumbing run on train-manifest maps.

One realized world per state is used here to test causal stage ordering and
learned execution. This is not final belief-conditioned or CVaR training.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .algorithm import (a4_teacher, a5_teacher, acquire_to_evidence,
                        delivery_memory_pair, post_reading_branch)
from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from .planning import candidate_viewpoints
from .value_model import (VariableSetValueNet, decision_score,
                          observation_map_tensor, quantile_huber_loss)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e2_dev"
TRAIN = HERE / "results" / "manifests_v2" / "train.json"
AGENTS = ("scout", "carrier")


def local_inputs(observation, agent):
    kin = np.asarray(observation["kinematics"], dtype=np.float32)
    state = np.concatenate((kin, np.asarray([
        observation["remaining_steps"] / 1300.,
        observation["measurements_remaining"] / 4.,
        observation["team_packets_remaining"] / 4.,
        float(agent == "carrier")], dtype=np.float32)))
    memory = np.zeros((1, 12), dtype=np.float32)
    evidence = observation["evidence"]
    memory[0, 0] = len(observation["owned_evidence_ids"]) / 4.
    memory[0, 1] = len(observation["received_evidence_ids"]) / 4.
    memory[0, 2] = len(observation["acknowledged_evidence_ids"]) / 4.
    if evidence:
        memory[0, 3] = np.mean([item["modality"] == "geometry"
                                for item in evidence])
        memory[0, 4] = np.mean([item["acquired_step"] / 1300.
                                for item in evidence])
        memory[0, 5] = np.mean([len(item["cells"]) / 128.
                                for item in evidence])
    return (observation_map_tensor(observation)[None],
            torch.from_numpy(state[None]),
            torch.from_numpy(memory[None]),
            torch.ones((1, 1), dtype=torch.bool))


def acquisition_menu(env, agent, max_candidates=4):
    obs = env.observations()[agent]
    candidates = candidate_viewpoints(obs, env.config, agent,
                                      max_candidates=max_candidates)
    choices = tuple(AuditChoice(agent, item.modality, item.viewpoint,
                                item.target_region) for item in candidates)
    sets = [(-1, -1)] + [(i, -1) for i in range(len(choices))]
    sets += [(i, j) for i in range(len(choices))
             for j in range(i + 1, len(choices))
             if choices[i].target_region == choices[j].target_region]
    return candidates, choices, sets


def evidence_features(env, agent, evidence_id):
    evidence = env.owned[agent][evidence_id]
    values = np.asarray(evidence.values, dtype=np.float32)
    source = np.asarray(evidence.location, dtype=np.float32)
    local = env.observations()[agent]
    teammate_position = np.asarray(local["kinematics"][:2] +
                                   local["kinematics"][4:6], dtype=np.float32)
    relative = teammate_position - source
    return (source[0] / 6., source[1] / 6.,
            float(evidence.modality == "geometry"),
            len(evidence.cells) / 128., float(values.mean()),
            float(values.std()), env.step_index / 1300.,
            (env.step_index - evidence.acquired_step) / 1300.,
            float(relative[0]) / 12., float(relative[1]) / 12.,
            env.config.communication_range / 12.,
            env.config.communication_delay_steps / 1300.)


def sample(env, agent, candidate_features, sets, rows, phase):
    if not len(candidate_features):
        return None
    obs = env.observations()[agent]
    features = torch.tensor(np.asarray(candidate_features, dtype=np.float32))[None]
    mission = torch.tensor([row["mission_cost"] for row in rows],
                           dtype=torch.float32)[None]
    risk = torch.tensor([row["material_exposure"] for row in rows],
                        dtype=torch.float32)[None, :, None]
    return {"phase": phase, "inputs": local_inputs(obs, agent),
            "candidates": features, "sets": torch.tensor(sets),
            "mission": mission, "risk": risk}


def collect_acquisition(env, agent, phase):
    candidates, choices, sets = acquisition_menu(env, agent)
    if not candidates:
        return None
    rows = a5_teacher(env, agent, choices, [tuple(i for i in pair if i >= 0)
                                              for pair in sets])
    return sample(env, agent, [item.features() for item in candidates],
                  sets, rows, phase)


def collect_group(group):
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, 120)
    if env.done:
        return [], [], {"group": group["group_id"], "status": "terminated"}
    acquisition = [item for agent in AGENTS
                   if (item := collect_acquisition(env, agent, "A3")) is not None]
    candidates, choices, _ = acquisition_menu(env, "scout")
    if not choices:
        return acquisition, [], {"group": group["group_id"],
                                  "status": "no_scout_choice"}
    measured = acquire_to_evidence(env, choices[0])
    if measured is None:
        return acquisition, [], {"group": group["group_id"],
                                  "status": "acquisition_failed"}
    result = a4_teacher(measured.env, measured.evidence_id)
    share = sample(measured.env, "scout",
                   [evidence_features(measured.env, "scout",
                                      measured.evidence_id)],
                   [(-1, -1), (0, -1)],
                   [result["hold"], result["send"]], "A4")
    states = []
    delivered = result["delivered_snapshot"]
    if measured.evidence_id in delivered.received["carrier"]:
        retained, withheld, recipient = delivery_memory_pair(
            measured.env, delivered, measured.evidence_id)
        for label, state in (("retained", retained), ("withheld", withheld)):
            item = collect_acquisition(state, recipient, "A5_" + label)
            if item is not None:
                states.append(item)
    return acquisition + states, [share], {
        "group": group["group_id"], "status": "collected",
        "a3_states": len(acquisition), "a5_states": len(states),
        "share_teacher": "send" if result["send"]["score_single_world"] <
                                  result["hold"]["score_single_world"] else "hold",
        "delivered": measured.evidence_id in delivered.received["carrier"]}


def model_outputs(model, item):
    mission, quantiles = model(*item["inputs"], item["candidates"],
                               item["sets"])
    return mission, quantiles


def fit(model, examples, steps=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        losses = []
        for item in examples:
            mission, quantiles = model_outputs(model, item)
            losses.append((mission - item["mission"]).square().mean() +
                          quantile_huber_loss(quantiles, item["risk"]))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def choose(model, env, agent, candidates, sets, phase):
    obs = env.observations()[agent]
    if phase == "share":
        features = [evidence_features(env, agent, candidates[0])]
    else:
        features = [item.features() for item in candidates]
    item = sample(env, agent, features, sets,
                  [{"mission_cost": 0., "material_exposure": 0.}
                   for _ in sets], "inference")
    with torch.no_grad():
        mission, quantiles = model_outputs(model, item)
        return int(torch.argmin(decision_score(mission, quantiles)))


def execute_stack(group, acquisition_model, sharing_model,
                  force_first_acquisition=False, force_share_send=False):
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, 120)
    return execute_snapshot(env, acquisition_model, sharing_model,
                            selected_index=1 if force_first_acquisition else None,
                            force_share_send=force_share_send)


def execute_snapshot(env, acquisition_model, sharing_model,
                     selected_index=None, force_share_send=False):
    """Execute a local policy, or evaluate one fixed A3 set, on an exact clone."""
    env = env.clone()
    candidates, choices, sets = acquisition_menu(env, "scout")
    if not candidates:
        return {"status": "no_choices"}
    a3 = (choose(acquisition_model, env, "scout", candidates, sets, "acquire")
          if selected_index is None else selected_index)
    selected = tuple(i for i in sets[a3] if i >= 0)
    if not selected:
        outcome = run_continuation(env, (), ())
        return {"status": "complete", "a3": list(sets[a3]),
                "a4": "not_reached", "a5": "not_reached",
                "mission_cost": outcome["mission_cost"],
                "material_exposure": outcome["material_exposure"],
                "cost": outcome["score_single_world"],
                "success": outcome["success"]}
    current = env
    acquired = []
    ordered = sorted(selected, key=lambda index: candidates[index].travel_distance)
    for index in ordered:
        measured = acquire_to_evidence(current, choices[index])
        if measured is None:
            break
        current = measured.env
        acquired.append(measured.evidence_id)
    if not acquired:
        return {"status": "acquisition_failed", "a3": list(sets[a3])}
    if len(acquired) != len(selected):
        return {"status": "partial_pair_acquisition",
                "a3": list(sets[a3]), "acquired": len(acquired)}
    share_decisions = []
    delivered = current
    delivered_any = False
    for evidence_id in acquired:
        a4 = (1 if force_share_send else choose(
            sharing_model, delivered, "scout", [evidence_id],
            [(-1, -1), (0, -1)], "share"))
        share_decisions.append("send" if a4 else "hold")
        if a4:
            _, delivered = post_reading_branch(delivered, evidence_id, True)
            delivered_any |= evidence_id in delivered.received["carrier"]
            if delivered.done:
                break
    if not delivered_any or delivered.done:
        outcome = (run_continuation(delivered, (), ()) if not delivered.done
                   else {"mission_cost": delivered.ledger.mission_cost,
                         "material_exposure": delivered.ledger.material_exposure,
                         "score_single_world": delivered.ledger.mission_cost +
                         delivered.ledger.material_exposure,
                         "success": delivered.success})
        return {"status": "complete", "a3": list(sets[a3]),
                "a4": share_decisions, "a5": "not_delivered",
                "mission_cost": outcome["mission_cost"],
                "material_exposure": outcome["material_exposure"],
                "cost": outcome["score_single_world"],
                "success": outcome["success"]}
    next_candidates, next_choices, next_sets = acquisition_menu(delivered,
                                                                  "carrier")
    if not next_candidates:
        outcome = run_continuation(delivered, (), ())
        return {"status": "complete", "a3": list(sets[a3]),
                "a4": share_decisions, "a5": "no_choices",
                "mission_cost": outcome["mission_cost"],
                "material_exposure": outcome["material_exposure"],
                "cost": outcome["score_single_world"],
                "success": outcome["success"]}
    a5 = choose(acquisition_model, delivered, "carrier", next_candidates,
                next_sets, "acquire")
    outcome = run_continuation(delivered, next_choices,
                               tuple(i for i in next_sets[a5] if i >= 0))
    return {"status": "complete", "a3": list(sets[a3]),
            "a4": share_decisions,
            "a5": list(next_sets[a5]),
            "mission_cost": outcome["mission_cost"],
            "material_exposure": outcome["material_exposure"],
            "cost": outcome["score_single_world"],
            "success": outcome["success"]}


def main():
    torch.manual_seed(1702)
    torch.set_num_threads(1)
    groups = json.loads(TRAIN.read_text())["groups"]
    selected = [groups[i] for i in (0, 1, 200, 201, 350, 351)]
    acquisition = []
    sharing = []
    collection = []
    for group in selected:
        a, s, report = collect_group(group)
        acquisition.extend(a)
        sharing.extend(s)
        collection.append(report)
        print(report, flush=True)
    if not acquisition or not sharing:
        raise RuntimeError("A3/A4/A5 dev teachers unavailable")
    acq_model = VariableSetValueNet()
    share_model = VariableSetValueNet()
    loss_acq = fit(acq_model, acquisition)
    loss_share = fit(share_model, sharing)
    # Separate training-manifest groups; all learned weights remain disposable.
    ordinary = execute_stack(groups[3], acq_model, share_model)
    forced = execute_stack(groups[3], acq_model, share_model, True, True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"status": "engineering_smoke_only", "collection": collection,
              "acquisition_examples": len(acquisition),
              "sharing_examples": len(sharing),
              "final_train_losses": {"acquisition": loss_acq,
                                     "sharing": loss_share},
              "unforced_replay": ordinary, "forced_stage_replay": forced,
              "limitations": ["single realized world per teacher branch",
                              "scripted A4 teacher continuation",
                              "no on-policy refresh or final risk calibration"]}
    path = OUT / "stack_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(acq_model.state_dict(), OUT / "stack_v1_a3_a5_disposable.pt")
    torch.save(share_model.state_dict(), OUT / "stack_v1_a4_disposable.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
