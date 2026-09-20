"""One disposable on-policy counterfactual refresh of the E2-dev stack."""

import json

import torch

from .algorithm import (a4_teacher, acquire_to_evidence,
                        delivery_memory_pair, post_reading_branch)
from .e2_dev_stack import (AGENTS, OUT, TRAIN, acquisition_menu,
                           choose, collect_acquisition, evidence_features,
                           execute_snapshot, execute_stack, fit, sample)
from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition, run_continuation
from .value_model import VariableSetValueNet


def collect_acquisition_under_policy(env, acq_model, share_model, phase):
    candidates, _, sets = acquisition_menu(env, "scout")
    if not candidates:
        return None
    rows = [execute_snapshot(env, acq_model, share_model, index)
            for index in range(len(sets))]
    if any(row["status"] != "complete" for row in rows):
        return None
    return sample(env, "scout", [item.features() for item in candidates],
                  sets, rows, phase)


def a4_teacher_under_policy(snapshot, evidence_id, acq_model):
    held, _ = post_reading_branch(snapshot, evidence_id, False)
    sent, delivered = post_reading_branch(snapshot, evidence_id, True)
    if evidence_id in delivered.received["carrier"] and not delivered.done:
        candidates, choices, sets = acquisition_menu(delivered, "carrier")
        if candidates:
            selected = choose(acq_model, delivered, "carrier", candidates,
                              sets, "acquire")
            sent = run_continuation(delivered, choices,
                                    tuple(i for i in sets[selected] if i >= 0))
    return {"hold": held, "send": sent, "delivered_snapshot": delivered}


def collect_policy_histories(group, acq_model, share_model):
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, 120)
    acq_examples = []
    share_examples = []
    initial = collect_acquisition_under_policy(env, acq_model, share_model,
                                               "A3_on_policy")
    if initial is not None:
        acq_examples.append(initial)
    candidates, choices, sets = acquisition_menu(env, "scout")
    if not candidates:
        return acq_examples, share_examples, {"status": "no_choices"}
    selected_index = choose(acq_model, env, "scout", candidates, sets,
                            "acquire")
    selected = tuple(i for i in sets[selected_index] if i >= 0)
    if not selected:
        advance_without_acquisition(env, 360)
        if not env.done:
            later = collect_acquisition_under_policy(env, acq_model,
                                                     share_model,
                                                     "A3_after_skip")
            if later is not None:
                acq_examples.append(later)
        return acq_examples, share_examples, {
            "status": "skip", "selected": list(sets[selected_index]),
            "examples": len(acq_examples)}
    measured = acquire_to_evidence(env, choices[selected[0]])
    if measured is None:
        return acq_examples, share_examples, {"status": "measurement_failed"}
    teacher = a4_teacher_under_policy(measured.env, measured.evidence_id,
                                      acq_model)
    share_examples.append(sample(
        measured.env, "scout",
        [evidence_features(measured.env, "scout", measured.evidence_id)],
        [(-1, -1), (0, -1)], [teacher["hold"], teacher["send"]],
        "A4_on_policy"))
    share_index = choose(share_model, measured.env, "scout",
                         [measured.evidence_id], [(-1, -1), (0, -1)], "share")
    if share_index == 0:
        return acq_examples, share_examples, {"status": "held",
                                               "examples": len(acq_examples)}
    _, delivered = post_reading_branch(measured.env, measured.evidence_id,
                                       True)
    if measured.evidence_id not in delivered.received["carrier"]:
        return acq_examples, share_examples, {"status": "not_delivered"}
    retained, withheld, recipient = delivery_memory_pair(
        measured.env, delivered, measured.evidence_id)
    for name, state in (("retained", retained), ("withheld", withheld)):
        item = collect_acquisition(state, recipient, "A5_on_policy_" + name)
        if item is not None:
            acq_examples.append(item)
    return acq_examples, share_examples, {"status": "delivered",
                                           "examples": len(acq_examples)}


def main():
    torch.manual_seed(1703)
    torch.set_num_threads(1)
    groups = json.loads(TRAIN.read_text())["groups"]
    acq_model = VariableSetValueNet()
    share_model = VariableSetValueNet()
    acq_model.load_state_dict(torch.load(OUT / "stack_v1_a3_a5_disposable.pt",
                                         map_location="cpu"))
    share_model.load_state_dict(torch.load(OUT / "stack_v1_a4_disposable.pt",
                                           map_location="cpu"))
    acq_examples, share_examples, paths = [], [], []
    for index in (4, 204, 354):
        a, s, info = collect_policy_histories(groups[index],
                                               acq_model, share_model)
        acq_examples.extend(a)
        share_examples.extend(s)
        paths.append({"group": groups[index]["group_id"], **info})
        print(paths[-1], flush=True)
    acq_loss = fit(acq_model, acq_examples, steps=40) if acq_examples else None
    share_loss = fit(share_model, share_examples, steps=40) if share_examples else None
    report = {"status": "engineering_smoke_only", "policy_histories": paths,
              "new_acquisition_examples": len(acq_examples),
              "new_sharing_examples": len(share_examples),
              "refresh_train_loss": {"acquisition": acq_loss,
                                     "sharing": share_loss},
              "replay": execute_stack(groups[5], acq_model, share_model),
              "limitations": ["single-world counterfactual labels",
                              "one refresh round and staged first acquisition",
                              "no final CVaR or multi-seed claim"]}
    path = OUT / "stack_refresh_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(acq_model.state_dict(), OUT / "stack_refresh_v1_a3_a5_disposable.pt")
    torch.save(share_model.state_dict(), OUT / "stack_refresh_v1_a4_disposable.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
