"""Train/freeze A3/A5 using v3 Natural maps and material-pH continuations.

This is deliberately a modest, reproducible gate.  It uses train and
validation manifests only; a teacher label is one deterministic realized-world
continuation, so reported regret is realized teacher regret, not population
CVaR.  The confirmatory test manifests are never opened.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .algorithm import acquire_to_evidence, delivery_memory_pair
from .e2_dev_stack import (AGENTS, acquisition_menu, collect_acquisition,
                           model_outputs)
from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition
from .value_model import VariableSetValueNet, decision_score

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "results" / "manifests_v3_natural"
OUT = HERE / "results" / "material_campaign"
CALIBRATION = HERE / "results" / "material_ph_calibration_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_source_hashes():
    names = ("generator.py", "scenario.py", "environment.py", "planning.py",
             "execution.py", "material_executor.py", "algorithm.py",
             "value_model.py", "train_material_a35.py")
    return {name: sha(HERE/name) for name in names}


def spread(groups, count):
    """Deterministic family-balanced prefix selection."""
    buckets = {}
    for row in groups:
        buckets.setdefault(row["family"], []).append(row)
    result = []
    while len(result) < count and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(result) < count:
                result.append(buckets[family].pop(0))
    return result


def collect(group, executor_device="cpu"):
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="material_ph", device=executor_device)
    advance_without_acquisition(env, 120)
    if env.done:
        return [], {"group": group["group_id"], "status": "terminated"}
    examples = []
    for agent in AGENTS:
        item = collect_acquisition(env, agent, "A3")
        if item is not None:
            examples.append(item)
    # Teacher-force one legal measurement solely to create paired A5 histories.
    _, choices, _ = acquisition_menu(env, "scout")
    if not choices:
        return examples, {"group": group["group_id"], "status": "a3_only"}
    measured = acquire_to_evidence(env, choices[0])
    if measured is None:
        return examples, {"group": group["group_id"], "status": "a3_only_acquire_failed"}
    # Delivery is teacher-forced for coverage; this is not an evaluation action.
    from .algorithm import post_reading_branch
    _, delivered = post_reading_branch(measured.env, measured.evidence_id,
                                       True, continue_mission=False)
    changed = False
    if measured.evidence_id in delivered.received["carrier"]:
        retained, withheld, recipient = delivery_memory_pair(
            measured.env, delivered, measured.evidence_id)
        pair = []
        for label, state in (("retained", retained), ("withheld", withheld)):
            item = collect_acquisition(state, recipient, "A5_" + label)
            if item is not None:
                examples.append(item); pair.append(item)
        if len(pair) == 2:
            changed = int(torch.argmin(pair[0]["mission"])) != int(torch.argmin(pair[1]["mission"]))
    return examples, {"group": group["group_id"], "status": "collected",
                      "examples": len(examples), "memory_optimum_changed": bool(changed)}


def metrics(model, examples):
    regrets, correct, pairs = [], 0, 0
    phases = {}
    for item in examples:
        with torch.no_grad():
            mission, quantiles = model_outputs(model, item)
            selected = int(torch.argmin(decision_score(mission, quantiles)))
        truth = item["mission"][0] + item["risk"][0, :, 0]
        oracle = int(torch.argmin(truth))
        regrets.append(float(truth[selected] - truth[oracle]))
        correct += selected == oracle
        pairs += int(oracle > 0 and item["sets"][oracle, 1] >= 0)
        phases.setdefault(item["phase"], []).append(float(truth[selected]-truth[oracle]))
    return {"count": len(examples), "mean_realized_regret": float(np.mean(regrets)),
            "max_realized_regret": float(np.max(regrets)),
            "exact_action_rate": correct / len(examples),
            "pair_optimal_states": pairs,
            "regret_by_phase": {k: float(np.mean(v)) for k, v in phases.items()}}


def fit_model(model, examples, steps, device):
    """Fit on GPU when requested; simulation collection remains separately selectable."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        optimizer.zero_grad(); losses = []
        for original in examples:
            item = dict(original)
            item["inputs"] = tuple(x.to(device) for x in original["inputs"])
            item["candidates"] = original["candidates"].to(device)
            item["sets"] = original["sets"].to(device)
            item["mission"] = original["mission"].to(device)
            item["risk"] = original["risk"].to(device)
            mission, quantiles = model_outputs(model, item)
            from .value_model import quantile_huber_loss
            losses.append((mission-item["mission"]).square().mean() +
                          quantile_huber_loss(quantiles, item["risk"]))
        loss = torch.stack(losses).mean(); loss.backward(); optimizer.step()
    model.cpu()
    return float(loss.detach().cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-groups", type=int, default=6)
    p.add_argument("--validation-groups", type=int, default=3)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--executor-device", default="cpu",
                   help="Use cuda only after VMAS/material collection equivalence is verified")
    args = p.parse_args()
    torch.manual_seed(2701); np.random.seed(2701); torch.set_num_threads(1)
    train_path, val_path = MANIFESTS/"train.json", MANIFESTS/"validation.json"
    # Preserve the immutable map allocation and verify its own archived digest.
    # The old manifest's source hash intentionally predates material_ph; current
    # source hashes below define this derived campaign version.
    for path in (train_path, val_path):
        sidecar = path.with_suffix(".sha256")
        if sha(path) != sidecar.read_text().strip():
            raise RuntimeError(f"base manifest digest mismatch: {path}")
    train_rows = spread(json.loads(train_path.read_text())["groups"], args.train_groups)
    val_rows = spread(json.loads(val_path.read_text())["groups"], args.validation_groups)
    train, validation, logs = [], [], {"train": [], "validation": []}
    for split, rows, target in (("train", train_rows, train),
                                ("validation", val_rows, validation)):
        for row in rows:
            samples, log = collect(row, args.executor_device)
            target.extend(samples); logs[split].append(log)
            print(split, log, flush=True)
    if not train or not validation:
        raise RuntimeError("insufficient material-pH teacher states")
    model = VariableSetValueNet()
    loss = fit_model(model, train, args.steps, args.train_device)
    train_metrics, val_metrics = metrics(model, train), metrics(model, validation)
    OUT.mkdir(parents=True, exist_ok=True)
    memory_switches = sum(row.get("memory_optimum_changed", False)
                          for split in logs.values() for row in split)
    gate = {"validation_mean_regret_below_0.02":
            val_metrics["mean_realized_regret"] < .02,
            "validation_exact_action_rate_at_least_0.90":
            val_metrics["exact_action_rate"] >= .90,
            "pair_optimal_state_present":
            train_metrics["pair_optimal_states"] + val_metrics["pair_optimal_states"] > 0,
            "memory_optimum_switch_present": memory_switches > 0}
    passed = all(gate.values())
    checkpoint = OUT/("a3_a5_frozen.pt" if passed else "a3_a5_candidate_failed.pt")
    torch.save({"model_state_dict": model.state_dict(), "architecture": "VariableSetValueNet",
                "seed": 2701, "executor": "material_ph", "schema_version": 1}, checkpoint)
    report = {"status": ("passed_frozen_material_ph_a3_a5_gate" if passed
                         else "failed_material_ph_a3_a5_gate"), "scope":
              "single-world deterministic teacher; train/validation only; not confirmatory CVaR",
              "executor": "material_ph", "calibration_sha256": sha(CALIBRATION),
              "manifests": {"train_sha256": sha(train_path), "validation_sha256": sha(val_path)},
              "base_manifest_source_status":
              "map allocation reused; current material-pH sources versioned below",
              "current_source_sha256": current_source_hashes(),
              "arguments": vars(args), "groups": logs, "final_train_loss": loss,
              "train": train_metrics, "validation": val_metrics,
              "gate": gate, "memory_optimum_switches": memory_switches,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
              "test_manifests_opened": False}
    (OUT/"a3_a5_training_log.json").write_text(json.dumps(report, indent=2)+"\n")
    sidecar = ("a3_a5_frozen.sha256" if passed else
               "a3_a5_candidate_failed.sha256")
    (OUT/sidecar).write_text(sha(checkpoint)+"  "+checkpoint.name+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
