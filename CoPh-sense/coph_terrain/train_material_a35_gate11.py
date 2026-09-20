"""One predeclared three-stratum A3/A5 coverage rerun under material-pH.

This is a development gate, not a final evaluation. The two challenge strata
are fixed by public information structure and seed before teacher outcomes are
computed; no row is retained or rejected using its optimal action.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .algorithm import acquire_to_evidence, delivery_memory_pair, post_reading_branch
from .complementarity import ChallengeSpec, ComplementarityEnv, TARGETS
from .e2_dev_stack import sample, model_outputs
from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition
from .planning import SensingCandidate
from .train_material_a35 import (CALIBRATION, HERE, MANIFESTS, collect,
                                 current_source_hashes, fit_model, metrics,
                                 sha, spread)
from .value_model import VariableSetValueNet, quantile_huber_loss
from .algorithm import a5_teacher


OUT = HERE / "results" / "material_campaign" / "gate11"
CHOICE_SETS = ((-1, -1), (0, -1), (1, -1), (0, 1))


def _rows(split, natural_groups, challenge_parents):
    base = MANIFESTS / ("train.json" if split == "train" else "validation.json")
    if sha(base) != base.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Natural manifest digest mismatch")
    natural = spread(json.loads(base.read_text())["groups"], natural_groups)
    return [{"stratum": "Natural", "group": row} for row in natural] + [
        {"stratum": stratum, "parent_seed": seed,
         "world_bits": list(bits)}
        for stratum in ("Complementarity", "Moving-A5")
        for seed in challenge_parents
        for bits in ([(False, False, g2, t2)
                      for g2,t2 in product((False, True), repeat=2)]
                     if stratum == "Complementarity" else
                     # Fixed four-corner risk patterns; not selected by Q.
                     ((False,False,False,False),
                      (True,True,False,False),
                      (False,False,True,True),
                      (True,True,True,True)))]


def _manifest(split, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (split + "_manifest.json")
    payload = {"split": split, "scope": "predeclared information-structure strata",
               "executor": "material_ph", "challenge_targets": TARGETS,
               "collector_sha256": sha(Path(__file__)), "rows": rows}
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError("existing Gate 1.1 manifest differs")
    else:
        path.write_text(encoded)
    return path


def _challenge_env(row):
    g1, t1, g2, t2 = row["world_bits"]
    spec = ChallengeSpec(row["parent_seed"], (g1, g2), (t1, t2),
                         known_second=row["stratum"] == "Complementarity")
    return ComplementarityEnv(spec, executor="material_ph", device="cpu")


def _choices(env, region, agent=None):
    # Approved challenge targets, with the unchanged terrain sensor and prices.
    target = TARGETS[region]
    scan_pose = (target[0] - 2, target[1])
    owner = agent or ("scout" if region == 0 else "carrier")
    raw = (AuditChoice(owner, "geometry",
                       scan_pose, target),
           AuditChoice(owner, "traction",
                       target, target))
    observation = env.observations()[raw[0].agent]
    position = np.asarray(observation["kinematics"][:2], dtype=float)
    from .environment import cell_to_position
    result = []
    for choice in raw:
        travel = float(np.linalg.norm(cell_to_position(choice.viewpoint)-position))
        speed = .75 if choice.agent == "scout" else .55
        dwell = env.config.dt * (env.config.scan_dwell_steps if choice.modality == "geometry"
                                  else env.config.probe_dwell_steps)
        price = env.config.scan_cost if choice.modality == "geometry" else env.config.probe_cost
        result.append(SensingCandidate(choice.modality, choice.viewpoint,
                                       travel, travel/speed, dwell, price, 1., 0.,
                                       observation["remaining_steps"]*env.config.dt
                                       - travel/speed-dwell, target))
    return raw, tuple(candidate.features() for candidate in result)


def _teacher_item(env, region, phase):
    choices, features = _choices(env, region)
    selected = [tuple(i for i in pair if i >= 0) for pair in CHOICE_SETS]
    rows = a5_teacher(env, choices[0].agent, choices, selected)
    return sample(env, choices[0].agent, features, CHOICE_SETS, rows, phase)


def _memory_item(env, phase):
    first, first_features = _choices(env, 0, "carrier")
    second, second_features = _choices(env, 1, "carrier")
    choices = first + second
    sets = ((-1,-1), (0,-1), (1,-1), (0,1),
            (2,-1), (3,-1), (2,3))
    selected = [tuple(i for i in pair if i >= 0) for pair in sets]
    rows = a5_teacher(env, "carrier", choices, selected)
    return sample(env, "carrier", first_features + second_features,
                  sets, rows, phase)


def _collect_challenge(row):
    env = _challenge_env(row)
    stratum = row["stratum"]
    if stratum == "Complementarity":
        # Ex-ante Q: average the same physical choices over all four hidden
        # first-region realizations. No label may depend on the realized bit.
        g2, t2 = row["world_bits"][2:]
        worlds = [ComplementarityEnv(
            ChallengeSpec(row["parent_seed"], (g1,g2), (t1,t2), True),
            executor="material_ph", device="cpu")
            for g1,t1 in product((False, True), repeat=2)]
        items = [_teacher_item(world, 0, stratum) for world in worlds]
        for other in items[1:]:
            for left, right in zip(items[0]["inputs"], other["inputs"]):
                if not torch.equal(left, right):
                    raise RuntimeError("complementarity worlds have different legal inputs")
        item = dict(items[0])
        item["mission"] = torch.stack([x["mission"] for x in items]).mean(0)
        item["risk"] = torch.stack([x["risk"] for x in items]).mean(0)
        return [item], {"stratum": stratum, "parent_seed": row["parent_seed"],
                        "world_bits": row["world_bits"], "examples": 1,
                        "memory_optimum_changed": False}
    # Physical first-region acquisition and packet delivery are teacher-forced
    # only to make both legal memory states available for training.
    first, _ = _choices(env, 0)
    measured = acquire_to_evidence(env, first[0])
    if measured is None:
        return [], {"stratum": stratum, "parent_seed": row["parent_seed"],
                    "world_bits": row["world_bits"], "status": "measurement_failed"}
    _, delivered = post_reading_branch(measured.env, measured.evidence_id,
                                       True, continue_mission=False)
    if measured.evidence_id not in delivered.received["carrier"]:
        return [], {"stratum": stratum, "parent_seed": row["parent_seed"],
                    "world_bits": row["world_bits"], "status": "delivery_failed"}
    retained, withheld, _ = delivery_memory_pair(measured.env, delivered,
                                                 measured.evidence_id)
    items = [_memory_item(state, "Moving-A5_"+label)
             for label, state in (("retained", retained), ("withheld", withheld))]
    optimum = [int(torch.argmin(x["mission"][0] + x["risk"][0,:,0]))
               for x in items]
    return items, {"stratum": stratum, "parent_seed": row["parent_seed"],
                   "world_bits": row["world_bits"], "examples": 2,
                   "memory_optimum_changed": optimum[0] != optimum[1]}


def _per_stratum(model, examples):
    return {key: metrics(model, [x for x in examples
                                 if x["phase"].split("_")[0] == key])
            for key in ("Natural", "Complementarity", "Moving-A5")
            if any(x["phase"].split("_")[0] == key for x in examples)}


def _fit_stratified(model, examples, steps, device):
    # Keep the exact architecture and value/quantile loss. Equal stratum mass
    # prevents Natural SKIP prevalence from erasing the challenge states.
    buckets = {key: [x for x in examples if x["phase"].split("_")[0] == key]
               for key in ("Natural", "Complementarity", "Moving-A5")}
    if not all(buckets.values()):
        raise RuntimeError("one or more Gate 1.1 strata has no teacher states")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    moved = {}
    for key, bucket in buckets.items():
        moved[key] = []
        for original in bucket:
            item = dict(original)
            item["inputs"] = tuple(x.to(device) for x in original["inputs"])
            for name in ("candidates", "sets", "mission", "risk"):
                item[name] = original[name].to(device)
            moved[key].append(item)
    for _ in range(steps):
        optimizer.zero_grad()
        strata_losses = []
        for bucket in moved.values():
            losses = []
            for item in bucket:
                mission, quantiles = model_outputs(model, item)
                losses.append((mission-item["mission"]).square().mean() +
                              quantile_huber_loss(quantiles, item["risk"]))
            strata_losses.append(torch.stack(losses).mean())
        loss = torch.stack(strata_losses).mean()
        loss.backward(); optimizer.step()
    model.cpu()
    return float(loss.detach().cpu()), {k: len(v) for k,v in buckets.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--natural-train-groups", type=int, default=6)
    parser.add_argument("--natural-validation-groups", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(2701); np.random.seed(2701); torch.set_num_threads(1)
    train_rows = _rows("train", args.natural_train_groups, (1100000,))
    val_rows = _rows("validation", args.natural_validation_groups, (1200000,))
    paths = {"train": _manifest("train", train_rows),
             "validation": _manifest("validation", val_rows)}
    datasets, logs = {}, {}
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        examples, records = [], []
        for row in rows:
            if row["stratum"] == "Natural":
                items, record = collect(row["group"])
                for item in items:
                    item["phase"] = "Natural_" + item["phase"]
            else:
                items, record = _collect_challenge(row)
            examples.extend(items); records.append(record)
            print(split, record, flush=True)
        datasets[split], logs[split] = examples, records
    model = VariableSetValueNet()
    loss, counts = _fit_stratified(model, datasets["train"], args.steps,
                                   args.train_device)
    train_metrics = _per_stratum(model, datasets["train"])
    val_metrics = _per_stratum(model, datasets["validation"])
    overall = metrics(model, datasets["validation"])
    macro_regret = float(np.mean([x["mean_realized_regret"] for x in val_metrics.values()]))
    macro_exact = float(np.mean([x["exact_action_rate"] for x in val_metrics.values()]))
    pair_count = val_metrics["Complementarity"]["pair_optimal_states"]
    switches = sum(x.get("memory_optimum_changed", False)
                   for x in logs["validation"])
    gate = {"global_mean_regret_below_0.02":
            overall["mean_realized_regret"] < .02,
            "global_exact_rate_at_least_0.90": overall["exact_action_rate"] >= .90,
            "complementarity_pair_optimal_present": pair_count > 0,
            "moving_a5_memory_switch_present": switches > 0,
            "natural_mean_regret_below_0.02":
            val_metrics["Natural"]["mean_realized_regret"] < .02}
    status = "passed" if all(gate.values()) else "failed"
    checkpoint = OUT / ("a3_a5_frozen.pt" if status == "passed"
                        else "a3_a5_candidate_failed.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "architecture": "VariableSetValueNet", "seed": 2701,
                "executor": "material_ph", "schema_version": 1}, checkpoint)
    report = {"status": status, "scope": "Gate 1.1 development rerun; no test/OOD",
              "arguments": vars(args), "stratified_train_counts": counts,
              "manifests": {k: sha(v) for k,v in paths.items()},
              "calibration_sha256": sha(CALIBRATION),
              "source_sha256": {**current_source_hashes(),
                                "train_material_a35_gate11.py": sha(Path(__file__))},
              "train": train_metrics, "validation": val_metrics,
              "validation_global": overall,
              "validation_macro_mean_regret": macro_regret,
              "validation_macro_exact_rate": macro_exact,
              "validation_memory_switches": switches, "gate": gate,
              "collection": logs, "train_loss": loss,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
              "test_manifests_opened": False}
    (OUT/"gate11_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": status, "gate": gate,
                      "validation": val_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
