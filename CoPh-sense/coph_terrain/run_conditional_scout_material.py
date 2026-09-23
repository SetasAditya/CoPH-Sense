"""Admission audit and one-shot training gate for material-pH scout dispatch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .conditional_scout_material import (causal_material_rollout,
    conditional_dispatch_teacher, dispatch_candidates, evaluate_dispatch_critic,
    fit_dispatch_critic, save_gate_record)
from .environment import CoPHTerrainEnv, TerrainConfig
from .physical_audit import advance_without_acquisition


def collect(split_seed, states, worlds, family="open"):
    rows, audit = [], []
    for state_index in range(states):
        parent = split_seed + 101 * state_index
        env = CoPHTerrainEnv(
            family, parent, 0, seed=parent+17,
            config=TerrainConfig(carrier_primary=True),
            executor="material_ph", device="cpu")
        advance_without_acquisition(env, 60 + 15 * (state_index % 4))
        if env.done:
            continue
        candidates = dispatch_candidates(env)
        state_id = f"{split_seed}-{state_index}"
        state_rows = []
        for candidate in candidates:
            label = conditional_dispatch_teacher(
                env, candidate, [10000+split_seed+state_index*worlds+i
                                 for i in range(worlds)])
            row = {"state_id": state_id, "candidate_id": candidate.candidate_id,
                   "features": list(candidate.features), "value": label["value"],
                   "world_values": label["world_values"]}
            rows.append(row); state_rows.append(row)
        if state_rows:
            audit.append({"state_id": state_id,
                          "best_value": max(r["value"] for r in state_rows),
                          "candidates": len(state_rows)})
    return rows, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="coph_terrain/results/conditional_scout_material")
    parser.add_argument("--train-states", type=int, default=24)
    parser.add_argument("--val-states", type=int, default=8)
    parser.add_argument("--test-states", type=int, default=12)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, seed, count in (("train", 21000, args.train_states),
                              ("validation", 31000, args.val_states),
                              ("test", 41000, args.test_states)):
        rows, audit = collect(seed, count, args.worlds)
        splits[name] = rows
        (out/f"{name}.json").write_text(json.dumps(rows, indent=2)+"\n")
        (out/f"{name}_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    values = [r["value"] for r in splits["train"] if r["candidate_id"] != "idle"]
    coverage = {"positive": int(np.sum(np.asarray(values) > 0)),
                "negative": int(np.sum(np.asarray(values) <= 0))}
    record = {"coverage": coverage, "trained": False,
              "gate": "blocked_missing_value_sign_coverage"}
    if coverage["positive"] and coverage["negative"]:
        model, validation = fit_dispatch_critic(
            splits["train"], splits["validation"], epochs=args.epochs)
        test = evaluate_dispatch_critic(model, splits["test"])
        torch.save(model.state_dict(), out/"dispatch_value.pt")
        record.update({"trained": True, "validation": validation, "test": test,
                       "gate": "passed" if (
                           test["mean_regret"] < .02 and test["exact_rate"] >= .90
                           and (test["useful_recall"] or 0) >= .90
                           and (test["unnecessary_rejection"] or 0) >= .90)
                       else "failed_metrics"})
        useful = [r for r in splits["test"] if r["candidate_id"] != "idle"
                  and r["value"] > 0]
        if useful:
            # Reconstruct the first positive test state for a causal trace.
            index = int(useful[0]["state_id"].split("-")[-1])
            parent = 41000 + 101 * index
            env = CoPHTerrainEnv("open", parent, 0, seed=parent+17,
                config=TerrainConfig(carrier_primary=True),
                executor="material_ph", device="cpu")
            advance_without_acquisition(env, 60 + 15 * (index % 4))
            candidate = next(c for c in dispatch_candidates(env)
                             if c.candidate_id == useful[0]["candidate_id"])
            record["causal_rollout"] = causal_material_rollout(env, candidate)
    record["record_sha256"] = save_gate_record(out/"gate.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
