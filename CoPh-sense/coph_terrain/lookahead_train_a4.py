"""Restricted A4 learning from paired physical SEND/HOLD dev labels.

The first sensing action is teacher-forced in both training and evaluation.
This evaluates selective post-reading communication, not a complete A3/A5
policy or a confirmatory E2 benchmark.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .algorithm import acquire_to_evidence
from .lookahead import LookaheadEnv, LookaheadSpec
from .lookahead_dev_audit import VIEW
from .lookahead_policy import RidgeMessageValue, message_features


def features_for_record(record):
    if "actor_features" in record:
        return record["actor_features"]
    spec = LookaheadSpec(record["parent_seed"],
                         surface=(record["surface"], .06),
                         traction=(record["traction"], .90),
                         scout_start=tuple(record.get("scout_start", (12, 27))))
    reading = acquire_to_evidence(LookaheadEnv(spec), VIEW)
    if reading is None:
        raise RuntimeError("stored acquired example did not replay")
    packet_id = next(iter(reading.env.innovation_packets))
    return message_features(reading.env, packet_id).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--train-max-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [row for row in json.loads(args.audit.read_text())["records"]
            if row.get("acquired") and "branches" in row]
    for row in rows:
        row["actor_features"] = features_for_record(row)
    train = [row for row in rows
             if row["parent_seed"] <= args.train_max_seed]
    test = [row for row in rows
            if row["parent_seed"] > args.train_max_seed]
    if len(train) < 2 or not test:
        raise ValueError("need nonempty seed-disjoint train and test groups")
    model = RidgeMessageValue.fit(train)
    choices = []
    for row in test:
        predicted = model.predict(row["actor_features"])
        branch = row["branches"]
        chosen = "innovation" if predicted > 0. and \
            row.get("sender_estimated_kl_nats", 1.) > 0. else "hold"
        costs = {"no_communication": branch["hold"]["cost"],
                 "raw_broadcast": branch["raw"]["cost"],
                 "novelty_only": branch["innovation"]["cost"],
                 "learned_task_value": branch[chosen]["cost"],
                 "exact_message_oracle": min(branch["hold"]["cost"],
                                             branch["innovation"]["cost"])}
        choices.append({"parent_seed": row["parent_seed"],
                        "surface": row["surface"],
                        "prediction": predicted,
                        "selected": chosen, "costs": costs})
    methods = tuple(choices[0]["costs"])
    report = {
        "scope": "disposable, forced first acquisition; local A4 only",
        "train_seeds": sorted({row["parent_seed"] for row in train}),
        "test_seeds": sorted({row["parent_seed"] for row in test}),
        "train_count": len(train), "test_count": len(test),
        "coefficients": model.coefficients.tolist(),
        "mean_test_cost": {name: float(np.mean(
            [row["costs"][name] for row in choices])) for name in methods},
        "mean_regret_to_exact_message_oracle": {
            name: float(np.mean([
                row["costs"][name] - row["costs"]["exact_message_oracle"]
                for row in choices])) for name in methods},
        "choices": choices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mean_test_cost"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
