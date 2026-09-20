"""Prospective A4 learning on the frozen large pool.

All fitting and checkpoint selection use train/validation seeds only. The
structured model uses public recipient-response features and exact radio
accounting. Its three heads are an inductive decomposition; without extra
interventions they must not be interpreted as identified causal components.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lookahead_a4_collect import source_hashes
from .lookahead_a4_learning import (fit_mlp, method_metrics, predict, regret)
from .lookahead_policy import RidgeMessageValue


class StructuredValue(nn.Module):
    """Predicted avoided repeat + new-support value - delay - exact radio."""

    AVOID = (1, 3, 6, 7, 8, 10, 12, 22)
    NEW = (1, 2, 3, 6, 7, 8, 10, 12, 13, 20, 21, 23)
    DELAY = (6, 7, 8, 9, 10, 13, 21)

    def __init__(self):
        super().__init__()
        self.avoid = nn.Linear(len(self.AVOID), 1)
        self.new = nn.Linear(len(self.NEW), 1)
        self.delay = nn.Linear(len(self.DELAY), 1)
        nn.init.constant_(self.avoid.bias, -2.)
        nn.init.constant_(self.delay.bias, -2.)

    def forward(self, normalized, raw):
        repeated_region = raw[:, 19].clamp(0, 1)
        new_support = (raw[:, 20] > 0).float()
        avoided = repeated_region * .02 * F.softplus(
            self.avoid(normalized[:, self.AVOID]).squeeze(-1))
        new_value = new_support * .02 * self.new(
            normalized[:, self.NEW]).squeeze(-1)
        delay = .02 * F.softplus(
            self.delay(normalized[:, self.DELAY]).squeeze(-1))
        radio = raw[:, 14]
        return avoided + new_value - delay - radio, (avoided, new_value,
                                                      delay, radio)


def fit_structured(train, validation, seed, epochs=500):
    torch.manual_seed(seed)
    x = np.asarray([row["response_features"] for row in train], np.float32)
    vx = np.asarray([row["response_features"] for row in validation], np.float32)
    y = np.asarray([row["decision_value"] for row in train], np.float32)
    vy = np.asarray([row["decision_value"] for row in validation], np.float32)
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-3)
    tx = torch.from_numpy(x)
    tv = torch.from_numpy(vx)
    nx = torch.from_numpy((x-mean)/std)
    nv = torch.from_numpy((vx-mean)/std)
    ty = torch.from_numpy(y)
    model = StructuredValue()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015,
                                  weight_decay=.03)
    best = None
    for epoch in range(epochs):
        model.train()
        predicted, _ = model(nx, tx)
        loss = ((predicted-ty)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20:
            continue
        model.eval()
        with torch.no_grad():
            scores = model(nv, tv)[0].numpy()
        candidate = (regret(vy, scores > 0), epoch+1,
                     copy.deepcopy(model.state_dict()))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    model.load_state_dict(best[2])
    return {"model": model, "mean": mean, "std": std,
            "selected_epoch": best[1], "validation_regret": best[0]}


def predict_structured(fitted, rows):
    x = np.asarray([row["response_features"] for row in rows], np.float32)
    raw = torch.from_numpy(x)
    normalized = torch.from_numpy((x-fitted["mean"])/fitted["std"])
    fitted["model"].eval()
    with torch.no_grad():
        score, parts = fitted["model"](normalized, raw)
    return score.numpy(), np.stack([part.numpy() for part in parts], axis=1)


def paired_seed_bootstrap(rows, methods, reference, candidate,
                          replicates=2000):
    seeds = sorted({row["seed"] for row in rows})
    by_seed = {seed: [index for index, row in enumerate(rows)
                      if row["seed"] == seed] for seed in seeds}
    values = np.asarray([row["decision_value"] for row in rows])
    def losses(name):
        decisions = np.asarray(methods[name]["selected_send"], bool)
        return np.where(decisions, np.maximum(0., -values),
                        np.maximum(0., values))
    difference = losses(candidate)-losses(reference)
    rng = np.random.default_rng(4144)
    samples = []
    for _ in range(replicates):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        indices = [index for seed in chosen for index in by_seed[seed]]
        samples.append(float(difference[indices].mean()))
    return {"candidate_minus_reference": float(difference.mean()),
            "seed_bootstrap_95pct": np.quantile(samples, [.025, .975]).tolist()}


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--tag", default="large_v1")
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()
    directory = args.dataset_dir.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("teacher/environment source differs from frozen collection")
    splits = {name: json.loads((directory / f"{name}.json").read_text())["rows"]
              for name in ("train", "validation", "test")}
    for name, rows in splits.items():
        first = manifest["splits"][name]["first_seed"]
        count = manifest["splits"][name]["seed_count"]
        if {row["seed"] for row in rows} != set(range(first, first+count)):
            raise ValueError(f"{name}: seed coverage mismatch")
        for row in rows:
            row["sender_padded_features"] = row["actor_features"] + [0.] * 13
            row["recipient_padded_features"] = row["recipient_features"] + [0.] * 8
    train, validation, test = (splits[name] for name in
                               ("train", "validation", "test"))
    values = np.asarray([row["decision_value"] for row in test])
    methods = {
        "never_send": method_metrics(test, np.zeros(len(test), bool)),
        "always_compact_send": method_metrics(test, np.ones(len(test), bool)),
        "paired_hindsight_oracle": method_metrics(test, values > 0)}
    raw_dir = directory / "raw_broadcast"
    if (raw_dir / "manifest.json").exists() and not args.skip_raw:
        raw_rows = []
        for path in sorted(raw_dir.glob("raw_test_*.json")):
            raw_rows.extend(json.loads(path.read_text())["rows"])
        key = lambda row: (row["seed"], row["risk"], row["age_steps"],
                           row["receiver_condition"])
        by_key = {key(row): row for row in raw_rows}
        if len(by_key) != len(test) or any(key(row) not in by_key
                                           for row in test):
            raise ValueError("raw comparator is incomplete")
        alternatives = []
        bytes_per_send = []
        for row in test:
            raw = by_key[key(row)]
            alternative = dict(row)
            alternative["send"] = raw["raw_send"]
            alternative["decision_value"] = (
                row["hold"]["cost"]-raw["raw_send"]["cost"])
            alternative["communicated_lookahead_gain_m"] = raw[
                "raw_lookahead_gain_m"]
            alternatives.append(alternative)
            bytes_per_send.append(raw["raw_bytes"])
        summary = method_metrics(alternatives, np.ones(len(test), bool),
                                 bytes_per_send=bytes_per_send)
        summary["mean_regret"] = None
        summary["mean_cost_gap_to_compact_oracle"] = float(np.mean([
            raw["raw_send"]["cost"]-
            min(row["hold"]["cost"], row["send"]["cost"])
            for row, raw in zip(test, (by_key[key(row)] for row in test))]))
        methods["raw_broadcast"] = summary
    novelty_val = np.asarray([row["actor_features"][4] for row in validation])
    novelty_test = np.asarray([row["actor_features"][4] for row in test])
    thresholds = np.unique(np.r_[-np.inf, novelty_val, np.inf])
    threshold = min(thresholds, key=lambda t: regret(
        [row["decision_value"] for row in validation], novelty_val > t))
    methods["novelty_threshold"] = method_metrics(test,
                                                  novelty_test > threshold)
    ridge = RidgeMessageValue.fit([
        {"actor_features": row["actor_features"],
         "decision_value": row["decision_value"]} for row in train])
    ridge_scores = np.asarray([ridge.predict(row["actor_features"])
                               for row in test])
    methods["sender_ridge"] = method_metrics(test, ridge_scores > 0,
                                              ridge_scores)
    for name, feature in (("recipient_ridge", "recipient_features"),
                          ("response_ridge", "response_features")):
        fitted_ridge = RidgeMessageValue.fit([
            {"actor_features": row[feature],
             "decision_value": row["decision_value"]} for row in train])
        scores = np.asarray([fitted_ridge.predict(row[feature])
                             for row in test])
        methods[name] = method_metrics(test, scores > 0, scores)
    fitting = {}
    for name, feature, sign_weight in (
        ("sender_mlp_value", "sender_padded_features", 0.),
        ("recipient_mlp_basic", "recipient_padded_features", 0.),
        ("recipient_mlp_value", "response_features", 0.),
        ("recipient_mlp_value_sign", "response_features", .003)):
        predictions, fits = [], []
        for seed in (0, 1, 2):
            fitted = fit_mlp(train, validation, feature, seed, sign_weight)
            predictions.append(predict(fitted, test, feature))
            fits.append({"seed": seed, "selected_epoch": fitted["selected_epoch"],
                         "validation_regret": fitted["validation_regret"]})
            torch.save({"state_dict": fitted["model"].state_dict(),
                        "mean": fitted["mean"], "std": fitted["std"],
                        "feature_key": feature},
                       directory / f"{args.tag}_{name}_seed{seed}.pt")
        scores = np.mean(predictions, axis=0)
        methods[name] = method_metrics(test, scores > 0, scores)
        fitting[name] = fits
    predictions, components, fits = [], [], []
    for seed in (0, 1, 2):
        fitted = fit_structured(train, validation, seed)
        scores, parts = predict_structured(fitted, test)
        predictions.append(scores)
        components.append(parts)
        fits.append({"seed": seed, "selected_epoch": fitted["selected_epoch"],
                     "validation_regret": fitted["validation_regret"]})
        torch.save({"state_dict": fitted["model"].state_dict(),
                    "mean": fitted["mean"], "std": fitted["std"]},
                   directory / f"{args.tag}_structured_seed{seed}.pt")
    scores = np.mean(predictions, axis=0)
    methods["structured_value"] = method_metrics(test, scores > 0, scores)
    fitting["structured_value"] = fits
    checks = {name: paired_seed_bootstrap(test, methods, name,
                                         "structured_value")
              for name in ("never_send", "sender_ridge",
                           "recipient_ridge", "response_ridge",
                           "recipient_mlp_value")}
    strata = {}
    for risk in ("safe", "risky"):
        for condition in ("natural", "private_direct_overlap",
                          "acked_duplicate"):
            for age in ((0, 8, 16, 32) if condition == "natural" else (None,)):
                indices = [i for i, row in enumerate(test)
                           if row["risk"] == risk and
                           row["receiver_condition"] == condition and
                           (age is None or row["age_steps"] == age)]
                if not indices:
                    continue
                subset = [test[i] for i in indices]
                key = f"{risk}/{condition}/{age if age is not None else 'all'}"
                strata[key] = {"n": len(indices), "positive_send": sum(
                    row["decision_value"] > 0 for row in subset),
                    "mean_value": float(np.mean([
                        row["decision_value"] for row in subset])),
                    "regret": {name: regret(
                        [row["decision_value"] for row in subset],
                        np.asarray(methods[name]["selected_send"])[indices])
                        for name in ("never_send", "sender_ridge",
                                     "recipient_ridge", "response_ridge",
                                     "structured_value")}}
    report = {
        "scope": "prospective frozen static look-ahead A4; no positive selection",
        "oracle_scope": "per-realized-world paired hindsight; not a deployable decentralized optimum",
        "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        "seed_counts": {name: manifest["splits"][name]["seed_count"]
                        for name in splits},
        "row_counts": {name: len(rows) for name, rows in splits.items()},
        "positive_counts": {name: sum(row["decision_value"] > 0 for row in rows)
                            for name, rows in splits.items()},
        "novelty_threshold": float(threshold), "fitting": fitting,
        "methods": methods, "paired_seed_bootstrap": checks,
        "predeclared_strata": strata,
        "structured_components": {
            "names": ["avoided_duplicate", "new_region", "delay_reconnect",
                      "exact_radio"],
            "predictions": np.mean(components, axis=0).tolist(),
            "identifiability": "inductive heads; not identified causal effects"},
        "test_rows": [{"seed": row["seed"], "risk": row["risk"],
                       "age_steps": row["age_steps"],
                       "receiver_condition": row["receiver_condition"],
                       "value": row["decision_value"],
                       "delta_tau": row["delta_tau"],
                       "duplicate_avoided": row["duplicate_sensing_avoided"],
                       "new_region_gained": row["new_region_sensing_gained"],
                       "disjoint_gain": row["delta_disjoint_support"]}
                      for row in test]}
    path = directory / f"learning_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    for name, metrics in methods.items():
        print(name, "regret", metrics["mean_regret"],
              "recall", metrics["useful_send_recall"],
              "false", metrics["false_send_rate"])


if __name__ == "__main__":
    main()
