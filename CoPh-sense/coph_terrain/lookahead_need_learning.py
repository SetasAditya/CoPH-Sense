"""Single frozen A4 learning comparison for the admitted NEED task."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


SEED = 2027
RESPONSE_CHARGED_BYTES = 16 + 48 + 8
NEED_CHARGED_BYTES = 16 + 32 + 8


class ValueMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load(path):
    return json.loads(path.read_text())["rows"]


def matrix(rows, with_need):
    values = [row["sender_features"] +
              (row["need_features"] if with_need else []) for row in rows]
    return np.asarray(values, dtype=np.float32)


def fit(train, validation, with_need):
    torch.manual_seed(SEED + int(with_need))
    np.random.seed(SEED + int(with_need))
    x = matrix(train, with_need)
    y = np.asarray([row["decision_value"] for row in train], np.float32)
    xv = matrix(validation, with_need)
    yv = np.asarray([row["decision_value"] for row in validation], np.float32)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    xt = torch.tensor((x-mean)/scale)
    yt = torch.tensor(y)
    xvt = torch.tensor((xv-mean)/scale)
    model = ValueMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3,
                                 weight_decay=1e-4)
    best = None
    best_loss = float("inf")
    for step in range(1001):
        optimizer.zero_grad()
        loss = ((model(xt)-yt)**2).mean()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                value = float(((model(xvt)-torch.tensor(yv))**2).mean())
            if value < best_loss:
                best_loss = value
                best = deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, mean, scale, best_loss


def predict(fitted, rows, with_need):
    model, mean, scale, _ = fitted
    with torch.no_grad():
        return model(torch.tensor((matrix(rows, with_need)-mean)/scale)).numpy()


def request_match(row):
    need = row["need"]
    return bool(row["need_features"][-1] and need["uncertainty"] > 0 and
                need["task_relevance"] > 0 and row["need_features"][4] > 0 and
                not row["sender_features"][9])


def bayes_lookup(train, rows):
    def key(row):
        # Finite legal information class, excluding seed-specific floating
        # kinematics. Posterior risk, direction, request semantics, age and ACK
        # are all available to the sender.
        return (row["direction"], int(row["sender_features"][0] > .5),
                int(row["need"]["region"]),
                int(row["need"]["uncertainty"] > 0),
                int(row["need"]["task_relevance"] > 0),
                int(row["need_features"][-1]), int(row["need_age"]),
                int(row["sender_features"][9]))
    groups = {}
    for row in train:
        groups.setdefault(key(row), []).append(row["decision_value"])
    fallback = float(np.mean([row["decision_value"] for row in train]))
    return np.asarray([np.mean(groups.get(key(row), [fallback])) for row in rows])


def report_method(name, rows, decisions, predictions=None):
    details = []
    for row, send in zip(rows, decisions):
        chosen = row["send"] if send else row["hold"]
        regret = chosen["cost"]-min(row["hold"]["cost"], row["send"]["cost"])
        details.append({"direction": row["direction"], "context": row["context"],
                        "send": bool(send), "regret": regret,
                        "cost": chosen["cost"], "radio": chosen["radio"],
                        "sensing": chosen["sensing"],
                        "bytes": NEED_CHARGED_BYTES +
                        (RESPONSE_CHARGED_BYTES if send else 0)})
    def aggregate(items):
        useful = [x for x in items if x["context"] == "useful"]
        negative = [x for x in items if x["context"] != "useful"]
        return {"n": len(items),
                "mean_regret": float(np.mean([x["regret"] for x in items])),
                "team_cost": float(np.mean([x["cost"] for x in items])),
                "radio_cost": float(np.mean([x["radio"] for x in items])),
                "sensing_cost": float(np.mean([x["sensing"] for x in items])),
                "charged_bytes": float(np.mean([x["bytes"] for x in items])),
                "send_rate": float(np.mean([x["send"] for x in items])),
                "useful_send_recall": float(np.mean([x["send"] for x in useful])),
                "false_send_rate": float(np.mean([x["send"] for x in negative])),
                "duplicate_scans_avoided": int(sum(
                    x["send"] and x["context"] == "useful" for x in items))}
    result = {"name": name, "overall": aggregate(details),
              "by_direction": {direction: aggregate([
                  x for x in details if x["direction"] == direction])
                  for direction in ("scout_to_carrier", "carrier_to_scout")}}
    if predictions is not None:
        result["value_mse"] = float(np.mean((predictions-np.asarray(
            [row["decision_value"] for row in rows]))**2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    d = args.directory.resolve()
    train = load(d/"admission_train_v2.json")
    validation = load(d/"admission_validation_v2.json")
    test = load(d/"admission_confirmatory_v2.json")
    sender = fit(train, validation, False)
    need = fit(train, validation, True)
    sender_pred = predict(sender, test, False)
    need_pred = predict(need, test, True)
    bayes_pred = bayes_lookup(train, test)
    methods = [
        report_method("never_send", test, np.zeros(len(test), bool)),
        report_method("always_compact", test, np.ones(len(test), bool)),
        report_method("request_match", test,
                      np.asarray([request_match(row) for row in test])),
        report_method("sender_only_a4", test, sender_pred > 0, sender_pred),
        report_method("history_plus_need_a4", test, need_pred > 0, need_pred),
        report_method("bayes_lookup_reference", test, bayes_pred > 0, bayes_pred),
        report_method("paired_hindsight_oracle", test, np.asarray([
            row["decision_value"] > 0 for row in test]))]
    rng = np.random.default_rng(SEED)
    # Reconstruct paired regrets directly; seed bootstrap preserves all
    # contexts belonging to the same physical world seed.
    def regrets(decisions):
        return np.asarray([(row["send"] if send else row["hold"])["cost"] -
                           min(row["hold"]["cost"], row["send"]["cost"])
                           for row, send in zip(test, decisions)])
    need_decisions = need_pred > 0
    heuristic_decisions = np.asarray([request_match(row) for row in test])
    paired = regrets(need_decisions)-regrets(heuristic_decisions)
    seeds = sorted({row["seed"] for row in test})
    by_seed = np.asarray([paired[[row["seed"] == seed for row in test]].mean()
                          for seed in seeds])
    bootstrap = np.asarray([rng.choice(by_seed, len(by_seed), replace=True).mean()
                            for _ in range(10000)])
    paired_comparison = {
        "history_need_minus_request_match_regret": float(paired.mean()),
        "seed_bootstrap_95_interval": [float(np.quantile(bootstrap, .025)),
                                        float(np.quantile(bootstrap, .975))],
        "interpretation": "negative favors history+NEED"}
    report = {"scope": "single frozen NEED-conditioned A4 learning run",
              "random_seed": SEED,
              "training": {"architecture": "2x32 tanh value MLP",
                           "optimizer": "Adam(lr=0.003, weight_decay=0.0001)",
                           "steps": 1000,
                           "selection": "lowest validation value MSE every 10 steps",
                           "sender_validation_mse": sender[3],
                           "need_validation_mse": need[3]},
              "train_rows": len(train), "validation_rows": len(validation),
              "test_rows": len(test), "methods": methods,
              "paired_comparison": paired_comparison}
    (d/"learning_frozen_v1.json").write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"state_dict": sender[0].state_dict(), "mean": sender[1],
                "scale": sender[2]}, d/"sender_only_a4_v1.pt")
    torch.save({"state_dict": need[0].state_dict(), "mean": need[1],
                "scale": need[2]}, d/"history_need_a4_v1.pt")
    print(d/"learning_frozen_v1.json")
    for method in methods:
        print(method["name"], method["overall"])


if __name__ == "__main__":
    main()
