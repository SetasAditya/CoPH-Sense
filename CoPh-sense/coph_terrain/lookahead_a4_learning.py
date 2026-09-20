"""Development-only A4 learning on physically executed receiver responses.

The original 80 inspected states are training data. The next prospective block
is split by whole seed into validation and untouched test groups before model
selection. No simulator truth enters actor inputs.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .algorithm import acquire_to_evidence, post_reading_branch
from .lookahead import LookaheadEnv, LookaheadSpec
from .lookahead_dev_audit import VIEW
from .lookahead_policy import RidgeMessageValue
from .lookahead_recipient_a4 import (advance_without_message,
                                      response_features)
from .physical_audit import AuditChoice
from .lookahead_a4_downstream import acquisition_continuation


OUT = Path(__file__).resolve().parent / "results" / "lookahead_recipient_a4"


def augment_response(rows, raw_broadcast=False):
    """Replay source histories once per seed/risk and verify stored inputs."""
    groups = {}
    for row in rows:
        key = (row["seed"], row["risk"], row["scout_start_x"])
        groups.setdefault(key, []).append(row)
    for (seed, risk, scout_x), members in groups.items():
        surface, traction = ((.04, .90) if risk == "safe" else
                             (.95, .20))
        spec = LookaheadSpec(seed, surface=(surface, .06),
                             traction=(traction, .90),
                             scout_start=(scout_x, 27))
        acquired = acquire_to_evidence(LookaheadEnv(spec), VIEW)
        if acquired is None:
            raise RuntimeError("source acquisition failed on replay")
        base = acquired.env
        packet_id = next(iter(base.innovation_packets))
        private_state = None
        for row in members:
            if row["receiver_condition"] == "natural":
                state = advance_without_message(base, row["age_steps"])
            elif row["receiver_condition"] == "private_direct_overlap":
                if private_state is None:
                    private_state = acquire_to_evidence(
                        base, AuditChoice("carrier", "geometry",
                                          (15, 27), (18, 27)))
                if private_state is None:
                    raise RuntimeError("private overlap replay failed")
                state = private_state.env
            elif row["receiver_condition"] == "acked_duplicate":
                _, state = post_reading_branch(
                    base, packet_id, True, continue_mission=False)
                state = advance_without_message(
                    state, max(1, state.config.communication_delay_steps)+1)
            else:
                raise ValueError(row["receiver_condition"])
            if state.step_index != row["step"]:
                raise AssertionError("source step changed during replay")
            from .lookahead_policy import message_features
            np.testing.assert_allclose(message_features(state, packet_id),
                                       row["actor_features"], atol=1e-9)
            row["response_features"] = response_features(
                state, packet_id).tolist()
            row["sender_padded_features"] = (
                row["actor_features"] +
                [0.] * (len(row["response_features"]) -
                         len(row["actor_features"])))
            row["recipient_padded_features"] = (
                row["recipient_features"] +
                [0.] * (len(row["response_features"]) -
                         len(row["recipient_features"])))
            if raw_broadcast:
                _, delivered = post_reading_branch(
                    state, packet_id, True, continue_mission=False)
                lookahead = delivered.observations()["carrier"]["lookahead"]
                row["communicated_lookahead_gain_m"] = max(
                    0., lookahead["effective_lookahead"] -
                    lookahead["local_lookahead"])
                raw_id = state.innovation_packets[packet_id].provenance
                row["raw_send"] = acquisition_continuation(
                    state, packet_id, True, evidence_id=raw_id)
                _, raw_delivered = post_reading_branch(
                    state, raw_id, True, continue_mission=False)
                raw_lookahead = raw_delivered.observations()["carrier"][
                    "lookahead"]
                row["raw_lookahead_gain_m"] = max(
                    0., raw_lookahead["effective_lookahead"] -
                    raw_lookahead["local_lookahead"])
                raw_evidence = state.owned["scout"][raw_id]
                row["raw_bytes"] = (state.config.header_bytes +
                                    16 + 4*len(raw_evidence.cells) +
                                    state.config.ack_bytes)
            # This metric is recomputed from the saved executed branch, using
            # the whole summarized region as already explored by the scout.
            for branch in ("hold", "send"):
                b = row[branch]
                b["new_disjoint_support"] = (
                    b["new_carrier_support"] if b["choice_region"] == 1 else 0)
            row["delta_disjoint_support"] = (
                row["send"]["new_disjoint_support"] -
                row["hold"]["new_disjoint_support"])


class ValueNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.SiLU(),
                                 nn.Linear(32, 32), nn.SiLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def regret(values, decisions):
    values = np.asarray(values)
    decisions = np.asarray(decisions, dtype=bool)
    return float(np.mean(np.where(decisions,
                                  np.maximum(0., -values),
                                  np.maximum(0., values))))


def fit_mlp(train, val, feature_key, seed, sign_weight=0., epochs=250):
    torch.manual_seed(seed)
    x = np.asarray([row[feature_key] for row in train], np.float32)
    v = np.asarray([row[feature_key] for row in val], np.float32)
    y = np.asarray([row["decision_value"] for row in train], np.float32)
    vy = np.asarray([row["decision_value"] for row in val], np.float32)
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-3)
    train_x = torch.from_numpy((x-mean)/std)
    val_x = torch.from_numpy((v-mean)/std)
    train_y = torch.from_numpy(y)
    model = ValueNet(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005,
                                  weight_decay=.01)
    positive = max(1., float((y <= 0).sum()) /
                   max(1, (y > 0).sum()))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive))
    best = None
    for epoch in range(epochs):
        model.train()
        predicted = model(train_x)
        loss = ((predicted-train_y)**2).mean()
        if sign_weight:
            loss = loss + sign_weight*bce(
                predicted/.03, (train_y > 0).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x).numpy()
        score = regret(vy, val_prediction > 0)
        candidate = (score, epoch+1, copy.deepcopy(model.state_dict()))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    model.load_state_dict(best[2])
    return {"model": model, "mean": mean, "std": std,
            "selected_epoch": best[1], "validation_regret": best[0],
            "positive_weight": positive, "sign_weight": sign_weight}


def predict(fitted, rows, feature_key):
    x = np.asarray([row[feature_key] for row in rows], np.float32)
    tensor = torch.from_numpy((x-fitted["mean"])/fitted["std"])
    fitted["model"].eval()
    with torch.no_grad():
        return fitted["model"](tensor).numpy()


def method_metrics(rows, decisions, scores=None, bytes_per_send=120):
    decisions = np.asarray(decisions, dtype=bool)
    values = np.asarray([r["decision_value"] for r in rows])
    positive = values > 0
    chosen = [row["send"] if send else row["hold"]
              for row, send in zip(rows, decisions)]
    result = {"n": len(rows), "selected_send": decisions.tolist(),
              "send_count": int(decisions.sum()),
              "positive_count": int(positive.sum()),
              "mean_regret": regret(values, decisions),
              "useful_send_recall": (float(decisions[positive].mean())
                                     if positive.any() else None),
              "false_send_rate": (float(decisions[~positive].mean())
                                  if (~positive).any() else None),
              "send_precision": (float(positive[decisions].mean())
                                 if decisions.any() else None),
              "mean_team_cost": float(np.mean([b["cost"] for b in chosen])),
              "success_rate": float(np.mean([b["success"] for b in chosen])),
              "mean_radio_cost": float(np.mean([b["radio"] for b in chosen])),
              "mean_bytes": float(np.mean(
                  np.asarray(bytes_per_send)*decisions)),
              "net_value_per_transmitted_byte": (
                  float(np.sum(values[decisions]) /
                        np.sum(np.asarray(bytes_per_send)*decisions))
                  if decisions.any() else None),
              "mean_duplicate_region_scans": float(np.mean([
                  b["duplicate_region_scan"] and b["acquisition_executed"]
                  for b in chosen])),
              "mean_new_region_scans": float(np.mean([
                  b["choice_region"] == 1 and b["acquisition_executed"]
                  for b in chosen])),
              "mean_disjoint_support": float(np.mean([
                  b["new_disjoint_support"] for b in chosen])),
              "mean_communicated_lookahead_gain_m": float(np.mean([
                  (row.get("communicated_lookahead_gain_m", 0.) if send else 0.)
                  for row, send in zip(rows, decisions)])),
              "mean_risk_exposure": float(np.mean([
                  b["risk_exposure"] for b in chosen]))}
    if scores is not None:
        result["value_mse"] = float(np.mean((np.asarray(scores)-values)**2))
        result["predicted_values"] = np.asarray(scores).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path,
                        default=OUT / "downstream_dev12.json")
    parser.add_argument("--prospective", type=Path,
                        default=OUT / "downstream_prospective12.json")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    path = OUT / f"learning_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    development = json.loads(args.development.read_text())["rows"]
    prospective = json.loads(args.prospective.read_text())["rows"]
    train = [copy.deepcopy(row) for row in development]
    val = [copy.deepcopy(row) for row in prospective if row["seed"] < 981018]
    test = [copy.deepcopy(row) for row in prospective if row["seed"] >= 981018]
    if (len({row["seed"] for row in train}) != 12 or
            len({row["seed"] for row in val}) != 6 or
            len({row["seed"] for row in test}) != 6):
        raise ValueError("expected disjoint 12/6/6-seed split")
    augment_response(train)
    augment_response(val)
    augment_response(test, raw_broadcast=True)
    values = np.asarray([row["decision_value"] for row in test])
    methods = {
        "never_send": method_metrics(test, np.zeros(len(test), bool)),
        "always_compact_send": method_metrics(test, np.ones(len(test), bool)),
        "restricted_oracle": method_metrics(test, values > 0),
    }
    raw_rows = []
    for row in test:
        alternative = dict(row)
        alternative["send"] = row["raw_send"]
        alternative["decision_value"] = (
            row["hold"]["cost"]-row["raw_send"]["cost"])
        alternative["communicated_lookahead_gain_m"] = row[
            "raw_lookahead_gain_m"]
        raw_rows.append(alternative)
    raw_summary = method_metrics(
        raw_rows, np.ones(len(raw_rows), bool),
        bytes_per_send=[row["raw_bytes"] for row in test])
    raw_summary["mean_regret"] = None
    raw_summary["mean_cost_gap_to_compact_oracle"] = float(np.mean([
        row["raw_send"]["cost"] -
        min(row["hold"]["cost"], row["send"]["cost"])
        for row in test]))
    methods["raw_broadcast"] = raw_summary
    novelty_val = np.asarray([row["actor_features"][4] for row in val])
    novelty_test = np.asarray([row["actor_features"][4] for row in test])
    thresholds = np.unique(np.r_[-np.inf, novelty_val, np.inf])
    best_threshold = min(thresholds, key=lambda t: regret(
        [row["decision_value"] for row in val], novelty_val > t))
    methods["novelty_threshold"] = method_metrics(
        test, novelty_test > best_threshold)
    ridge = RidgeMessageValue.fit([
        {"actor_features": row["actor_features"],
         "decision_value": row["decision_value"]} for row in train])
    ridge_scores = [ridge.predict(row["actor_features"]) for row in test]
    methods["sender_ridge"] = method_metrics(test,
                                              np.asarray(ridge_scores) > 0,
                                              ridge_scores)
    model_specs = (("sender_mlp_value", "sender_padded_features", 0.),
                   ("recipient_mlp_basic", "recipient_padded_features", 0.),
                   ("recipient_mlp_value", "response_features", 0.),
                   ("recipient_mlp_value_sign", "response_features", .003))
    fitting = {}
    for name, feature, sign_weight in model_specs:
        predictions = []
        fits = []
        for seed in (0, 1, 2):
            fitted = fit_mlp(train, val, feature, seed, sign_weight)
            scores = predict(fitted, test, feature)
            predictions.append(scores)
            fits.append({"seed": seed,
                         "validation_regret": fitted["validation_regret"],
                         "selected_epoch": fitted["selected_epoch"]})
            torch.save({"state_dict": fitted["model"].state_dict(),
                        "mean": fitted["mean"], "std": fitted["std"],
                        "feature_key": feature},
                       OUT / f"{args.tag}_{name}_seed{seed}.pt")
        scores = np.mean(predictions, axis=0)
        methods[name] = method_metrics(test, scores > 0, scores)
        fitting[name] = fits
    report = {"scope": "development-trained, prospective seed-disjoint A4",
              "train_seeds": sorted({row["seed"] for row in train}),
              "validation_seeds": sorted({row["seed"] for row in val}),
              "test_seeds": sorted({row["seed"] for row in test}),
              "train_count": len(train), "validation_count": len(val),
              "test_count": len(test),
              "novelty_threshold": float(best_threshold),
              "fitting": fitting, "methods": methods,
              "test_rows": [{"seed": row["seed"], "risk": row["risk"],
                             "age_steps": row["age_steps"],
                             "receiver_condition": row["receiver_condition"],
                             "value": row["decision_value"],
                             "delta_tau": row["delta_tau"],
                             "duplicate_avoided": row["duplicate_sensing_avoided"],
                             "new_region_gained": row["new_region_sensing_gained"],
                             "disjoint_gain": row["delta_disjoint_support"]}
                            for row in test]}
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)
    for name, metrics in methods.items():
        score = metrics["mean_regret"]
        print(name, "regret", None if score is None else round(score, 6),
              "recall", metrics["useful_send_recall"],
              "false", metrics["false_send_rate"])


if __name__ == "__main__":
    main()
