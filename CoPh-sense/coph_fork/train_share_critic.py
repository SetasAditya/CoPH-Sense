#!/usr/bin/env python3
"""A4 exact-label learning and matched post-observation message baselines."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.critic import SUBSETS, SetCostCritic, parameter_count  # noqa: E402
from coph_fork.oracle import ExactOracleConfig, VARIABLES  # noqa: E402
from coph_fork.share_oracle import ExactShareOracle, ShareState  # noqa: E402


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-states", type=int, default=600)
    parser.add_argument("--validation-states", type=int, default=150)
    parser.add_argument("--test-states", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "a4")
    return parser.parse_args()


def build_features(config, prior, share_state):
    acquired = dict(share_state.acquired)
    acknowledged = dict(share_state.acknowledged)
    p_g = sum(prob for world, prob in prior.items() if world[0])
    p_t = sum(prob for world, prob in prior.items() if world[1])
    state = [
        p_g, p_t, config.packet_cost, share_state.timely_probability,
        float(share_state.deadline_open), config.top_safe_cost,
        config.bottom_safe_cost, config.unsafe_route_cost,
        float(config.decision_mode == "expected_cost"),
    ]
    for name in VARIABLES:
        state.extend((float(name in acquired), float(acquired.get(name, False))))
    for name in VARIABLES:
        state.extend((float(name in acknowledged), float(acknowledged.get(name, False))))
    candidates = []
    for name in VARIABLES:
        candidates.append([
            float(name.startswith("top")), float(name.startswith("bottom")),
            float(name.endswith("geometry")), float(name.endswith("traction")),
            float(acquired.get(name, False)), float(name in acknowledged),
        config.packet_cost, share_state.timely_probability,
        ])
    return np.asarray(state, np.float32), np.asarray(candidates, np.float32)


def build_features_v2(config, prior, share_state):
    """Sufficient finite-fork input: full public prior and receiver decision context."""
    state, candidates = build_features(config, prior, share_state)
    oracle = ExactShareOracle(share_state, config=config, prior=prior)
    receiver_worlds = oracle._consistent_mass(dict(share_state.acknowledged))
    mass = sum(receiver_worlds.values())
    route_costs = [
        sum(p * oracle.route_oracle.execution_cost(route, world)
            for world, p in receiver_worlds.items()) / mass
        for route in ("bottom", "top")
    ]
    no_message_route = oracle._receiver_decision(())
    joint = [
        prior.get((geometry, traction, True, True), 0.0)
        for geometry in (False, True) for traction in (False, True)
    ]
    ack_attempt = config.communication_attempt_cost + config.byte_cost * (
        config.header_bytes + config.ack_payload_bytes
    )
    effective_packet_cost = config.packet_cost + share_state.timely_probability * ack_attempt
    extra = np.asarray(
        joint + [float(no_message_route == "top"), route_costs[1] - route_costs[0],
                 effective_packet_cost], dtype=np.float32,
    )
    state = np.concatenate((state, extra))
    candidate_extra = np.full((len(VARIABLES), 1), effective_packet_cost, np.float32)
    return state, np.concatenate((candidates, candidate_extra), axis=1)


def encode_context(config, prior, share_state, feature_builder=build_features):
    available = [name for name, _ in share_state.acquired]
    acknowledged = share_state.acknowledged
    oracle = ExactShareOracle(share_state, config=config, prior=prior)
    optimum, rows = oracle.solve()
    labels = np.zeros((len(SUBSETS), 3), np.float32)
    feasible = np.zeros(len(SUBSETS), np.bool_)
    for row in rows:
        index = SUBSETS.index(tuple(name for name in VARIABLES if name in row.message_set))
        labels[index] = (
            row.expected_execution_cost,
            row.expected_communication_cost,
            0.0,
        )
        feasible[index] = True
    state, candidates = feature_builder(config, prior, share_state)
    optimal_index = SUBSETS.index(tuple(name for name in VARIABLES if name in optimum.message_set))
    category = "stale" if not share_state.deadline_open or share_state.timely_probability < 0.03 else (
        "duplicate" if all(name in dict(acknowledged) for name in available) else (
            "complementary" if len(optimum.message_set) >= 2 else (
                "novel_useful" if optimum.message_set else "irrelevant_or_costly"
            )
        )
    )
    return state, candidates, labels, feasible, optimal_index, category


def sample_context(rng, feature_builder=build_features):
    p_g = float(rng.uniform(0.12, 0.98))
    p_t = float(rng.uniform(0.12, 0.98))
    world = (bool(rng.rand() < p_g), bool(rng.rand() < p_t), True, True)
    prior = {
        (geometry, traction, True, True):
            (p_g if geometry else 1 - p_g) * (p_t if traction else 1 - p_t)
        for geometry in (False, True) for traction in (False, True)
    }
    available = [name for name in VARIABLES if rng.rand() < 0.7]
    if not available:
        available = [VARIABLES[int(rng.randint(0, len(VARIABLES)))]]
    acquired = tuple((name, world[VARIABLES.index(name)]) for name in available)
    acknowledged = tuple(
        (name, world[VARIABLES.index(name)])
        for name in VARIABLES if rng.rand() < 0.25
    )
    config = ExactOracleConfig(
        communication_attempt_cost=float(rng.uniform(0.002, 0.12)),
        byte_cost=float(rng.uniform(0.0, 0.0006)),
        timely_delivery_probability=1.0,  # state-specific channel below
        top_safe_cost=float(rng.uniform(0.6, 1.6)),
        bottom_safe_cost=float(rng.uniform(2.0, 4.4)),
        unsafe_route_cost=float(rng.uniform(4.8, 8.0)),
        decision_mode="expected_cost" if rng.rand() < 0.25 else "certified",
    )
    share_state = ShareState(
        acquired=acquired,
        acknowledged=acknowledged,
        timely_probability=float(rng.uniform(0.0, 1.0)),
        deadline_open=bool(rng.rand() >= 0.12),
    )
    return encode_context(config, prior, share_state, feature_builder)


def generate(count, seed, feature_builder=build_features):
    rng = np.random.RandomState(seed)
    categories = (
        "complementary", "duplicate", "irrelevant_or_costly",
        "novel_useful", "stale",
    )
    quota = {name: count // len(categories) for name in categories}
    for name in categories[:count % len(categories)]:
        quota[name] += 1
    samples = []
    observed = {name: 0 for name in categories}
    attempts = 0
    while len(samples) < count:
        sample = sample_context(rng, feature_builder)
        category = sample[5]
        if observed[category] < quota[category]:
            samples.append(sample)
            observed[category] += 1
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not fill stratified post-observation dataset")
    return (
        torch.tensor(np.stack([row[0] for row in samples])),
        torch.tensor(np.stack([row[1] for row in samples])),
        torch.tensor(np.stack([row[2] for row in samples])),
        torch.tensor(np.stack([row[3] for row in samples]), dtype=torch.bool),
        np.asarray([row[4] for row in samples]),
        np.asarray([row[5] for row in samples]),
    )


def sample_warning_context(rng, feature_builder=build_features_v2):
    """Construct a bad reading that averts a likely unsafe top-route choice."""
    for _ in range(1000):
        p_g = float(rng.uniform(0.75, 0.98))
        p_t = float(rng.uniform(0.75, 0.98))
        prior = {
            (geometry, traction, True, True):
                (p_g if geometry else 1 - p_g) * (p_t if traction else 1 - p_t)
            for geometry in (False, True) for traction in (False, True)
        }
        config = ExactOracleConfig(
            communication_attempt_cost=float(rng.uniform(0.002, 0.06)),
            byte_cost=float(rng.uniform(0.0, 0.0006)),
            top_safe_cost=float(rng.uniform(0.6, 1.2)),
            bottom_safe_cost=float(rng.uniform(2.4, 4.0)),
            unsafe_route_cost=float(rng.uniform(4.8, 7.0)),
            decision_mode="expected_cost",
        )
        acknowledged = (("top_traction", True),) if rng.rand() < 0.3 else ()
        state = ShareState(
            acquired=(("top_geometry", False),),
            acknowledged=acknowledged,
            timely_probability=float(rng.uniform(0.7, 1.0)),
        )
        oracle = ExactShareOracle(state, config=config, prior=prior)
        optimum, _ = oracle.solve()
        if (
            oracle._receiver_decision(()) == "top"
            and optimum.message_set == ("top_geometry",)
            and oracle.evaluate(()).total_cost - optimum.total_cost > 0.25
        ):
            sample = encode_context(config, prior, state, feature_builder)
            return sample[:5] + ("bad_warning",)
    raise RuntimeError("could not construct useful bad-warning state")


def generate_warning_curriculum(count, warning_count, seed, feature_builder=build_features_v2):
    if not 0 <= warning_count <= count:
        raise ValueError("warning_count must be within dataset size")
    base = generate(count - warning_count, seed, feature_builder)
    rng = np.random.RandomState(seed + 100000)
    warning = [sample_warning_context(rng, feature_builder) for _ in range(warning_count)]
    if not warning:
        return base
    return (
        torch.cat((base[0], torch.tensor(np.stack([row[0] for row in warning])))),
        torch.cat((base[1], torch.tensor(np.stack([row[1] for row in warning])))),
        torch.cat((base[2], torch.tensor(np.stack([row[2] for row in warning])))),
        torch.cat((base[3], torch.tensor(np.stack([row[3] for row in warning]), dtype=torch.bool))),
        np.concatenate((base[4], np.asarray([row[4] for row in warning]))),
        np.concatenate((base[5], np.asarray([row[5] for row in warning]))),
    )


def model_metrics(model, data):
    state, candidates, labels, feasible, optimal, categories = data
    model.eval()
    with torch.no_grad():
        prediction = model(state, candidates)
        predicted_cost = prediction.sum(-1).masked_fill(~feasible, float("inf"))
        true_cost = labels.sum(-1)
        selected = predicted_cost.argmin(-1)
        indices = torch.arange(len(state))
        regret = true_cost[indices, selected] - true_cost[indices, torch.tensor(optimal)]
        no_send = true_cost[:, 0]
        sent = selected != 0
        wasteful = sent & (true_cost[indices, selected] >= no_send - 1e-8)
        missed = (selected == 0) & (true_cost[indices, torch.tensor(optimal)] < no_send - 1e-8)
        advantage = (no_send - true_cost[indices, torch.tensor(optimal)]).clamp_min(0)
        critical = advantage > 0.5
        critical_recalled = torch.tensor([
            set(SUBSETS[int(optimal[j])]).issubset(SUBSETS[int(selected[j])])
            for j in range(len(state))
        ], dtype=torch.float32)
        bytes_sent = np.asarray([
            len(SUBSETS[int(i)]) * (48 + 40 * float(state[j, 3]))
            for j, i in enumerate(selected)
        ])
        by_category = {}
        for category in sorted(set(categories)):
            mask = categories == category
            by_category[category] = {
                "count": int(mask.sum()),
                "mean_regret": float(regret.numpy()[mask].mean()),
                "top1": float(np.mean(selected.numpy()[mask] == optimal[mask])),
            }
        return {
            "mean_regret": float(regret.mean()),
            "p90_regret": float(torch.quantile(regret, 0.9)),
            "top1_accuracy": float(np.mean(selected.numpy() == optimal)),
            "wasteful_send_rate": float(wasteful.float().mean()),
            "missed_useful_rate": float(missed.float().mean()),
            "missed_useful_value": float((advantage * missed.float()).mean()),
            "critical_recall_at_0_5": float(critical_recalled[critical].mean()) if critical.any() else None,
            "critical_count": int(critical.sum()),
            "mean_bytes": float(bytes_sent.mean()),
            "value_rmse": float(torch.mean((prediction[feasible].sum(-1) - true_cost[feasible]) ** 2).sqrt()),
            "by_category": by_category,
        }


def baseline_metrics(data):
    state, candidates, labels, feasible, optimal, categories = data
    true_cost = labels.sum(-1).numpy()
    feasible = feasible.numpy()
    acquired = state[:, 9:17:2].numpy() > 0.5
    acknowledged = state[:, 17:25:2].numpy() > 0.5
    selection = {}
    selection["never_send"] = np.zeros(len(state), dtype=int)
    selection["always_send"] = np.asarray([
        SUBSETS.index(tuple(name for i, name in enumerate(VARIABLES) if acquired[j, i]))
        if feasible[j, SUBSETS.index(tuple(name for i, name in enumerate(VARIABLES) if acquired[j, i]))] else 0
        for j in range(len(state))
    ])
    selection["send_all_novel"] = np.asarray([
        SUBSETS.index(tuple(name for i, name in enumerate(VARIABLES) if acquired[j, i] and not acknowledged[j, i]))
        if feasible[j, SUBSETS.index(tuple(name for i, name in enumerate(VARIABLES) if acquired[j, i] and not acknowledged[j, i]))] else 0
        for j in range(len(state))
    ])
    # This entropy heuristic sends any novel measurement whose public prior is
    # uncertain, regardless of whether it changes the route decision.
    prior = state[:, :2].numpy()
    selection["uncertainty_threshold"] = np.asarray([
        SUBSETS.index(tuple(
            name for i, name in enumerate(VARIABLES)
            if acquired[j, i] and not acknowledged[j, i]
            and 0.2 < (prior[j, i] if i < 2 else 1.0) < 0.8
        )) if feasible[j, SUBSETS.index(tuple(
            name for i, name in enumerate(VARIABLES)
            if acquired[j, i] and not acknowledged[j, i]
            and 0.2 < (prior[j, i] if i < 2 else 1.0) < 0.8
        ))] else 0
        for j in range(len(state))
    ])
    result = {}
    for name, indices in selection.items():
        index = np.arange(len(state))
        regret = true_cost[index, indices] - true_cost[index, optimal]
        sent = indices != 0
        wasteful = sent & (true_cost[index, indices] >= true_cost[:, 0] - 1e-8)
        missed = (indices == 0) & (true_cost[index, optimal] < true_cost[:, 0] - 1e-8)
        advantage = np.maximum(0, true_cost[:, 0] - true_cost[index, optimal])
        critical = advantage > 0.5
        critical_recalled = np.asarray([
            set(SUBSETS[int(optimal[j])]).issubset(SUBSETS[int(indices[j])])
            for j in range(len(state))
        ])
        result[name] = {
            "mean_regret": float(regret.mean()),
            "top1_accuracy": float(np.mean(indices == optimal)),
            "wasteful_send_rate": float(wasteful.mean()),
            "missed_useful_rate": float(missed.mean()),
            "missed_useful_value": float(np.mean(advantage * missed)),
            "critical_recall_at_0_5": float(critical_recalled[critical].mean()) if critical.any() else None,
            "critical_count": int(critical.sum()),
            "mean_bytes": float(np.mean([
                len(SUBSETS[int(i)]) * (48 + 40 * float(state[j, 3]))
                for j, i in enumerate(indices)
            ])),
        }
    return result


def train_one(kind, train, validation, setting, critical_beta=0.0):
    torch.manual_seed(setting.seed)
    model = SetCostCritic(train[0].shape[-1], train[1].shape[-1], kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(setting.epochs):
        model.train()
        order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(setting.seed + epoch))
        for indices in order.split(setting.batch_size):
            state, candidates, targets, valid = (part[indices] for part in train[:4])
            prediction = model(state, candidates)
            predicted_q = prediction.sum(-1)
            true_q = targets.sum(-1)
            advantage = (
                true_q[:, 0] - true_q.masked_fill(~valid, float("inf")).min(-1).values
            ).clamp_min(0)
            weights = 1 + critical_beta * advantage
            weighted_valid = valid.float() * weights[:, None]
            component_error = (prediction - targets).square().mean(-1)
            value_error = (predicted_q - true_q).square()
            component_loss = (component_error * weighted_valid).sum() / weighted_valid.sum()
            value_loss = (value_error * weighted_valid).sum() / weighted_valid.sum()
            gap = true_q[:, :, None] - true_q[:, None, :]
            predicted_gap = predicted_q[:, :, None] - predicted_q[:, None, :]
            separated = valid[:, :, None] & valid[:, None, :] & (gap.abs() > 0.01)
            rank = torch.relu(0.01 - torch.sign(gap) * predicted_gap)
            rank_loss = rank[separated].mean() if separated.any() else value_loss * 0
            loss = component_loss + value_loss + 0.1 * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = model_metrics(model, validation)["mean_regret"]
        if score < best_score - 1e-8:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
    model.load_state_dict(best_state)
    return model, best_epoch


def main():
    setting = args()
    torch.set_num_threads(2)
    setting.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train = generate(setting.train_states, setting.seed)
    validation = generate(setting.validation_states, setting.seed + 1)
    test = generate(setting.test_states, setting.seed + 2)
    result = {
        "setup": {
            "seed": setting.seed,
            "train_states": setting.train_states,
            "validation_states": setting.validation_states,
            "test_states": setting.test_states,
            "data_seconds": time.time() - start,
            "selection_metric": "validation mean exact sharing regret",
            "split_unit": "independent realized post-acquisition state",
            "stratification": "equal quota across five diagnostic categories",
        },
        "baselines": baseline_metrics(test),
        "models": {},
    }
    for kind in ("additive", "deepsets", "pairwise"):
        model, epoch = train_one(kind, train, validation, setting)
        torch.save(model.state_dict(), setting.output_dir / (kind + ".pth"))
        result["models"][kind] = {
            "parameters": parameter_count(model),
            "selected_epoch": epoch,
            "validation": model_metrics(model, validation),
            "test": model_metrics(model, test),
        }
        print(kind, json.dumps(result["models"][kind]["test"]), flush=True)
    path = setting.output_dir / "report.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()
