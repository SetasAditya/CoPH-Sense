#!/usr/bin/env python3
"""Train A3 critics against held-out exact A2 information states."""

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

from coph_fork.critic import SUBSETS, SetCostCritic, contrast, parameter_count  # noqa: E402
from coph_fork.oracle import ExactAcquisitionOracle, ExactOracleConfig, default_prior  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-states", type=int, default=320)
    parser.add_argument("--validation-states", type=int, default=80)
    parser.add_argument("--test-states", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "a3")
    return parser.parse_args()


def sample_context(rng, return_context=False):
    p_geometry = float(rng.uniform(0.15, 0.85))
    p_traction = float(rng.uniform(0.15, 0.85))
    config = ExactOracleConfig(
        sensing_cost=float(rng.uniform(0.01, 0.13)),
        communication_attempt_cost=float(rng.uniform(0.002, 0.035)),
        byte_cost=float(rng.uniform(0.00001, 0.00035)),
        timely_delivery_probability=float(rng.uniform(0.40, 1.0)),
        top_safe_cost=float(rng.uniform(0.6, 1.6)),
        bottom_safe_cost=float(rng.uniform(2.2, 4.5)),
        unsafe_route_cost=float(rng.uniform(5.0, 8.0)),
    )
    prior = {
        (geometry, traction, True, True):
            (p_geometry if geometry else 1.0 - p_geometry)
            * (p_traction if traction else 1.0 - p_traction)
        for geometry in (False, True)
        for traction in (False, True)
    }
    encoded = encode_context(config, prior, p_geometry, p_traction)
    return encoded + (config, prior) if return_context else encoded


def encode_context(config, prior, p_geometry, p_traction):
    state, candidates = public_features(config, prior, p_geometry, p_traction)
    optimum, rows, _ = ExactAcquisitionOracle(config=config, prior=prior).solve()
    labels = np.asarray(
        [
            [row.expected_execution_cost, row.sensing_cost, row.expected_communication_cost]
            for row in rows
        ],
        dtype=np.float32,
    )
    if tuple(row.sense_subset for row in rows) != SUBSETS:
        raise RuntimeError("oracle and critic subset ordering disagree")
    return state, candidates, labels, SUBSETS.index(optimum.sense_subset)


def public_features(config, prior, p_geometry, p_traction):
    """A3 actor inputs without regenerating exact A2 training labels."""
    state = np.asarray(
        [
            p_geometry,
            p_traction,
            config.sensing_cost,
            config.packet_cost,
            config.timely_delivery_probability,
            config.top_safe_cost,
            config.bottom_safe_cost,
            config.unsafe_route_cost,
            *[prior[key] for key in sorted(prior)],
        ],
        dtype=np.float32,
    )
    # One-hot route/modality fields plus public action price, channel quality,
    # and prior probability. No realized probe readings appear here.
    candidates = np.asarray(
        [
            [
                float(name.startswith("top")),
                float(name.startswith("bottom")),
                float(name.endswith("geometry")),
                float(name.endswith("traction")),
                config.sensing_cost,
                config.packet_cost,
                config.timely_delivery_probability,
                p_geometry if name == "top_geometry" else
                p_traction if name == "top_traction" else 1.0,
            ]
            for name in ("top_geometry", "top_traction", "bottom_geometry", "bottom_traction")
        ],
        dtype=np.float32,
    )
    return state, candidates


def canonical_a2_state():
    return encode_context(ExactOracleConfig(), default_prior(), 0.5, 0.5)


def generate(count, seed):
    rng = np.random.RandomState(seed)
    states, candidates, labels, choices = [], [], [], []
    for _ in range(count):
        state, candidate, label, choice = sample_context(rng)
        states.append(state)
        candidates.append(candidate)
        labels.append(label)
        choices.append(choice)
    return (
        torch.tensor(np.asarray(states)),
        torch.tensor(np.asarray(candidates)),
        torch.tensor(np.asarray(labels)),
        np.asarray(choices),
    )


def metrics(model, data):
    states, candidates, labels, choices = data
    model.eval()
    with torch.no_grad():
        prediction = model(states, candidates)
        target_cost = labels.sum(-1)
        predicted_cost = prediction.sum(-1)
        selected_tensor = predicted_cost.argmin(-1)
        selected = selected_tensor.numpy()
        regret = target_cost.gather(1, selected_tensor[:, None]).squeeze(1) - target_cost.min(-1).values
        pair_index = SUBSETS.index(("top_geometry", "top_traction"))
        baseline_index = SUBSETS.index(())
        geometry_index = SUBSETS.index(("top_geometry",))
        traction_index = SUBSETS.index(("top_traction",))
        complementary = (
            (target_cost[:, geometry_index] >= target_cost[:, baseline_index] - 1e-8)
            & (target_cost[:, traction_index] >= target_cost[:, baseline_index] - 1e-8)
            & (target_cost[:, pair_index] < target_cost[:, baseline_index] - 1e-8)
        )
        advantageous = target_cost.min(-1).values < target_cost[:, baseline_index] - 1e-8
        harmful = (selected_tensor != baseline_index) & (
            target_cost.gather(1, selected_tensor[:, None]).squeeze(1)
            >= target_cost[:, baseline_index] - 1e-8
        )
        target_contrast = contrast(target_cost)
        predicted_contrast = contrast(predicted_cost)
        ranking = []
        for first in range(len(SUBSETS)):
            for second in range(first + 1, len(SUBSETS)):
                margin = target_cost[:, first] - target_cost[:, second]
                valid = margin.abs() > 1e-4
                if valid.any():
                    correct = (
                        torch.sign(predicted_cost[:, first] - predicted_cost[:, second])
                        == torch.sign(margin)
                    )
                    ranking.append((correct & valid).sum().item() / valid.sum().item())
        return {
            "value_rmse": float(torch.mean((predicted_cost - target_cost) ** 2).sqrt()),
            "component_rmse": [
                float(torch.mean((prediction[:, :, index] - labels[:, :, index]) ** 2).sqrt())
                for index in range(3)
            ],
            "top1_accuracy": float(np.mean(selected == choices)),
            "mean_acquisition_regret": float(regret.mean()),
            "p90_acquisition_regret": float(torch.quantile(regret, 0.9)),
            "complementary_states": int(complementary.sum()),
            "pair_selection_recall": float(np.mean(selected[complementary.numpy()] == pair_index))
            if complementary.any() else None,
            "harmful_sensing_rate": float(harmful.float().mean()),
            "ranking_accuracy": float(np.mean(ranking)),
            "complementarity_sign_accuracy": float(
                ((predicted_contrast > 0) == (target_contrast > 0)).float().mean()
            ),
            "positive_information_value_states": int(advantageous.sum()),
        }


def train_one(kind, train, validation, args):
    torch.manual_seed(args.seed)
    model = SetCostCritic(train[0].shape[-1], train[1].shape[-1], kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best = float("inf")
    best_state = None
    history = []
    states, candidates, labels, _ = train
    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        permutation = torch.randperm(len(states), generator=generator)
        for indices in permutation.split(args.batch_size):
            prediction = model(states[indices], candidates[indices])
            target = labels[indices]
            predicted_cost = prediction.sum(-1)
            target_cost = target.sum(-1)
            value_loss = nn.functional.mse_loss(prediction, target)
            q_loss = nn.functional.mse_loss(predicted_cost, target_cost)
            contrast_loss = nn.functional.mse_loss(
                contrast(predicted_cost), contrast(target_cost)
            )
            # All acquisition alternatives are exactly labeled at each state.
            # Pairwise ranking emphasizes differences exceeding 0.01 cost unit.
            difference = target_cost[:, :, None] - target_cost[:, None, :]
            pred_difference = predicted_cost[:, :, None] - predicted_cost[:, None, :]
            separated = difference.abs() > 0.01
            rank_loss = torch.relu(0.01 - torch.sign(difference) * pred_difference)
            rank_loss = rank_loss[separated].mean() if separated.any() else q_loss * 0
            loss = value_loss + q_loss + 0.1 * rank_loss + 0.05 * contrast_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_result = metrics(model, validation)
        score = validation_result["mean_acquisition_regret"]
        history.append({"epoch": epoch + 1, **validation_result})
        if score < best - 1e-8:
            best = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main():
    args = arguments()
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train = generate(args.train_states, args.seed)
    validation = generate(args.validation_states, args.seed + 1)
    test = generate(args.test_states, args.seed + 2)
    data_seconds = time.time() - start
    report = {
        "setup": {
            "train_states": args.train_states,
            "validation_states": args.validation_states,
            "test_states": args.test_states,
            "exact_labels_per_state": len(SUBSETS),
            "data_generation_seconds": data_seconds,
            "seed": args.seed,
            "split_unit": "independent pre-acquisition information state",
            "selection_metric": "validation mean acquisition regret",
            "oracle_mode": "certified",
        },
        "models": {},
    }
    for kind in ("additive", "deepsets", "pairwise"):
        start = time.time()
        model, history = train_one(kind, train, validation, args)
        torch.save(model.state_dict(), args.output_dir / (kind + ".pth"))
        report["models"][kind] = {
            "parameters": parameter_count(model),
            "training_seconds": time.time() - start,
            "selected_epoch": min(
                history,
                key=lambda row: (row["mean_acquisition_regret"], row["epoch"]),
            )["epoch"],
            "validation": metrics(model, validation),
            "test": metrics(model, test),
        }
        state, candidates, labels, optimal_index = canonical_a2_state()
        model.eval()
        with torch.no_grad():
            estimate = model(
                torch.tensor(state[None, :]),
                torch.tensor(candidates[None, :, :]),
            ).sum(-1)[0].numpy()
        subset_indices = [
            SUBSETS.index(subset)
            for subset in ((), ("top_geometry",), ("top_traction",), ("top_geometry", "top_traction"))
        ]
        report["models"][kind]["canonical_a2"] = {
            "predicted_q": {str(SUBSETS[i]): float(estimate[i]) for i in subset_indices},
            "true_q": {str(SUBSETS[i]): float(labels[i].sum()) for i in subset_indices},
            "selected_subset": list(SUBSETS[int(estimate.argmin())]),
            "oracle_subset": list(SUBSETS[optimal_index]),
            "selected_regret": float(labels[int(estimate.argmin())].sum() - labels[optimal_index].sum()),
            "predicted_complementarity": float(
                estimate[subset_indices[1]] + estimate[subset_indices[2]]
                - estimate[subset_indices[3]] - estimate[subset_indices[0]]
            ),
        }
        print(kind, json.dumps(report["models"][kind]["test"], sort_keys=True), flush=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()
