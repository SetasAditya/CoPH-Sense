#!/usr/bin/env python3
"""Fine-tune A3 on exact labels under the frozen learned A4 continuation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.critic import SUBSETS, SetCostCritic, contrast  # noqa: E402
from coph_fork.nested_oracle import nested_values, pair_ordering  # noqa: E402
from coph_fork.train_critic import sample_context  # noqa: E402
from coph_fork.train_share_critic import build_features_v2  # noqa: E402
from coph_fork.oracle import ExactOracleConfig, default_prior  # noqa: E402
from coph_fork.share_oracle import ShareState  # noqa: E402


def load_a3(seed, state, candidates):
    folder = HERE / "results" / "a3"
    folder = folder if seed == 1701 else folder / ("seed" + str(seed))
    model = SetCostCritic(len(state), candidates.shape[-1], "pairwise")
    model.load_state_dict(torch.load(
        folder / "pairwise.pth", map_location="cpu", weights_only=True
    ))
    return model


def load_a4(seed):
    state, candidates = build_features_v2(
        ExactOracleConfig(), default_prior(), ShareState(acquired=())
    )
    model = SetCostCritic(len(state), candidates.shape[-1], "pairwise")
    path = HERE / "results" / "a4_v2" / ("seed" + str(seed)) / "pairwise_weighted.pth"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def dataset(count, seed, a4_model):
    rng = np.random.RandomState(seed)
    states, candidates, components, exact_q, learned_q = [], [], [], [], []
    for _ in range(count):
        state, candidate, _, _, config, prior = sample_context(rng, True)
        q_star, q_learned, _, learned_parts = nested_values(
            config, prior, a4_model, return_components=True
        )
        states.append(state)
        candidates.append(candidate)
        components.append(learned_parts)
        exact_q.append(q_star)
        learned_q.append(q_learned)
    return (
        torch.tensor(np.stack(states)),
        torch.tensor(np.stack(candidates)),
        torch.tensor(np.stack(components), dtype=torch.float32),
        np.stack(exact_q),
        np.stack(learned_q),
    )


def selection_metrics(model, data):
    model.eval()
    states, candidates, _, exact_q, q = data
    with torch.no_grad():
        selected = model(states, candidates).sum(-1).argmin(-1).numpy()
    index = np.arange(len(selected))
    regret = q[index, selected] - q.min(-1)
    pair_relevant = np.asarray([pair_ordering(row) for row in q])
    pair_index = SUBSETS.index(("top_geometry", "top_traction"))
    return {
        "mean_nested_regret": float(regret.mean()),
        "p90_nested_regret": float(np.quantile(regret, 0.9)),
        "max_nested_regret": float(regret.max()),
        "mean_executed_normalized_cost": float(q[index, selected].mean()),
        "optimal_set_accuracy": float((selected == q.argmin(-1)).mean()),
        "pair_relevant_count": int(pair_relevant.sum()),
        "mean_nested_regret_on_pair_relevant": float(regret[pair_relevant].mean())
        if pair_relevant.any() else None,
        "p90_nested_regret_on_pair_relevant": float(np.quantile(regret[pair_relevant], 0.9))
        if pair_relevant.any() else None,
        "pair_selection_recall": float((selected[pair_relevant] == pair_index).mean())
        if pair_relevant.any() else None,
        "extraneous_bottom_sensing_rate": float(np.mean([
            any(name.startswith("bottom") for name in SUBSETS[int(choice)])
            for choice in selected
        ])),
        "mean_oracle_sharing_gap_at_selection": float(
            (q[index, selected] - exact_q[index, selected]).mean()
        ),
    }


def fit(model, train, validation, seed, epochs=80):
    states, candidates, targets, _, _ = train
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_metric = selection_metrics(model, validation)["mean_nested_regret"]
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(states), generator=torch.Generator().manual_seed(seed + epoch))
        for indices in order.split(32):
            prediction = model(states[indices], candidates[indices])
            target = targets[indices]
            predicted_q = prediction.sum(-1)
            true_q = target.sum(-1)
            component_loss = nn.functional.mse_loss(prediction, target)
            total_loss = nn.functional.mse_loss(predicted_q, true_q)
            difference = true_q[:, :, None] - true_q[:, None, :]
            predicted_difference = predicted_q[:, :, None] - predicted_q[:, None, :]
            separated = difference.abs() > 0.01
            ranking = torch.relu(0.01 - torch.sign(difference) * predicted_difference)
            ranking_loss = ranking[separated].mean() if separated.any() else total_loss * 0
            interaction_loss = nn.functional.mse_loss(
                contrast(predicted_q), contrast(true_q)
            )
            loss = component_loss + total_loss + 0.1 * ranking_loss + 0.05 * interaction_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = selection_metrics(model, validation)["mean_nested_regret"]
        if score < best_metric - 1e-8:
            best_metric = score
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_epoch, best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-seed", type=int, required=True)
    parser.add_argument("--a4-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    a4_model = load_a4(args.a4_seed)
    train = dataset(320, args.a3_seed, a4_model)
    validation = dataset(80, args.a3_seed + 1, a4_model)
    # Fresh confirmation seeds were not used in A3, A4, or nested diagnosis.
    confirmation_seed = 2100 + (args.a3_seed - 1700)
    confirmation = dataset(120, confirmation_seed, a4_model)
    original = load_a3(args.a3_seed, train[0][0].numpy(), train[1][0].numpy())
    before = selection_metrics(original, confirmation)
    before_validation = selection_metrics(original, validation)
    epoch, score = fit(original, train, validation, args.a3_seed)
    after = selection_metrics(original, confirmation)
    torch.save(original.state_dict(), args.output_dir / "pairwise_nested.pth")
    report = {
        "a3_seed": args.a3_seed,
        "a4_seed": args.a4_seed,
        "confirmation_seed": confirmation_seed,
        "train_states": 320,
        "validation_states": 80,
        "confirmation_states": 120,
        "continuation": "frozen A4-v2 weighted pairwise; exact expected finite outcome labels",
        "checkpoint_rule": "lowest validation mean nested regret, including epoch 0",
        "selected_epoch": epoch,
        "validation_before": before_validation,
        "validation_best_regret": score,
        "confirmation_before": before,
        "confirmation_after": after,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
