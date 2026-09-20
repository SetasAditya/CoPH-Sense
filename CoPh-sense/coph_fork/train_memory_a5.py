#!/usr/bin/env python3
"""Train A5-v2 memory-aware critics against exact post-delivery action values."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.memory_a5 import EvidenceLedger, EvidenceRecord, exact_evaluation  # noqa: E402
from coph_fork.memory_critic_a5 import (  # noqa: E402
    MemorySetCritic, action_costs, delivery_posteriors, encode_memory,
)
from coph_fork.oracle import ExactOracleConfig  # noqa: E402
from coph_fork.run_memory_a5 import load_models  # noqa: E402


OUTPUT = HERE / "results" / "a5_memory_v2"


def ledger_from_pattern(pattern):
    records = []
    for j, value in enumerate(pattern):
        if value == -1:
            continue
        records.append(EvidenceRecord(
            evidence_id="training-{}".format(j), region=1 + j // 2,
            modality="geometry" if j % 2 == 0 else "traction",
            value=bool(value), uncertainty=0.0, acquisition_time=0,
            delivery_time=1, source="scout" if j < 2 else "carrier",
        ))
    return EvidenceLedger(records)


def sample_prior(rng):
    if rng.rand() < 0.60:
        return np.full(16, 1 / 16, dtype=np.float64)
    # Positive correlated priors test whether the full posterior, rather than
    # four independent marginals, is sufficient for the exact action values.
    return rng.dirichlet(np.full(16, rng.uniform(0.25, 2.5)))


def sample_row(rng, default=False):
    pattern = tuple(int(rng.choice((-1, 0, 1), p=(0.42, 0.29, 0.29))) for _ in range(4))
    memory = ledger_from_pattern(pattern)
    prior = np.full(16, 1 / 16, dtype=np.float64) if default else sample_prior(rng)
    config = ExactOracleConfig() if default else replace(
        ExactOracleConfig(),
        sensing_cost=float(rng.uniform(0.02, 0.11)),
        top_safe_cost=float(rng.uniform(0.8, 1.5)),
        bottom_safe_cost=float(rng.uniform(2.4, 4.0)),
    )
    state, candidates, posterior = encode_memory(memory, config, prior)
    return state, candidates, action_costs(posterior, memory, config)


def data(count, seed):
    rng = np.random.RandomState(seed)
    rows = [sample_row(rng, default=(i % 3 == 0)) for i in range(count)]
    return tuple(torch.tensor(np.asarray([row[j] for row in rows])) for j in range(3))


def sufficiency_audit():
    config = ExactOracleConfig()
    seen = {}
    collisions = 0
    for pattern in product((-1, 0, 1), repeat=4):
        memory = ledger_from_pattern(pattern)
        state, candidates, posterior = encode_memory(memory, config)
        key = (state.tobytes(), candidates.tobytes())
        values = action_costs(posterior, memory, config)
        if key in seen and not np.allclose(seen[key], values, atol=1e-7):
            collisions += 1
        seen[key] = values
    return {"patterns_checked": 81, "distinct_inputs": len(seen),
            "conflicting_exact_labels": collisions}


def decision_metrics(model, dataset):
    state, candidates, target = dataset
    with torch.no_grad():
        estimate = model(state, candidates)
    selected = estimate.argmin(1)
    optimal = target.argmin(1)
    regret = target.gather(1, selected[:, None])[:, 0] - target.min(1).values
    return {"regret": float(regret.mean()),
            "optimal_action_accuracy": float((selected == optimal).float().mean()),
            "value_rmse": float(torch.mean((estimate - target) ** 2).sqrt())}


def delivery_training_data(a4_model):
    config = ExactOracleConfig()
    posteriors, examples, _ = delivery_posteriors(a4_model, config)
    rows = []
    for key, posterior in posteriors.items():
        memory = examples[key]
        state, candidates, _ = encode_memory(memory, config, posterior=posterior)
        target = action_costs(posterior, memory, config)
        # Equal signature support: common empty histories must not crowd out
        # fully delivered states in a naturally sampled batch.
        rows.extend([(state, candidates, target)] * 80)
    return tuple(torch.tensor(np.asarray([row[j] for row in rows])) for j in range(3))


def train(kind, seed, train_set, validation_set):
    torch.manual_seed(seed)
    model = MemorySetCritic(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best = None
    best_regret = float("inf")
    selected_epoch = 0
    for epoch in range(1, 121):
        order = torch.randperm(len(train_set[0]))
        model.train()
        for indices in order.split(64):
            prediction = model(train_set[0][indices], train_set[1][indices])
            target = train_set[2][indices]
            # Values and near-optimal distinctions both matter for selection.
            value_loss = nn.functional.mse_loss(prediction, target)
            relative = prediction - prediction[:, :1]
            target_relative = target - target[:, :1]
            loss = value_loss + nn.functional.mse_loss(relative, target_relative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        metrics = decision_metrics(model, validation_set)
        if metrics["regret"] < best_regret - 1e-7:
            best_regret = metrics["regret"]
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            selected_epoch = epoch
    model.load_state_dict(best)
    model.eval()
    return model, selected_epoch


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = sufficiency_audit()
    if audit["conflicting_exact_labels"]:
        raise RuntimeError("A5-v2 representation has contradictory exact labels")
    heldout = data(360, 91501)
    results = {}
    for seed, a4_seed in ((1701, 1801), (1702, 1802), (1703, 1803)):
        train_set = data(1800, seed)
        validation_set = data(360, seed + 10000)
        a3, a4 = load_models(seed, a4_seed)
        delivery_set = delivery_training_data(a4)
        train_set = tuple(torch.cat((train_set[j], delivery_set[j]), 0) for j in range(3))
        posteriors, _, _ = delivery_posteriors(a4, ExactOracleConfig())
        results[str(seed)] = {}
        for kind in ("additive", "deepsets", "pairwise"):
            model, epoch = train(kind, seed, train_set, validation_set)
            checkpoint = OUTPUT / ("{}_{}.pth".format(kind, seed))
            torch.save(model.state_dict(), checkpoint)
            executed = exact_evaluation("learned_memory_v2", a3, a4,
                                        ExactOracleConfig(), model, posteriors)
            executed.pop("branches")
            results[str(seed)][kind] = {
                "selected_epoch": epoch,
                "validation": decision_metrics(model, validation_set),
                "heldout": decision_metrics(model, heldout),
                "executed": executed,
            }
            print(seed, kind, "heldout regret", round(results[str(seed)][kind]["heldout"]["regret"], 4),
                  "team cost", round(executed["expected_total_team_cost"], 4), flush=True)
    report = {
        "scope": "post-delivery carrier acquisition in exact two-region finite task",
        "continuation": "frozen A4-v2 selects scout messages before carrier acquisition; exact carrier posterior includes A4 selection and packet loss; fixed evidence-based route executor follows carrier acquisition; no later A4 stage exists in this protocol",
        "sufficiency_audit": audit,
        "models": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
