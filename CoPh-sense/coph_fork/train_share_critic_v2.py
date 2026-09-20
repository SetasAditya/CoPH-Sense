#!/usr/bin/env python3
"""A4 development revision: sufficient input and warning-balanced supervision."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.critic import SetCostCritic, SUBSETS, parameter_count  # noqa: E402
from coph_fork.train_share_critic import (  # noqa: E402
    build_features_v2, generate, generate_warning_curriculum,
    model_metrics, sample_warning_context, train_one,
)


def warning_test(count, seed, feature_builder=build_features_v2):
    rng = np.random.RandomState(seed)
    rows = [sample_warning_context(rng, feature_builder) for _ in range(count)]
    return (
        torch.tensor(np.stack([row[0] for row in rows])),
        torch.tensor(np.stack([row[1] for row in rows])),
        torch.tensor(np.stack([row[2] for row in rows])),
        torch.tensor(np.stack([row[3] for row in rows]), dtype=torch.bool),
        np.asarray([row[4] for row in rows]),
        np.asarray([row[5] for row in rows]),
    )


def warning_recall(model, data):
    state, candidates, labels, feasible, optimal, _ = data
    model.eval()
    with torch.no_grad():
        selected = model(state, candidates).sum(-1).masked_fill(
            ~feasible, float("inf")
        ).argmin(-1).numpy()
    required = np.asarray([
        "top_geometry" in SUBSETS[int(selected[j])]
        for j in range(len(selected))
    ])
    return {
        "count": len(selected),
        "recall": float(required.mean()),
        "mean_regret": model_metrics(model, data)["mean_regret"],
    }


def last_rows(data, count):
    return tuple(part[-count:] for part in data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "a4_v2")
    setting = parser.parse_args()
    setting.batch_size = 48
    setting.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    begin = time.time()
    train = generate_warning_curriculum(600, 120, setting.seed)
    validation = generate_warning_curriculum(150, 30, setting.seed + 1)
    test = generate(200, setting.seed + 2, build_features_v2)
    warnings = warning_test(100, setting.seed + 30000)
    report = {
        "setup": {
            "seed": setting.seed,
            "train_states": 600,
            "train_warning_states": 120,
            "validation_states": 150,
            "validation_warning_states": 30,
            "standard_stratified_test_states": 200,
            "held_out_warning_states": 100,
            "feature_version": "v2 full four-world public prior, no-message route/margin, effective evidence-plus-ACK price",
            "selection_metric": "balanced validation exact regret",
            "critical_beta": 1.0,
            "data_seconds": time.time() - begin,
        },
        "models": {},
    }
    for label, kind, beta in (
        ("additive_unweighted", "additive", 0.0),
        ("additive_weighted", "additive", 1.0),
        ("pairwise_weighted", "pairwise", 1.0),
    ):
        model, epoch = train_one(kind, train, validation, setting, critical_beta=beta)
        torch.save(model.state_dict(), setting.output_dir / (label + ".pth"))
        report["models"][label] = {
            "architecture": kind,
            "critical_beta": beta,
            "parameters": parameter_count(model),
            "selected_epoch": epoch,
            "validation": model_metrics(model, validation),
            "standard_test": model_metrics(model, test),
            "training_warning": warning_recall(model, last_rows(train, 120)),
            "warning_test": warning_recall(model, warnings),
        }
        print(label, report["models"][label]["standard_test"]["mean_regret"],
              report["models"][label]["warning_test"], flush=True)
    (setting.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
