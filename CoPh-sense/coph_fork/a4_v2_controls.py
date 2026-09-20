#!/usr/bin/env python3
"""Matched A4 controls for warning curriculum and representation."""

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.train_share_critic import (  # noqa: E402
    build_features, build_features_v2, generate, generate_warning_curriculum,
    model_metrics, train_one,
)
from coph_fork.train_share_critic_v2 import warning_recall, warning_test  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    setting = parser.parse_args()
    setting.epochs = 100
    setting.batch_size = 48
    setting.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    report = {"seed": setting.seed, "models": {}}
    for label, builder, warning_count in (
        ("v1_with_warning_curriculum", build_features, 120),
        ("v2_without_warning_curriculum", build_features_v2, 0),
    ):
        if warning_count:
            train = generate_warning_curriculum(600, warning_count, setting.seed, builder)
            validation = generate_warning_curriculum(150, 30, setting.seed + 1, builder)
        else:
            train = generate(600, setting.seed, builder)
            validation = generate(150, setting.seed + 1, builder)
        test = generate(200, setting.seed + 2, builder)
        warnings = warning_test(100, setting.seed + 30000, builder)
        model, epoch = train_one("pairwise", train, validation, setting, critical_beta=1.0)
        torch.save(model.state_dict(), setting.output_dir / (label + ".pth"))
        report["models"][label] = {
            "selected_epoch": epoch,
            "standard_test": model_metrics(model, test),
            "warning_test": warning_recall(model, warnings),
        }
        print(label, report["models"][label]["standard_test"]["mean_regret"],
              report["models"][label]["warning_test"], flush=True)
    (setting.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
