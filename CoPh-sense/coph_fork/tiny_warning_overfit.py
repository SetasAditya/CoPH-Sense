#!/usr/bin/env python3
"""Small capacity check: can the pairwise critic fit bad-warning examples?"""

import json
from pathlib import Path
from types import SimpleNamespace

from coph_fork.train_share_critic import generate_warning_curriculum, model_metrics, train_one
from coph_fork.train_share_critic_v2 import last_rows, warning_recall, warning_test


def main():
    train = generate_warning_curriculum(32, 16, 9917)
    setting = SimpleNamespace(seed=9917, epochs=200, batch_size=16)
    model, epoch = train_one("pairwise", train, train, setting)
    report = {
        "setup": "16 constructed warnings and 16 other states; same data for checkpoint selection, so this is a capacity audit only",
        "selected_epoch": epoch,
        "training_warning": warning_recall(model, last_rows(train, 16)),
        "training_regret": model_metrics(model, train)["mean_regret"],
        "independent_warning": warning_recall(model, warning_test(100, 19917)),
    }
    path = Path(__file__).resolve().parent / "results" / "a4_v2" / "tiny_overfit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
