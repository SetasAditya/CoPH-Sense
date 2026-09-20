#!/usr/bin/env python3
"""Aggregate matched A4-v2 attribution controls."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "results" / "a4_v2_controls"


def main():
    paths = sorted(ROOT.glob("seed*/report.json"))
    if len(paths) != 3:
        raise SystemExit("expected three control seed reports")
    reports = [json.loads(path.read_text()) for path in paths]
    output = {"seeds": [row["seed"] for row in reports], "controls": {}}
    for name in reports[0]["models"]:
        entries = [row["models"][name] for row in reports]
        output["controls"][name] = {
            metric: {
                "by_seed": values,
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
            for metric, values in {
                "standard_regret": [e["standard_test"]["mean_regret"] for e in entries],
                "held_out_warning_recall": [e["warning_test"]["recall"] for e in entries],
                "held_out_warning_regret": [e["warning_test"]["mean_regret"] for e in entries],
            }.items()
        }
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
