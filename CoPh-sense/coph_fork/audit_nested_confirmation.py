#!/usr/bin/env python3
"""Refresh fresh-seed A3→A4 metrics, including complementary-state regret."""

import json
from pathlib import Path

import torch

from coph_fork.finetune_a3_for_a4 import dataset, load_a3, load_a4, selection_metrics


ROOT = Path(__file__).resolve().parent / "results" / "nested_a3_a4"


def main():
    torch.set_num_threads(2)
    for a3_seed, a4_seed in ((1701, 1801), (1702, 1802), (1703, 1803)):
        folder = ROOT / ("finetune" + str(a3_seed))
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        data = dataset(120, report["confirmation_seed"], load_a4(a4_seed))
        original = load_a3(a3_seed, data[0][0].numpy(), data[1][0].numpy())
        revised = load_a3(a3_seed, data[0][0].numpy(), data[1][0].numpy())
        revised.load_state_dict(torch.load(
            folder / "pairwise_nested.pth", map_location="cpu", weights_only=True
        ))
        report["confirmation_before"] = selection_metrics(original, data)
        report["confirmation_after"] = selection_metrics(revised, data)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(a3_seed,
              report["confirmation_before"]["mean_nested_regret_on_pair_relevant"],
              report["confirmation_after"]["mean_nested_regret_on_pair_relevant"],
              flush=True)


if __name__ == "__main__":
    main()
