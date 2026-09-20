"""Render the Natural-A4 hindsight-versus-actionable-value result."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    directory = args.dataset_dir.resolve()
    rows = json.loads((directory / "test.json").read_text())["rows"]
    values = np.asarray([
        row["decision_value"] for row in rows
        if row["risk"] == "risky" and row["scout_start_x"] == 14 and
        row["age_steps"] == 0 and row["receiver_condition"] == "natural"
    ], dtype=float)
    if len(values) != 200:
        raise RuntimeError(f"expected 200 strongest-context rows, got {len(values)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35),
                             gridspec_kw={"width_ratios": (1.7, 1.)})
    axes[0].hist(values, bins=22, color="#4C78A8", alpha=.88)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].axvline(values.mean(), color="#E45756", linewidth=2,
                    label=f"conditional mean = {values.mean():.4f}")
    axes[0].set(xlabel=r"realized $J_{HOLD}-J_{SEND}$",
                ylabel="held-out worlds",
                title="Realized SEND effects")
    axes[0].legend(frameon=False, fontsize=8)

    positive = float(np.mean(values > 0))
    upper = float(values.mean() + 1.6448536269514722 *
                  values.std(ddof=1) / np.sqrt(len(values)))
    axes[1].bar([0], [positive], color="#72B7B2", width=.58)
    axes[1].set(xticks=[0], xticklabels=["hindsight\nSEND-positive"],
                ylabel="fraction of worlds", ylim=(0, .55),
                title="43% helped by chance")
    axes[1].text(0, positive + .025, f"{positive:.0%}", ha="center")
    fig.suptitle("Hindsight usefulness is not actionable information value\n"
                 f"one-sided 95% upper bound on conditional mean: {upper:.4f}",
                 fontsize=11)
    fig.tight_layout()
    out = directory / "figures" / "natural_a4_hindsight_vs_actionable.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
