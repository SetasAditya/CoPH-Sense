"""Plots for the disposable static look-ahead mechanism audit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for path in args.audit
            for r in json.loads(path.read_text())["records"]
            if r.get("acquired") and "branches" in r]
    if not rows:
        raise ValueError("audit has no complete message cases")
    risky = np.asarray([r["surface"] > .5 for r in rows])
    color = np.where(risky, "#b63d3d", "#3577a7")
    kl = np.asarray([r["conditional_kl_nats"] for r in rows])
    value = np.asarray([r["decision_value"] for r in rows])
    delta = np.asarray([r["delta_tau_at_reading"] for r in rows])
    lookahead_gain = np.asarray([
        r["branches"]["innovation"]["carrier_effective_lookahead"]
        - r["branches"]["innovation"]["carrier_local_lookahead"]
        for r in rows])
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4))
    ax = axes[0, 0]
    ax.scatter(kl, value, c=color)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Receiver-conditioned KL proxy (nats)",
           ylabel="HOLD cost − innovation SEND cost",
           title="Novelty and task value")
    ax = axes[0, 1]
    ax.scatter(delta, lookahead_gain, c=color)
    ax.set(xlabel="Leader − follower x-progress (m)",
           ylabel="Extra effective look-ahead (m)",
           title="Communicated look-ahead")
    ax = axes[0, 2]
    ax.scatter(delta, value, c=color)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Leader − follower x-progress (m)",
           ylabel="Message decision value",
           title="Value versus progress gap")
    ax = axes[0, 3]
    ax.scatter(kl, value / np.asarray([r["summary_bytes"] for r in rows]),
               c=color)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Receiver-conditioned KL proxy (nats)",
           ylabel="Decision value / payload byte",
           title="Value per byte")
    ax = axes[1, 0]
    raw = np.asarray([r["raw_bytes"] for r in rows])
    compact = np.asarray([r["summary_bytes"] for r in rows])
    ax.bar(["Raw footprint", "Innovation"],
           [raw.mean(), compact.mean()], color=["#999999", "#2b8c8c"])
    ax.set(ylabel="Payload bytes", title="Wire payload")
    ax = axes[1, 1]
    support = [r["branches"]["innovation"]["support"] for r in rows]
    ax.bar(["Overlapping", "Disjoint", "New to carrier"],
           [np.mean([s["overlap_cells"] for s in support]),
            np.mean([s["disjoint_cells"] for s in support]),
            np.mean([s["carrier_new_received_cells"] for s in support])],
           color=["#999999", "#3577a7", "#2b8c8c"])
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="Grid cells", title="Explored support")
    ax = axes[1, 2]
    for name, marker, tone in (("hold", "o", "#333333"),
                               ("raw", "s", "#999999"),
                               ("innovation", "^", "#2b8c8c")):
        communication = [r["branches"][name].get("communication_cost", 0.)
                         for r in rows]
        cost = [r["branches"][name]["cost"] for r in rows]
        ax.scatter(communication, cost, marker=marker, color=tone,
                   label=name, alpha=.75)
    ax.set(xlabel="Charged communication cost", ylabel="Full team cost",
           title="Cost trade-off")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 3]
    for name, marker, tone in (("hold", "o", "#333333"),
                               ("raw", "s", "#999999"),
                               ("innovation", "^", "#2b8c8c")):
        communication = [r["branches"][name].get("communication_cost", 0.)
                         for r in rows]
        exposure = [r["branches"][name]["risk_exposure"] for r in rows]
        ax.scatter(communication, exposure, marker=marker, color=tone,
                   label=name, alpha=.75)
    ax.set(xlabel="Charged communication cost", ylabel="Material exposure",
           title="Risk trade-off")
    for axis in axes.flat:
        axis.grid(alpha=.18)
    fig.suptitle("CoPH-Lookahead v1: static two-lane development diagnostic\n"
                 "Blue: safe first region; red: risky first region",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
