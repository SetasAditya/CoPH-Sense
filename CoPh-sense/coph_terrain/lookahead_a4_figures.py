"""Offline figures for corrected A4; reads frozen JSON, never simulates."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("learning", type=Path)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selected-method", default="recipient_mlp_value_sign")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.learning.read_text())
    pooled = (json.loads(args.development.read_text())["rows"] +
              json.loads(args.prospective.read_text())["rows"])
    test = report["test_rows"]
    selected = report["methods"][args.selected_method]["selected_send"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for row, send in zip(test, selected):
        positive = row["value"] > 0
        changed = bool(row["new_region_gained"])
        ax.scatter(row["delta_tau"], row["value"],
                   s=22 if len(test) > 500 else 60,
                   marker="^" if changed else "o",
                   edgecolor="tab:green" if positive else "tab:red",
                   facecolor=("tab:green" if positive else "tab:red")
                   if send else "white", linewidth=.8 if len(test) > 500 else 1.3,
                   alpha=.42 if len(test) > 500 else .85)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="leader–follower progress gap Δτ (m)",
           ylabel="J(HOLD) − J(compact SEND)",
           title="Held-out A4 value: filled = learned SEND")
    ax.text(.02, .02, "green: useful  red: costly  triangle: sensing switch",
            transform=ax.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    fig.tight_layout()
    fig.savefig(args.out / "a4_value_vs_gap.png", dpi=180)
    fig.savefig(args.out / "a4_value_vs_gap.pdf")
    plt.close(fig)

    useful = [row for row in pooled if row["decision_value"] > 0]
    physical = np.mean([row["physical_benefit_excluding_radio"]
                        for row in useful])
    fixed = np.mean([row["fixed_header_ack_radio"] for row in useful])
    payload = np.mean([row["payload_radio"] for row in useful])
    net = physical-fixed-payload
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    vals = [physical, -fixed, -payload, net]
    labels = ["physical\nbenefit", "fixed +\nheader + ACK",
              "48 B\npayload", "net\nSEND value"]
    ax.bar(range(4), vals, color=["tab:green", "tab:red", "tab:orange",
                                  "tab:blue"])
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(4), labels)
    ax.set(ylabel="team cost difference", title="Useful SEND states only")
    for index, value in enumerate(vals):
        if value >= 0:
            ax.annotate(f"{value:+.4f}", (index, value),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9)
        else:
            ax.text(index, -.011 if index == 1 else -.005,
                    f"{value:+.4f}", ha="center", fontsize=9,
                    color="white" if index == 1 else "black")
    fig.tight_layout()
    fig.savefig(args.out / "a4_value_decomposition.png", dpi=180)
    fig.savefig(args.out / "a4_value_decomposition.pdf")
    plt.close(fig)

    natural = [row for row in pooled if row["receiver_condition"] == "natural"]
    switched = [row for row in natural if
                row["hold"]["duplicate_region_scan"] and
                row["send"]["choice_region"] == 1 and
                row["send"]["acquisition_executed"]]
    gain = np.mean([row["delta_disjoint_support"] for row in switched])
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for x, color, face, title, detail in (
        (0.25, "#666666", "#f2f2f2", "HOLD", "Duplicate Region 1 scan"),
        (5.65, "#1565a4", "#e8f3fb", "SEND", "New Region 2 scan")):
        ax.add_patch(FancyBboxPatch((x, .65), 4.1, 1.7,
                                   boxstyle="round,pad=0.14,rounding_size=.13",
                                   edgecolor=color, facecolor=face, linewidth=1.7))
        ax.text(x+.2, 1.9, title, color=color, fontsize=12, weight="bold")
        ax.text(x+.2, 1.2, detail, color="#202020", fontsize=11)
    ax.add_patch(FancyArrowPatch((4.55, 1.5), (5.35, 1.5),
                                 arrowstyle="-|>", mutation_scale=17,
                                 linewidth=1.6, color="#333333"))
    ax.text(5.0, .12, f"+{gain:.0f} disjoint cells · {len(switched)}/{len(natural)} matched states switch",
            ha="center", fontsize=10, color="#145c87")
    fig.suptitle("Delivered innovation reallocates the carrier's next physical scan",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "a4_exploration_switch.png", dpi=180)
    fig.savefig(args.out / "a4_exploration_switch.pdf")
    plt.close(fig)

    names = ["never_send", "always_compact_send", "novelty_threshold",
             "sender_ridge", "sender_mlp_value", "recipient_mlp_basic",
             "recipient_mlp_value",
             "recipient_mlp_value_sign"]
    display = ["Never", "Always", "Novelty", "Sender ridge",
               "Sender MLP", "Recipient basic", "Response MLP",
               "Response MLP + sign"]
    for name, label in (("recipient_ridge", "Recipient ridge"),
                        ("response_ridge", "Response ridge")):
        if name in report["methods"]:
            names.append(name)
            display.append(label)
    if "structured_value" in report["methods"]:
        names.append("structured_value")
        display.append("Structured value")
    names.append("paired_hindsight_oracle" if
                 "paired_hindsight_oracle" in report["methods"] else
                 "restricted_oracle")
    display.append("Hindsight oracle")
    regrets = [report["methods"][name]["mean_regret"] for name in names]
    recalls = [report["methods"][name]["useful_send_recall"] or 0.
               for name in names]
    false_rates = [report["methods"][name]["false_send_rate"] or 0.
                   for name in names]
    fig, axes = plt.subplots(1, 2,
                             figsize=(13 if len(names) > 10 else 10, 4.5))
    x = np.arange(len(names))
    regret_cap = .008
    axes[0].bar(x, np.minimum(regrets, regret_cap), color="tab:blue")
    axes[0].set_ylim(0, regret_cap*1.08)
    axes[0].annotate(f"{regrets[1]:.3f}", (1, regret_cap*.82),
                     ha="center", color="white", fontsize=8,
                     rotation=90)
    axes[0].set_ylabel("mean paired decision regret")
    axes[1].bar(x-.18, recalls, width=.36, color="tab:green",
                label="useful-SEND recall")
    axes[1].bar(x+.18, false_rates, width=.36, color="tab:red",
                label="false-SEND rate")
    axes[1].set_ylabel("fraction")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, display, rotation=55, ha="right", fontsize=8)
    fig.suptitle("Seed-disjoint A4 decision quality")
    fig.tight_layout()
    fig.savefig(args.out / "a4_methods.png", dpi=180)
    fig.savefig(args.out / "a4_methods.pdf")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
