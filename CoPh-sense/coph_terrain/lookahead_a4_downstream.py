"""A4 SEND/HOLD labels with the receiver's next acquisition executed.

Lookahead environment, radio, and pH dynamics are unchanged. Both branches
use the same transparent local sensing rule after their own delivered histories.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .algorithm import post_reading_branch
from .lookahead import region_cells
from .lookahead_policy import choose_region_to_inspect, message_features
from .lookahead_recipient_a4 import (make_pool, model_report,
                                      recipient_features, response_features)
from .physical_audit import run_continuation


OUT = Path(__file__).resolve().parent / "results" / "lookahead_recipient_a4"


def acquisition_continuation(snapshot, packet_id, send, evidence_id=None):
    """Execute one receiver-selected scan, then the same scripted pH mission."""
    transmitted_id = packet_id if evidence_id is None else evidence_id
    _, state = post_reading_branch(snapshot, transmitted_id, send,
                                   continue_mission=False)
    decision_step = state.step_index
    delivered = transmitted_id in state.received["carrier"]
    choice = None if state.done else choose_region_to_inspect(state, "carrier")
    choices = () if choice is None else (choice,)
    selected = () if choice is None else (0,)
    result, final = run_continuation(state, choices, selected, transmit=False,
                                     return_environment=True)
    new_carrier_support = (final.local_support["carrier"] -
                           snapshot.local_support["carrier"])
    packet = snapshot.innovation_packets[packet_id]
    # The scout's region-level posterior represents the entire marked band,
    # although its raw footprint occupies fewer cells. Exclude that complete
    # region when measuring genuinely disjoint carrier exploration.
    scout_known = (snapshot.local_support["scout"] |
                   snapshot.passive_support["scout"] |
                   region_cells(packet.region))
    new_disjoint = new_carrier_support - scout_known
    duplicate_region = (choice is not None and
                        choice.target_region in region_cells(packet.region))
    return {"cost": float(result["score_single_world"]),
            "success": bool(result["success"]),
            "decision_step": decision_step,
            "delivered": delivered,
            "choice": None if choice is None else asdict(choice),
            "choice_region": None if choice is None else
            (0 if choice.target_region == (18, 27) else 1),
            "acquisition_executed": bool(
                result["target_region_covered"] and
                result["target_region_covered"][0]),
            "new_carrier_support": len(new_carrier_support),
            "new_disjoint_support": len(new_disjoint),
            "duplicate_region_scan": duplicate_region,
            "radio": float(result["ledger"]["communication"]),
            "sensing": float(result["ledger"]["sensing"]),
            "risk_exposure": float(result["material_exposure"]),
            "mission_cost": float(result["mission_cost"]),
            "path_length": result["path_length"],
            "steps": result["steps"],
            "ledger": result["ledger"]}


def evaluate_downstream(snapshot, packet_id, meta):
    hold = acquisition_continuation(snapshot, packet_id, False)
    sent = acquisition_continuation(snapshot, packet_id, True)
    value = hold["cost"]-sent["cost"]
    physical_benefit = ((hold["cost"]-hold["radio"]) -
                        (sent["cost"]-sent["radio"]))
    incremental_radio = sent["radio"]-hold["radio"]
    if abs(value-(physical_benefit-incremental_radio)) > 1e-7:
        raise AssertionError("A4 downstream value accounting does not close")
    packet = snapshot.innovation_packets[packet_id]
    accepted = int(sent["radio"] > hold["radio"])
    payload_cost = (accepted * snapshot.config.byte_cost *
                    packet.payload_bytes)
    return {**meta, "step": snapshot.step_index,
            "actor_features": message_features(snapshot, packet_id).tolist(),
            "recipient_features": recipient_features(snapshot, packet_id).tolist(),
            "response_features": response_features(snapshot, packet_id).tolist(),
            "decision_value": value,
            "hold": hold, "send": sent,
            "physical_benefit_excluding_radio": physical_benefit,
            "incremental_radio": incremental_radio,
            "payload_radio": payload_cost,
            "fixed_header_ack_radio": incremental_radio-payload_cost,
            "duplicate_sensing_avoided": int(
                hold["duplicate_region_scan"] and
                hold["acquisition_executed"] and
                not (sent["duplicate_region_scan"] and
                     sent["acquisition_executed"])),
            "new_region_sensing_gained": int(
                sent["choice_region"] == 1 and
                hold["choice_region"] != 1 and
                sent["acquisition_executed"]),
            "delta_disjoint_support": (sent["new_disjoint_support"]-
                                       hold["new_disjoint_support"]),
            "delta_risk_exposure": (hold["risk_exposure"]-
                                    sent["risk_exposure"]),
            "delta_tau": snapshot.observations()["scout"]
            ["lookahead"]["delta_tau"],
            "receiver_direct_overlap_evaluator_only": bool(
                snapshot.local_support["carrier"] &
                set(packet.support_cells))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=981000)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--schedule-origin", type=int)
    parser.add_argument("--collection-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--ages", type=int, nargs="+", default=(0, 8, 16))
    parser.add_argument("--include-acked-duplicate", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"downstream_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    rows, failures = make_pool(args.start, args.seeds,
                               evaluator=evaluate_downstream,
                               schedule_origin=args.schedule_origin,
                               verbose=not args.collection_only,
                               ages=tuple(args.ages),
                               include_acked_duplicate=args.include_acked_duplicate)
    if args.collection_only:
        report = {"start": args.start, "seeds": args.seeds,
                  "schedule_origin": args.schedule_origin,
                  "ages": args.ages,
                  "include_acked_duplicate": args.include_acked_duplicate,
                  "rows": rows, "failures": failures}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, separators=(",", ":"))+"\n")
        os.replace(temporary, path)
        print(path, "rows", len(rows), "positive_send",
              sum(row["decision_value"] > 0 for row in rows))
        return
    seeds = sorted({row["seed"] for row in rows})
    cut = min(len(seeds)-1, max(1, len(seeds)//2))
    train = [row for row in rows if row["seed"] in seeds[:cut]]
    test = [row for row in rows if row["seed"] in seeds[cut:]]
    report = {"scope": "disposable restricted A4 with one executed A5 heuristic scan",
              "continuation": "identical local region-priority heuristic in both branches; no automatic acquired-evidence rebroadcast",
              "train_seeds": seeds[:cut], "test_seeds": seeds[cut:],
              "failures": failures, "rows": rows,
              "positive_send": sum(row["decision_value"] > 0 for row in rows),
              "sender_only": model_report(train, test, "actor_features"),
              "recipient_conditioned": model_report(train, test,
                                                     "recipient_features")}
    path.write_text(json.dumps(report, indent=2)+"\n")
    fig_path = args.out_dir / f"downstream_value_vs_gap_{args.tag}.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for overlap in (False, True):
        for changed in (False, True):
            subset = [row for row in rows if
                      row["receiver_direct_overlap_evaluator_only"] == overlap
                      and (row["hold"]["choice_region"] !=
                           row["send"]["choice_region"]) == changed]
            if subset:
                ax.scatter([row["delta_tau"] for row in subset],
                           [row["decision_value"] for row in subset],
                           color="tab:orange" if overlap else "tab:blue",
                           marker="^" if changed else "o", alpha=.7,
                           label=f"overlap={overlap}, sensing changes={changed}")
    ax.axhline(0., color="black", linewidth=.8)
    ax.set(xlabel="leader–follower progress gap Δτ (m)",
           ylabel="downstream J(HOLD) − J(SEND)",
           title="A4 value with receiver sensing physically executed")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(path)
    print(fig_path)
    print("positive SEND", report["positive_send"], "/", len(rows))
    for key in ("sender_only", "recipient_conditioned"):
        print(key, {k: v for k, v in report[key].items() if k != "rows"})


if __name__ == "__main__":
    main()
