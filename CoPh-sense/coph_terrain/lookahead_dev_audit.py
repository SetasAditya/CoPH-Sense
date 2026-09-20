"""Disposable paired physical audit for static look-ahead communication.

This evaluates actual VMAS/pH continuations. The exact SEND/HOLD comparison is
an evaluator diagnostic, not a deployed A4 actor. No cases are selected by a
CoPH checkpoint or by whether SEND happens to win.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .algorithm import acquire_to_evidence, post_reading_branch
from .lookahead import LookaheadEnv, LookaheadSpec
from .lookahead_policy import message_features, sender_estimated_novelty
from .physical_audit import AuditChoice


OUT = Path(__file__).resolve().parent / "results" / "lookahead_dev"
VIEW = AuditChoice("scout", "geometry", (15, 27), (18, 27))


def evaluate_case(seed, surface, traction, scout_start=(12, 27)):
    spec = LookaheadSpec(seed, surface=(surface, .06),
                         traction=(traction, .90),
                         scout_start=scout_start)
    initial = LookaheadEnv(spec)
    reading = acquire_to_evidence(initial, VIEW)
    record = {"parent_seed": seed, "surface": surface,
              "traction": traction, "scout_start": scout_start,
              "acquired": reading is not None}
    if reading is None:
        return record
    env = reading.env
    ids = tuple(env.innovation_packets)
    if len(ids) != 1:
        record["reason"] = f"expected one summary, got {len(ids)}"
        return record
    summary_id = ids[0]
    record.update({
        "reading_step": env.step_index,
        "delta_tau_at_reading": env.observations()["scout"]
        ["lookahead"]["delta_tau"],
        "conditional_kl_nats": env.conditional_information_gain(
            "carrier", summary_id),
        "sender_estimated_kl_nats": sender_estimated_novelty(
            env, summary_id),
        "actor_features": message_features(env, summary_id).tolist(),
        "summary_bytes": env.innovation_packets[summary_id].payload_bytes,
        "raw_bytes": 16 + 4 * len(env.owned["scout"][reading.evidence_id].cells),
    })
    branches = {}
    for label, evidence_id, send in (("hold", reading.evidence_id, False),
                                     ("raw", reading.evidence_id, True),
                                     ("innovation", summary_id, True)):
        result, snapshot = post_reading_branch(env, evidence_id, send)
        prefix_length = {agent: float(sum(np.linalg.norm(
            np.asarray(b["positions"][agent]) -
            np.asarray(a["positions"][agent]))
            for a, b in zip(snapshot.state_trace[:-1],
                            snapshot.state_trace[1:])))
            for agent in ("scout", "carrier")}
        # run_continuation clones the entire state_trace, so its path_length
        # already includes the acquisition and delivery prefix.
        full_path = {agent: result.get("path_length", {}).get(
                         agent, prefix_length[agent])
                     for agent in ("scout", "carrier")}
        branches[label] = {"cost": result["score_single_world"],
                           "success": result["success"],
                           "mission_cost": result["mission_cost"],
                           "risk_exposure": result["material_exposure"],
                           "communication_cost": result.get("ledger", {}).get(
                               "communication", snapshot.ledger.communication),
                           "sensing_cost": result.get("ledger", {}).get(
                               "sensing", snapshot.ledger.sensing),
                           "path_length": full_path,
                           "delivered": bool(send and evidence_id in
                                             snapshot.received["carrier"]),
                           "carrier_effective_lookahead": snapshot.observations()
                           ["carrier"]["lookahead"]["effective_lookahead"],
                           "carrier_local_lookahead": snapshot.observations()
                           ["carrier"]["lookahead"]["local_lookahead"],
                           "support": snapshot.support_metrics()}
    record["branches"] = branches
    record["decision_value"] = (branches["hold"]["cost"] -
                                 branches["innovation"]["cost"])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=980000)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--scout-xs", type=int, nargs="+", default=[12])
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    records = []
    for seed in range(args.start, args.start + args.count):
        for scout_x in args.scout_xs:
            for surface, traction in ((.04, .90), (.95, .20)):
                row = evaluate_case(seed, surface, traction,
                                    (scout_x, 27))
                records.append(row)
                print(seed, scout_x, surface,
                      round(row.get("decision_value", 0.), 5),
                      row["acquired"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"audit_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({
        "status": "disposable development audit",
        "oracle_scope": "paired complete physical continuation, realized world",
        "records": records,
    }, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
