"""Recipient-conditioned A4 audit on the frozen static look-ahead task.

Only the sender-visible feature map is used at deployment. Paired physical
continuations and the receiver's private state are evaluator-only diagnostics.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .algorithm import acquire_to_evidence, post_reading_branch
from .environment import TerrainAction, cell_to_position
from .lookahead import LookaheadEnv, LookaheadSpec, REGIONS
from .lookahead_dev_audit import VIEW
from .lookahead_policy import (RidgeMessageValue, message_features,
                               choose_region_to_inspect)
from .pilot_environment import waypoint
from .physical_audit import AuditChoice
from .planning import plan_path


OUT = Path(__file__).resolve().parent / "results" / "lookahead_recipient_a4"
FEATURE_NAMES = ("bias", "innovation_mean", "innovation_precision",
                 "recipient_prior_variance", "acked_same_region",
                 "acked_same_provenance", "receiver_progress", "progress_gap",
                 "distance_to_region", "distance_to_wall", "message_age",
                 "receiver_near_region", "decision_risk_shift", "delivery_slack",
                 "radio_cost", "payload_bytes")


def recipient_features(env, packet_id):
    """Estimate K^{sender->receiver} without reading receiver private maps.

    Public teammate pose is part of sender kinematics. ACKs establish only
    receipt of previous packets, never unreported private measurements.
    """
    packet = env.innovation_packets[packet_id]
    sender = packet.source
    obs = env.observations()[sender]
    kin = np.asarray(obs["kinematics"], dtype=float)
    own_position = kin[:2]
    receiver_position = own_position + kin[4:6]
    x0, x1, y0, y1 = REGIONS[packet.region]
    region_center = cell_to_position(((x0 + x1) // 2, (y0 + y1) // 2))
    wall_position = cell_to_position((16, 30))[0]
    acked_ids = obs["acknowledged_evidence_ids"]
    same_region = sum(1 for item in acked_ids
                      if item in env.innovation_packets and
                      env.innovation_packets[item].region == packet.region and
                      env.innovation_packets[item].modality == packet.modality)
    same_provenance = any(item in env.innovation_packets and
                          env.innovation_packets[item].provenance == packet.provenance
                          for item in acked_ids)
    receiver_distance = float(np.linalg.norm(receiver_position-region_center))
    distance_to_wall = float(wall_position-receiver_position[0])
    # The public prior is the only guaranteed receiver belief absent ACKs.
    prior_variance = 1. / (4. + sum(
        env.innovation_packets[item].wire_factor()[0] for item in acked_ids
        if item in env.innovation_packets and
        env.innovation_packets[item].region == packet.region and
        env.innovation_packets[item].modality == packet.modality))
    mean_shift = packet.posterior_mean-packet.prior_mean
    near_route = float(receiver_distance < 3. and distance_to_wall > -.5)
    estimated_steps_to_wall = max(0., distance_to_wall) / .06 / env.config.dt
    delivery_slack = (estimated_steps_to_wall -
                      env.config.communication_delay_steps)
    radio_cost = (2 * env.config.communication_attempt_cost +
                  env.config.byte_cost * (env.config.header_bytes +
                                          packet.payload_bytes +
                                          env.config.ack_bytes))
    return np.asarray((1., packet.posterior_mean,
                       packet.delta_precision / 400., prior_variance,
                       float(same_region), float(same_provenance),
                       receiver_position[0],
                       own_position[0]-receiver_position[0],
                       receiver_distance, distance_to_wall,
                       (env.step_index-packet.acquired_step)*env.config.dt,
                       near_route, mean_shift*near_route,
                       delivery_slack*env.config.dt, radio_cost,
                       packet.payload_bytes/1024.), dtype=np.float64)


def response_features(env, packet_id):
    """Predict the receiver's next scan from sender-visible public knowledge.

    The estimate starts from the declared public prior and incorporates only
    ACKed innovation factors. It deliberately cannot see private carrier
    scans. The same region-priority rule is applied before and after the
    candidate message, without a hidden-world rollout.
    """
    from .lookahead import GaussianRegionBelief, region_cells
    from .lookahead_policy import LOOKAHEAD_VIEWS

    packet = env.innovation_packets[packet_id]
    local = env.observations()[packet.source]
    kin = np.asarray(local["kinematics"], dtype=float)
    receiver_position = kin[:2] + kin[4:6]
    acked = local["acknowledged_evidence_ids"]
    beliefs = [GaussianRegionBelief(4., 4. * .35)
               for _ in REGIONS]
    acked_provenance = set()
    for item in acked:
        old = env.innovation_packets.get(item)
        if old is None or old.modality != "geometry":
            continue
        if old.provenance in acked_provenance:
            continue
        acked_provenance.add(old.provenance)
        beliefs[old.region].add(*old.wire_factor())

    def priority(region, variance):
        view = LOOKAHEAD_VIEWS[region][0]
        travel = float(np.linalg.norm(cell_to_position(view)-receiver_position))
        return variance/(.25+travel), travel

    hold_scores = [priority(i, b.variance)[0]
                   for i, b in enumerate(beliefs)]
    hold_action = int(np.argmax(hold_scores))
    send_variances = [b.variance for b in beliefs]
    if (packet.modality == "geometry" and
            packet.provenance not in acked_provenance):
        factor_precision, _ = packet.wire_factor()
        send_variances[packet.region] = 1. / (
            beliefs[packet.region].precision + factor_precision)
    send_scores = [priority(i, variance)[0]
                   for i, variance in enumerate(send_variances)]
    send_action = int(np.argmax(send_scores))
    hold_travel = priority(hold_action,
                           beliefs[hold_action].variance)[1]
    send_travel = priority(send_action,
                           send_variances[send_action])[1]
    return np.concatenate((recipient_features(env, packet_id), np.asarray((
        float(hold_action), float(send_action),
        float(hold_action != send_action),
        float(hold_action == packet.region and
              send_action != packet.region),
        float(len(region_cells(send_action)) if
              send_action != packet.region else 0),
        send_travel-hold_travel,
        hold_scores[0]-hold_scores[1],
        send_scores[0]-send_scores[1],
    ), dtype=np.float64)))


def advance_without_message(env, steps):
    env = env.clone()
    targets = {agent: waypoint(env, agent) for agent in ("scout", "carrier")}
    for _ in range(steps):
        if env.done:
            break
        if env.step_index % 20 == 0:
            targets = {agent: waypoint(env, agent)
                       for agent in ("scout", "carrier")}
        env.step({agent: TerrainAction(waypoint=targets[agent])
                  for agent in ("scout", "carrier")})
    return env


def branch_cost(snapshot, packet_id, delay=None):
    if delay is not None:
        snapshot = snapshot.clone()
        snapshot.config = replace(snapshot.config,
                                  communication_delay_steps=delay)
    result, delivery_state = post_reading_branch(snapshot, packet_id, True)
    radio = float(result.get("ledger", {}).get(
        "communication", delivery_state.ledger.communication))
    accepted = [event for event in delivery_state.events if
                event["type"] == "transmission" and
                event["evidence_id"] == packet_id and event["accepted"]]
    payload = (snapshot.config.byte_cost *
               snapshot.innovation_packets[packet_id].payload_bytes *
               len(accepted))
    return {"cost": float(result["score_single_world"]),
            "radio": radio, "payload_radio": payload,
            "fixed_header_ack_radio": radio-payload,
            "success": bool(result["success"]),
            "delivered": packet_id in delivery_state.received["carrier"],
            "delivery_step": next((event["step"] for event in
                                   delivery_state.events if
                                   event["type"] == "packet_delivery" and
                                   event["evidence_id"] == packet_id and
                                   event["kind"] == "evidence" and
                                   event["delivered"]), None)}


def evaluate_state(env, packet_id, meta):
    packet = env.innovation_packets[packet_id]
    hold, _ = post_reading_branch(env, packet_id, False)
    held = float(hold["score_single_world"])
    actual = branch_cost(env, packet_id)
    immediate = branch_cost(env, packet_id, delay=0)
    n0 = immediate["cost"]-immediate["radio"]
    nd = actual["cost"]-actual["radio"]
    physical_gain = held-n0
    delay_penalty = nd-n0
    value = held-actual["cost"]
    if abs(value-(physical_gain-delay_penalty-actual["radio"])) > 1e-7:
        raise AssertionError("SEND value decomposition does not close")
    receiver = env.observations()["carrier"]
    before_path, _ = plan_path(receiver, goal=(55, 29))
    delivered_copy = env.clone()
    delivered_copy._apply_evidence(
        "carrier", delivered_copy.owned[packet.source][packet_id])
    after_path, _ = plan_path(delivered_copy.observations()["carrier"],
                              goal=(55, 29))
    def lane_at_region(path):
        return next((cell[1] for cell in path if cell[0] >= 18), None)
    route_changed = lane_at_region(before_path) != lane_at_region(after_path)
    before_scan = choose_region_to_inspect(env, "carrier")
    after_scan = choose_region_to_inspect(delivered_copy, "carrier")
    sensing_changed = (None if before_scan is None else before_scan.target_region) != (
        None if after_scan is None else after_scan.target_region)
    actual_receiver_var = receiver["lookahead"]["local_region_beliefs"][
        f"{packet.region}:{packet.modality}"][1]
    return {**meta, "step": env.step_index,
            "actor_features": message_features(env, packet_id).tolist(),
            "recipient_features": recipient_features(env, packet_id).tolist(),
            "decision_value": value, "hold_cost": held,
            "send": actual, "immediate_send": immediate,
            "physical_gain_zero_delay": physical_gain,
            "delay_penalty_nonradio": delay_penalty,
            "radio_cost": actual["radio"],
            "actual_receiver_variance_evaluator_only": actual_receiver_var,
            "receiver_acked_overlap": recipient_features(env, packet_id)[4],
            "route_changed_evaluator_only": route_changed,
            "sensing_changed_evaluator_only": sensing_changed,
            "receiver_direct_overlap_evaluator_only": bool(
                env.local_support["carrier"] & set(packet.support_cells)),
            "delta_tau": env.observations()["scout"]["lookahead"]["delta_tau"]}


def make_pool(start=981000, seeds=12, ages=(0, 8, 16),
              evaluator=evaluate_state, schedule_origin=None, verbose=True,
              include_acked_duplicate=False):
    rows = []
    failures = []
    if schedule_origin is None:
        schedule_origin = start
    for seed in range(start, start+seeds):
        scout_x = (10, 12, 14)[(seed-schedule_origin) % 3]
        for risk_name, surface, traction in (("safe", .04, .90),
                                             ("risky", .95, .20)):
            spec = LookaheadSpec(seed, surface=(surface, .06),
                                 traction=(traction, .90),
                                 scout_start=(scout_x, 27))
            acquired = acquire_to_evidence(LookaheadEnv(spec), VIEW)
            if acquired is None:
                failures.append({"seed": seed, "risk": risk_name,
                                 "reason": "scout acquisition failed"})
                continue
            base = acquired.env
            packet_id = next(iter(base.innovation_packets))
            for age in ages:
                state = advance_without_message(base, age)
                if state.done:
                    continue
                meta = {"seed": seed, "risk": risk_name,
                        "scout_start_x": scout_x, "age_steps": age,
                        "receiver_condition": "natural"}
                rows.append(evaluator(state, packet_id, meta))
                if verbose:
                    print(seed, risk_name, age,
                          round(rows[-1]["decision_value"], 5), flush=True)
            # A physically obtained private receiver scan tests whether a
            # sender can cope with overlap it has not learned via an ACK.
            if (seed-schedule_origin) % 3 == 0:
                carrier_view = AuditChoice("carrier", "geometry",
                                           (15, 27), (18, 27))
                overlapped = acquire_to_evidence(base, carrier_view)
                if overlapped is None:
                    failures.append({"seed": seed, "risk": risk_name,
                                     "reason": "carrier overlap acquisition failed"})
                elif not overlapped.env.done:
                    meta = {"seed": seed, "risk": risk_name,
                            "scout_start_x": scout_x, "age_steps":
                            overlapped.env.step_index-base.step_index,
                            "receiver_condition": "private_direct_overlap"}
                    rows.append(evaluator(overlapped.env, packet_id, meta))
            # A delivered packet followed by its ACK is a sender-visible
            # duplicate candidate. This varies receiver knowledge without
            # exposing the receiver's private scans to the sender.
            if include_acked_duplicate and (seed-schedule_origin) % 3 == 1:
                _, acked = post_reading_branch(base, packet_id, True,
                                               continue_mission=False)
                acked = advance_without_message(
                    acked, max(1, acked.config.communication_delay_steps)+1)
                visible_acks = acked.observations()["scout"][
                    "acknowledged_evidence_ids"]
                if packet_id not in visible_acks:
                    failures.append({"seed": seed, "risk": risk_name,
                                     "reason": "duplicate ACK unavailable"})
                elif not acked.done:
                    meta = {"seed": seed, "risk": risk_name,
                            "scout_start_x": scout_x, "age_steps":
                            acked.step_index-base.step_index,
                            "receiver_condition": "acked_duplicate"}
                    rows.append(evaluator(acked, packet_id, meta))
    return rows, failures


def model_report(train, test, feature_key):
    fitted = RidgeMessageValue.fit([
        {"actor_features": row[feature_key],
         "decision_value": row["decision_value"]} for row in train])
    results = []
    for row in test:
        predicted = fitted.predict(row[feature_key])
        actual = row["decision_value"]
        selected_send = predicted > 0
        results.append({"seed": row["seed"], "risk": row["risk"],
                        "age_steps": row["age_steps"],
                        "predicted": predicted, "actual": actual,
                        "selected_send": selected_send,
                        "regret": (max(0., actual) if not selected_send else
                                   max(0., -actual))})
    positive = [row for row in results if row["actual"] > 0]
    return {"features": feature_key,
            "mean_regret": float(np.mean([r["regret"] for r in results])),
            "mean_squared_value_error": float(np.mean([
                (r["predicted"]-r["actual"])**2 for r in results])),
            "send_count": sum(r["selected_send"] for r in results),
            "positive_count": len(positive),
            "positive_recall": (sum(r["selected_send"] for r in positive) /
                                len(positive) if positive else None),
            "false_send_count": sum(r["selected_send"] and r["actual"] <= 0
                                    for r in results),
            "rows": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=981000)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pool_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    rows, failures = make_pool(args.start, args.seeds)
    unique = sorted({row["seed"] for row in rows})
    cut = min(len(unique)-1, max(1, len(unique)//2))
    train_seeds = unique[:cut]
    train = [row for row in rows if row["seed"] in train_seeds]
    test = [row for row in rows if row["seed"] not in train_seeds]
    if not test:
        raise RuntimeError("need seed-disjoint test rows")
    report = {"scope": "disposable A4, forced scout geometry on fixed lookahead v1",
              "train_seeds": train_seeds,
              "test_seeds": unique[cut:], "failures": failures,
              "feature_names": FEATURE_NAMES,
              "rows": rows,
              "sender_only": model_report(train, test, "actor_features"),
              "recipient_conditioned": model_report(train, test,
                                                     "recipient_features")}
    path.write_text(json.dumps(report, indent=2)+"\n")
    plot_path = OUT / f"value_vs_gap_{args.tag}.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for overlap in (False, True):
        for changed in (False, True):
            subset = [row for row in rows if
                      row["receiver_direct_overlap_evaluator_only"] == overlap
                      and (row["route_changed_evaluator_only"] or
                           row["sensing_changed_evaluator_only"]) == changed]
            if subset:
                ax.scatter([row["delta_tau"] for row in subset],
                           [row["decision_value"] for row in subset],
                           c="tab:orange" if overlap else "tab:blue",
                           marker="^" if changed else "o", alpha=.65,
                           label=f"overlap={overlap}, route/sensing change={changed}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="leader–follower progress gap Δτ (m)",
           ylabel="J(HOLD) − J(SEND)",
           title="Frozen Lookahead v1: recipient-conditioned SEND value")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(path)
    print(plot_path)
    for key in ("sender_only", "recipient_conditioned"):
        print(key, {k: v for k, v in report[key].items() if k != "rows"})


if __name__ == "__main__":
    main()
