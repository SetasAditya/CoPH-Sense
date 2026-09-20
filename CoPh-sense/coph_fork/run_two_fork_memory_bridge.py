#!/usr/bin/env python3
"""Minimal moving two-fork A5 bridge with frozen information actors."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from coph_fork.memory_critic_a5 import WORLDS, posterior_for
from coph_fork.nested_oracle import learned_message_set
from coph_fork.oracle import ExactOracleConfig, default_prior
from coph_fork.run_memory_a5 import load_models
from coph_fork.run_moving_bridge_a65 import control, state
from coph_fork.share_oracle import ExactShareOracle, ShareState
from coph_fork.two_fork_environment import (
    TwoForkAction, TwoForkConfig, TwoForkEnv, TwoForkWorld,
)
from coph_fork.two_fork_memory import CarrierA5Adapter, actor_ledger, delivered_ledger
from coph_fork.two_fork_motion import carrier_motion, scout_motion
from coph_fork.memory_a5 import CANDIDATES


OUT = Path(__file__).resolve().parent / "results" / "a5_moving_bridge"


def hold_carrier(env):
    position, velocity = state(env, "carrier")
    return control(position, velocity, (-1.08, 0.), gain=2.4, damping=1.4)


def step(env, scout_phase="navigate", carrier_target=None,
         scout_sense=None, scout_send=None, carrier_sense=None):
    scout = TwoForkAction(motion=scout_motion(env, scout_phase))
    if scout_sense:
        scout.sense_region, scout.sense_route, scout.sense_modality = scout_sense
    if scout_send:
        scout.send_observation_id = scout_send
    if carrier_target is None:
        carrier_command = carrier_motion(env)
    elif isinstance(carrier_target, str) and carrier_target == "hold":
        carrier_command = hold_carrier(env)
    else:
        position, velocity = state(env, "carrier")
        carrier_command = control(position, velocity, carrier_target,
                                  gain=2.8, damping=1.5)
    carrier = TwoForkAction(motion=carrier_command)
    if carrier_sense:
        carrier.sense_region, carrier.sense_route, carrier.sense_modality = carrier_sense
    return env.step({"scout": scout, "carrier": carrier})


def stage_record(env, phase, condition, budget):
    received = delivered_ledger(env)
    acquisition = actor_ledger(received, condition)
    return {
        "phase": phase, "step": env.step_index,
        "positions": {key: value.tolist() for key, value in env.agent_positions.items()},
        "delivered_evidence": [asdict(record) for record in received.records],
        "acquisition_memory": [asdict(record) for record in acquisition.records],
        "evidence_only_posterior": posterior_for(acquisition).tolist(),
        "acks": {key: sorted(value) for key, value in env.acknowledged.items()},
        "remaining_measurements": budget,
        "route_commitments": dict(env.route_commitments),
        "ledger": asdict(env.ledger),
    }


def a4_share_config(config, delivery_probability):
    return ExactOracleConfig(
        sensing_cost=config.sensing_cost,
        communication_attempt_cost=config.communication_attempt_cost,
        byte_cost=config.byte_cost,
        header_bytes=config.header_bytes,
        payload_bytes=config.payload_bytes,
        ack_payload_bytes=config.ack_payload_bytes,
        timely_delivery_probability=delivery_probability)


def a4_selected_names(model, config, values, delivery_probability):
    oracle = ExactShareOracle(
        ShareState(acquired=tuple((name, values[name])
                                  for name in ("top_geometry", "top_traction")),
                   timely_probability=delivery_probability),
        config=a4_share_config(config, delivery_probability),
        prior=default_prior(),
    )
    return tuple(learned_message_set(model, oracle))


def a4_message_ids(env, model, delivery_probability):
    values, ids = {}, {}
    for observation_id, item in env.evidence["scout"].items():
        if item.region != 1 or item.route != "top":
            continue
        name = "top_" + item.modality
        values[name] = (bool(item.value) if item.modality == "geometry"
                        else bool(float(item.value) >= env.config.traction_safe_threshold))
        ids[name] = observation_id
    if set(values) != {"top_geometry", "top_traction"}:
        raise RuntimeError("A4 must run after both valid R1 readings")
    return [ids[name] for name in a4_selected_names(
        model, env.config, values, delivery_probability)]


def protocol_posterior(memory, condition, model, config, prior=None):
    """Bayes update for A4's selected messages, including informative silence.

    This bridge's fixed pre-fork phase has a public send schedule. A sent
    reading arrives independently with the declared packet probability.
    Neither actual hidden terrain nor failed send attempts are observed.
    """
    base = (np.full(16, 1 / 16, dtype=np.float64) if prior is None
            else np.asarray(prior, dtype=np.float64))
    if base.shape != (16,) or np.any(base < 0) or not np.isclose(base.sum(), 1):
        raise ValueError("prior must be a normalized 16-world distribution")
    if condition in ("independent", "shared_reset"):
        return base.copy()
    if condition == "shuffled":
        return posterior_for(memory, base)
    delivered = {"top_" + item.modality: bool(item.value)
                 for item in memory.records if item.region == 1}
    p = 1 - config.packet_drop_probability
    masses = np.zeros(16, dtype=np.float64)
    for wi, bits in enumerate(WORLDS):
        values = {"top_geometry": bool(bits[0]),
                  "top_traction": bool(bits[1])}
        selected = set(a4_selected_names(model, config, values, p))
        if any(name not in selected or values[name] != value
               for name, value in delivered.items()):
            continue
        likelihood = p ** len(delivered) * (1 - p) ** (len(selected) - len(delivered))
        masses[wi] = likelihood * base[wi]
    if masses.sum() <= 0:
        raise ValueError("delivered evidence is impossible under A4 protocol")
    return masses / masses.sum()


def run_episode(world, condition, seed=1701, config=None,
                a4_model=None, a5_adapter=None, save_replay=None,
                forced_action=None, oracle_reference=None, prior=None):
    if condition not in ("independent", "shared_reset", "persistent", "shuffled",
                         "oracle_memory"):
        raise ValueError("unsupported bridge condition")
    config = config or TwoForkConfig()
    if a4_model is None:
        _, a4_model = load_models(1701, 1801)
    if a5_adapter is None and forced_action is None:
        a5_adapter = CarrierA5Adapter()
    env = TwoForkEnv(config, world, seed)
    stages = []
    budget = 2
    stages.append(stage_record(env, "APPROACH_R1", condition, budget))
    viewpoint = TwoForkEnv.probe_position(1, "top")
    while np.linalg.norm(state(env, "scout")[0] - viewpoint) > .10:
        step(env, "probe_1", "hold")
        if env.done:
            raise RuntimeError("scout failed to reach Region-1 viewpoint")
    stages.append(stage_record(env, "SCOUT_R1_DECISION", condition, budget))
    # The acquisition gate is actionable here: the carrier remains at the
    # public pre-fork staging state and the scout is in radio range.
    for modality in ("geometry", "traction"):
        step(env, "probe_1", "hold", scout_sense=(1, "top", modality))
    stages.append(stage_record(env, "SCOUT_R1_ACQUIRE", condition, budget))
    selected = ([] if condition == "independent" else
                a4_message_ids(env, a4_model, 1 - config.packet_drop_probability))
    for observation_id in selected:
        step(env, "probe_1", "hold", scout_send=observation_id)
    for _ in range(config.communication_delay_steps + 2):
        step(env, "probe_1", "hold")
    stages.append(stage_record(env, "SCOUT_R1_SHARE", condition, budget))
    valid_memory = delivered_ledger(env)
    acquisition_memory = actor_ledger(valid_memory, condition)
    posterior = protocol_posterior(acquisition_memory, condition, a4_model,
                                   config, prior)
    if forced_action is not None:
        chosen = tuple(forced_action)
    elif condition == "oracle_memory":
        if oracle_reference is None:
            raise ValueError("oracle_memory requires a frozen physical oracle reference")
        signature = sorted((item.region, item.modality, bool(item.value))
                           for item in acquisition_memory.records)
        key = json.dumps([list(item) for item in signature])
        index = oracle_reference["by_memory_signature"][key]["best_action_index"]
        chosen = CANDIDATES[index]
    else:
        chosen = a5_adapter.choose(acquisition_memory, posterior=posterior)
    stages.append(stage_record(env, "CARRIER_ACQUISITION_DECISION", condition, budget))
    stages[-1]["protocol_posterior"] = posterior.tolist()
    if chosen:
        region = chosen[0][0]
        viewpoint = TwoForkEnv.probe_position(region, "top")
        while np.linalg.norm(state(env, "carrier")[0] - viewpoint) > .10:
            step(env, "navigate", viewpoint)
            if env.done or env.route_commitments[1] is not None:
                raise RuntimeError("carrier sensing viewpoint became unreachable before Fork 1")
        for _, modality in chosen:
            step(env, "navigate", viewpoint,
                 carrier_sense=(region, "top", modality))
            budget -= 1
    stages.append(stage_record(env, "CARRIER_ACQUIRE", condition, budget))
    while not env.done:
        step(env)
    stages.append(stage_record(env, "FINISH", condition, budget))
    if save_replay is not None:
        env.save_replay(save_replay)
    return {
        "condition": condition, "world": asdict(world),
        "success": env.success, "team_cost": env.ledger.total,
        "ledger": asdict(env.ledger), "steps": env.step_index,
        "routes": dict(env.route_commitments),
        "chosen_acquisition": [list(atom) for atom in chosen],
        "sent_ids": selected,
        "received_ids": sorted(env.received["carrier"]),
        "stage_log": stages,
        "events": env.events,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="symmetric_v2")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"canonical_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    world = TwoForkWorld()
    _, a4 = load_models(1701, 1801)
    a5 = CarrierA5Adapter()
    oracle_path = OUT / "physical_oracle_all_groups_symmetric_v2.json"
    oracle_reference = json.loads(oracle_path.read_text()) if oracle_path.exists() else None
    conditions = ["independent", "shared_reset", "persistent", "shuffled"]
    if oracle_reference is not None:
        conditions.append("oracle_memory")
    rows = [run_episode(world, condition, a4_model=a4,
                        a5_adapter=a5,
                        oracle_reference=oracle_reference,
                        save_replay=OUT / f"{condition}_canonical_{args.tag}_replay.json")
            for condition in conditions]
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print([(r["condition"], r["chosen_acquisition"], r["team_cost"],
            r["success"], r["ledger"]["collision"]) for r in rows])


if __name__ == "__main__":
    main()
