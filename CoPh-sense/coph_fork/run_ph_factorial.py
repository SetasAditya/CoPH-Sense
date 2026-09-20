#!/usr/bin/env python3
"""Paired information-policy x direct/pH execution on the frozen two-fork map."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .ph_factorial_controller import ForceConfig, execute_step
from .ph_factorial_environment import FactorialEnv
from .run_two_fork_memory_bridge import a4_message_ids, protocol_posterior
from .run_memory_a5 import load_models
from .two_fork_environment import TwoForkAction, TwoForkConfig, TwoForkWorld
from .two_fork_memory import CarrierA5Adapter, delivered_ledger


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "ph_factorial"


def carrier_target(env):
    q = env.agent_positions["carrier"]
    x = float(q[0])
    evidence = env.observations()["carrier"]["evidence"]
    def high(region):
        item = evidence[region]["top"]
        return (item["geometry"] is True and item["traction"] is not None
                and item["traction"] >= env.config.traction_safe_threshold)
    if x < -.20:
        y = .43 if high(1) else env.config.backup_y1
        return ((-.82, y) if x < -.58 and abs(float(q[1]) - y) > .10
                else (-.12, y))
    if x < -.10:
        return (.02, .43 if high(1) else env.config.backup_y1)
    if x < .24:
        y = .43 if high(2) else env.config.backup_y2
        # Complete the lane switch before the second island begins at x=.28.
        return ((-.04, y) if abs(float(q[1]) - y) > .08 else (.32, y))
    if x < .65:
        return (.70, .43 if high(2) else env.config.backup_y2)
    return (1.17, .17 if env.route_commitments[2] == "top" else -.17)


def scout_target(env, phase):
    if phase == "probe":
        return env.probe_position(1, "top")
    x = float(env.agent_positions["scout"][0])
    if x < -.77:
        return (-.72, .72)
    if x < -.20:
        return (-.12, .72)
    if x < .22:
        return (.25, .72)
    if x < .67:
        return (.70, .72)
    return (1.17, -.17 if env.route_commitments[2] == "top" else .17)


def advance(env, phase="navigate", carrier_override=None, scout_sense=None,
            scout_send=None, carrier_sense=None):
    scout = TwoForkAction()
    carrier = TwoForkAction()
    if scout_sense:
        scout.sense_region, scout.sense_route, scout.sense_modality = scout_sense
    if carrier_sense:
        carrier.sense_region, carrier.sense_route, carrier.sense_modality = carrier_sense
    scout.send_observation_id = scout_send
    targets = {"scout": scout_target(env, phase),
               "carrier": (carrier_target(env) if carrier_override is None
                           else carrier_override)}
    return env.step({"scout": scout, "carrier": carrier}, targets)


def public_viewpoint_forecast(env, region):
    """No-hidden-terrain rollout with the deployed integrator and unit traction.

    It estimates reaching the viewpoint and retaining positive pre-Fork-1
    slack. A physical guard also checks the actual commitment during execution.
    """
    actor = env.physics.world.agents[1]
    q = env.agent_positions["carrier"].astype(float)
    v = actor.state.vel[0].detach().cpu().numpy().copy().astype(float)
    target = env.probe_position(region, "top")
    for k in range(400):
        if np.linalg.norm(q - target) <= .10:
            return {"reachable": True, "travel_steps": k,
                    "precommit_x_margin": float(env.config.decision_x - q[0])}
        q, v, _ = execute_step(q, v, target, env, "carrier", env.scheme,
                               env.force_config, float(actor.mass),
                               float(actor.shape.radius), float(actor.max_speed), 1.)
        if q[0] >= env.config.decision_x:
            break
    return {"reachable": False, "travel_steps": k + 1,
            "precommit_x_margin": float(env.config.decision_x - q[0])}


def run_episode(world, info_policy, scheme, seed, config, force_config,
                a4_model, a5_adapter, ordinary_action=((1, "geometry"), (1, "traction"))):
    env = FactorialEnv(config=config, world=world, seed=seed, scheme=scheme,
                       force_config=force_config)
    hold = (-1.08, 0.)
    probe = env.probe_position(1, "top")
    while np.linalg.norm(env.agent_positions["scout"] - probe) > .10:
        advance(env, "probe", hold)
        if env.done:
            raise RuntimeError("scout failed to reach R1 probe")
    for modality in ("geometry", "traction"):
        advance(env, "probe", hold, scout_sense=(1, "top", modality))
    if info_policy == "ordinary":
        selected = list(env.evidence["scout"])
    else:
        selected = a4_message_ids(env, a4_model,
                                  1 - config.packet_drop_probability)
    for observation_id in selected:
        advance(env, "probe", hold, scout_send=observation_id)
    for _ in range(config.communication_delay_steps + 2):
        advance(env, "probe", hold)
    if info_policy == "ordinary":
        chosen = ordinary_action
    else:
        memory = delivered_ledger(env)
        posterior = protocol_posterior(memory, "persistent", a4_model, config)
        chosen = a5_adapter.choose(memory, posterior=posterior)
    forecast = (public_viewpoint_forecast(env, chosen[0][0]) if chosen
                else {"reachable": True, "travel_steps": 0,
                      "precommit_x_margin": float(config.decision_x - env.agent_positions["carrier"][0])})
    if chosen and forecast["reachable"]:
        viewpoint = env.probe_position(chosen[0][0], "top")
        while np.linalg.norm(env.agent_positions["carrier"] - viewpoint) > .10:
            advance(env, "navigate", viewpoint)
            if env.done or env.route_commitments[1] is not None:
                raise RuntimeError("carrier missed actionable viewpoint")
        for region, modality in chosen:
            advance(env, "navigate", viewpoint,
                    carrier_sense=(region, "top", modality))
    elif chosen:
        chosen = ()
    while not env.done:
        advance(env)
    energies = [row["carrier"]["energy_after"] for row in env.energy_trace]
    acquisitions = [e for e in env.events if e["type"] == "acquisition" and e["valid"]]
    return {"info_policy": info_policy, "scheme": scheme, "seed": seed,
            "world": asdict(world), "success": env.success,
            "team_cost": env.ledger.total, "steps": env.step_index,
            "ledger": asdict(env.ledger), "routes": env.route_commitments,
            "chosen_acquisition": [list(v) for v in chosen],
            "sent_count": len(selected), "forecast": forecast,
            "valid_acquisitions": len(acquisitions),
            "collision_cost": env.ledger.collision,
            "collisions": [e for e in env.events if e["type"] == "collision"],
            "final_positions": {k: v.tolist() for k, v in env.agent_positions.items()},
            "energy_start": energies[0], "energy_end": energies[-1],
            "energy_max": max(energies)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    layout = json.loads((HERE / "results/a5_moving_bridge/physical_search/frozen_physical_layout_v1.json").read_text())
    p = layout["layout"]
    config = TwoForkConfig(horizon=layout["horizon"], backup_y1=p["backup_y1"],
                           backup_y2=p["backup_y2"], sensing_cost=p["sensing_cost"])
    _, a4 = load_models(1701, 1801)
    a5 = CarrierA5Adapter()
    force = ForceConfig()
    worlds = [TwoForkWorld(geometry_1=bool((i >> 0) & 1),
                           traction_1=.9 if (i >> 1) & 1 else .3,
                           geometry_2=bool((i >> 2) & 1),
                           traction_2=.9 if (i >> 3) & 1 else .3,
                           backup_traction_1=p["backup_traction_1"],
                           backup_traction_2=p["backup_traction_2"])
              for i in range(args.start, min(args.start + args.worlds, 16))]
    rows = []
    cells = (("ordinary", "direct"), ("ordinary", "ph")) if args.calibrate else (
        ("ordinary", "direct"), ("ordinary", "ph"),
        ("coph", "direct"), ("coph", "ph"))
    for i, world in enumerate(worlds, start=args.start):
        for info_policy, scheme in cells:
            row = run_episode(world, info_policy, scheme, 1701 + i, config,
                              force, a4, a5)
            rows.append(row)
            print(i, info_policy, scheme, row["success"],
                  round(row["team_cost"], 4), row["steps"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("controller_calibration.json" if args.calibrate else "paired_factorial.json")
    path.write_text(json.dumps({"config": asdict(config),
                                "force_config": asdict(force), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
