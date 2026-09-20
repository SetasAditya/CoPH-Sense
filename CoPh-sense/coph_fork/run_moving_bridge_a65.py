#!/usr/bin/env python3
"""A6.5 one-fork VMAS bridge for the frozen timing gate and A4 sharing actor.

This is a moving-agent timing test, not the two-region A5 or pH benchmark.
"""

from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.environment import CoPHForkEnv, ForkAction, ForkConfig, ForkWorld  # noqa: E402
from coph_fork.nested_oracle import learned_message_set  # noqa: E402
from coph_fork.oracle import ExactOracleConfig, default_prior  # noqa: E402
from coph_fork.run_memory_a5 import load_models  # noqa: E402
from coph_fork.share_oracle import ExactShareOracle, ShareState  # noqa: E402
from coph_fork.timing_a6 import TimingContext  # noqa: E402
from coph_fork.train_timing_gate_a6 import ScoutTimingGate  # noqa: E402


OUT = HERE / "results" / "a65_moving_bridge"


def control(position, velocity, target, gain=2.8, damping=1.4, scale=1.0):
    command = (gain * (np.asarray(target, dtype=np.float32) - position)
               - damping * velocity)
    norm = np.linalg.norm(command)
    return tuple((scale * command / max(1.0, norm)).tolist())


def state(env, name):
    actor = env.physics.world.agents[0 if name == "scout" else 1]
    return (actor.state.pos[0].detach().cpu().numpy().copy(),
            actor.state.vel[0].detach().cpu().numpy().copy())


def forecast_commit_slack(env):
    """Roll out the declared provisional, no-new-evidence controllers.

    This forecast reads only the current public kinematics and delivered
    evidence. The cloned simulator has the actual world for physical fidelity,
    but no hidden terrain values are inspected by this function or controller.
    """
    shadow = copy.deepcopy(env)
    initial_step = shadow.step_index
    while not shadow.done and shadow.route_commitment is None:
        scout_pos, scout_vel = state(shadow, "scout")
        carrier_pos, carrier_vel = state(shadow, "carrier")
        shadow.step({
            "scout": ForkAction(motion=scout_motion(
                shadow, "navigate", scout_pos, scout_vel)),
            "carrier": ForkAction(motion=carrier_motion(
                shadow, carrier_pos, carrier_vel)),
        })
    return (shadow.step_index - initial_step) * env.config.dt


def timing_context(env, rollout_forecast=False):
    scout_pos, _ = state(env, "scout")
    carrier_pos, carrier_vel = state(env, "carrier")
    separation = float(np.linalg.norm(scout_pos - carrier_pos))
    # Public online estimate from the current carrier position and positive
    # x velocity. A nominal floor handles early near-zero velocity.
    forward_speed = max(0.10, float(carrier_vel[0]))
    remaining = max(0.0, env.config.decision_x - float(carrier_pos[0]))
    return TimingContext(
        scout_position=separation, carrier_position=0.0,
        communication_range=env.config.communication_range,
        network_delay=env.config.communication_delay_steps * env.config.dt,
        delivery_probability=1 - env.config.packet_drop_probability,
        decision_slack=(forecast_commit_slack(env) if rollout_forecast
                        else remaining / forward_speed),
        acquisition_duration=2 * env.config.dt,
        return_speed=0.8, return_effort_per_distance=0.12,
        waiting_cost_per_second=0.03,
    )


def load_actors():
    _, a4 = load_models(1701, 1801)
    gate = ScoutTimingGate()
    gate.load_state_dict(torch.load(
        HERE / "results" / "a6_timing" / "scout_timing_gate.pth",
        map_location="cpu", weights_only=True
    ))
    gate.eval()
    return a4, gate


def a4_messages(env, a4, context):
    readings = {}
    ids = {}
    for observation_id, item in env.evidence["scout"].items():
        name = "top_" + item.modality
        readings[name] = bool(item.value) if item.modality == "geometry" else \
            bool(float(item.value) >= env.config.traction_safe_threshold)
        ids[name] = observation_id
    if set(readings) != {"top_geometry", "top_traction"}:
        raise RuntimeError("A4 cannot act until both physical readings exist")
    # This is the *current* physical range and predicted commit slack, after
    # the sensing actions. A4 remains frozen; only its public timing input moves.
    effective_probability = context.delivery_probability if context.timely else 0.0
    config = ExactOracleConfig(
        sensing_cost=env.config.sensing_cost,
        communication_attempt_cost=env.config.communication_attempt_cost,
        byte_cost=env.config.byte_cost,
        header_bytes=env.config.header_bytes,
        payload_bytes=env.config.payload_bytes,
        ack_payload_bytes=env.config.ack_payload_bytes,
        timely_delivery_probability=effective_probability,
    )
    acquired = tuple((name, readings[name]) for name in ("top_geometry", "top_traction"))
    oracle = ExactShareOracle(ShareState(acquired=acquired,
        timely_probability=effective_probability), config=config, prior=default_prior())
    return [ids[name] for name in learned_message_set(a4, oracle)]


def carrier_motion(env, position, velocity, force_top=False):
    if env.route_commitment is None:
        evidence = env.observations()["carrier"]["evidence"]["top"]
        informed_good = force_top or (evidence["geometry"] is True and evidence["traction"] is not None
            and evidence["traction"] >= env.config.traction_safe_threshold)
        if (informed_good and position[0] < -0.40
                and position[1] < env.config.carrier_top_y - 0.01):
            # Turn above the island before driving through the upper corridor.
            target = (-0.47, env.config.carrier_top_y + 0.05)
        else:
            target = (-0.03, env.config.carrier_top_y if informed_good
                      else env.config.bottom_route_y)
        return control(position, velocity, target, gain=3.0, damping=1.5, scale=0.9)
    sign = 1 if env.route_commitment == "top" else -1
    if position[0] < 0.46:
        target = (0.55, env.config.carrier_top_y if sign > 0
                  else env.config.bottom_route_y)
    elif env.route_commitment == "top" and env.config.direct_top_exit:
        target = (0.91, env.config.carrier_goal_y)
    elif env.route_commitment == "top" and position[1] > 0.02:
        # Pass below the scout before entering the shared goal region.
        target = (0.62, -0.10)
    else:
        target = (0.91, env.config.carrier_goal_y)
    return control(position, velocity, target, gain=3.6, damping=1.6)


def scout_motion(env, phase, position, velocity):
    if phase == "approach":
        return control(position, velocity, (-0.34, 0.43))
    if phase == "return":
        carrier_pos, _ = state(env, "carrier")
        return control(position, velocity, carrier_pos + np.array([-0.05, 0.04]),
                       gain=2.5, damping=1.2)
    if phase in ("sense_geometry", "sense_traction", "choose_share", "send"):
        return control(position, velocity, (-0.34, 0.43), gain=2.0, damping=1.2)
    evidence = env.observations()["scout"]["evidence"]["top"]
    informed_good = evidence["geometry"] is True and evidence["traction"] is not None \
        and evidence["traction"] >= env.config.traction_safe_threshold
    if informed_good:
        target = ((0.55, 0.63) if position[0] < 0.46
                  else (0.91, env.config.scout_goal_y))
    elif position[0] < -0.30 and position[1] > env.config.scout_bottom_y + 0.09:
        # Cross from the inspection viewpoint to the safe lower route left of
        # the island; without evidence the scout takes this conservative path.
        target = (env.config.scout_detour_x, env.config.scout_bottom_y)
    else:
        target = ((0.62, env.config.scout_bottom_y)
                  if position[0] < 0.60 else (0.91, env.config.scout_goal_y))
    return control(position, velocity, target, gain=3.6, damping=1.6)


def gate_snapshot(world, config, seed):
    """Return the actual VMAS state just before the probe decision."""
    env = CoPHForkEnv(config=config, world=world, seed=seed)
    viewpoint = np.array([-0.34, 0.43])
    while not env.done:
        scout_pos, scout_vel = state(env, "scout")
        if np.linalg.norm(scout_pos - viewpoint) < 0.10:
            return env
        carrier_pos, carrier_vel = state(env, "carrier")
        env.step({
            "scout": ForkAction(motion=scout_motion(
                env, "approach", scout_pos, scout_vel)),
            "carrier": ForkAction(motion=carrier_motion(
                env, carrier_pos, carrier_vel)),
        })
    raise RuntimeError("scout did not reach the inspection viewpoint")


def run_episode(world, config, policy, seed, a4, gate, save_replay=None,
                rollout_forecast=False, starting_env=None,
                forced_evidence_drops=None):
    if starting_env is not None and (starting_env.world_spec != world
                                     or starting_env.config != config):
        raise ValueError("cloned rollout world/config must match the decision state")
    env = (CoPHForkEnv(config=config, world=world, seed=seed)
           if starting_env is None else copy.deepcopy(starting_env))
    if forced_evidence_drops is not None:
        env.forced_evidence_drops = tuple(forced_evidence_drops)
        env._forced_evidence_drop_index = 0
    phase = "approach"
    pending_ids = []
    gate_log = None
    sharing_log = None
    actual_commit_step = None
    while not env.done:
        scout_pos, scout_vel = state(env, "scout")
        carrier_pos, carrier_vel = state(env, "carrier")
        if phase == "approach" and np.linalg.norm(scout_pos - np.array([-0.34, 0.43])) < 0.10:
            context = timing_context(env, rollout_forecast)
            if policy == "learned":
                inspect = gate.chooses_inspect(context)
            elif policy == "always":
                inspect = True
            elif policy == "never":
                inspect = False
            else:
                raise ValueError("unknown bridge policy")
            gate_log = {"step": env.step_index, "scout_position": scout_pos.tolist(),
                        "carrier_position": carrier_pos.tolist(),
                        "carrier_velocity": carrier_vel.tolist(),
                        "timing": asdict(context), "predicted_delivery_time": context.delivery_time,
                        "predicted_commit_slack": context.decision_slack,
                        "inspect": inspect}
            phase = "sense_geometry" if inspect else "navigate"
        if phase == "choose_share":
            context = timing_context(env, rollout_forecast)
            pending_ids = a4_messages(env, a4, context)
            sharing_log = {"step": env.step_index, "timing": asdict(context),
                           "selected_observation_ids": list(pending_ids)}
            phase = "return" if pending_ids and context.return_distance > 0.03 else \
                "send" if pending_ids else "navigate"
        if phase == "return":
            if np.linalg.norm(scout_pos - carrier_pos) <= config.communication_range - 0.02:
                phase = "send"
        scout = ForkAction(motion=scout_motion(env, phase, scout_pos, scout_vel))
        carrier = ForkAction(motion=carrier_motion(env, carrier_pos, carrier_vel))
        if phase == "sense_geometry":
            scout.sense_route = "top"
            scout.sense_modality = "geometry"
            phase = "sense_traction"
        elif phase == "sense_traction":
            scout.sense_route = "top"
            scout.sense_modality = "traction"
            phase = "choose_share"
        elif phase == "send" and pending_ids:
            scout.send_observation_id = pending_ids.pop(0)
            if not pending_ids:
                phase = "navigate"
        env.step({"scout": scout, "carrier": carrier})
        if env.route_commitment is not None and actual_commit_step is None:
            actual_commit_step = env.step_index
    if save_replay is not None:
        env.save_replay(save_replay)
        replayed = CoPHForkEnv.replay(json.loads(Path(save_replay).read_text()))
        if replayed.replay_payload()["sha256"] != env.replay_payload()["sha256"]:
            raise RuntimeError("VMAS moving bridge replay mismatch")
    events = env.events
    delivered_before = [event for event in events if event["type"] == "packet_delivery"
                        and event["kind"] == "evidence" and event["delivered"]
                        and event["usable_before_decision"]]
    delivered_after = [event for event in events if event["type"] == "packet_delivery"
                       and event["kind"] == "evidence" and event["delivered"]
                       and not event["usable_before_decision"]]
    actual_slack = None if actual_commit_step is None or gate_log is None else \
        (actual_commit_step - gate_log["step"]) * config.dt
    return {
        "policy": policy, "world": asdict(world), "config": asdict(config),
        "rollout_forecast": rollout_forecast,
        "seed": seed, "success": env.success, "steps": env.step_index,
        "route_commitment": env.route_commitment,
        "actual_commit_step": actual_commit_step,
        "actual_commit_slack_from_gate": actual_slack,
        "gate": gate_log, "sharing": sharing_log,
        "delivered_before_commit": len(delivered_before),
        "delivered_after_commit": len(delivered_after),
        "evidence_deliveries": [{"step": event["step"],
                                 "usable_before_decision": event["usable_before_decision"]}
                                for event in delivered_before + delivered_after],
        "scout_acquisitions": sum(event["type"] == "acquisition"
                                  for event in events),
        "ledger": asdict(env.ledger), "team_cost": env.ledger.total,
        "trajectory": env.state_trace,
        "event_counts": {name: sum(event["type"] == name for event in events)
                         for name in ("acquisition", "transmission", "packet_delivery",
                                      "route_commitment")},
    }


def plot_trajectories(rows, output):
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.2),
                             constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        path = row["trajectory"]
        scout = np.asarray([step["scout_position"] for step in path])
        carrier = np.asarray([step["carrier_position"] for step in path])
        ax.plot(scout[:, 0], scout[:, 1], color="tab:green", label="scout")
        ax.plot(carrier[:, 0], carrier[:, 1], color="tab:blue", label="carrier")
        ax.scatter([-0.34], [0.43], marker="x", color="black", label="viewpoint")
        ax.axvline(row["config"]["decision_x"], color="tab:red", ls="--")
        ax.set(xlim=(-1.05, 1.04), ylim=(-.75, .75),
               title="{}: {} | cost {:.2f}".format(row["policy"],
                    row["route_commitment"], row["team_cost"]),
               xlabel="x", ylabel="y")
        ax.set_aspect("equal")
    axes[0].legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    a4, gate = load_actors()
    world = ForkWorld(top_geometry=True, top_traction=0.9,
                      bottom_geometry=True, bottom_traction=0.50)
    config = ForkConfig(horizon=900, communication_range=0.9,
                        communication_delay_steps=3, packet_drop_probability=0.0)
    rows = [run_episode(world, config, policy, 1701, a4, gate,
            save_replay=OUT / (policy + "_replay.json"))
            for policy in ("learned", "always", "never")]
    (OUT / "pilot.json").write_text(json.dumps(rows, indent=2) + "\n")
    plot_trajectories(rows, OUT / "pilot_trajectories.png")
    for row in rows:
        print(row["policy"], "gate", row["gate"]["inspect"],
              "commit", row["route_commitment"], "steps", row["steps"],
              "cost", round(row["team_cost"], 3), "delivered before",
              row["delivered_before_commit"])
    intermediate = ForkConfig(horizon=900, communication_range=0.9,
                              communication_delay_steps=100,
                              packet_drop_probability=0.0)
    intermediate_rows = [run_episode(world, intermediate, policy, 1701, a4, gate)
                         for policy in ("learned", "always", "never")]
    for row in intermediate_rows:
        row.pop("trajectory")
    (OUT / "intermediate_delay.json").write_text(
        json.dumps(intermediate_rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
