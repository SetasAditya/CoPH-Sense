#!/usr/bin/env python3
"""Run a causal acquire-send-route scripted policy and save A1 artifacts."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork import CoPHForkEnv, ForkAction, ForkConfig, ForkWorld  # noqa: E402


def control(position, velocity, target, gain=3.2, damping=1.5):
    command = gain * (np.asarray(target) - position) - damping * velocity
    norm = np.linalg.norm(command)
    return tuple((command / max(1.0, norm)).tolist())


def actor_state(env, name):
    agent = env.physics.world.agents[0 if name == "scout" else 1]
    return (
        agent.state.pos[0].detach().cpu().numpy(),
        agent.state.vel[0].detach().cpu().numpy(),
    )


def latest_unsent(env, sent):
    candidates = [key for key in env.evidence["scout"] if key not in sent]
    return candidates[0] if candidates else None


def policy(env, state):
    scout_pos, scout_vel = actor_state(env, "scout")
    carrier_pos, carrier_vel = actor_state(env, "carrier")
    scout_action = ForkAction()
    carrier_action = ForkAction()
    probe = np.array([-0.34, 0.43])

    if state["phase"] == "approach_probe":
        scout_action.motion = control(scout_pos, scout_vel, probe)
        if np.linalg.norm(scout_pos - probe) < 0.12:
            state["phase"] = "sense_geometry"
    elif state["phase"] == "sense_geometry":
        scout_action.sense_route = "top"
        scout_action.sense_modality = "geometry"
        state["phase"] = "sense_traction"
    elif state["phase"] == "sense_traction":
        scout_action.sense_route = "top"
        scout_action.sense_modality = "traction"
        state["phase"] = "send"
    elif state["phase"] == "send":
        observation_id = latest_unsent(env, state["sent"])
        if observation_id is not None:
            scout_action.send_observation_id = observation_id
            state["sent"].add(observation_id)
        if len(state["sent"]) == 2:
            state["phase"] = "navigate"
    else:
        scout_waypoints = [(-0.27, 0.46), (0.48, 0.46), (0.78, 0.24), (0.91, 0.0)]
        index = min(state["scout_waypoint"], len(scout_waypoints) - 1)
        target = np.asarray(scout_waypoints[index])
        if np.linalg.norm(scout_pos - target) < 0.12 and index + 1 < len(scout_waypoints):
            state["scout_waypoint"] += 1
            target = np.asarray(scout_waypoints[state["scout_waypoint"]])
        scout_action.motion = control(scout_pos, scout_vel, target)

    received = env.observations()["carrier"]["evidence"]["top"]
    informed = received["geometry"] is not None and received["traction"] is not None
    if informed:
        route = "top" if received["geometry"] and received["traction"] >= env.config.traction_safe_threshold else "bottom"
        sign = 1.0 if route == "top" else -1.0
        carrier_waypoints = [(-0.28, 0.46 * sign), (0.50, 0.46 * sign), (0.80, 0.22 * sign), (0.91, 0.0)]
        index = min(state["carrier_waypoint"], len(carrier_waypoints) - 1)
        target = np.asarray(carrier_waypoints[index])
        if np.linalg.norm(carrier_pos - target) < 0.10 and index + 1 < len(carrier_waypoints):
            state["carrier_waypoint"] += 1
            target = np.asarray(carrier_waypoints[state["carrier_waypoint"]])
        carrier_action.motion = control(carrier_pos, carrier_vel, target, gain=3.6, damping=1.8)
    else:
        # Brake before the route commitment plane until evidence is delivered.
        carrier_action.motion = tuple(np.clip(-2.0 * carrier_vel, -1.0, 1.0).tolist())
    return {"scout": scout_action, "carrier": carrier_action}


def render_summary(env, output):
    states = env.state_trace
    scout = np.asarray([row["scout_position"] for row in states])
    carrier = np.asarray([row["carrier_position"] for row in states])
    fig, ax = plt.subplots(figsize=(10, 6))
    for route, y, traction in (
        ("top", 0.43, env.world_spec.top_traction),
        ("bottom", -0.43, env.world_spec.bottom_traction),
    ):
        color = plt.cm.RdYlGn(traction)
        ax.fill_between([-0.28, 0.58], y - 0.16, y + 0.16, color=color, alpha=0.28)
        ax.text(0.18, y, "{}: traction {:.2f}".format(route, traction), ha="center")
    ax.add_patch(plt.Rectangle((-0.26, -0.17), 0.62, 0.34, color="0.35"))
    ax.plot(scout[:, 0], scout[:, 1], color="tab:green", lw=2, label="scout")
    ax.plot(carrier[:, 0], carrier[:, 1], color="tab:blue", lw=3, label="carrier")
    ax.scatter([-0.34, -0.34], [0.43, -0.43], marker="x", s=90, color="black", label="probe")
    ax.scatter([0.91], [0], marker="*", s=240, color="tab:red", label="goal")
    ax.axvline(env.config.decision_x, color="black", ls="--", alpha=0.6, label="decision plane")
    ax.set(xlim=(-1.08, 1.08), ylim=(-0.82, 0.82), xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("CoPH-Fork A1: acquire, deliver, then commit")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    env = CoPHForkEnv(
        config=ForkConfig(horizon=600, communication_delay_steps=3),
        world=ForkWorld(
            top_geometry=True,
            top_traction=0.90,
            bottom_geometry=True,
            bottom_traction=0.28,
        ),
        seed=1701,
    )
    state = {"phase": "approach_probe", "sent": set(), "scout_waypoint": 0, "carrier_waypoint": 0}
    frames = []
    while not env.done:
        actions = policy(env, state)
        _, _, _, info = env.step(actions)
        if env.step_index % 4 == 0:
            image = Image.fromarray(env.physics.render(mode="rgb_array"))
            draw = ImageDraw.Draw(image)
            label = "step {} | cost {:.3f} | evidence {} | route {}".format(
                env.step_index,
                info["episode_total"],
                len(env.received["carrier"]),
                env.route_commitment or "open",
            )
            draw.rectangle((0, 0, 700, 27), fill="white")
            draw.text((8, 7), label, fill="black")
            frames.append(image)

    replay_path = output / "demo_replay.json"
    env.save_replay(replay_path)
    replayed = CoPHForkEnv.replay(json.loads(replay_path.read_text()))
    if replayed.replay_payload()["sha256"] != env.replay_payload()["sha256"]:
        raise RuntimeError("deterministic replay digest mismatch")
    gif_path = output / "coph_fork_a1.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    render_summary(env, output / "coph_fork_a1_summary.png")
    summary = {
        "success": env.success,
        "steps": env.step_index,
        "route_commitment": env.route_commitment,
        "ledger": env.replay_payload()["ledger"],
        "total_cost": env.ledger.total,
        "observations_acquired": len(env.evidence["scout"]),
        "observations_delivered": len(env.received["carrier"]),
        "replay_sha256": env.replay_payload()["sha256"],
    }
    (output / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
