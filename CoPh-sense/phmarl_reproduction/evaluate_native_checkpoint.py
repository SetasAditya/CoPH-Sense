#!/usr/bin/env python3
"""Reproduce and visualize the published pH-MARL VMAS navigation checkpoint.

This wrapper intentionally leaves the upstream repository unchanged.  It fixes
only evaluation portability issues: CPU checkpoint remapping, a correct
availability check, one simulator step per loop iteration, and headless RGB
rendering.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from vmas import make_env


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=here.parent / "external" / "phMARL",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-seed", type=int, default=9)
    parser.add_argument("--gif-stride", type=int, default=2)
    return parser.parse_args()


def force_cpu(module):
    """Update legacy device attributes in addition to moving parameters."""
    module.to("cpu")
    for child in module.modules():
        if hasattr(child, "device"):
            child.device = torch.device("cpu")
    return module


def positions(env):
    agents = np.stack([a.state.pos[0].detach().cpu().numpy() for a in env.world.agents])
    landmarks = np.stack(
        [l.state.pos[0].detach().cpu().numpy() for l in env.world.landmarks]
    )
    return agents, landmarks


def geometry_metrics(env):
    agents, landmarks = positions(env)
    distances = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    assigned = distances[rows, cols]
    nearest = distances.min(axis=0)
    return {
        "mean_assignment_distance": float(assigned.mean()),
        "max_assignment_distance": float(assigned.max()),
        "mean_nearest_landmark_distance": float(nearest.mean()),
        "landmark_coverage_at_0.15": float((nearest <= 0.15).mean()),
    }


def collision_pairs(env):
    count = 0
    for i, first in enumerate(env.world.agents):
        for second in env.world.agents[i + 1 :]:
            count += int(env.world.is_overlapping(first, second)[0].item())
    return count


def observation_batch(observations):
    stacked = torch.stack(observations)
    if stacked.ndim == 3:
        stacked = stacked.squeeze(1)
    return stacked.unsqueeze(0)


def run_episode(actor, policy_name, seed, args, render=False):
    env = make_env(
        scenario_name="simple_spread",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        max_steps=args.horizon,
        n_agents=args.agents,
        share_reward=True,
    )
    observations = env.reset(seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    frames = []
    trajectory = []
    mean_agent_reward_sum = 0.0
    upstream_reward_sum = 0.0
    collision_pair_steps = 0

    if render:
        frames.append(Image.fromarray(env.render(mode="rgb_array")))

    for step in range(args.horizon):
        if actor is None:
            actions = torch.empty((1, args.agents, 2)).uniform_(
                -1.0, 1.0, generator=generator
            )
        else:
            with torch.no_grad():
                actions = actor(observation_batch(observations)).mean

        observations, rewards, dones, _ = env.step(list(actions.transpose(0, 1)))
        reward_values = torch.stack(rewards).detach().cpu().numpy().reshape(-1)
        # The native scenario combines a common landmark term with an
        # agent-specific collision term. The upstream script sums agents;
        # retain that convention and a team-size-normalized mean-agent return.
        mean_agent_reward_sum += float(reward_values.mean())
        upstream_reward_sum += float(reward_values.sum())
        collision_pair_steps += collision_pairs(env)
        agent_pos, _ = positions(env)
        trajectory.append(agent_pos)

        if render and (step + 1) % args.gif_stride == 0:
            frames.append(Image.fromarray(env.render(mode="rgb_array")))
        if bool(torch.as_tensor(dones).any()):
            break

    result = {
        "policy": policy_name,
        "seed": seed,
        "steps": step + 1,
        "mean_agent_return": mean_agent_reward_sum,
        "upstream_summed_agent_return": upstream_reward_sum,
        "collision_pair_steps": collision_pair_steps,
        **geometry_metrics(env),
    }
    return result, np.asarray(trajectory), frames


def summarize(rows):
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "seed"]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in numeric
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream_dir.resolve()
    sys.path.insert(0, str(upstream))
    # Required so torch.load can resolve functions.LEMURS_actor from the
    # upstream whole-object pickle.
    import functions  # noqa: F401, PLC0415

    checkpoint = upstream / "data" / "actor_net_simple_spread_1400000_LEMURS.pth"
    actor = force_cpu(torch.load(checkpoint, map_location="cpu"))
    actor.na = args.agents
    actor.eval()

    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        for policy_name, policy in (("phmarl_checkpoint", actor), ("uniform_random", None)):
            result, trajectory, _ = run_episode(policy, policy_name, seed, args)
            rows.append(result)
            np.save(args.output_dir / f"trajectory_{policy_name}_seed{seed}.npy", trajectory)

    result, trajectory, frames = run_episode(
        actor, "phmarl_checkpoint", args.render_seed, args, render=True
    )
    np.save(args.output_dir / f"trajectory_phmarl_checkpoint_seed{args.render_seed}.npy", trajectory)
    gif_path = args.output_dir / f"simple_spread_phmarl_seed{args.render_seed}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=40 * args.gif_stride,
        loop=0,
        optimize=False,
    )

    by_policy = {
        name: [row for row in rows if row["policy"] == name]
        for name in ("phmarl_checkpoint", "uniform_random")
    }
    configuration = vars(args).copy()
    configuration.update({"upstream_dir": str(upstream), "output_dir": str(args.output_dir)})
    report = {
        "provenance": {
            "upstream": "https://github.com/EduardoSebastianRodriguez/phMARL.git",
            "checkpoint": str(checkpoint),
            "scenario": "VMAS 1.2.6 simple_spread (Navigation)",
            "checkpoint_training_frame": 1_400_000,
            "torch": torch.__version__,
        },
        "configuration": configuration,
        "episodes": rows,
        "summary": {name: summarize(policy_rows) for name, policy_rows in by_policy.items()},
        "rendered_episode": result,
        "artifact": str(gif_path),
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {gif_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
