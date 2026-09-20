"""Controller-only unit calibration for the transferred material pH checkpoint."""

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .environment import CoPHTerrainEnv, position_to_cell
from .material_executor import (DEFAULT_CHECKPOINT, MaterialExecutorConfig,
                                MaterialHamiltonianExecutor)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)),
            "p99": float(np.quantile(values, .99))}


def development_states():
    manifest = []
    for family_index, family in enumerate(("open", "bottleneck", "labyrinth")):
        for offset in range(4):
            seed = 910000 + 1000 * family_index + offset
            env = CoPHTerrainEnv(family, seed, 0, seed + 17)
            obs = env.observations()["carrier"]
            q = env.positions["carrier"]
            # Public fixed-information controller states only.
            for progress, velocity in ((.6, (0., 0.)), (1.2, (.25, 0.)),
                                       (2.0, (.45, .08))):
                target = np.minimum(q + (progress, .15), (5.4, 5.4))
                manifest.append((family, seed, q.copy(), np.asarray(velocity),
                                 target, copy.deepcopy(obs)))
    return manifest


def raw_channel_audit(executor, states):
    magnitudes = {name: [] for name in ("goal", "geometry", "soft", "hard", "damping")}
    total = []
    for _, _, q, velocity, target, obs in states:
        components, _ = executor.force_components(q, velocity, target, obs)
        for name, value in components.items():
            magnitudes[name].append(float(np.linalg.norm(value)))
        total.append(float(np.linalg.norm(sum(components.values()))))
    return {name: percentile_summary(values) for name, values in magnitudes.items()}, \
        percentile_summary(total)


def derive_scales(channels):
    gg = max(1e-8, channels["goal"]["p99"] + channels["geometry"]["p99"])
    context = max(1e-8, channels["soft"]["p99"] + channels["hard"]["p99"])
    damping = max(1e-8, channels["damping"]["p99"])
    # Leave headroom for channel addition under the immutable 4 N clip.
    return {"goal_geometry_scale": min(1., 2.5 / gg),
            "context_scale": min(8., 1.0 / context),
            "damping_scale": min(1., .75 / damping)}


def rollout_probe(executor, observation, start, goal, risk_variant=None,
                  steps=320, dt=.05, mass=1.8, radius=.1, max_speed=.55):
    obs = copy.deepcopy(observation)
    if risk_variant == "zero":
        obs["appearance"][:] = .86
        obs["known_surface"][:] = 0.
        obs["known_traction"][:] = .55
    elif risk_variant == "upper":
        obs["known_surface"][15:42, 30:45] = 1.
        obs["known_traction"][15:42, 30:45] = .15
    q, v = np.asarray(start, float), np.zeros(2)
    clip, speed_clip, collisions, force_norms, trajectory = 0, 0, 0, [], [q.copy()]
    radial_previous = None
    oscillations = 0
    for _ in range(steps):
        force, state = executor.proposal(q, v, goal, obs, mass, radius)
        clip += int(state["clipped"])
        force_norms.append(float(np.linalg.norm(force)))
        v = v + dt * force / mass
        speed = float(np.linalg.norm(v))
        if speed > max_speed:
            speed_clip += 1
            v *= max_speed / speed
        q = q + dt * v
        trajectory.append(q.copy())
        cell = position_to_cell(q)
        collisions += int(obs["known_geometry"][cell] == 1.)
        radial = float(np.dot(v, np.asarray(goal)-q))
        if radial_previous is not None and radial * radial_previous < 0:
            oscillations += 1
        radial_previous = radial
        if np.linalg.norm(q-np.asarray(goal)) <= .35 and speed < .12:
            break
    trajectory = np.asarray(trajectory)
    distance = np.linalg.norm(trajectory-np.asarray(goal), axis=1)
    return {"success": bool(distance[-1] <= .4), "steps": len(trajectory)-1,
            "final_distance": float(distance[-1]), "overshoot": float(max(0.,
                np.linalg.norm(trajectory[-1]-trajectory[np.argmin(distance)]))),
            "clip_fraction": clip/max(1, len(trajectory)-1),
            "speed_saturation_fraction": speed_clip/max(1, len(trajectory)-1),
            "collisions": collisions, "oscillations": oscillations,
            "force_p99": float(np.quantile(force_norms, .99)),
            "trajectory": trajectory.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="coph_terrain/results/material_ph_calibration_v1.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    states = development_states()
    raw_config = MaterialExecutorConfig(goal_geometry_scale=1., context_scale=1.,
                                        damping_scale=1.)
    raw = MaterialHamiltonianExecutor(raw_config, device=args.device)
    channels, total = raw_channel_audit(raw, states)
    scales = derive_scales(channels)
    config = replace(raw.config, **scales)
    calibrated = MaterialHamiltonianExecutor(config, args.device)
    base_env = CoPHTerrainEnv("open", 910000, 0, 910017)
    observation = base_env.observations()["carrier"]
    start, goal = base_env.positions["carrier"], np.asarray((-3.8, -.25))
    goal_probe = copy.deepcopy(observation)
    goal_probe["known_geometry"][:] = 0.
    probes = {
        "goal_only": rollout_probe(calibrated, goal_probe, start, goal, "zero"),
        "geometry_only": rollout_probe(calibrated, observation, start, goal, "zero"),
        "material_zero": rollout_probe(calibrated, observation, start, goal, "zero"),
        "material_visible_risk": rollout_probe(calibrated, observation, start, goal, "upper"),
    }
    zero_path = np.asarray(probes["material_zero"].pop("trajectory"))
    risk_path = np.asarray(probes["material_visible_risk"].pop("trajectory"))
    for name in ("goal_only", "geometry_only"):
        probes[name].pop("trajectory")
    n = min(len(zero_path), len(risk_path))
    belief_deflection = float(np.max(np.linalg.norm(zero_path[:n]-risk_path[:n], axis=1)))
    checkpoint = Path(config.checkpoint)
    acceptance = {
        "goal_success": probes["goal_only"]["success"],
        "geometry_collision_free": probes["geometry_only"]["collisions"] == 0,
        "ordinary_clipping_below_5pct": probes["geometry_only"]["clip_fraction"] < .05,
        "no_sustained_oscillation": probes["geometry_only"]["oscillations"] <= 2,
        "belief_changes_trajectory": belief_deflection > 1e-4,
    }
    payload = {"schema": "material_ph_calibration_v1",
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
               "coordinate_scale_m_per_pixel": .2,
               **scales, "force_clip": config.max_force, "dt": .05, "mass": 1.8,
               "development_seed_manifest": sorted(set((f, s) for f, s, *_ in states)),
               "raw_channel_magnitudes": channels, "raw_total_magnitude": total,
               "controller_probes": probes, "belief_trajectory_deflection": belief_deflection,
               "acceptance": acceptance, "passed": all(acceptance.values()),
               "selection_policy": "analytic p99 channel normalization; no information-policy metrics"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
