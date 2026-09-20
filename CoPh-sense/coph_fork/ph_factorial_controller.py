"""Matched direct and split pH executors for the two-fork VMAS factorial.

Both schemes use the same conservative force and damping channels. The split
scheme updates position and momentum itself; VMAS is used for the world,
observations, collision queries, rendering, and the episode ledger.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceConfig:
    beta_goal: float = 4.0
    alpha_ipc: float = 1.0
    lambda_risk: float = .35
    lambda_hard: float = 1.0
    gamma: float = 1.5
    d_hat: float = .18
    max_force: float = 4.0
    max_barrier_grad: float = 16.0
    max_barrier_energy: float = 10.0
    risk_sigma_x: float = .24
    risk_sigma_y: float = .19


def clip_norm(vector, limit):
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


def box_clearance(position, center, half_size, radius):
    """Signed disc-to-box clearance and an outward distance gradient."""
    delta = np.asarray(position, dtype=float) - np.asarray(center, dtype=float)
    half = np.asarray(half_size, dtype=float)
    outside = np.maximum(np.abs(delta) - half, 0.)
    distance = float(np.linalg.norm(outside))
    if distance > 1e-10:
        gradient = np.sign(delta) * outside / distance
        return distance - radius, gradient
    # Inside the box: choose the nearest face as a well-defined repulsion.
    clearance_to_face = half - np.abs(delta)
    axis = int(np.argmin(clearance_to_face))
    gradient = np.zeros(2, dtype=float)
    gradient[axis] = 1. if delta[axis] >= 0 else -1.
    return -float(clearance_to_face[axis]) - radius, gradient


def ipc_barrier(distance, config):
    dh = config.d_hat
    if distance >= dh:
        return 0., 0.
    safe = max(distance, 1e-5)
    energy = -(safe - dh) ** 2 * np.log(safe / dh)
    derivative = -(2 * (safe - dh) * np.log(safe / dh)
                   + (safe - dh) ** 2 / safe)
    return (min(float(energy), config.max_barrier_energy),
            max(float(derivative), -config.max_barrier_grad))


def public_boxes(env, agent_name):
    """Known fixed islands plus only blockers revealed by delivered geometry."""
    result = []
    evidence = env.observations()[agent_name]["evidence"]
    for region, x in ((1, -.42), (2, .42)):
        lower = getattr(env.config, f"backup_y{region}") + .27
        upper = .16
        result.append((np.asarray((x, (lower + upper) / 2)),
                       np.asarray((.14, (upper - lower) / 2)), "geometry"))
        if evidence[region]["top"]["geometry"] is False:
            result.append((np.asarray((x, .43)), np.asarray((.10, .16)),
                           "hazard"))
    return result


def traction_risk_amplitude(env, agent_name, region):
    reading = env.observations()[agent_name]["evidence"][region]["top"]["traction"]
    threshold = env.config.traction_safe_threshold
    if reading is None:
        probability_safe = (.65 if region == 1 else .60)
        return (1 - probability_safe) * max(0., threshold - .3)
    return max(0., threshold - float(reading))


def conservative_field(position, target, env, agent_name, config, radius):
    q = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    difference = q - target
    potential_goal = .5 * config.beta_goal * float(difference @ difference)
    force_goal = -config.beta_goal * difference
    force_obstacle = np.zeros(2, dtype=float)
    force_hard = np.zeros(2, dtype=float)
    potential_obstacle = 0.
    potential_hard = 0.
    for center, half_size, kind in public_boxes(env, agent_name):
        distance, gradient = box_clearance(q, center, half_size, radius)
        value, derivative = ipc_barrier(distance, config)
        if kind == "geometry":
            potential_obstacle += config.alpha_ipc * value
            force_obstacle -= config.alpha_ipc * derivative * gradient
        else:
            potential_hard += config.lambda_hard * value
            force_hard -= config.lambda_hard * derivative * gradient
    force_risk = np.zeros(2, dtype=float)
    potential_risk = 0.
    for region, x in ((1, -.42), (2, .42)):
        amplitude = traction_risk_amplitude(env, agent_name, region)
        center = np.asarray((x, .43))
        offset = q - center
        scaled = np.asarray((offset[0] / config.risk_sigma_x,
                             offset[1] / config.risk_sigma_y))
        value = amplitude * float(np.exp(-.5 * (scaled @ scaled)))
        potential_risk += config.lambda_risk * value
        force_risk += config.lambda_risk * value * np.asarray((
            offset[0] / config.risk_sigma_x ** 2,
            offset[1] / config.risk_sigma_y ** 2))
    components = {
        "goal": force_goal, "obstacle": force_obstacle,
        "risk": force_risk, "hard": force_hard,
    }
    potential = potential_goal + potential_obstacle + potential_hard + potential_risk
    return potential, components


def execute_step(position, velocity, target, env, agent_name, scheme,
                 config, mass, radius, max_speed, traction):
    """Return q+,v+ and a diagnostic for a frozen-belief integration stage."""
    if scheme not in ("direct", "ph"):
        raise ValueError(scheme)
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    dt = env.config.dt
    potential_before, components_before = conservative_field(
        q, target, env, agent_name, config, radius)
    force_before = clip_norm(sum(components_before.values()), config.max_force)
    momentum = mass * v
    energy_before = .5 * mass * float(v @ v) + potential_before
    if scheme == "direct":
        p_next = momentum + dt * (traction * force_before - config.gamma * v)
        v_next = clip_norm(p_next / mass, max_speed)
        q_next = q + dt * v_next
    else:
        p_half = momentum + .5 * dt * (traction * force_before - config.gamma * v)
        # Enforce the common speed cap before evaluating the second kick.
        # The force must be evaluated at the position actually executed.
        q_next = q + dt * clip_norm(p_half / mass, max_speed)
        _, components_after = conservative_field(
            q_next, target, env, agent_name, config, radius)
        force_after = clip_norm(sum(components_after.values()), config.max_force)
        p_next = p_half + .5 * dt * (traction * force_after
                                   - config.gamma * p_half / mass)
        v_next = clip_norm(p_next / mass, max_speed)
    potential_after, _ = conservative_field(
        q_next, target, env, agent_name, config, radius)
    energy_after = .5 * mass * float(v_next @ v_next) + potential_after
    return q_next, v_next, {
        "energy_before": energy_before, "energy_after": energy_after,
        "potential_before": potential_before,
        "force_components": {key: value.tolist()
                             for key, value in components_before.items()},
        "force_norm": float(np.linalg.norm(force_before)),
        "dissipation_estimate": dt * config.gamma * float(v @ v),
        "traction": float(traction),
    }
