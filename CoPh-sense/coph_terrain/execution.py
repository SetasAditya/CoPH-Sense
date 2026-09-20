"""Belief-conditioned split pH waypoint execution on procedural terrain."""

from dataclasses import dataclass

import numpy as np

from coph_fork.ph_factorial_controller import (
    box_clearance, clip_norm, ipc_barrier,
)

from .generator import GRID_SIZE, RESOLUTION


@dataclass(frozen=True)
class TerrainForceConfig:
    beta_goal: float = 4.
    alpha_ipc: float = 1.
    lambda_risk: float = .35
    gamma: float = 1.5
    d_hat: float = .18
    max_force: float = 4.
    max_barrier_grad: float = 16.
    max_barrier_energy: float = 10.
    coordination_clearance: float = .35


def _interpolate(grid, position):
    xy = np.clip((np.asarray(position) + 6.) / RESOLUTION - .5,
                 0., GRID_SIZE - 1.000001)
    x0, y0 = np.floor(xy).astype(int)
    x1, y1 = min(x0 + 1, GRID_SIZE - 1), min(y0 + 1, GRID_SIZE - 1)
    tx, ty = xy - (x0, y0)
    return float((1 - tx) * (1 - ty) * grid[x0, y0]
                 + tx * (1 - ty) * grid[x1, y0]
                 + (1 - tx) * ty * grid[x0, y1]
                 + tx * ty * grid[x1, y1])


def belief_risk_grid(observation):
    appearance_risk = np.full((GRID_SIZE, GRID_SIZE), .35, dtype=np.float64)
    appearance = observation["appearance"]
    visible = np.isfinite(appearance)
    appearance_risk[visible] = np.clip((.86 - appearance[visible]) / .65, 0., 1.)
    surface = appearance_risk.copy()
    traction_risk = appearance_risk.copy()
    measured_surface = observation["known_surface"]
    measured_traction = observation["known_traction"]
    known_surface = np.isfinite(measured_surface)
    known_traction = np.isfinite(measured_traction)
    surface[known_surface] = measured_surface[known_surface]
    traction_risk[known_traction] = np.clip(
        (.55 - measured_traction[known_traction]) / .40, 0., 1.)
    return np.clip(.4 * surface + .4 * traction_risk
                   + .2 * surface * traction_risk, 0., 1.)


def potential_force(q, target, observation, config, radius, risk_grid=None):
    q = np.asarray(q, dtype=float)
    target = np.asarray(target, dtype=float)
    displacement = q - target
    potential = .5 * config.beta_goal * float(displacement @ displacement)
    force = -config.beta_goal * displacement
    geometry = observation["known_geometry"]
    ix, iy = np.floor((q + 6.) / RESOLUTION).astype(int)
    for x in range(max(0, ix - 4), min(GRID_SIZE, ix + 5)):
        for y in range(max(0, iy - 4), min(GRID_SIZE, iy + 5)):
            if geometry[x, y] != 1.:
                continue
            center = np.asarray((-6. + (x + .5) * RESOLUTION,
                                 -6. + (y + .5) * RESOLUTION))
            distance, gradient = box_clearance(q, center, (.1, .1), radius)
            value, derivative = ipc_barrier(distance, config)
            potential += config.alpha_ipc * value
            force -= config.alpha_ipc * derivative * gradient
    teammate = (np.asarray(observation["kinematics"][:2], dtype=float)
                + np.asarray(observation["kinematics"][4:6], dtype=float))
    delta = q - teammate
    distance = float(np.linalg.norm(delta))
    other_radius = .10 if radius < .08 else .065
    separation = distance - radius - other_radius
    if separation < config.coordination_clearance and distance > 1e-9:
        # Same public/local teammate kinematics for both executors. This is a
        # coordination potential, not privileged collision geometry.
        barrier_config = TerrainForceConfig(
            d_hat=config.coordination_clearance,
            max_barrier_grad=config.max_barrier_grad,
            max_barrier_energy=config.max_barrier_energy)
        value, derivative = ipc_barrier(separation, barrier_config)
        potential += value
        force -= derivative * delta / distance
    risk_grid = belief_risk_grid(observation) if risk_grid is None else risk_grid
    risk = _interpolate(risk_grid, q)
    step = .02
    gradient = np.asarray([
        (_interpolate(risk_grid, q + (step, 0)) -
         _interpolate(risk_grid, q - (step, 0))) / (2 * step),
        (_interpolate(risk_grid, q + (0, step)) -
         _interpolate(risk_grid, q - (0, step))) / (2 * step),
    ])
    potential += config.lambda_risk * risk
    force -= config.lambda_risk * gradient
    return potential, clip_norm(force, config.max_force)


def ph_step(q, v, target, observation, mass, radius, max_speed,
            traction, dt, config=None):
    """One belief-frozen kick–drift–kick step; traction is a plant port."""
    config = config or TerrainForceConfig()
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    risk_grid = belief_risk_grid(observation)
    potential_before, force_before = potential_force(
        q, target, observation, config, radius, risk_grid)
    p_half = mass * v + .5 * dt * (traction * force_before - config.gamma * v)
    q_next = q + dt * clip_norm(p_half / mass, max_speed)
    potential_after, force_after = potential_force(
        q_next, target, observation, config, radius, risk_grid)
    p_next = p_half + .5 * dt * (traction * force_after
                                - config.gamma * p_half / mass)
    v_next = clip_norm(p_next / mass, max_speed)
    return q_next, v_next, {
        "storage_before": .5 * mass * float(v @ v) + potential_before,
        "storage_after": .5 * mass * float(v_next @ v_next) + potential_after,
        "force_norm": float(np.linalg.norm(force_before)),
        "traction_port_gain": float(traction),
    }
