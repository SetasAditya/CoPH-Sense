"""Batched Torch version of the CoPH-Terrain split pH step.

Inputs stay on one device for an entire batched rollout. This kernel alone
does not make the present Python planner/packet/environment loop GPU-native.
"""

import torch

from .execution import TerrainForceConfig


GRID_SIZE = 60
RESOLUTION = .2


def _clip_norm(vector, limit):
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector * torch.clamp(limit / norm.clamp_min(1e-12), max=1.)


def _interpolate(grid, position):
    xy = torch.clamp((position + 6.) / RESOLUTION - .5,
                     min=0., max=GRID_SIZE - 1.000001)
    low = torch.floor(xy).long()
    high = torch.clamp(low + 1, max=GRID_SIZE - 1)
    frac = xy - low
    flat = grid.reshape(grid.shape[0], -1)

    def at(x, y):
        return flat.gather(1, (x * GRID_SIZE + y).unsqueeze(1)).squeeze(1)

    x0, y0 = low.unbind(-1)
    x1, y1 = high.unbind(-1)
    tx, ty = frac.unbind(-1)
    return ((1 - tx) * (1 - ty) * at(x0, y0)
            + tx * (1 - ty) * at(x1, y0)
            + (1 - tx) * ty * at(x0, y1)
            + tx * ty * at(x1, y1))


def _barrier(distance, config):
    safe = distance.clamp_min(1e-5)
    delta = safe - config.d_hat
    log = torch.log(safe / config.d_hat)
    energy = -(delta.square()) * log
    derivative = -(2 * delta * log + delta.square() / safe)
    active = distance < config.d_hat
    return (torch.where(active, energy.clamp(max=config.max_barrier_energy),
                        torch.zeros_like(distance)),
            torch.where(active, derivative.clamp(min=-config.max_barrier_grad),
                        torch.zeros_like(distance)))


def _disc_boxes(q, cells, radius):
    center = -6. + (cells.to(q.dtype) + .5) * RESOLUTION
    delta = q[:, None, :] - center
    outside = (delta.abs() - .1).clamp_min(0.)
    distance = torch.linalg.vector_norm(outside, dim=-1)
    outside_gradient = delta.sign() * outside / distance[..., None].clamp_min(1e-10)
    clearance_to_face = .1 - delta.abs()
    axis = clearance_to_face.argmin(dim=-1)
    face_gradient = torch.zeros_like(delta)
    face_gradient.scatter_(-1, axis[..., None],
                           torch.gather(delta.sign(), -1, axis[..., None]).where(
                               torch.gather(delta, -1, axis[..., None]) != 0,
                               torch.ones_like(axis[..., None], dtype=q.dtype)))
    inside_distance = -clearance_to_face.gather(-1, axis[..., None]).squeeze(-1)
    signed = torch.where(distance > 1e-10, distance, inside_distance) - radius[:, None]
    gradient = torch.where((distance > 1e-10)[..., None],
                           outside_gradient, face_gradient)
    return signed, gradient


def potential_force_batch(q, target, geometry, risk_grid, teammate,
                          radius, config=None):
    """Return [B] potential and [B,2] force with per-branch public beliefs."""
    config = config or TerrainForceConfig()
    batch = q.shape[0]
    displacement = q - target
    potential = .5 * config.beta_goal * displacement.square().sum(-1)
    force = -config.beta_goal * displacement

    cell = torch.floor((q + 6.) / RESOLUTION).long()
    offsets = torch.stack(torch.meshgrid(
        torch.arange(-4, 5, device=q.device),
        torch.arange(-4, 5, device=q.device), indexing="ij"), -1).reshape(1, 81, 2)
    nearby = cell[:, None, :] + offsets
    valid = ((nearby >= 0) & (nearby < GRID_SIZE)).all(-1)
    nearby = nearby.clamp(0, GRID_SIZE - 1)
    flat_index = nearby[..., 0] * GRID_SIZE + nearby[..., 1]
    occupied = geometry.reshape(batch, -1).gather(1, flat_index) == 1.
    occupied = occupied & valid
    signed, gradient = _disc_boxes(q, nearby, radius)
    energy, derivative = _barrier(signed, config)
    mask = occupied.to(q.dtype)
    potential = potential + config.alpha_ipc * (energy * mask).sum(-1)
    force = force - config.alpha_ipc * (derivative * mask)[..., None].mul(
        gradient).sum(1)

    delta = q - teammate
    distance = torch.linalg.vector_norm(delta, dim=-1)
    other_radius = torch.where(radius < .08, .10, .065)
    separation = distance - radius - other_radius
    teammate_config = TerrainForceConfig(
        d_hat=config.coordination_clearance,
        max_barrier_grad=config.max_barrier_grad,
        max_barrier_energy=config.max_barrier_energy)
    team_energy, team_derivative = _barrier(separation, teammate_config)
    team_mask = (separation < config.coordination_clearance) & (distance > 1e-9)
    potential = potential + team_energy * team_mask.to(q.dtype)
    force = force - (team_derivative * team_mask.to(q.dtype))[:, None] \
        * delta / distance[:, None].clamp_min(1e-9)

    risk = _interpolate(risk_grid, q)
    step = .02
    shift_x = q.new_tensor((step, 0.))
    shift_y = q.new_tensor((0., step))
    risk_gradient = torch.stack((
        (_interpolate(risk_grid, q + shift_x) -
         _interpolate(risk_grid, q - shift_x)) / (2 * step),
        (_interpolate(risk_grid, q + shift_y) -
         _interpolate(risk_grid, q - shift_y)) / (2 * step)), -1)
    potential = potential + config.lambda_risk * risk
    force = force - config.lambda_risk * risk_gradient
    return potential, _clip_norm(force, config.max_force)


def ph_step_batch(q, v, target, geometry, risk_grid, teammate,
                  mass, radius, max_speed, traction, dt, config=None):
    """Kick–drift–kick for B independent branches; all tensors remain on device."""
    config = config or TerrainForceConfig()
    potential_before, force_before = potential_force_batch(
        q, target, geometry, risk_grid, teammate, radius, config)
    p_half = mass[:, None] * v + .5 * dt * (
        traction[:, None] * force_before - config.gamma * v)
    q_next = q + dt * _clip_norm(p_half / mass[:, None], max_speed[:, None])
    potential_after, force_after = potential_force_batch(
        q_next, target, geometry, risk_grid, teammate, radius, config)
    p_next = p_half + .5 * dt * (
        traction[:, None] * force_after - config.gamma * p_half / mass[:, None])
    v_next = _clip_norm(p_next / mass[:, None], max_speed[:, None])
    diagnostics = {
        "storage_before": .5 * mass * v.square().sum(-1) + potential_before,
        "storage_after": .5 * mass * v_next.square().sum(-1) + potential_after,
        "force_norm": torch.linalg.vector_norm(force_before, dim=-1),
    }
    return q_next, v_next, diagnostics
