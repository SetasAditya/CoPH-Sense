"""Adapter for the repository's trained material-aware Hamiltonian executor.

The adapter deliberately accepts an actor observation rather than an environment.
Consequently its coefficient network and force proposal cannot read latent terrain.
True traction remains a separate plant port, applied only by :func:`step`.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import ndimage

from .execution import belief_risk_grid
from .generator import GRID_SIZE, RESOLUTION


DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[2]
                      / "repair_experiments/outputs/"
                        "behavioral_soft_force_risk_encoder_recall_full/best.pt")


@dataclass(frozen=True)
class MaterialExecutorConfig:
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    patch_size: int = 32
    d_hat: float = 3.0
    d_hat_sdf: float = 3.0
    margin_factor: float = 0.5
    max_force: float = 4.0
    # Frozen by material_ph_calibration_v1.json using controller-only states.
    goal_geometry_scale: float = 0.010887502178108369
    context_scale: float = 8.0
    damping_scale: float = 0.17588677369457986


def _material_module():
    root = Path(__file__).resolve().parents[2]
    source = root / "full_code"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import train_material
    return train_material


@lru_cache(maxsize=4)
def _load_model(checkpoint, device):
    module = _material_module()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    model = module.CoefEnergyNetMaterial(
        patch_size=int(cfg.get("patch_size", 32)),
        lam_soft_max=float(cfg.get("lam_soft_max", 5.0)),
        lam_hard_max=float(cfg.get("lam_hard_max", 10.0)),
        mu_lat_max=float(cfg.get("mu_lat_max", 5.0)),
    ).to(device)
    state = payload.get("model_state_dict", payload.get("model", payload))
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _world_to_grid(position):
    """Convert VMAS metres (x,y) to the checkpoint's pixel (col,row) frame."""
    return (np.asarray(position, dtype=np.float32) + 6.0) / RESOLUTION - 0.5


def _extract_patch(grid, centre_xy, size, fill):
    # CoPH grids use [x,y]; torch image tensors use [row=y,col=x].
    image = np.asarray(grid, dtype=np.float32).T
    cx, cy = np.rint(centre_xy).astype(int)
    half = size // 2
    result = np.full((size, size), fill, dtype=np.float32)
    x0, y0 = cx - half, cy - half
    sx0, sx1 = max(0, x0), min(GRID_SIZE, x0 + size)
    sy0, sy1 = max(0, y0), min(GRID_SIZE, y0 + size)
    result[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = image[sy0:sy1, sx0:sx1]
    return result


def belief_patches(observation, position, patch_size=32):
    """Build network and rollout patches solely from delivered/public belief."""
    risk = belief_risk_grid(observation).astype(np.float32)
    occupied = np.asarray(observation["known_geometry"] == 1.0)
    # Distance and its gradient are public topology features. Distances are in
    # grid units, matching the coordinate system used by the source executor.
    sdf = ndimage.distance_transform_edt(~occupied).astype(np.float32)
    risk_yx, sdf_yx = risk.T, sdf.T
    grad_ry, grad_rx = np.gradient(risk_yx)
    grad_sy, grad_sx = np.gradient(sdf_yx)
    centre = _world_to_grid(position)
    rp = np.stack([
        _extract_patch(risk, centre, patch_size, .35),
        _extract_patch(sdf, centre, patch_size, 50.),
        _extract_patch(grad_rx.T, centre, patch_size, 0.),
        _extract_patch(grad_ry.T, centre, patch_size, 0.),
        _extract_patch(grad_sx.T, centre, patch_size, 0.),
        _extract_patch(grad_sy.T, centre, patch_size, 0.),
    ])
    network = np.stack((rp[0], _extract_patch(
        occupied.astype(np.float32), centre, patch_size, 1.)))
    return network.astype(np.float32), rp.astype(np.float32)


def _geometry_tokens(observation, goal):
    """Approximate public obstacle components by checkpoint-compatible circles."""
    occupied = np.asarray(observation["known_geometry"] == 1.0)
    labels, count = ndimage.label(occupied)
    centres, radii, weights = [], [], []
    goal_px = _world_to_grid(goal)
    for label in range(1, count + 1):
        coords = np.argwhere(labels == label)  # (x,y)
        if not len(coords):
            continue
        low, high = coords.min(0), coords.max(0)
        centre = .5 * (low + high).astype(np.float32)
        radius = .5 * float(np.linalg.norm(high - low + 1))
        centres.append(centre)
        radii.append(radius)
        weights.append(1.0)
    if not centres:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), goal_px)
    return (np.asarray(centres, np.float32), np.asarray(radii, np.float32),
            np.asarray(weights, np.float32), goal_px)


class MaterialHamiltonianExecutor:
    """Frozen learned coefficients with the source material force equations."""

    def __init__(self, config=None, device="cpu"):
        self.config = config or MaterialExecutorConfig()
        self.device = str(device)
        self.model = _load_model(self.config.checkpoint, self.device)
        self.source = _material_module()

    def __deepcopy__(self, memo):
        # The deployed model and frozen config are immutable. Counterfactual
        # environment clones may therefore share this executor safely instead
        # of duplicating ~200k Torch parameters for every branch.
        memo[id(self)] = self
        return self

    @torch.inference_mode()
    def prepare(self, position, goal, observation):
        q = _world_to_grid(position)
        C, R, W, goal_px = _geometry_tokens(observation, goal)
        network_patch, rollout_patch = belief_patches(
            observation, position, self.config.patch_size)
        obs = np.concatenate((C, R[:, None], W[:, None], goal_px[None]-C), 1)
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(
            x, dtype=dtype, device=self.device).unsqueeze(0)
        outputs = self.model(
            tensor(obs), tensor(np.ones(len(C), bool), torch.bool),
            tensor(np.r_[goal_px-q, np.linalg.norm(goal_px-q), 1.]),
            tensor(network_patch))
        names = ("alphas", "beta", "gamma", "lambda_soft",
                 "lambda_hard", "mu_lateral")
        coefficients = {name: value.detach().cpu().numpy()[0]
                        for name, value in zip(names, outputs)}
        return {"q_px": q, "goal_px": goal_px, "centres": C, "radii": R,
                "weights": W, "network_patch": network_patch,
                "rollout_patch": rollout_patch, "coefficients": coefficients}

    @torch.inference_mode()
    def force_components(self, position, velocity, goal, observation,
                         mass=1., radius=.1):
        """Return unscaled checkpoint force channels in VMAS force units."""
        state = self.prepare(position, goal, observation)
        c = state["coefficients"]
        q = torch.tensor(state["q_px"], device=self.device).view(1, 2)
        v = torch.tensor(np.asarray(velocity)/RESOLUTION,
                         dtype=torch.float32, device=self.device).view(1, 2)
        goal_px = torch.tensor(state["goal_px"], device=self.device).view(1, 2)
        C = torch.tensor(state["centres"], device=self.device).view(1, -1, 2)
        R = torch.tensor(state["radii"], device=self.device).view(1, -1)
        mask = torch.ones_like(R, dtype=torch.bool)
        alphas = torch.tensor(c["alphas"], device=self.device).view(1, -1)
        sem = self.source.bilinear_sample_patch(
            torch.tensor(state["rollout_patch"], device=self.device).unsqueeze(0), q, q)
        goal_force = -torch.tensor(c["beta"], device=self.device) * (q-goal_px)
        geometry_force = torch.zeros_like(goal_force)
        if C.shape[1]:
            diff = q[:, None]-C
            norm = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
            distance = norm-(R+self.config.margin_factor*(radius/RESOLUTION))
            _, derivative = self.source.ipc_piecewise(
                distance, torch.tensor([[self.config.d_hat]], device=self.device))
            geometry_force = (-(alphas*derivative)[..., None]
                              *(diff/norm[..., None])).sum(1)
        soft_force = -torch.tensor(c["lambda_soft"], device=self.device) * sem[:, 2:4]
        _, db = self.source._sdf_barrier_grad(sem[:, 1], self.config.d_hat_sdf)
        hard_force = (-torch.tensor(c["lambda_hard"], device=self.device)
                      * db[:, None] * sem[:, 4:6])
        damping_force = -torch.tensor(c["gamma"], device=self.device) * v
        components = {"goal": goal_force, "geometry": geometry_force,
                      "soft": soft_force, "hard": hard_force,
                      "damping": damping_force}
        return ({name: value[0].cpu().numpy() * RESOLUTION
                 for name, value in components.items()}, state)

    def proposal(self, position, velocity, goal, observation, mass=1., radius=.1):
        """Return the calibrated learned force before the hidden traction port."""
        components, state = self.force_components(
            position, velocity, goal, observation, mass=mass, radius=radius)
        force_m = (self.config.goal_geometry_scale
                   * (components["goal"] + components["geometry"])
                   + self.config.context_scale
                   * (components["soft"] + components["hard"])
                   + self.config.damping_scale * components["damping"])
        norm = np.linalg.norm(force_m)
        if norm > self.config.max_force:
            force_m *= self.config.max_force / norm
        state["force_components"] = components
        state["unclipped_force"] = force_m.copy()
        state["clipped"] = bool(norm > self.config.max_force)
        return force_m.astype(float), state

    @torch.inference_mode()
    def rollout(self, position, velocity, goal, observation, horizon_steps,
                dt, mass=1., radius=.1):
        """Run the repository's original material surrogate integrator.

        This is the evaluator/teacher path. Positions, velocities, obstacle
        radii, and time are supplied in the source checkpoint's pixel frame;
        returned positions, velocities, clearance, and path length are mapped
        back to VMAS metres.
        """
        state = self.prepare(position, goal, observation)
        c = state["coefficients"]
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(
            x, dtype=dtype, device=self.device).unsqueeze(0)
        outputs = self.source.integrate_surrogate_material(
            o0=tensor(state["q_px"]),
            v0=tensor(np.asarray(velocity, np.float32) / RESOLUTION),
            goal=tensor(state["goal_px"]),
            C=tensor(state["centres"]), R=tensor(state["radii"]),
            mask=tensor(np.ones(len(state["radii"]), bool), torch.bool),
            alphas=tensor(c["alphas"]), beta=tensor(c["beta"]),
            gamma=tensor(c["gamma"]), lam_soft=tensor(c["lambda_soft"]),
            lam_hard=tensor(c["lambda_hard"]),
            rollout_patch=tensor(state["rollout_patch"]),
            d_hat=tensor(self.config.d_hat), dt=tensor(float(dt)),
            H=tensor(int(horizon_steps), torch.long),
            robot_radius=float(radius / RESOLUTION),
            margin_factor=self.config.margin_factor, mass=float(mass),
            d_hat_sdf=self.config.d_hat_sdf)
        o, v, clearance, risk, hard_count, path_length = outputs
        return {
            "position": (o[0].cpu().numpy() + .5) * RESOLUTION - 6.,
            "velocity": v[0].cpu().numpy() * RESOLUTION,
            "min_clearance": float(clearance[0]) * RESOLUTION,
            "cumulative_risk": float(risk[0]) * RESOLUTION,
            "hard_count": float(hard_count[0]),
            "path_length": float(path_length[0]) * RESOLUTION,
            "coefficients": c,
        }

    def step(self, position, velocity, goal, observation, mass, radius,
             max_speed, traction, dt):
        force, state = self.proposal(position, velocity, goal, observation,
                                     mass=mass, radius=radius)
        velocity = np.asarray(velocity, dtype=float)
        v_next = velocity + dt * float(traction) * force / float(mass)
        speed = np.linalg.norm(v_next)
        if speed > max_speed:
            v_next *= max_speed / speed
        q_next = np.asarray(position, dtype=float) + dt * v_next
        c = state["coefficients"]
        return q_next, v_next, {
            "force_norm": float(np.linalg.norm(force)),
            "traction_port_gain": float(traction),
            "learned_beta": float(c["beta"]),
            "learned_gamma": float(c["gamma"]),
            "learned_lambda_soft": float(c["lambda_soft"]),
            "learned_lambda_hard": float(c["lambda_hard"]),
            "checkpoint": self.config.checkpoint,
        }
