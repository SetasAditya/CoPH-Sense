"""Conditional-scout dispatch on the frozen material-pH continuation.

The actor-facing portion of this module only consumes public observations and
public task metadata.  Hidden terrain is confined to the paired evaluator.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .conditional_scout import (ScoutTask, run_carrier_primary_episode,
                                scout_dispatch_teacher,
                                scout_task_from_candidate)
from .e2_dev_stack import acquisition_menu
from .environment import CoPHTerrainEnv
from .generator import realize_map
from .pilot_environment import waypoint as mission_waypoint


@dataclass(frozen=True)
class DispatchCandidate:
    """Locally executable dispatch candidate; ``task=None`` denotes IDLE."""

    candidate_id: str
    task: Optional[ScoutTask]
    features: tuple


def _public_state_features(observation):
    kin = np.asarray(observation["kinematics"], dtype=np.float32)
    return (float(kin[0]) / 6., float(kin[1]) / 6.,
            float(kin[4]) / 6., float(kin[5]) / 6.,
            observation["remaining_steps"] / 1300.,
            observation["measurements_remaining"] / 4.,
            observation["team_packets_remaining"] / 4.)


def dispatch_candidates(env, max_candidates=4):
    """Construct IDLE and feasible scout tasks from public state only."""
    obs = env.observations()["scout"]
    candidates, _, _ = acquisition_menu(env, "scout", max_candidates)
    public = _public_state_features(obs)
    result = [DispatchCandidate("idle", None, public + (0.,) * 12)]
    for index, candidate in enumerate(candidates):
        task = scout_task_from_candidate(
            env, candidate, f"dispatch-{env.step_index}-{index}")
        result.append(DispatchCandidate(
            task.request_id, task, public + tuple(candidate.features())))
    return tuple(result)


def _replace_compatible_truth(env, realization_seed):
    """Create a paired posterior-support world with the same legal history.

    Already observed cells retain their realized values.  Unobserved material
    cells are drawn from another realization of the same public topology.  The
    actor observation, physics state, messages, and RNG snapshot are unchanged.
    This is a declared finite empirical conditional distribution, rather than
    a claim of exact Bayesian posterior sampling.
    """
    branch = env.clone()
    alternative = realize_map(branch.parent_map, int(realization_seed))
    surface = alternative.surface.copy()
    traction = alternative.traction.copy()
    for agent in ("scout", "carrier"):
        mask = np.isfinite(branch.known_surface[agent])
        surface[mask] = branch._truth.surface[mask]
        mask = np.isfinite(branch.known_traction[agent])
        traction[mask] = branch._truth.traction[mask]
    branch._truth = replace(branch._truth, surface=surface, traction=traction)
    return branch


def conditional_dispatch_teacher(snapshot, candidate, realization_seeds):
    """Estimate E[J_idle-J_dispatch | legal history] with paired branches."""
    if candidate.task is None:
        return {"candidate_id": candidate.candidate_id, "value": 0.,
                "world_values": [], "decision": "idle"}
    values, rows = [], []
    for seed in realization_seeds:
        world = _replace_compatible_truth(snapshot, seed)
        paired = scout_dispatch_teacher(world, candidate.task)
        value = float(paired["decision"]["value"])
        values.append(value)
        rows.append({"realization_seed": int(seed), "value": value,
                     "idle": paired["idle"], "dispatch": paired["dispatch"]})
    mean = float(np.mean(values))
    return {"candidate_id": candidate.candidate_id, "value": mean,
            "world_values": values, "decision": "dispatch" if mean > 0 else "idle",
            "branches": rows}


class DispatchValueNet(nn.Module):
    """Shared variable-candidate value scorer with IDLE fixed at zero."""

    def __init__(self, feature_dim=19, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def fit_dispatch_critic(rows, validation_rows, seed=0, epochs=300,
                        ranking_weight=.5, device="cpu"):
    """Fit value and within-state ranking losses; return model and diagnostics."""
    torch.manual_seed(int(seed))
    model = DispatchValueNet(len(rows[0]["features"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32, device=device)
    y = torch.tensor([r["value"] for r in rows], dtype=torch.float32, device=device)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["state_id"], []).append(index)
    for _ in range(int(epochs)):
        pred = model(x)
        loss = torch.mean((pred-y) ** 2)
        pairs = []
        for indices in groups.values():
            for a in indices:
                for b in indices:
                    if y[a] > y[b] + 1e-8:
                        pairs.append(torch.nn.functional.softplus(-(pred[a]-pred[b])))
        if pairs:
            loss = loss + ranking_weight * torch.stack(pairs).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model, evaluate_dispatch_critic(model, validation_rows, device)


def evaluate_dispatch_critic(model, rows, device="cpu"):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["state_id"], []).append(row)
    regrets, exact, useful, rejected = [], [], [], []
    with torch.inference_mode():
        for candidates in grouped.values():
            features = torch.tensor([r["features"] for r in candidates],
                                    dtype=torch.float32, device=device)
            predicted = model(features).cpu().numpy()
            true = np.asarray([r["value"] for r in candidates])
            chosen = int(np.argmax(predicted)); oracle = int(np.argmax(true))
            regrets.append(float(true[oracle]-true[chosen]))
            exact.append(chosen == oracle)
            if true[oracle] > 0:
                useful.append(chosen == oracle)
            if true[oracle] <= 0:
                rejected.append(chosen == oracle)
    return {"mean_regret": float(np.mean(regrets)),
            "exact_rate": float(np.mean(exact)),
            "useful_recall": float(np.mean(useful)) if useful else None,
            "unnecessary_rejection": float(np.mean(rejected)) if rejected else None,
            "states": len(grouped)}


def save_gate_record(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def make_material_env(family, parent_seed, realization_seed, seed, config):
    """Canonical carrier-primary environment used by the integrated gate."""
    return CoPHTerrainEnv(family, parent_seed, realization_seed, seed=seed,
                          config=replace(config, carrier_primary=True),
                          executor="material_ph", device="cpu")


def causal_material_rollout(snapshot, candidate):
    """Execute one free dispatch and audit belief -> force -> trajectory causality."""
    if candidate.task is None:
        raise ValueError("causal rollout requires a non-IDLE candidate")
    env = snapshot.clone()
    observations = env.observations()
    carrier_obs = observations["carrier"]
    goal = mission_waypoint(env, "carrier")
    before_force, _ = env._material_executor.proposal(
        env.positions["carrier"], np.zeros(2), goal, carrier_obs)
    before_surface = carrier_obs["known_surface"].copy()
    before_traction = carrier_obs["known_traction"].copy()
    delivered = {"seen": False, "force": None, "belief_cells": 0,
                 "step": None, "carrier_position": None}

    def observe(current, _controller):
        if delivered["seen"] or not current.received["carrier"]:
            return
        obs = current.observations()["carrier"]
        surface_changed = np.isfinite(obs["known_surface"]) & ~np.isfinite(before_surface)
        traction_changed = np.isfinite(obs["known_traction"]) & ~np.isfinite(before_traction)
        delivered["belief_cells"] = int(surface_changed.sum()+traction_changed.sum())
        delivered["force"], _ = current._material_executor.proposal(
            current.positions["carrier"], np.zeros(2),
            mission_waypoint(current, "carrier"), obs)
        delivered["step"] = int(current.step_index)
        delivered["carrier_position"] = current.positions["carrier"].tolist()
        delivered["seen"] = True

    outcome = run_carrier_primary_episode(env, candidate.task, observer=observe)
    after_force = delivered["force"]
    return {"outcome": outcome, "report_delivered": delivered["seen"],
            "delivered_step": delivered["step"],
            "belief_cells_changed": delivered["belief_cells"],
            "force_before": before_force.tolist(),
            "force_after_delivery": None if after_force is None else after_force.tolist(),
            "force_delta_norm": None if after_force is None else float(
                np.linalg.norm(after_force-before_force)),
            "carrier_at_delivery": delivered["carrier_position"],
            "carrier_final": env.positions["carrier"].tolist(),
            "trajectory_changed_after_report": bool(
                delivered["seen"] and np.linalg.norm(
                    env.positions["carrier"]-np.asarray(delivered["carrier_position"])) > .05)}
