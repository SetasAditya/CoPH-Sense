"""Finite-prior conditional-scout dispatch with material-pH continuations."""

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.special import log_ndtr
from torch import nn

from .conditional_scout import (ConditionalScoutController, ScoutMode,
                                ScoutTask, scout_dispatch_teacher,
                                scout_task_from_candidate)
from .environment import (AGENTS, CoPHTerrainEnv, TerrainAction, TerrainConfig,
                          TerrainLedger)
from .generator import realize_map
from .planning import candidate_viewpoints, plan_path
from .pilot_environment import waypoint as mission_waypoint


@dataclass(frozen=True)
class DispatchCandidate:
    candidate_id: str
    task: Optional[ScoutTask]
    features: tuple
    feasible: bool = True
    heuristic_score: float = 0.
    unknown_fraction: float = 0.


@dataclass(frozen=True)
class TerrainHypothesis:
    realization_seed: int
    prior_weight: float
    map_sha256: str


@dataclass(frozen=True)
class FiniteTerrainPrior:
    family: str
    parent_seed: int
    hypotheses: tuple

    @classmethod
    def build(cls, snapshot, realization_seeds, weights=None):
        seeds = tuple(int(seed) for seed in realization_seeds)
        if not seeds:
            raise ValueError("finite terrain prior cannot be empty")
        raw = np.ones(len(seeds), dtype=float) if weights is None else np.asarray(weights, float)
        if raw.shape != (len(seeds),) or np.any(raw <= 0) or not np.isfinite(raw).all():
            raise ValueError("prior weights must be finite and positive")
        raw /= raw.sum()
        entries = [TerrainHypothesis(seed, float(weight),
                     realize_map(snapshot.parent_map, seed).digest())
                   for seed, weight in zip(seeds, raw)]
        return cls(snapshot.family, snapshot.parent_seed, tuple(entries))

    def manifest(self):
        return {"family": self.family, "parent_seed": self.parent_seed,
                "hypotheses": [asdict(item) for item in self.hypotheses]}


@dataclass
class DispatchHistory:
    """Carrier-legal history; uncommunicated scout evidence is excluded."""

    family: str
    parent_seed: int
    seed: int
    config: dict
    executor: str
    actions: tuple
    carrier_observation: dict
    public_position_trace: np.ndarray
    carrier_evidence: tuple
    step_index: int

    @classmethod
    def from_env(cls, env):
        observation = env.observations()["carrier"]
        evidence = tuple(sorted(
            (item.evidence_id, item.modality, tuple(item.cells),
             tuple(float(value) for value in item.values), item.acquired_step)
            for item in list(env.owned["carrier"].values())
            + list(env.received["carrier"].values())))
        trace = np.asarray([[row["positions"][name] for name in AGENTS]
                            for row in env.state_trace], dtype=np.float32)
        return cls(env.family, env.parent_seed, env.seed, asdict(env.config),
                   env.executor, tuple(copy.deepcopy(env.action_trace)),
                   copy.deepcopy(observation), trace, evidence, int(env.step_index))


def _clipped_gaussian_log_likelihood(observed, mean, sigma):
    observed, mean = np.asarray(observed, float), np.asarray(mean, float)
    sigma = float(sigma)
    if sigma <= 0:
        return 0. if np.allclose(observed, np.clip(mean, 0., 1.)) else -np.inf
    z = (observed-mean)/sigma
    result = -.5*z*z-math.log(sigma)-.5*math.log(2.*math.pi)
    low, high = observed <= 0., observed >= 1.
    result[low] = log_ndtr(-mean[low]/sigma)
    result[high] = log_ndtr((mean[high]-1.)/sigma)
    return float(result.sum())


def _carrier_evidence_signature(env):
    return tuple(sorted(
        (item.evidence_id, item.modality, tuple(item.cells),
         tuple(float(value) for value in item.values), item.acquired_step)
        for item in list(env.owned["carrier"].values())
        + list(env.received["carrier"].values())))


def replay_hypothesis(history, hypothesis, position_tolerance=.025):
    """Replay one complete map and test compatibility with the legal history."""
    env = CoPHTerrainEnv(history.family, history.parent_seed,
                         hypothesis.realization_seed, seed=history.seed,
                         config=TerrainConfig(**history.config),
                         executor=history.executor, device="cpu")
    try:
        for row in history.actions:
            env.step({name: TerrainAction(**row[name]) for name in AGENTS})
    except Exception as error:
        return None, {"reason": "replay_error", "detail": str(error)}
    replay_trace = np.asarray([[row["positions"][name] for name in AGENTS]
                               for row in env.state_trace], dtype=np.float32)
    if replay_trace.shape != history.public_position_trace.shape:
        return None, {"reason": "public_trace_shape"}
    max_error = float(np.max(np.linalg.norm(
        replay_trace-history.public_position_trace, axis=-1)))
    if max_error > position_tolerance:
        return None, {"reason": "public_motion", "max_position_error": max_error}
    if _carrier_evidence_signature(env) != history.carrier_evidence:
        return None, {"reason": "carrier_measurement"}
    observed = np.asarray(history.carrier_observation["appearance"])
    mask = np.isfinite(observed)
    predicted = np.clip(.86-.65*env._truth.risk, 0., 1.)
    likelihood = _clipped_gaussian_log_likelihood(
        observed[mask], predicted[mask], env.config.appearance_noise_std)
    return env, {"reason": None, "log_likelihood": likelihood,
                 "appearance_cell_count": int(mask.sum()),
                 "max_position_error": max_error}


def posterior_support(history, prior):
    if prior.family != history.family or prior.parent_seed != history.parent_seed:
        raise ValueError("prior and history use different parent maps")
    accepted, rejected = [], []
    for hypothesis in prior.hypotheses:
        env, audit = replay_hypothesis(history, hypothesis)
        if env is None or not np.isfinite(audit.get("log_likelihood", -np.inf)):
            rejected.append({"realization_seed": hypothesis.realization_seed, **audit})
        else:
            accepted.append((hypothesis, env, audit))
    if not accepted:
        raise RuntimeError("finite-prior posterior has empty compatible support")
    logs = np.asarray([math.log(row[0].prior_weight)+row[2]["log_likelihood"]
                       for row in accepted])
    weights = np.exp(logs-logs.max()); weights /= weights.sum()
    entries = [{"hypothesis": hypothesis, "env": env,
                "posterior_weight": float(weight), "audit": audit}
               for weight, (hypothesis, env, audit) in zip(weights, accepted)]
    return entries, {"accepted": len(entries), "rejected": rejected,
                     "effective_sample_size": float(1./np.sum(weights**2)),
                     "weights": [{"realization_seed": row["hypothesis"].realization_seed,
                                  "weight": row["posterior_weight"]}
                                 for row in entries]}


def _public_state_features(carrier, scout):
    carrier_kin = np.asarray(carrier["kinematics"], dtype=np.float32)
    scout_kin = np.asarray(scout["kinematics"], dtype=np.float32)
    return tuple(np.r_[carrier_kin[:4]/6., scout_kin[:4]/6.,
                       carrier["remaining_steps"]/1300.,
                       carrier["measurements_remaining"]/4.,
                       carrier["team_packets_remaining"]/4.,
                       len(carrier["received_evidence_ids"])/4.,
                       len(carrier["acknowledged_evidence_ids"])/4.].astype(float))


def _belief_task_features(carrier, candidate):
    region = candidate.target_region or candidate.viewpoint
    x, y = region
    window = (slice(max(0, x-3), min(60, x+4)),
              slice(max(0, y-3), min(60, y+4)))
    values = []
    for name in ("appearance", "known_surface", "known_traction"):
        observed = np.asarray(carrier[name][window], dtype=float)
        finite = observed[np.isfinite(observed)]
        values.extend((float(finite.mean()) if finite.size else 0.,
                       float(finite.std()) if finite.size else 0.,
                       float(finite.size/observed.size)))
    return tuple(values)


def dispatch_candidates(env, max_candidates=4, maneuver_lead_seconds=1.0):
    """Build carrier-selected scout tasks from carrier belief and public state."""
    observations = env.observations()
    carrier, scout = observations["carrier"], observations["scout"]
    hybrid = copy.deepcopy(carrier)
    hybrid["kinematics"] = scout["kinematics"].copy()
    hybrid["measurements_remaining"] = scout["measurements_remaining"]
    public = _public_state_features(carrier, scout)
    proposed = candidate_viewpoints(hybrid, env.config, "scout",
                                    max_candidates=max_candidates)
    result = [DispatchCandidate("idle", None, public+(0.,)*21, True, 0., 0.)]
    carrier_path, _ = plan_path(carrier)
    for index, candidate in enumerate(proposed):
        if not carrier_path or candidate.target_region is None:
            continue
        region = np.asarray(candidate.target_region)
        route_index = int(np.argmin([np.linalg.norm(np.asarray(cell)-region)
                                     for cell in carrier_path]))
        useful_seconds = max(0., route_index*.2/.55-float(maneuver_lead_seconds))
        deadline = env.step_index+int(useful_seconds/env.config.dt)
        required = (candidate.travel_time+candidate.dwell_time
                    + env.config.communication_delay_steps*env.config.dt)
        feasible = bool(useful_seconds > required and deadline > env.step_index)
        task = scout_task_from_candidate(env, candidate,
            f"dispatch-{env.step_index}-{index}", deadline_step=deadline)
        task = replace(task, actionability_slack_steps=max(0, deadline-env.step_index),
                       provenance="carrier_public_belief")
        heuristic = (candidate.unknown_fraction-candidate.sensing_cost
                     -candidate.travel_time*env.config.time_cost_per_step/env.config.dt)
        result.append(DispatchCandidate(task.request_id, task,
            public+tuple(candidate.features())+_belief_task_features(carrier, candidate),
            feasible, float(heuristic), float(candidate.unknown_fraction)))
    return tuple(result)


def conditional_dispatch_teacher(history, prior, candidate,
                                 packet_seeds=(0, 1, 2, 3)):
    """Compute finite-prior conditional value with common packet schedules."""
    if candidate.task is None:
        return {"candidate_id": "idle", "value": 0., "stderr": 0.,
                "interval": (0., 0.), "classification": "idle",
                "posterior": None, "branches": []}
    if not candidate.feasible:
        return {"candidate_id": candidate.candidate_id, "value": -np.inf,
                "stderr": 0., "interval": (-np.inf, -np.inf),
                "classification": "infeasible", "posterior": None, "branches": []}
    support, posterior = posterior_support(history, prior)
    branches, means, weights, within = [], [], [], []
    for entry in support:
        values = []
        for packet_seed in packet_seeds:
            env = entry["env"].clone()
            rng = np.random.default_rng(np.random.SeedSequence([
                prior.parent_seed, entry["hypothesis"].realization_seed,
                int(packet_seed), 7441]))
            env.set_packet_random_schedule(rng.random(64))
            paired = scout_dispatch_teacher(env, candidate.task)
            value = float(paired["decision"]["value"]); values.append(value)
            branches.append({"realization_seed": entry["hypothesis"].realization_seed,
                             "packet_seed": int(packet_seed), "value": value,
                             "idle": paired["idle"], "dispatch": paired["dispatch"]})
        means.append(float(np.mean(values)))
        within.append(float(np.var(values, ddof=1)) if len(values) > 1 else 0.)
        weights.append(entry["posterior_weight"])
    weights, means = np.asarray(weights), np.asarray(means)
    value = float(weights@means)
    finite_variance = float(weights@((means-value)**2))
    mc_variance = float(np.sum(weights**2*np.asarray(within)/max(1, len(packet_seeds))))
    stderr = math.sqrt(max(0., finite_variance/max(1., posterior["effective_sample_size"])
                           + mc_variance))
    interval = (value-1.96*stderr, value+1.96*stderr)
    classification = "useful" if interval[0] > 0 else (
        "unnecessary" if interval[1] < 0 else "unresolved")
    return {"candidate_id": candidate.candidate_id, "value": value,
            "stderr": stderr, "interval": interval, "classification": classification,
            "posterior": posterior, "finite_prior_variance": finite_variance,
            "packet_mc_variance": mc_variance, "branches": branches}


def _continue_with_controller(env, controller, replan_period=20):
    carrier_target = mission_waypoint(env, "carrier")
    while not env.done:
        if env.step_index % replan_period == 0:
            carrier_target = mission_waypoint(env, "carrier")
        env.step({"scout": controller.action(env),
                  "carrier": TerrainAction(waypoint=carrier_target)})
    return {"score": float(env.ledger.mission_cost+env.ledger.material_exposure),
            "success": bool(env.success), "failure_reason": env.failure_reason,
            "ledger": asdict(env.ledger), "recovered": bool(
                np.linalg.norm(env.positions["scout"]-env.positions["carrier"])
                <= env.config.scout_recovery_radius),
            "outstanding_packets": len(env.pending)}


def delivery_causal_intervention(snapshot, candidate):
    """Compare delivered/withheld evidence at one identical physical state."""
    if candidate.task is None or not candidate.feasible:
        raise ValueError("causal intervention requires a feasible task")
    env = snapshot.clone()
    controller = ConditionalScoutController(home=tuple(env.positions["scout"]))
    controller.dispatch(env, candidate.task)
    carrier_target = mission_waypoint(env, "carrier")
    target_packet = None
    while not env.done:
        due = [packet for packet in env.pending if packet.kind == "evidence"
               and packet.delivery_step <= env.step_index]
        if due:
            target_packet = due[0]
            break
        if env.step_index % 20 == 0:
            carrier_target = mission_waypoint(env, "carrier")
        env.step({"scout": controller.action(env),
                  "carrier": TerrainAction(waypoint=carrier_target)})
    if target_packet is None:
        return {"passed": False, "reason": "no_pre_delivery_state"}
    delivered, withheld = env.clone(), env.clone()
    delivered_controller, withheld_controller = copy.deepcopy(controller), copy.deepcopy(controller)
    incremental = TerrainLedger()
    delivered._deliver_packets(incremental)
    delivered.ledger.add(incremental)
    withheld.pending = [packet for packet in withheld.pending
                        if not (packet.kind == "evidence"
                                and packet.evidence_id == target_packet.evidence_id)]
    for branch_controller, outcome in ((delivered_controller, "delivered"),
                                       (withheld_controller, "withheld")):
        branch_controller.mode = ScoutMode.RETURN
        branch_controller.last_outcome = outcome
    delivered_obs = delivered.observations()["carrier"]
    withheld_obs = withheld.observations()["carrier"]
    position = delivered.positions["carrier"]
    velocity = delivered.physics.world.agents[1].state.vel[0].detach().cpu().numpy().copy()
    goal = mission_waypoint(delivered, "carrier")
    force_delivered, _ = delivered._material_executor.proposal(
        position, velocity, goal, delivered_obs)
    force_withheld, _ = withheld._material_executor.proposal(
        position, velocity, goal, withheld_obs)
    start_delivered, start_withheld = len(delivered.state_trace), len(withheld.state_trace)
    delivered_result = _continue_with_controller(delivered, delivered_controller)
    withheld_result = _continue_with_controller(withheld, withheld_controller)
    delivered_trace = np.asarray([[row["positions"]["carrier"]
                                   for row in delivered.state_trace[start_delivered:]]])
    withheld_trace = np.asarray([[row["positions"]["carrier"]
                                 for row in withheld.state_trace[start_withheld:]]])
    count = min(delivered_trace.shape[1], withheld_trace.shape[1])
    divergence = (0. if count == 0 else float(np.max(np.linalg.norm(
        delivered_trace[0, :count]-withheld_trace[0, :count], axis=-1))))
    belief_changed = bool(target_packet.evidence_id in delivered.received["carrier"]
                          and target_packet.evidence_id not in withheld.received["carrier"])
    return {"passed": bool(belief_changed
                           and np.linalg.norm(force_delivered-force_withheld) > 1e-8
                           and divergence > 1e-4),
            "evidence_id": target_packet.evidence_id,
            "same_position": bool(np.array_equal(delivered.positions["carrier"],
                                                   withheld.positions["carrier"])),
            "same_velocity": True, "belief_changed": belief_changed,
            "force_delivered": force_delivered.tolist(),
            "force_withheld": force_withheld.tolist(),
            "force_delta_norm": float(np.linalg.norm(force_delivered-force_withheld)),
            "trajectory_max_divergence": divergence,
            "delivered": delivered_result, "withheld": withheld_result}


class DispatchValueNet(nn.Module):
    def __init__(self, feature_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def candidate_scores(model, candidates, device="cpu"):
    scores = np.full(len(candidates), -np.inf); scores[0] = 0.
    indices = [i for i, candidate in enumerate(candidates[1:], 1) if candidate.feasible]
    if indices:
        x = torch.tensor([candidates[i].features for i in indices],
                         dtype=torch.float32, device=device)
        with torch.inference_mode(): scores[indices] = model(x).cpu().numpy()
    return scores


def fit_dispatch_critic(rows, validation_rows, seed=0, epochs=300,
                        ranking_weight=.5, device="cpu"):
    train = [row for row in rows if row["candidate_id"] != "idle"
             and np.isfinite(row["value"])]
    if not train: raise ValueError("no finite non-IDLE training rows")
    torch.manual_seed(int(seed))
    model = DispatchValueNet(len(train[0]["features"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.tensor([row["features"] for row in train], dtype=torch.float32, device=device)
    y = torch.tensor([row["value"] for row in train], dtype=torch.float32, device=device)
    groups = {}
    for index, row in enumerate(train): groups.setdefault(row["state_id"], []).append(index)
    for _ in range(int(epochs)):
        prediction = model(x); loss = torch.mean((prediction-y)**2)
        pairs = [torch.nn.functional.softplus(-(prediction[a]-prediction[b]))
                 for indices in groups.values() for a in indices for b in indices
                 if y[a] > y[b]+1e-8]
        if pairs: loss = loss+ranking_weight*torch.stack(pairs).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model, evaluate_dispatch_critic(model, validation_rows, device)


def _optimal_indices(values, tolerance=1e-8):
    best = float(np.max(values))
    return {index for index, value in enumerate(values) if value >= best-tolerance}


def evaluate_dispatch_critic(model, rows, device="cpu"):
    grouped = {}
    for row in rows: grouped.setdefault(row["state_id"], []).append(row)
    regrets, exact, useful, rejected = [], [], [], []
    for candidates in grouped.values():
        candidates = sorted(candidates, key=lambda row: row["candidate_index"])
        predicted = np.full(len(candidates), -np.inf); predicted[0] = 0.
        feasible = [i for i, row in enumerate(candidates[1:], 1)
                    if row["feasible"] and np.isfinite(row["value"])]
        if feasible:
            x = torch.tensor([candidates[i]["features"] for i in feasible],
                             dtype=torch.float32, device=device)
            with torch.inference_mode(): predicted[feasible] = model(x).cpu().numpy()
        true = np.asarray([row["value"] for row in candidates])
        chosen = int(np.argmax(predicted)); optimal = _optimal_indices(true)
        regrets.append(float(np.max(true)-true[chosen])); exact.append(chosen in optimal)
        if np.max(true[1:]) > 0: useful.append(chosen in optimal and chosen != 0)
        if np.max(true[1:]) <= 0: rejected.append(chosen == 0)
    return {"mean_regret": float(np.mean(regrets)), "exact_rate": float(np.mean(exact)),
            "useful_recall": float(np.mean(useful)) if useful else None,
            "useful_denominator": len(useful),
            "unnecessary_rejection": float(np.mean(rejected)) if rejected else None,
            "unnecessary_denominator": len(rejected), "states": len(grouped)}


def save_gate_record(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)+"\n"
    path.write_text(text); return hashlib.sha256(text.encode()).hexdigest()
