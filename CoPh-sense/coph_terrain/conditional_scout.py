"""Carrier-primary navigation with an on-demand sequential scout.

The carrier is the persistent mission planner.  The scout is dormant unless a
carrier-side value gate dispatches a concrete sensing task.  The scout then
travels, acquires one physical measurement, returns/transmits as required by
the existing radio model, and goes back to its home position.

This module deliberately keeps the scout controller simple and inspectable.
The learning target is the *ex-ante dispatch value*, not an end-to-end MARL
policy over two continuously active agents.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .environment import (AGENTS, TerrainAction, cell_to_position,
                          position_to_cell)
from .planning import plan_path
from .pilot_environment import waypoint as mission_waypoint


class ScoutMode(str, Enum):
    IDLE = "idle"
    DISPATCH = "dispatch"
    ACQUIRE = "acquire"
    REPORT = "report"
    RETURN = "return"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ScoutTask:
    """One causal information request issued by the carrier.

    ``viewpoint`` is the requested grid cell under the existing sensing API.
    ``latest_useful_step`` is a decision deadline: evidence arriving after this
    point is not allowed to count as a successful information intervention.
    ``predicted_value`` and ``max_task_cost`` are policy-visible bookkeeping
    fields and are never populated from hidden simulator truth at deployment.
    """

    request_id: str
    modality: str
    viewpoint: Tuple[int, int]
    latest_useful_step: int
    target_region: Optional[Tuple[int, int]] = None
    predicted_value: float = 0.0
    max_task_cost: float = float("inf")
    acquisition_dwell_steps: int = 0
    estimated_delivery_cost: float = 0.0
    provenance: str = "public_candidate"
    actionability_slack_steps: int = 0

    def __post_init__(self):
        if self.modality not in ("geometry", "traction"):
            raise ValueError(self.modality)
        if self.latest_useful_step < 0:
            raise ValueError("latest_useful_step must be nonnegative")

    def sense_payload(self):
        return (self.modality, int(self.viewpoint[0]), int(self.viewpoint[1]))


@dataclass(frozen=True)
class DispatchDecision:
    dispatch: bool
    idle_cost: float
    dispatch_cost: float
    value: float
    margin: float


def dispatch_decision(idle_cost: float, dispatch_cost: float,
                      margin: float = 0.0) -> DispatchDecision:
    """Value gate for the ex-ante scout decision.

    Positive ``value`` means the complete dispatched continuation is cheaper
    than the no-scout continuation.  The dispatch branch must include scout
    travel, dwell, radio, carrier delay/waiting, and downstream navigation.
    """

    idle_cost = float(idle_cost)
    dispatch_cost = float(dispatch_cost)
    value = idle_cost - dispatch_cost
    return DispatchDecision(value > float(margin), idle_cost, dispatch_cost,
                            value, float(margin))


def _grid_waypoint(observation, target_cell, lookahead=5):
    path, _ = plan_path(observation, goal=tuple(target_cell))
    if not path:
        return tuple(float(x) for x in observation["kinematics"][:2])
    cell = path[min(len(path) - 1, int(lookahead))]
    return tuple(float(x) for x in cell_to_position(cell))


class ConditionalScoutController:
    """Deterministic execution state machine for one dispatched scout task."""

    def __init__(self, home=(-5.0, 0.25), replan_period=10):
        self.home = tuple(float(x) for x in home)
        self.replan_period = int(replan_period)
        self.mode = ScoutMode.IDLE
        self.task: Optional[ScoutTask] = None
        self.evidence_id: Optional[str] = None
        self._owned_before = set()
        self._report_attempted = False
        self._target_waypoint = self.home
        self.last_outcome = None

    @property
    def active(self):
        return self.mode not in (ScoutMode.IDLE, ScoutMode.EXPIRED)

    def dispatch(self, env, task: ScoutTask):
        if self.active:
            raise RuntimeError("scout already has an active task")
        if env.step_index >= task.latest_useful_step:
            self.mode = ScoutMode.EXPIRED
            self.task = task
            self.last_outcome = "deadline_before_dispatch"
            return False
        self.task = task
        self.mode = ScoutMode.DISPATCH
        self.evidence_id = None
        self._owned_before = set(env.owned["scout"])
        self._report_attempted = False
        self._target_waypoint = tuple(float(x) for x in env.positions["scout"])
        self.last_outcome = None
        env.scout_ever_dispatched = True
        env.events.append({"type": "scout_dispatch", "step": env.step_index,
                           "request_id": task.request_id,
                           "modality": task.modality,
                           "viewpoint": tuple(task.viewpoint),
                           "latest_useful_step": task.latest_useful_step,
                           "predicted_value": task.predicted_value})
        return True

    def _expire(self, env, reason):
        env.events.append({"type": "scout_task_expired", "step": env.step_index,
                           "request_id": None if self.task is None else self.task.request_id,
                           "reason": reason})
        self.last_outcome = reason
        self.mode = ScoutMode.RETURN

    def action(self, env) -> TerrainAction:
        """Return the scout's next action using only its legal observation."""
        if self.mode in (ScoutMode.IDLE, ScoutMode.EXPIRED):
            return TerrainAction(waypoint=tuple(float(x) for x in env.positions["scout"]))

        if self.task is None:
            raise RuntimeError("active scout has no task")

        if env.step_index >= self.task.latest_useful_step and self.mode != ScoutMode.RETURN:
            self._expire(env, "decision_deadline")

        obs = env.observations()["scout"]
        pos = np.asarray(env.positions["scout"], dtype=float)

        if self.mode == ScoutMode.DISPATCH:
            target = cell_to_position(self.task.viewpoint)
            velocity = np.asarray(obs["kinematics"][2:4], dtype=float)
            if (np.linalg.norm(pos - target) <= .5 * env.config.probe_range and
                    np.linalg.norm(velocity) <= .12 and
                    env.active_sense["scout"] is None):
                self.mode = ScoutMode.ACQUIRE
                return TerrainAction(waypoint=tuple(float(x) for x in pos),
                                     sense=self.task.sense_payload())
            if env.step_index % self.replan_period == 0:
                self._target_waypoint = _grid_waypoint(obs, self.task.viewpoint)
            return TerrainAction(waypoint=self._target_waypoint)

        if self.mode == ScoutMode.ACQUIRE:
            new_ids = set(env.owned["scout"]) - self._owned_before
            if new_ids:
                self.evidence_id = sorted(new_ids)[0]
                evidence = env.owned["scout"][self.evidence_id]
                target = self.task.target_region or self.task.viewpoint
                if target not in evidence.cells:
                    self._expire(env, "target_not_covered")
                    return TerrainAction(waypoint=tuple(float(x) for x in pos))
                self.mode = ScoutMode.REPORT
                env.events.append({"type": "scout_evidence_ready",
                                   "step": env.step_index,
                                   "request_id": self.task.request_id,
                                   "evidence_id": self.evidence_id})
            elif env.active_sense["scout"] is None:
                self._expire(env, "acquisition_failed")
            return TerrainAction(waypoint=tuple(float(x) for x in pos))

        if self.mode == ScoutMode.REPORT:
            if self.evidence_id is None:
                self._expire(env, "missing_evidence")
                return TerrainAction(waypoint=tuple(float(x) for x in pos))
            if self.evidence_id in env.received["carrier"]:
                self.mode = ScoutMode.RETURN
                self.last_outcome = "delivered"
                env.events.append({"type": "scout_report_delivered",
                                   "step": env.step_index,
                                   "request_id": self.task.request_id,
                                   "evidence_id": self.evidence_id,
                                   "timely": env.step_index < self.task.latest_useful_step})
                return TerrainAction(waypoint=tuple(
                    float(x) for x in env.positions["carrier"]))
            carrier = np.asarray(env.positions["carrier"], dtype=float)
            in_range = np.linalg.norm(pos - carrier) <= env.config.communication_range
            pending = any(packet.kind == "evidence" and
                          packet.evidence_id == self.evidence_id
                          for packet in env.pending)
            if in_range and not pending:
                # Retry only after a dropped attempt has cleared from the queue.
                self._report_attempted = True
                return TerrainAction(waypoint=tuple(float(x) for x in pos),
                                     send_evidence_id=self.evidence_id)
            if not in_range:
                carrier_cell = position_to_cell(carrier)
                if env.step_index % self.replan_period == 0:
                    self._target_waypoint = _grid_waypoint(obs, carrier_cell)
                return TerrainAction(waypoint=self._target_waypoint)
            return TerrainAction(waypoint=tuple(float(x) for x in pos))

        if self.mode == ScoutMode.RETURN:
            recovery = np.asarray(env.positions["carrier"], dtype=float)
            if np.linalg.norm(pos - recovery) <= env.config.scout_recovery_radius:
                request_id = self.task.request_id
                env.events.append({"type": "scout_recovered",
                                   "step": env.step_index,
                                   "request_id": request_id,
                                   "outcome": self.last_outcome})
                self.mode = ScoutMode.IDLE
                self.task = None
                self.evidence_id = None
                return TerrainAction(waypoint=tuple(float(x) for x in pos))
            home_cell = position_to_cell(recovery)
            if env.step_index % self.replan_period == 0:
                self._target_waypoint = _grid_waypoint(obs, home_cell)
            return TerrainAction(waypoint=self._target_waypoint)

        raise RuntimeError(self.mode)


def scout_task_from_candidate(env, candidate, request_id,
                              value_estimate=0.0, deadline_step=None):
    """Convert an existing observation-only sensing candidate into a task."""
    if deadline_step is None:
        slack_steps = max(1, int(float(candidate.deadline_slack) / env.config.dt))
        deadline_step = min(env.config.horizon_steps, env.step_index + slack_steps)
    return ScoutTask(
        request_id=str(request_id),
        modality=candidate.modality,
        viewpoint=tuple(candidate.viewpoint),
        target_region=(None if candidate.target_region is None else
                       tuple(candidate.target_region)),
        latest_useful_step=int(deadline_step),
        predicted_value=float(value_estimate),
        acquisition_dwell_steps=(env.config.scan_dwell_steps
                                 if candidate.modality == "geometry"
                                 else env.config.probe_dwell_steps),
        estimated_delivery_cost=float(
            env.config.communication_attempt_cost
            + env.config.byte_cost * (env.config.header_bytes + 20)),
        provenance="carrier_public_belief",
        actionability_slack_steps=max(0, int(deadline_step-env.step_index)),
    )


def scout_dispatch_teacher(snapshot, task: ScoutTask, margin=0.0,
                           hold_carrier_until_report=False):
    """Paired ex-ante teacher for IDLE versus one complete scout task.

    Unlike the historical A4 SEND/HOLD teacher, this intervention occurs
    *before* scout travel and sensing, so every acquisition and motion cost is
    included in the dispatch branch.  Hidden simulator state is used only to
    evaluate the paired continuations, never as a deployed actor input.
    """
    idle_env = snapshot.clone()
    dispatch_env = snapshot.clone()
    idle = run_carrier_primary_episode(
        idle_env, task=None, hold_carrier_until_report=False)
    dispatched = run_carrier_primary_episode(
        dispatch_env, task=task,
        hold_carrier_until_report=hold_carrier_until_report)
    decision = dispatch_decision(idle["score"], dispatched["score"], margin)
    return {
        "idle": idle,
        "dispatch": dispatched,
        "decision": {
            "dispatch": decision.dispatch,
            "idle_cost": decision.idle_cost,
            "dispatch_cost": decision.dispatch_cost,
            "value": decision.value,
            "margin": decision.margin,
        },
    }


def run_carrier_primary_episode(env, task: Optional[ScoutTask] = None,
                                hold_carrier_until_report=False,
                                replan_period=20, observer=None):
    """Small executable reference policy for conditional-scout experiments.

    This is intentionally a baseline, not a learned policy.  The carrier uses
    the repository's observation-only mission planner.  The scout remains at
    home unless ``task`` is supplied.
    """
    if not env.config.carrier_primary:
        raise ValueError("run_carrier_primary_episode requires carrier_primary=True")
    controller = ConditionalScoutController(home=tuple(env.positions["scout"]))
    if task is not None:
        controller.dispatch(env, task)
    carrier_target = mission_waypoint(env, "carrier")
    while not env.done:
        if env.step_index % int(replan_period) == 0:
            carrier_target = mission_waypoint(env, "carrier")
        if hold_carrier_until_report and controller.active and controller.last_outcome != "delivered":
            carrier_target = tuple(float(x) for x in env.positions["carrier"])
        scout_action = controller.action(env)
        env.step({"scout": scout_action,
                  "carrier": TerrainAction(waypoint=carrier_target)})
        if observer is not None:
            observer(env, controller)
    return {
        "success": bool(env.success),
        "failure_reason": env.failure_reason,
        "steps": int(env.step_index),
        "mission_cost": float(env.ledger.mission_cost),
        "material_exposure": float(env.ledger.material_exposure),
        "score": float(env.ledger.mission_cost + env.ledger.material_exposure),
        "scout_outcome": controller.last_outcome,
        "scout_final_mode": controller.mode.value,
        "measurements": dict(env.measurement_count),
        "packets": int(env.packet_attempts),
        "recovered": bool(not env.scout_ever_dispatched or
                          np.linalg.norm(env.positions["scout"]-
                                         env.positions["carrier"])
                          <= env.config.scout_recovery_radius),
        "stranded": bool(env.scout_ever_dispatched and
                         np.linalg.norm(env.positions["scout"]-
                                        env.positions["carrier"])
                         > env.config.scout_recovery_radius),
        "outstanding_packets": int(len(env.pending)),
        "ledger": {name: float(getattr(env.ledger, name)) for name in (
            "time", "motion", "sensing", "communication", "collision",
            "failure", "material_exposure")},
    }
