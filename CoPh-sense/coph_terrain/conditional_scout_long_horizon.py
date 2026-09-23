"""Long-horizon geometric point-navigation benchmark for the conditional scout.

This module is deliberately a diagnostic between the exact admission benchmark
and the final VMAS/pH experiment.  Compared with the first visualizer it adds
three pieces that matter for the scientific claim:

1. the scout is *docked to the carrier* until dispatch and returns to the
   moving carrier after reporting;
2. both point agents obey hard geometric clearance constraints around fixed
   obstacles and moving objects;
3. dynamic objects are physical disks in the world, not only latent route-cost
   variables used by the admission model.

The learned quantity is still the ex-ante dispatch value.  The carrier route is
chosen from the same exact case semantics, while the rollout checks whether the
chosen continuation is physically executable in a cluttered dynamic scene.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from .conditional_scout_benchmark import default_cases, evaluate_case, top_route_cost
from .conditional_scout_learning import load_model, predict_value


@dataclass
class KinematicState:
    q: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class DiskObstacle:
    center: np.ndarray
    radius: float
    name: str
    dynamic: bool = False


# Two route homotopy classes around the fixed central geometry.
ROUTES = {
    "top": np.asarray([
        [-5.0, 0.0], [-2.25, 0.0], [-1.45, 1.05], [0.45, 1.45],
        [2.35, 1.35], [3.55, 0.85], [5.0, 0.0],
    ], dtype=float),
    "bottom": np.asarray([
        [-5.0, 0.0], [-2.25, 0.0], [-1.45, -1.05], [0.45, -1.45],
        [2.35, -1.35], [3.55, -0.85], [5.0, 0.0],
    ], dtype=float),
}

START = np.asarray([-5.0, 0.0], dtype=float)
GOAL = np.asarray([5.0, 0.0], dtype=float)
HOLD_POINT = np.asarray([-2.25, 0.0], dtype=float)
DOCK_OFFSET = np.asarray([0.0, 0.22], dtype=float)

# Conditional-scout motion parameters.  The sensing site is NOT a fixed world
# landmark.  It is generated when the carrier issues a request, ahead along the
# carrier's current intended route and constrained by the remaining report
# deadline.
SCOUT_MAX_SPEED = 1.70
SCOUT_REQUEST_DELAY = 0.70
SCOUT_REPORT_MARGIN = 0.30
SCOUT_VANTAGE_OFFSET = 0.28

# These are always present.  They make the fork geometric rather than two
# dashed, unconstrained curves.  One rock also lies close to the scout's
# straight-line path, so the scout must execute obstacle avoidance as well.
STATIC_OBSTACLES = (
    DiskObstacle(np.asarray([-0.10, 0.00]), 0.72, "central_island"),
    DiskObstacle(np.asarray([2.30, 0.00]), 0.58, "goal_island"),
    DiskObstacle(np.asarray([-3.72, 0.78]), 0.38, "scout_side_rock"),
    DiskObstacle(np.asarray([0.90, 2.35]), 0.40, "upper_rock"),
    DiskObstacle(np.asarray([0.90, -2.35]), 0.40, "lower_rock"),
)

WORLD_X = (-5.7, 5.7)
WORLD_Y = (-3.5, 3.5)
AGENT_RADIUS = 0.16
SAFETY_MARGIN = 0.08
INFLUENCE_RANGE = 0.85


def _background_patrol(t: float) -> DiskObstacle:
    """A moving object common to all scenes.

    It oscillates vertically near the second half of the map.  It is not the
    hidden variable used by the dispatch teacher; its purpose is to ensure the
    long-horizon geometry remains dynamic even in static-information cases.
    """
    y = 1.85 * np.sin(0.42 * float(t) + 0.55)
    return DiskObstacle(np.asarray([3.05, y], dtype=float), 0.28, "patrol", True)


def _case_obstacles(case, mode, t: float) -> list[DiskObstacle]:
    out = list(STATIC_OBSTACLES)
    out.append(_background_patrol(t))

    # Hidden static blockage is a physical disk on the upper corridor.
    if case.kind in ("static_block", "local_known") and bool(mode.blocked):
        out.append(DiskObstacle(np.asarray([1.78, 1.34]), 0.46, "hidden_block", False))

    # Information-specific moving blocker.  The exact benchmark reasons about
    # its future occupancy at carrier arrival; the rollout now enforces it as a
    # hard moving obstacle as well.
    if case.kind == "moving_blocker":
        y = float(mode.phase + mode.velocity * t)
        out.append(DiskObstacle(np.asarray([1.80, y]), float(case.dynamic_radius),
                                "moving_blocker", True))
    return out


def _min_clearance(q: np.ndarray, obstacles: Iterable[DiskObstacle],
                   agent_radius: float = AGENT_RADIUS,
                   safety_margin: float = SAFETY_MARGIN) -> float:
    vals = []
    for obs in obstacles:
        vals.append(float(np.linalg.norm(q - obs.center) -
                          (obs.radius + agent_radius + safety_margin)))
    return min(vals) if vals else float("inf")


def _repulsive_accel(q: np.ndarray, obstacles: Iterable[DiskObstacle],
                     agent_radius: float = AGENT_RADIUS,
                     safety_margin: float = SAFETY_MARGIN,
                     influence: float = INFLUENCE_RANGE) -> np.ndarray:
    rep = np.zeros(2, dtype=float)
    for obs in obstacles:
        vec = q - obs.center
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            vec = np.asarray([1.0, 0.0])
            dist = 1e-9
        n = vec / dist
        clr = dist - (obs.radius + agent_radius + safety_margin)
        if clr < influence:
            # Bounded inverse-clearance potential.  It shapes motion before the
            # hard projection below becomes active.
            safe = max(clr, 0.025)
            mag = 0.32 * (1.0 / safe - 1.0 / influence) / (safe * safe)
            rep += np.clip(mag, 0.0, 7.0) * n
    return rep


def _project_feasible(q: np.ndarray, v: np.ndarray,
                      obstacles: Iterable[DiskObstacle],
                      agent_radius: float = AGENT_RADIUS,
                      safety_margin: float = SAFETY_MARGIN):
    """Project a candidate point outside all inflated disks.

    This is the hard geometric safety layer for the toy point system.  The
    number of projections is logged as a safety intervention rather than being
    silently ignored.
    """
    q = np.asarray(q, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    interventions = 0
    for _ in range(3):
        changed = False
        for obs in obstacles:
            min_dist = float(obs.radius + agent_radius + safety_margin)
            vec = q - obs.center
            dist = float(np.linalg.norm(vec))
            if dist < min_dist:
                if dist < 1e-9:
                    n = np.asarray([1.0, 0.0])
                else:
                    n = vec / dist
                q = obs.center + n * (min_dist + 1e-4)
                inward = float(np.dot(v, n))
                if inward < 0.0:
                    v = v - inward * n
                interventions += 1
                changed = True
        if not changed:
            break
    q[0] = np.clip(q[0], WORLD_X[0] + agent_radius, WORLD_X[1] - agent_radius)
    q[1] = np.clip(q[1], WORLD_Y[0] + agent_radius, WORLD_Y[1] - agent_radius)
    return q, v, interventions


def _advance(state: KinematicState, target, dt, max_speed, obstacles,
             kp=2.8, kd=2.0):
    target = np.asarray(target, dtype=float)
    a = kp * (target - state.q) - kd * state.v
    a += _repulsive_accel(state.q, obstacles)
    an = float(np.linalg.norm(a))
    if an > 4.5:
        a = a * (4.5 / an)
    v_new = state.v + dt * a
    speed = float(np.linalg.norm(v_new))
    if speed > max_speed:
        v_new = v_new * (max_speed / speed)
    q_new = state.q + dt * v_new
    q_new, v_new, nproj = _project_feasible(q_new, v_new, obstacles)
    state.v = v_new
    state.q = q_new
    return nproj


def _route_target(route, q, index):
    pts = ROUTES[route]
    i = int(index)
    while i < len(pts) - 1 and np.linalg.norm(q - pts[i]) < .30:
        i += 1
    return pts[i], i


def _choose_mode(case, rng, mode_name=None):
    if mode_name is not None:
        for mode in case.modes:
            if mode.name == mode_name:
                return mode
        raise ValueError(f"unknown mode {mode_name!r}; choose from {[m.name for m in case.modes]}")
    probs = np.asarray([m.probability for m in case.modes], dtype=float)
    probs /= probs.sum()
    return case.modes[int(rng.choice(len(case.modes), p=probs))]


def _mode_route(case, mode):
    return "top" if top_route_cost(case, mode) < case.bottom_cost else "bottom"


def _prior_route(case):
    row = evaluate_case(case)
    return row["idle_route"] if row["idle_route"] in ("top", "bottom") else "bottom"


def _dispatch(policy, case, ckpt, device):
    if policy == "never":
        return False, None
    if policy == "always":
        return True, None
    if policy == "oracle":
        row = evaluate_case(case)
        return bool(row["dispatch"]), row["scout_value"]
    if policy == "learned":
        if ckpt is None:
            raise ValueError("--ckpt is required for policy=learned")
        model, payload = load_model(ckpt, device)
        value = float(predict_value(model, payload, [case], device)[0])
        return value > 0.0, value
    raise ValueError(policy)


def _docked_pose(carrier_q: np.ndarray) -> np.ndarray:
    return np.asarray(carrier_q, dtype=float) + DOCK_OFFSET



def _project_to_polyline_s(q: np.ndarray, pts: np.ndarray):
    """Return arclength coordinate and tangent at the closest polyline point."""
    pts = np.asarray(pts, dtype=float)
    q = np.asarray(q, dtype=float)
    seg = pts[1:] - pts[:-1]
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    best = None
    for i, (a, d, L) in enumerate(zip(pts[:-1], seg, lens)):
        if L < 1e-9:
            continue
        u = float(np.clip(np.dot(q - a, d) / (L * L), 0.0, 1.0))
        proj = a + u * d
        err = float(np.linalg.norm(q - proj))
        if best is None or err < best[0]:
            best = (err, cum[i] + u * L, d / L)
    if best is None:
        return 0.0, np.asarray([1.0, 0.0])
    return float(best[1]), np.asarray(best[2], dtype=float)


def _point_at_polyline_s(pts: np.ndarray, s: float):
    pts = np.asarray(pts, dtype=float)
    seg = pts[1:] - pts[:-1]
    lens = np.linalg.norm(seg, axis=1)
    total = float(lens.sum())
    s = float(np.clip(s, 0.0, total))
    acc = 0.0
    for a, d, L in zip(pts[:-1], seg, lens):
        if L < 1e-9:
            continue
        if s <= acc + L:
            u = (s - acc) / L
            return a + u * d, d / L
        acc += L
    d = pts[-1] - pts[-2]
    d /= max(float(np.linalg.norm(d)), 1e-9)
    return pts[-1].copy(), d


def _generate_scout_site(carrier_q: np.ndarray, carrier_v: np.ndarray, route: str,
                         case, mode, t: float, obstacles: Iterable[DiskObstacle]):
    """Generate a safe remote sensing site in front of the carrier.

    The site is the farthest useful route-relative vantage point that the scout
    can visit and still finish dwell + communication before the route commitment
    deadline.  Candidates are displaced slightly to the outside of the route so
    the scout is visibly separate from, but still spatially coupled to, the
    carrier trajectory.
    """
    pts = ROUTES[route]
    s0, route_tangent = _project_to_polyline_s(carrier_q, pts)
    speed = float(np.linalg.norm(carrier_v))
    if speed > 0.12:
        forward = carrier_v / speed
        # Avoid a transient velocity direction that points opposite the route.
        if float(np.dot(forward, route_tangent)) < 0.25:
            forward = route_tangent
    else:
        forward = route_tangent

    remaining = (float(case.commit_time) - float(t) - float(case.scout_dwell_time)
                 - float(case.communication_delay) - SCOUT_REPORT_MARGIN)
    # Leave enough time for sensing/reporting; keep a useful minimum look-ahead
    # when the analytical case is already known to be timely.
    reachable = max(0.8, SCOUT_MAX_SPEED * max(remaining, 0.0))
    desired = 5.2 if case.kind in ("moving_blocker", "moving_cost") else 4.6
    lookahead = float(np.clip(reachable * 0.92, 1.2, desired))
    base, tangent = _point_at_polyline_s(pts, s0 + lookahead)

    # Put the vantage on the outside of the currently intended branch.
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    branch_sign = 1.0 if route == "top" else -1.0
    if branch_sign * normal[1] < 0.0:
        normal = -normal

    offsets = [SCOUT_VANTAGE_OFFSET, 0.48, 0.0, -0.28, 0.72]
    candidates = [base + off * normal for off in offsets]
    best = None
    for cand in candidates:
        cand = np.asarray(cand, dtype=float)
        cand[0] = np.clip(cand[0], WORLD_X[0] + .25, WORLD_X[1] - .25)
        cand[1] = np.clip(cand[1], WORLD_Y[0] + .25, WORLD_Y[1] - .25)
        front = float(np.dot(cand - carrier_q, forward))
        if front <= 0.75:
            continue
        clr = _min_clearance(cand, obstacles, agent_radius=AGENT_RADIUS * .82,
                             safety_margin=SAFETY_MARGIN)
        # Prefer clearance first, then forward progress, while staying near the
        # deadline-aware route anchor.
        score = 2.2 * min(clr, 1.5) + 0.18 * front - 0.12 * float(np.linalg.norm(cand - base))
        if best is None or score > best[0]:
            best = (score, cand.copy(), clr, front)
    if best is None:
        # Conservative fallback remains in front and is hard-projected later.
        cand = carrier_q + max(1.2, min(lookahead, 2.0)) * forward
        return cand, {"lookahead": lookahead, "forward_distance": float(np.dot(cand-carrier_q, forward)),
                      "clearance": _min_clearance(cand, obstacles), "remaining_time": remaining}
    return best[1], {"lookahead": lookahead, "forward_distance": float(best[3]),
                     "clearance": float(best[2]), "remaining_time": remaining}


def _dynamic_cost_center(case, mode, t):
    if case.kind != "moving_cost":
        return np.nan
    return float(mode.phase + mode.velocity * t)


def run_rollout(case_name, policy="oracle", ckpt=None, outdir=Path("rollout"),
                seed=0, dt=.05, max_steps=900, device="cpu", frame_stride=5,
                mode_name=None, render=True):
    cases = {c.name: c for c in default_cases()}
    if case_name not in cases:
        raise ValueError(f"unknown case {case_name}; choose from {sorted(cases)}")
    case = cases[case_name]
    rng = np.random.default_rng(seed)
    mode = _choose_mode(case, rng, mode_name=mode_name)
    dispatch, predicted_value = _dispatch(policy, case, ckpt, device)

    carrier = KinematicState(START.copy(), np.zeros(2))
    scout = KinematicState(_docked_pose(carrier.q), np.zeros(2))
    carrier_route = _prior_route(case)
    prior_route = carrier_route
    route_locked = False
    route_index = 1
    scout_phase = "docked" if not dispatch else "docked_waiting"
    scout_released = False
    request_issued = False
    request_t = None
    request_carrier_q = None
    scout_site = None
    scout_site_info = None
    route_change_t = None
    route_after_report = carrier_route
    carrier_wait_time = 0.0
    scout_arrival_t = None
    sensed_t = None
    report_t = None
    dwell_end_t = None
    message_due_t = None
    report_received = False
    collision = False
    blocked_by_safety = False
    cost_pulse_integral = 0.0
    scout_motion = 0.0
    carrier_motion = 0.0
    carrier_safety_interventions = 0
    scout_safety_interventions = 0
    min_clear_carrier = float("inf")
    min_clear_scout = float("inf")
    carrier_avoidance_steps = 0
    scout_avoidance_steps = 0
    carrier_goal_time = None
    records = []

    for step in range(int(max_steps)):
        t = step * dt
        prev_c = carrier.q.copy()
        prev_s = scout.q.copy()
        world_obs = _case_obstacles(case, mode, t)

        # Once the two agents are separated, they also act as mutual dynamic
        # obstacles.  During docking/launch overlap is intentional.
        carrier_obs = list(world_obs)
        scout_obs = list(world_obs)
        if scout_released and scout_phase not in ("return", "docked_returned") \
                and np.linalg.norm(carrier.q - scout.q) > .42:
            carrier_obs.append(DiskObstacle(scout.q.copy(), AGENT_RADIUS,
                                            "scout_body", True))
            scout_obs.append(DiskObstacle(carrier.q.copy(), AGENT_RADIUS,
                                          "carrier_body", True))

        # The scout stays physically docked while the carrier moves.  A policy
        # may decide that scouting has positive value, but deployment itself is
        # triggered only after the carrier reaches an information-request epoch.
        if not dispatch:
            scout_phase = "docked"
            scout.q = _docked_pose(carrier.q)
            scout.v = carrier.v.copy()
        else:
            if not request_issued:
                scout_phase = "docked_waiting"
                scout.q = _docked_pose(carrier.q)
                scout.v = carrier.v.copy()
                if t >= SCOUT_REQUEST_DELAY:
                    scout_site, scout_site_info = _generate_scout_site(
                        carrier.q, carrier.v, carrier_route, case, mode, t, world_obs)
                    request_issued = True
                    request_t = t
                    request_carrier_q = carrier.q.copy()
                    scout_phase = "launch"
            if request_issued and scout_phase == "launch":
                # Start from the *moving* carrier and sprint to a generated
                # route-relative sensing site ahead of the carrier.
                scout_released = True
                scout_phase = "outbound"
            if request_issued and scout_phase == "outbound":
                scout_safety_interventions += _advance(
                    scout, scout_site, dt, SCOUT_MAX_SPEED, scout_obs)
                if np.linalg.norm(scout.q - scout_site) < .18:
                    scout_phase = "dwell"
                    scout_arrival_t = t
                    dwell_end_t = t + case.scout_dwell_time
            elif request_issued and scout_phase == "dwell":
                scout_safety_interventions += _advance(
                    scout, scout_site, dt, .30, scout_obs)
                if t >= dwell_end_t:
                    sensed_t = t
                    scout_phase = "report_wait"
                    message_due_t = t + case.communication_delay
            elif request_issued and scout_phase == "report_wait":
                scout_safety_interventions += _advance(
                    scout, scout_site, dt, .20, scout_obs)
                if t >= message_due_t:
                    report_t = t
                    report_received = t < case.commit_time
                    if report_received:
                        new_route = _mode_route(case, mode)
                        if new_route != carrier_route:
                            route_change_t = t
                        carrier_route = new_route
                        route_after_report = new_route
                        # Re-enter route tracking near the current carrier pose
                        # instead of restarting from the beginning of the new path.
                        route_index = 1
                    scout_phase = "return"
            elif request_issued and scout_phase == "return":
                # Recover to the *moving carrier*, not a detached home point.
                recover_target = _docked_pose(carrier.q)
                scout_safety_interventions += _advance(
                    scout, recover_target, dt, 1.20, scout_obs)
                if np.linalg.norm(scout.q - recover_target) < .22:
                    scout_phase = "docked_returned"
                    scout_released = False
            elif request_issued and scout_phase == "docked_returned":
                scout.q = _docked_pose(carrier.q)
                scout.v = carrier.v.copy()

        # Carrier may progress to the hold/commit point while the scout works.
        if not route_locked:
            if t >= case.commit_time or report_received or not dispatch:
                route_locked = True
            carrier_safety_interventions += _advance(
                carrier, HOLD_POINT, dt, .72, carrier_obs)
            if dispatch and request_issued and not report_received and \
                    np.linalg.norm(carrier.q - HOLD_POINT) < .34:
                carrier_wait_time += dt
        else:
            target, route_index = _route_target(carrier_route, carrier.q, route_index)
            carrier_safety_interventions += _advance(
                carrier, target, dt, .72, carrier_obs)

        # If a non-dispatched scout is docked, follow the just-updated carrier.
        if not dispatch or scout_phase == "docked_returned":
            scout.q = _docked_pose(carrier.q)
            scout.v = carrier.v.copy()

        scout_motion += float(np.linalg.norm(scout.q - prev_s))
        carrier_motion += float(np.linalg.norm(carrier.q - prev_c))

        # Hard-safety clearances are logged against the physical obstacles.
        obs_after = _case_obstacles(case, mode, t)
        cclr = _min_clearance(carrier.q, obs_after)
        min_clear_carrier = min(min_clear_carrier, cclr)
        if cclr < INFLUENCE_RANGE:
            carrier_avoidance_steps += 1
        if scout_released:
            sclr = _min_clearance(scout.q, obs_after)
            min_clear_scout = min(min_clear_scout, sclr)
            if sclr < INFLUENCE_RANGE:
                scout_avoidance_steps += 1
        else:
            sclr = float("nan")

        # Moving-cost field remains a soft traversal cost; physical moving
        # objects are handled separately above.
        pulse_center = _dynamic_cost_center(case, mode, t)
        if case.kind == "moving_cost" and carrier_route == "top":
            sigma = max(case.dynamic_radius, 1e-6)
            local = np.exp(-.5 * ((carrier.q[0] - pulse_center) / sigma) ** 2)
            cost_pulse_integral += float(local * dt)

        # Numerical collision audit: the hard projection should keep this false
        # up to a tiny tolerance.  A violation indicates a bug in the safety
        # layer, not a desired failure mode.
        if cclr < -1e-4 or (scout_released and sclr < -1e-4):
            collision = True

        # A route can be physically blocked by a hidden/static object or moving
        # blocker.  We do not count the safety filter doing its job as a
        # collision; instead, sustained zero progress is reported separately.
        if len(records) > int(3.0 / dt):
            past = records[-int(3.0 / dt)]["carrier"]
            if route_locked and np.linalg.norm(carrier.q - past) < .05 and \
                    np.linalg.norm(carrier.q - GOAL) > .8 and cclr < .12:
                blocked_by_safety = True

        at_goal = np.linalg.norm(carrier.q - GOAL) < .28
        if at_goal and carrier_goal_time is None:
            carrier_goal_time = t

        records.append({
            "step": step, "time": t,
            "carrier": carrier.q.copy(), "carrier_v": carrier.v.copy(),
            "scout": scout.q.copy(), "scout_v": scout.v.copy(),
            "route": carrier_route, "route_locked": route_locked,
            "scout_phase": scout_phase, "report_received": report_received,
            "request_issued": request_issued,
            "request_origin": (request_carrier_q.copy() if request_carrier_q is not None
                               else np.asarray([np.nan, np.nan])),
            "prior_route": prior_route,
            "scout_site": (np.asarray(scout_site, dtype=float).copy()
                           if scout_site is not None else np.asarray([np.nan, np.nan])),
            "route_changed": route_change_t is not None,
            "pulse_center": pulse_center,
            "carrier_clearance": cclr, "scout_clearance": sclr,
            "obstacles": [(o.center.copy(), float(o.radius), o.name, o.dynamic)
                          for o in obs_after],
        })
        if collision:
            break
        # For visualization/recovery semantics, do not terminate while a
        # dispatched scout is still detached.  The carrier holds at the goal
        # and the faster scout catches and re-docks.
        if at_goal and (not dispatch or scout_phase == "docked_returned"):
            break

    reached = carrier_goal_time is not None and not collision
    total_time = records[-1]["time"] if records else 0.0
    route_cost_paid = top_route_cost(case, mode) if carrier_route == "top" else case.bottom_cost
    prior_route_cost = top_route_cost(case, mode) if prior_route == "top" else case.bottom_cost
    route_cost_improvement = float(prior_route_cost - route_cost_paid)
    score = (total_time + .10 * carrier_motion + .06 * scout_motion
             + (case.dispatch_cost if dispatch else 0.0)
             + (6.0 if collision else 0.0) + .35 * cost_pulse_integral)
    metrics = {
        "case": case.name, "kind": case.kind, "mode": mode.name,
        "policy": policy, "dispatch": bool(dispatch),
        "predicted_value": predicted_value,
        "scout_starts_docked": True,
        "scout_returns_to_carrier": True,
        "request_time": request_t,
        "request_carrier_position": (None if request_carrier_q is None else request_carrier_q.tolist()),
        "generated_scout_site": (None if scout_site is None else np.asarray(scout_site).tolist()),
        "scout_site_info": scout_site_info,
        "scout_site_is_forward": bool(scout_site_info is not None and scout_site_info.get("forward_distance", -1.0) > 0.0),
        "report_received_timely": bool(report_received),
        "scout_arrival_time": scout_arrival_t, "sense_time": sensed_t,
        "report_time": report_t, "commit_time": case.commit_time,
        "prior_carrier_route": prior_route,
        "carrier_route": carrier_route,
        "route_after_report": route_after_report,
        "route_change_time": route_change_t,
        "route_changed_due_to_report": bool(report_received and carrier_route != prior_route),
        "carrier_wait_time_for_scout": float(carrier_wait_time),
        "reached_goal": bool(reached),
        "collision": bool(collision), "blocked_by_safety": bool(blocked_by_safety),
        "steps": len(records), "time": total_time,
        "carrier_path_length": carrier_motion, "scout_path_length": scout_motion,
        "dynamic_cost_integral": cost_pulse_integral,
        "carrier_safety_interventions": int(carrier_safety_interventions),
        "scout_safety_interventions": int(scout_safety_interventions),
        "carrier_avoidance_steps": int(carrier_avoidance_steps),
        "scout_avoidance_steps": int(scout_avoidance_steps),
        "carrier_goal_time": carrier_goal_time,
        "scout_recovered_at_end": bool((not dispatch) or scout_phase == "docked_returned"),
        "final_carrier_scout_distance": float(np.linalg.norm(carrier.q - scout.q)),
        "min_carrier_clearance": float(min_clear_carrier),
        "min_scout_clearance": (None if not np.isfinite(min_clear_scout)
                                else float(min_clear_scout)),
        "true_route_cost": float(route_cost_paid),
        "prior_route_true_cost": float(prior_route_cost),
        "route_cost_improvement_from_report": route_cost_improvement,
        "visual_score": float(score),
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (outdir / "world_geometry.json").write_text(json.dumps({
        "static_obstacles": [
            {"name": o.name, "center": o.center.tolist(), "radius": o.radius}
            for o in STATIC_OBSTACLES
        ],
        "agent_radius": AGENT_RADIUS,
        "safety_margin": SAFETY_MARGIN,
        "scout_site_policy": "generated_ahead_of_carrier_at_request_time",
        "generated_scout_site": (None if scout_site is None else np.asarray(scout_site).tolist()),
        "request_time": request_t,
        "dock_offset": DOCK_OFFSET.tolist(),
    }, indent=2) + "\n")
    np.savez_compressed(
        outdir / "trace.npz",
        time=np.asarray([r["time"] for r in records], dtype=np.float32),
        carrier=np.asarray([r["carrier"] for r in records], dtype=np.float32),
        scout=np.asarray([r["scout"] for r in records], dtype=np.float32),
        scout_site=np.asarray([r["scout_site"] for r in records], dtype=np.float32),
        pulse_center=np.asarray([r["pulse_center"] for r in records], dtype=np.float32),
        carrier_clearance=np.asarray([r["carrier_clearance"] for r in records], dtype=np.float32),
        scout_clearance=np.asarray([r["scout_clearance"] for r in records], dtype=np.float32),
    )
    if render:
        _render(case, mode, records, metrics, outdir, max(1, int(frame_stride)))
    return metrics


def _draw_world(ax, case, mode, record):
    ax.set_xlim(*WORLD_X); ax.set_ylim(*WORLD_Y); ax.set_aspect("equal")
    prior_route = record.get("prior_route", record.get("route", "bottom"))
    active_route = record.get("route", prior_route)
    other_route = "bottom" if active_route == "top" else "top"
    ax.plot(ROUTES[other_route][:, 0], ROUTES[other_route][:, 1], linestyle="--",
            linewidth=.9, alpha=.28, label="other candidate route")
    ax.plot(ROUTES[prior_route][:, 0], ROUTES[prior_route][:, 1], linestyle=":",
            linewidth=1.4, alpha=.55, label="no-scout / prior route")
    if active_route != prior_route:
        ax.plot(ROUTES[active_route][:, 0], ROUTES[active_route][:, 1], linestyle="--",
                linewidth=1.8, alpha=.72, label="route after scout report")
    ax.scatter(*GOAL, marker="*", s=130, label="goal")
    site = np.asarray(record.get("scout_site", [np.nan, np.nan]), dtype=float)
    if np.isfinite(site).all():
        ax.scatter(*site, marker="s", s=60, label="generated sensing site")
        if record.get("request_issued", False):
            # Keep the original request location visible.  The target is generated
            # from the carrier motion at this point, so the line directly shows
            # that the sensing site lies ahead of the moving carrier.
            rq = np.asarray(record.get("request_origin", [np.nan, np.nan]), dtype=float)
            if np.isfinite(rq).all():
                ax.scatter(*rq, marker="D", s=32, label="carrier scout request")
                ax.plot([rq[0], site[0]], [rq[1], site[1]], linestyle=":",
                        linewidth=1.0, alpha=.55, label="forward scout task")
        if record.get("report_received", False):
            # Once a report is available, show the information return path.
            sq = np.asarray(record["scout"], dtype=float)
            cq = np.asarray(record["carrier"], dtype=float)
            ax.plot([sq[0], cq[0]], [sq[1], cq[1]], linestyle="-.",
                    linewidth=1.0, alpha=.55, label="report to carrier")
    ax.axvline(HOLD_POINT[0], linestyle=":", linewidth=1, alpha=.55,
               label="commit line")

    # Reconstruct the actual obstacles present at this frame from the record so
    # visualization and collision checking cannot silently diverge.
    for center, radius, name, dynamic in record["obstacles"]:
        if name == "moving_blocker":
            alpha = .50
        elif dynamic:
            alpha = .36
        else:
            alpha = .24
        ax.add_patch(Circle(tuple(center), radius, alpha=alpha))
        if name in ("moving_blocker", "hidden_block", "patrol"):
            ax.text(center[0], center[1] + radius + .08, name,
                    fontsize=7, ha="center")

    # Safety-inflated boundaries show the actual point-navigation constraint.
    for center, radius, name, dynamic in record["obstacles"]:
        inflated = radius + AGENT_RADIUS + SAFETY_MARGIN
        ax.add_patch(Circle(tuple(center), inflated, fill=False,
                            linewidth=.7, linestyle=":", alpha=.45))

    if case.kind == "moving_cost":
        center = float(record["pulse_center"])
        if np.isfinite(center):
            ax.axvspan(center - case.dynamic_radius, center + case.dynamic_radius,
                       alpha=.10)
    ax.set_xlabel("x"); ax.set_ylabel("y")


def _render(case, mode, records, metrics, outdir, stride):
    c = np.asarray([r["carrier"] for r in records])
    s = np.asarray([r["scout"] for r in records])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _draw_world(ax, case, mode, records[-1])
    ax.plot(c[:, 0], c[:, 1], linewidth=2, label="carrier")
    ax.plot(s[:, 0], s[:, 1], linewidth=2, label="scout")
    ax.scatter(*c[0], marker="o", s=45); ax.scatter(*s[0], marker="o", s=45)
    ax.scatter(*c[-1], marker="X", s=65); ax.scatter(*s[-1], marker="X", s=65)
    route_transition = metrics.get("prior_carrier_route", metrics["carrier_route"])
    if metrics.get("route_changed_due_to_report", False):
        route_transition += "→" + metrics["carrier_route"]
    ax.set_title(f"{case.name} | {metrics['policy']} | route={route_transition} | "
                 f"dispatch={metrics['dispatch']}")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(outdir / "trajectory.png", dpi=160)
    plt.close(fig)

    # Timing and geometry diagnostics.
    tt = np.asarray([r["time"] for r in records])
    goal_dist = np.linalg.norm(c - GOAL[None, :], axis=1)
    cclr = np.asarray([r["carrier_clearance"] for r in records], dtype=float)
    sclr = np.asarray([r["scout_clearance"] for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(tt, goal_dist, label="carrier goal distance")
    ax.plot(tt, cclr, label="carrier clearance", alpha=.75)
    if np.isfinite(sclr).any():
        ax.plot(tt, sclr, label="scout clearance", alpha=.75)
    ax.axhline(0.0, linestyle="--", linewidth=.8, label="safety boundary")
    if metrics["report_time"] is not None:
        ax.axvline(metrics["report_time"], linestyle="--", label="report")
    ax.axvline(metrics["commit_time"], linestyle=":", label="commit deadline")
    ax.set(xlabel="time", ylabel="distance / clearance", title="Timing and geometric safety")
    ax.legend(fontsize=8)
    fig.savefig(outdir / "timeseries.png", dpi=160)
    plt.close(fig)

    frames = []
    indices = list(range(0, len(records), stride))
    if indices[-1] != len(records) - 1:
        indices.append(len(records) - 1)
    for idx in indices:
        r = records[idx]
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=90, constrained_layout=True)
        _draw_world(ax, case, mode, r)
        ax.plot(c[:idx+1, 0], c[:idx+1, 1], linewidth=2, label="carrier")
        ax.plot(s[:idx+1, 0], s[:idx+1, 1], linewidth=2, label="scout")
        ax.add_patch(Circle(tuple(c[idx]), AGENT_RADIUS, alpha=.85))
        ax.add_patch(Circle(tuple(s[idx]), AGENT_RADIUS * .82, alpha=.85))
        change = " | route switched" if r.get("route_changed", False) else ""
        ax.set_title(f"t={r['time']:.1f}s | scout={r['scout_phase']} | "
                     f"route={r['route']} | report={r['report_received']}{change}")
        ax.legend(loc="upper left", fontsize=8)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba[..., :3]))
        plt.close(fig)
    frames[0].save(outdir / "trajectory.gif", save_all=True,
                   append_images=frames[1:], duration=80, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="moving_blocker_prediction",
                    choices=[c.name for c in default_cases()])
    ap.add_argument("--policy", choices=("never", "always", "oracle", "learned"),
                    default="oracle")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--outdir", type=Path,
                    default=Path("coph_terrain/results/conditional_scout_rollout"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=.05)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode-name", default=None,
                    help="Optional exact hidden mode, e.g. crossing or clearing")
    args = ap.parse_args()
    metrics = run_rollout(args.case, args.policy, args.ckpt, args.outdir,
                          args.seed, args.dt, args.max_steps, args.device,
                          args.frame_stride, args.mode_name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
