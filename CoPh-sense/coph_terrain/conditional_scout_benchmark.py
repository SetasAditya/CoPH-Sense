"""Exact small benchmarks for *when* an optional scout is necessary.

These cases are intentionally simpler than E2-Natural.  They establish the
information/timing inequality before any neural policy is trained:

    carrier only  >  conditional scout

on cooperation-required worlds, while retaining worlds where dispatch is
unnecessary or too late.  Dynamic cases use future route cost at the carrier's
arrival time, so the scout must provide a timely state/velocity observation;
they are not merely moving-object visualizations.
"""

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class HiddenMode:
    name: str
    probability: float
    # Parameters used by the selected dynamic model.
    phase: float = 0.0
    velocity: float = 0.0
    blocked: bool = False


@dataclass(frozen=True)
class ScoutNecessityCase:
    name: str
    kind: str  # static_block | moving_blocker | moving_cost | local_known
    modes: Tuple[HiddenMode, ...]
    commit_time: float
    carrier_route_arrival_time: float
    scout_travel_time: float
    scout_dwell_time: float
    communication_delay: float
    single_agent_remote_time: float
    dispatch_cost: float
    top_base_cost: float = 1.0
    bottom_cost: float = 3.0
    failure_top_cost: float = 7.0
    dynamic_radius: float = 0.35
    moving_cost_amplitude: float = 5.0
    carrier_knows_mode: bool = False

    @property
    def report_time(self):
        return self.scout_travel_time + self.scout_dwell_time + self.communication_delay

    @property
    def timely(self):
        return self.report_time < self.commit_time

    @property
    def single_agent_can_query(self):
        return self.single_agent_remote_time < self.commit_time


def _normalize_modes(modes: Iterable[HiddenMode]):
    modes = tuple(modes)
    total = sum(mode.probability for mode in modes)
    if total <= 0:
        raise ValueError("mode probabilities must have positive mass")
    return tuple((mode, mode.probability / total) for mode in modes)


def top_route_cost(case: ScoutNecessityCase, mode: HiddenMode) -> float:
    """True top-route cost at the carrier's future arrival time."""
    if case.kind == "static_block":
        return case.failure_top_cost if mode.blocked else case.top_base_cost

    if case.kind == "moving_blocker":
        # A blocker moves in y across the top corridor.  The scout observes its
        # state remotely; what matters is its predicted y when the carrier
        # reaches the crossing, not its position at sensing time.
        y_arrival = mode.phase + mode.velocity * case.carrier_route_arrival_time
        occupied = abs(y_arrival) <= case.dynamic_radius
        return case.failure_top_cost if occupied else case.top_base_cost

    if case.kind == "moving_cost":
        # A traversal-cost pulse advects along a 1-D route coordinate.  The
        # carrier crosses coordinate zero at route_arrival_time.  High cost is
        # localized and therefore sensitive to timing/information age.
        center = mode.phase + mode.velocity * case.carrier_route_arrival_time
        sigma = max(case.dynamic_radius, 1e-6)
        pulse = np.exp(-0.5 * (center / sigma) ** 2)
        return float(case.top_base_cost + case.moving_cost_amplitude * pulse)

    if case.kind == "local_known":
        return case.failure_top_cost if mode.blocked else case.top_base_cost

    raise ValueError(case.kind)


def _route_cost_with_mode(case, mode):
    return min(top_route_cost(case, mode), case.bottom_cost)


def no_scout_value(case: ScoutNecessityCase) -> Dict[str, float]:
    weighted = _normalize_modes(case.modes)
    if case.carrier_knows_mode:
        expected = sum(prob * _route_cost_with_mode(case, mode)
                       for mode, prob in weighted)
        return {"cost": float(expected), "route": "mode_dependent"}
    expected_top = sum(prob * top_route_cost(case, mode) for mode, prob in weighted)
    if expected_top < case.bottom_cost:
        return {"cost": float(expected_top), "route": "top"}
    return {"cost": float(case.bottom_cost), "route": "bottom"}


def dispatched_value(case: ScoutNecessityCase) -> Dict[str, float]:
    """Expected value of a complete scout task followed by optimal route use."""
    baseline = no_scout_value(case)
    if not case.timely:
        return {"cost": float(baseline["cost"] + case.dispatch_cost),
                "timely": False, "downstream_cost": float(baseline["cost"])}
    weighted = _normalize_modes(case.modes)
    downstream = sum(prob * _route_cost_with_mode(case, mode)
                     for mode, prob in weighted)
    return {"cost": float(case.dispatch_cost + downstream), "timely": True,
            "downstream_cost": float(downstream)}


def evaluate_case(case: ScoutNecessityCase, margin: float = 0.0):
    idle = no_scout_value(case)
    dispatched = dispatched_value(case)
    value = idle["cost"] - dispatched["cost"]
    dispatch = bool(case.timely and value > margin)
    conditional_cost = dispatched["cost"] if dispatch else idle["cost"]
    # Continuously active scout pays a standing/exploration charge even when
    # the measurement cannot alter the carrier decision.  The factor is only a
    # diagnostic comparator; the conditional-vs-idle result does not depend on it.
    always_active_overhead = .5 * case.dispatch_cost
    always_active_cost = dispatched["cost"] + always_active_overhead
    return {
        "name": case.name,
        "kind": case.kind,
        "idle_cost": float(idle["cost"]),
        "idle_route": idle["route"],
        "dispatch_cost_total": float(dispatched["cost"]),
        "dispatch_downstream_cost": float(dispatched["downstream_cost"]),
        "scout_value": float(value),
        "dispatch": dispatch,
        "timely": bool(case.timely),
        "report_time": float(case.report_time),
        "commit_time": float(case.commit_time),
        "single_agent_can_query": bool(case.single_agent_can_query),
        "conditional_cost": float(conditional_cost),
        "always_active_cost": float(always_active_cost),
        "top_cost_by_mode": {mode.name: top_route_cost(case, mode)
                             for mode in case.modes},
    }


def default_cases():
    half = .5
    return (
        ScoutNecessityCase(
            name="static_hidden_block_useful", kind="static_block",
            modes=(HiddenMode("clear", half, blocked=False),
                   HiddenMode("blocked", half, blocked=True)),
            commit_time=5.0, carrier_route_arrival_time=8.0,
            scout_travel_time=2.0, scout_dwell_time=.4,
            communication_delay=.3, single_agent_remote_time=7.0,
            dispatch_cost=.35),
        ScoutNecessityCase(
            name="static_local_known_no_need", kind="local_known",
            modes=(HiddenMode("clear", half, blocked=False),
                   HiddenMode("blocked", half, blocked=True)),
            commit_time=5.0, carrier_route_arrival_time=8.0,
            scout_travel_time=2.0, scout_dwell_time=.4,
            communication_delay=.3, single_agent_remote_time=7.0,
            dispatch_cost=.35, carrier_knows_mode=True),
        ScoutNecessityCase(
            name="static_hidden_block_too_late", kind="static_block",
            modes=(HiddenMode("clear", half, blocked=False),
                   HiddenMode("blocked", half, blocked=True)),
            commit_time=2.0, carrier_route_arrival_time=5.0,
            scout_travel_time=2.2, scout_dwell_time=.4,
            communication_delay=.3, single_agent_remote_time=7.0,
            dispatch_cost=.35),
        ScoutNecessityCase(
            name="moving_blocker_prediction", kind="moving_blocker",
            # One latent trajectory crosses the corridor at t=8, one clears it.
            modes=(HiddenMode("crossing", half, phase=1.6, velocity=-.20),
                   HiddenMode("clearing", half, phase=1.6, velocity=.20)),
            commit_time=5.0, carrier_route_arrival_time=8.0,
            scout_travel_time=1.8, scout_dwell_time=.4,
            communication_delay=.35, single_agent_remote_time=6.5,
            dispatch_cost=.40),
        ScoutNecessityCase(
            name="moving_cost_prediction", kind="moving_cost",
            # Pulse centered at the crossing in the adverse mode at t=8.
            modes=(HiddenMode("pulse_at_crossing", half, phase=-1.6, velocity=.20),
                   HiddenMode("pulse_moving_away", half, phase=1.6, velocity=.20)),
            commit_time=5.0, carrier_route_arrival_time=8.0,
            scout_travel_time=1.5, scout_dwell_time=.4,
            communication_delay=.3, single_agent_remote_time=6.0,
            dispatch_cost=.35, moving_cost_amplitude=5.0),
        ScoutNecessityCase(
            name="moving_blocker_stale_report", kind="moving_blocker",
            modes=(HiddenMode("crossing", half, phase=1.2, velocity=-.15),
                   HiddenMode("clearing", half, phase=1.2, velocity=.20)),
            commit_time=2.0, carrier_route_arrival_time=8.0,
            scout_travel_time=1.4, scout_dwell_time=.4,
            communication_delay=.5, single_agent_remote_time=6.5,
            dispatch_cost=.40),
    )


def run_suite(margin=0.0):
    return [evaluate_case(case, margin=margin) for case in default_cases()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = run_suite(args.margin)
    for row in rows:
        print(f"{row['name']:<32} dispatch={str(row['dispatch']):<5} "
              f"idle={row['idle_cost']:.3f} conditional={row['conditional_cost']:.3f} "
              f"value={row['scout_value']:+.3f} timely={row['timely']} "
              f"single_can_query={row['single_agent_can_query']}")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
