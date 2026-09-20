"""Restricted physical counterfactual audit on excluded pilot map groups.

This is a model-free continuation policy, not the optimal decentralized policy.
Each action set is rolled out with the same hidden world, appearance noise,
packet randomness, pH executor, and policy code. The evaluator alone compares
realized returns. No rollout result is shown to a deployed actor.
"""

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from .environment import CoPHTerrainEnv, TerrainAction, cell_to_position, position_to_cell
from .generator import FAMILIES, RESOLUTION
from .pilot_environment import waypoint
from .planning import (candidate_viewpoints, dijkstra, observed_costmap,
                       plan_path, GOAL_CELL, _public_line_of_sight)


OUT = Path(__file__).resolve().parent / "results" / "pilot"
AGENTS = ("scout", "carrier")


@dataclass(frozen=True)
class AuditChoice:
    agent: str
    modality: str
    viewpoint: tuple
    target_region: tuple = None

    def payload(self):
        return (self.modality, int(self.viewpoint[0]), int(self.viewpoint[1]))


def select_choices(env, region_ranks=(0, 0)):
    """Same local candidate generator as the learner; no hidden-map access."""
    choices = []
    observations = env.observations()
    for agent, rank in zip(AGENTS, region_ranks):
        candidates = candidate_viewpoints(observations[agent], env.config, agent)
        viewpoints = tuple(dict.fromkeys(candidate.viewpoint for candidate in candidates))
        if not viewpoints:
            continue
        target_viewpoint = viewpoints[min(rank, len(viewpoints) - 1)]
        # Each agent receives a same-region geometry/traction pair.
        for modality in ("geometry", "traction"):
            selected = next((candidate for candidate in candidates
                             if candidate.modality == modality
                             and candidate.viewpoint == target_viewpoint), None)
            if selected is not None:
                choices.append(AuditChoice(agent, modality, selected.viewpoint,
                                           selected.target_region))
    return tuple(choices)


def select_stress_choices(env, viewpoints_per_agent=5):
    """Nearest public geometry viewpoints for a physical budget stress."""
    choices = []
    observations = env.observations()
    for agent in AGENTS:
        candidates = candidate_viewpoints(observations[agent], env.config, agent,
                                          max_candidates=16)
        geometry = {candidate.viewpoint: candidate for candidate in candidates
                    if candidate.modality == "geometry"}
        ordered = sorted(geometry.values(),
                         key=lambda candidate: (candidate.travel_distance,
                                                candidate.viewpoint))
        for candidate in ordered[:viewpoints_per_agent]:
            choices.append(AuditChoice(agent, "geometry", candidate.viewpoint))
    return tuple(choices)


def advance_without_acquisition(env, decision_step):
    targets = {agent: waypoint(env, agent) for agent in AGENTS}
    while not env.done and env.step_index < decision_step:
        if env.step_index % 20 == 0:
            targets = {agent: waypoint(env, agent) for agent in AGENTS}
        env.step({agent: TerrainAction(waypoint=targets[agent]) for agent in AGENTS})
    return env


def audit_sets(choices):
    return ((),) + tuple((i,) for i in range(len(choices))) + tuple(
        combinations(range(len(choices)), 2))


def _public_travel_waypoint(env, agent, destination):
    """Route a sensing/radio detour through publicly known occupancy."""
    observation = env.observations()[agent]
    goal = position_to_cell(destination)
    path, _ = plan_path(observation, goal=goal)
    if not path:
        return tuple(float(v) for v in env.positions[agent])
    if len(path) <= 6:
        return tuple(float(v) for v in destination)
    return tuple(float(v) for v in cell_to_position(path[5]))


def _ordered_local_choices(snapshot, agent, choices):
    """Choose a two-sensor tour by public travel cost, including mission exit."""
    if len(choices) != 2:
        return choices
    observation = snapshot.observations()[agent]
    # Mission-integrated candidates already have a geometry pose on the
    # current route before the contact probe. Re-optimizing that pose over
    # nearby off-route cells would silently restore the old detour benchmark.
    mission_path, _ = plan_path(observation)
    route_order = {cell: index for index, cell in enumerate(mission_path)}
    if all(choice.viewpoint in route_order for choice in choices):
        return sorted(choices, key=lambda choice: route_order[choice.viewpoint])
    cost = observed_costmap(observation)
    start = position_to_cell(observation["kinematics"][:2])
    cost[start] = min(float(cost[start]), 1.)
    # For a common information target, geometry can be scanned from several
    # LOS poses while traction needs contact. Optimize the geometry pose and
    # the two possible orders jointly using only the public costmap.
    first_choice, second_choice = choices
    if (first_choice.agent == second_choice.agent and
            first_choice.modality != second_choice.modality and
            first_choice.target_region is not None and
            first_choice.target_region == second_choice.target_region):
        geometry = (first_choice if first_choice.modality == "geometry"
                    else second_choice)
        traction = (first_choice if first_choice.modality == "traction"
                    else second_choice)
        region = geometry.target_region
        radius = (snapshot.config.scan_range_scout if agent == "scout"
                  else snapshot.config.scan_range_carrier)
        reach = int(np.floor(radius / RESOLUTION))
        from_start = dijkstra(cost, start)[0]
        from_traction = dijkstra(cost, traction.viewpoint)[0]
        from_goal = dijkstra(cost, GOAL_CELL)[0]
        options = []
        for x in range(max(0, region[0] - reach),
                       min(cost.shape[0], region[0] + reach + 1)):
            for y in range(max(0, region[1] - reach),
                           min(cost.shape[1], region[1] + reach + 1)):
                pose = (x, y)
                if (np.linalg.norm((np.asarray(pose) - region) * RESOLUTION) > radius - 1e-5
                        or not np.isfinite(from_start[pose])
                        or not np.isfinite(from_goal[pose])
                        or not _public_line_of_sight(observation["known_geometry"],
                                                      pose, region)):
                    continue
                g_first = (from_start[pose] + from_traction[pose] +
                           from_traction[GOAL_CELL])
                t_first = (from_start[traction.viewpoint] +
                           from_traction[pose] + from_goal[pose])
                options.extend(((float(g_first), pose, 0),
                                (float(t_first), pose, 1)))
        if options:
            _, pose, order = min(options)
            geometry = AuditChoice(agent, "geometry", pose, region)
            return [geometry, traction] if order == 0 else [traction, geometry]
    points = (start, choices[0].viewpoint, choices[1].viewpoint)
    distances = [dijkstra(cost, point)[0] for point in points]
    first = (distances[0][points[1]] + distances[1][points[2]] +
             distances[2][GOAL_CELL])
    second = (distances[0][points[2]] + distances[2][points[1]] +
              distances[1][GOAL_CELL])
    return choices if first <= second else list(reversed(choices))


def run_continuation(snapshot, choices, selected, transmit=True,
                     hold_carrier_until_delivery=False, return_environment=False):
    env = snapshot.clone()
    queues = {agent: _ordered_local_choices(
        snapshot, agent, [choices[i] for i in selected
                          if choices[i].agent == agent]) for agent in AGENTS}
    executed_choices = {agent: [asdict(choice) for choice in queues[agent]]
                        for agent in AGENTS}
    seen = {agent: set(env.owned[agent]) for agent in AGENTS}
    unsent = {agent: [] for agent in AGENTS}
    targets = {agent: waypoint(env, agent) for agent in AGENTS}
    travel_goal = {agent: None for agent in AGENTS}
    carrier_staging_position = tuple(float(x) for x in env.positions["carrier"])
    initial_carrier_evidence = len(env.received["carrier"])
    required_scout_deliveries = sum(choices[i].agent == "scout" for i in selected)
    carrier_waiting = bool(hold_carrier_until_delivery and
                           required_scout_deliveries)
    while not env.done:
        actions = {}
        positions = env.positions
        for agent in AGENTS:
            receiver = "carrier" if agent == "scout" else "scout"
            queue = queues[agent]
            if env.measurement_count[agent] >= env.config.max_measurements_per_agent:
                queue.clear()
            active = env.active_sense[agent]
            if active is None and queue:
                target_viewpoint = queue[0].viewpoint
                target = cell_to_position(target_viewpoint)
                velocity = np.asarray(env.observations()[agent]["kinematics"][2:4],
                                      dtype=float)
                # A scan is validated at completion, not when requested.
                # Settle at the pose before starting its dwell so momentum
                # cannot carry the robot out of the validity radius.
                if (np.linalg.norm(positions[agent] - target) <=
                        .5 * env.config.probe_range and
                        np.linalg.norm(velocity) <= .12):
                    sense = queue.pop(0).payload()
                    targets[agent] = tuple(float(value) for value in positions[agent])
                    travel_goal[agent] = ("dwell", target_viewpoint)
                else:
                    sense = None
                    goal = ("sense", target_viewpoint)
                    if goal != travel_goal[agent] or env.step_index % 20 == 0:
                        targets[agent] = _public_travel_waypoint(env, agent, target)
                        travel_goal[agent] = goal
            else:
                sense = None
                if not queue and active is None:
                    if (transmit and unsent[agent]
                            and env.packet_attempts < env.config.max_team_packets
                            and np.linalg.norm(
                            positions[agent] - positions[receiver]) > env.config.communication_range):
                        goal = ("radio", position_to_cell(positions[receiver]))
                        if goal != travel_goal[agent] or env.step_index % 20 == 0:
                            targets[agent] = _public_travel_waypoint(
                                env, agent, positions[receiver])
                            travel_goal[agent] = goal
                    elif (travel_goal[agent] is not None and
                          travel_goal[agent][0] == "dwell") or env.step_index % 20 == 0:
                        targets[agent] = waypoint(env, agent)
                        travel_goal[agent] = None
            send = None
            if (transmit and unsent[agent] and env.packet_attempts < env.config.max_team_packets
                    and np.linalg.norm(positions[agent] - positions[receiver])
                    <= env.config.communication_range):
                send = unsent[agent].pop(0)
            if agent == "carrier" and carrier_waiting:
                enough = (len(env.received["carrier"]) - initial_carrier_evidence
                          >= required_scout_deliveries)
                if enough:
                    carrier_waiting = False
                    targets[agent] = waypoint(env, agent)
                else:
                    targets[agent] = carrier_staging_position
            actions[agent] = TerrainAction(waypoint=targets[agent], sense=sense,
                                            send_evidence_id=send)
        env.step(actions)
        for agent in AGENTS:
            new = set(env.owned[agent]) - seen[agent]
            unsent[agent].extend(sorted(new))
            seen[agent].update(new)
    path_length = {agent: float(sum(np.linalg.norm(
        np.asarray(next_record["positions"][agent]) -
        np.asarray(record["positions"][agent]))
        for record, next_record in zip(env.state_trace[:-1], env.state_trace[1:])))
        for agent in AGENTS}
    target_region_covered = [any(
        evidence.evidence_id not in snapshot.owned[choices[index].agent] and
        evidence.modality == choices[index].modality and
        (choices[index].target_region or choices[index].viewpoint) in evidence.cells
        for evidence in env.owned[choices[index].agent].values())
        for index in selected]
    evidence_locations = {agent: [{"modality": evidence.modality,
                                   "location": evidence.location,
                                   "cell_count": len(evidence.cells)}
                                  for evidence in env.owned[agent].values()
                                  if evidence.evidence_id not in snapshot.owned[agent]]
                          for agent in AGENTS}
    delivery_steps = [int(event["step"]) for event in env.events
                      if event.get("type") == "packet_delivery"
                      and event.get("receiver") == "carrier"
                      and event.get("kind") == "evidence"
                      and event.get("delivered")]
    carrier_cells_at_delivery = [position_to_cell(
        env.state_trace[min(step, len(env.state_trace) - 1)]
        ["positions"]["carrier"]) for step in delivery_steps]
    result = {"selected": list(selected), "mission_cost": env.ledger.mission_cost,
            "material_exposure": env.ledger.material_exposure,
            "score_single_world": env.ledger.mission_cost + env.ledger.material_exposure,
            "success": env.success, "failure_reason": env.failure_reason,
            "steps": env.step_index, "ledger": asdict(env.ledger),
            "path_length": path_length,
            "target_region_covered": target_region_covered,
            "evidence_locations": evidence_locations,
            "executed_choices": executed_choices,
            "measurements": sum(env.measurement_count.values()),
            "packets": env.packet_attempts,
            "acquired": sum(len(env.owned[name]) for name in AGENTS),
            "delivered": sum(len(env.received[name]) for name in AGENTS),
            "carrier_delivery_steps": delivery_steps,
            "carrier_cells_at_delivery": carrier_cells_at_delivery,
            "unsent_evidence": sum(len(ids) for ids in unsent.values()),
            "rejected_sense": sum(event["type"] == "rejected_sense"
                                  for event in env.events)}
    return (result, env) if return_environment else result


def audit_parent(family, parent_seed, realization_seed=0, limit_sets=None,
                 device="cpu", decision_step=0, region_ranks=(0, 0)):
    snapshot = CoPHTerrainEnv(family, parent_seed, realization_seed,
                              seed=parent_seed + 9, executor="ph", device=device)
    advance_without_acquisition(snapshot, decision_step)
    choices = select_choices(snapshot, region_ranks)
    sets = audit_sets(choices)
    if limit_sets is not None:
        sets = sets[:limit_sets]
    started = time.perf_counter()
    rows = [run_continuation(snapshot, choices, selected) for selected in sets]
    elapsed = time.perf_counter() - started
    eligible = [row for row in rows if row["success"]]
    best = min((row["score_single_world"] for row in eligible), default=None)
    winners = ([row for row in eligible
                if abs(row["score_single_world"] - best) < 1e-9]
               if best is not None else [])
    return {"family": family, "parent_seed": parent_seed,
            "realization_seed": realization_seed,
            "decision_step": snapshot.step_index,
            "region_ranks": list(region_ranks),
            "oracle_scope": "restricted scripted continuation; single-world cost",
            "choices": [asdict(choice) for choice in choices],
            "results": rows, "best_sets": [row["selected"] for row in winners],
            "no_success_option": best is None,
            "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=3)
    parser.add_argument("--limit-sets", type=int)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-start", type=int, default=660000)
    parser.add_argument("--decision-step", type=int, default=0)
    parser.add_argument("--region-ranks", type=int, nargs=2, default=(0, 0))
    args = parser.parse_args()
    records = []
    for index in range(args.parents):
        family = FAMILIES[index % 3]
        record = audit_parent(family, args.seed_start + index,
                              limit_sets=args.limit_sets, device=args.device,
                              decision_step=args.decision_step,
                              region_ranks=tuple(args.region_ranks))
        records.append(record)
        print(family, record["parent_seed"], record["best_sets"],
              round(record["wall_seconds"], 2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"physical_audit_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
