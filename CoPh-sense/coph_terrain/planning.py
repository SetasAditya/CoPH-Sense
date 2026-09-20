"""Observation-only global costmap planning and sensing candidates."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from .environment import cell_to_position, position_to_cell
from .execution import belief_risk_grid
from .generator import GRID_SIZE, RESOLUTION


GOAL_CELL = (55, 30)
_EDGE_SRC = []
_EDGE_DST = []
for _x in range(GRID_SIZE):
    for _y in range(GRID_SIZE):
        for _nx, _ny in ((_x + 1, _y), (_x - 1, _y),
                         (_x, _y + 1), (_x, _y - 1)):
            if 0 <= _nx < GRID_SIZE and 0 <= _ny < GRID_SIZE:
                _EDGE_SRC.append(_x * GRID_SIZE + _y)
                _EDGE_DST.append(_nx * GRID_SIZE + _ny)
_EDGE_SRC = np.asarray(_EDGE_SRC, dtype=np.int32)
_EDGE_DST = np.asarray(_EDGE_DST, dtype=np.int32)


def observed_costmap(observation):
    geometry = observation["known_geometry"]
    risk = belief_risk_grid(observation)
    cost = 1. + 1.5 * risk + .25 * np.isnan(geometry)
    blocked = geometry == 1.
    # One-cell obstacle inflation accounts for the carrier disc and a small
    # clearance margin. Unknown cells remain traversable with a prior cost.
    inflated = blocked.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            inflated[max(0, dx):GRID_SIZE + min(0, dx),
                     max(0, dy):GRID_SIZE + min(0, dy)] |= blocked[
                         max(0, -dx):GRID_SIZE - max(0, dx),
                         max(0, -dy):GRID_SIZE - max(0, dy)]
    cost[inflated] = np.inf
    return cost


def dijkstra(cost, start):
    flat = np.asarray(cost, dtype=np.float64).reshape(-1)
    weights = .5 * (flat[_EDGE_SRC] + flat[_EDGE_DST]) * RESOLUTION
    valid = np.isfinite(weights)
    graph = csr_matrix((weights[valid], (_EDGE_SRC[valid], _EDGE_DST[valid])),
                       shape=(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE))
    start_index = start[0] * GRID_SIZE + start[1]
    distances, previous = sparse_dijkstra(
        graph, directed=True, indices=start_index, return_predecessors=True)
    return distances.reshape((GRID_SIZE, GRID_SIZE)), previous.reshape(
        (GRID_SIZE, GRID_SIZE))


def plan_path(observation, goal=GOAL_CELL):
    cost = observed_costmap(observation)
    start = position_to_cell(observation["kinematics"][:2])
    # A robot can occupy a cell inside the planner's conservative inflation
    # margin without colliding. Permit a path *out* of that fringe rather
    # than returning no route and holding its current position indefinitely.
    cost[start] = min(float(cost[start]), 1.)
    distances, previous = dijkstra(cost, start)
    if not np.isfinite(distances[goal]):
        return (), distances
    path = [goal]
    while path[-1] != start:
        preceding = int(previous[path[-1]])
        if preceding < 0:
            raise RuntimeError("predecessor missing on finite route")
        path.append((preceding // GRID_SIZE, preceding % GRID_SIZE))
    path.reverse()
    return tuple(path), distances


def _path_on_cost(cost, start, goal=GOAL_CELL):
    distances, previous = dijkstra(cost, start)
    if not np.isfinite(distances[goal]):
        return ()
    path = [goal]
    while path[-1] != start:
        index = int(previous[path[-1]])
        if index < 0:
            return ()
        path.append((index // GRID_SIZE, index % GRID_SIZE))
    return tuple(reversed(path))


@dataclass(frozen=True)
class SensingCandidate:
    modality: str
    viewpoint: tuple
    travel_distance: float
    travel_time: float
    dwell_time: float
    sensing_cost: float
    unknown_fraction: float
    route_distance: float
    deadline_slack: float
    target_region: tuple = None

    def features(self):
        region = self.target_region or self.viewpoint
        return (float(region[0]) / GRID_SIZE,
                float(region[1]) / GRID_SIZE,
                float(self.viewpoint[0]) / GRID_SIZE,
                float(self.viewpoint[1]) / GRID_SIZE,
                float(self.modality == "geometry"),
                self.travel_distance / 12., self.travel_time / 60.,
                self.dwell_time / 60., self.sensing_cost,
                self.unknown_fraction, self.route_distance / 12.,
                self.deadline_slack / 60.)


def _public_line_of_sight(geometry, start, destination):
    """Conservative grid visibility using only the known static topology."""
    source = cell_to_position(start)
    target = cell_to_position(destination)
    distance = float(np.linalg.norm(target - source))
    for fraction in np.linspace(0, 1, max(2, int(distance / .05)),
                                endpoint=False)[1:]:
        cell = position_to_cell(source + fraction * (target - source))
        if geometry[cell] == 1. and cell != destination:
            return False
    return True


def _acquisition_pose(cell, modality, distances, goal_distances, observation,
                      config, agent_name, mission_path=None):
    """Choose a feasible sensing pose, preferring the existing mission path."""
    if modality == "traction":
        return cell if (np.isfinite(distances[cell]) and
                        (mission_path is None or cell in mission_path)) else None
    radius = config.scan_range_scout if agent_name == "scout" else config.scan_range_carrier
    radius_cells = int(np.floor(radius / RESOLUTION))
    geometry = observation["known_geometry"]
    options = []
    route_order = {pose: index for index, pose in enumerate(mission_path or ())}
    target_order = route_order.get(cell, len(route_order))
    for x in range(max(0, cell[0] - radius_cells),
                   min(GRID_SIZE, cell[0] + radius_cells + 1)):
        for y in range(max(0, cell[1] - radius_cells),
                       min(GRID_SIZE, cell[1] + radius_cells + 1)):
            pose = (x, y)
            if (np.linalg.norm((np.asarray(pose) - cell) * RESOLUTION) > radius - 1e-5
                    or not np.isfinite(distances[pose])
                    or not np.isfinite(goal_distances[pose])
                    or not _public_line_of_sight(geometry, pose, cell)):
                continue
            if mission_path is not None and (pose not in route_order or
                                             route_order[pose] >= target_order):
                continue
            # Earliest visible point preserves time to act on the reading.
            options.append((route_order.get(pose, 0),
                            float(distances[pose] + goal_distances[pose]), pose))
    return min(options)[2] if options else None


def candidate_viewpoints(observation, config, agent_name, max_candidates=16,
                         advertised_path=None):
    """Propose sensor pairs encountered along the current public mission route."""
    if observation["measurements_remaining"] <= 0:
        return ()
    path, distances = plan_path(observation)
    if not path:
        return ()
    route = tuple(advertised_path) if advertised_path is not None else path
    if not route:
        route = path
    position = np.asarray(observation["kinematics"][:2], dtype=float)
    speed = .75 if agent_name == "scout" else .55
    chosen = []
    region_limit = max_candidates // 2
    if region_limit == 0:
        return ()
    # A target is physically encountered on the route already being taken.
    # Geometry is acquired at an earlier visible route cell; traction requires
    # contact at the target itself. Neither action adds a planned side trip.
    base_cost = observed_costmap(observation)
    start = position_to_cell(position)
    goal_distances, _ = dijkstra(base_cost, GOAL_CELL)
    support = {cell for cell in path[4:-4:2] if np.isfinite(distances[cell])}
    appearance = observation["appearance"]
    known_surface = observation["known_surface"]
    known_traction = observation["known_traction"]
    scored = []
    for cell in support:
        if np.linalg.norm(cell_to_position(cell) - position) < .35:
            continue
        x, y = cell
        near = (slice(max(0, x - 3), min(GRID_SIZE, x + 4)),
                slice(max(0, y - 3), min(GRID_SIZE, y + 4)))
        unknown = .5 * (float(np.isnan(known_surface[near]).mean()) +
                        float(np.isnan(known_traction[near]).mean()))
        local_appearance = appearance[near]
        observed = local_appearance[np.isfinite(local_appearance)]
        ambiguity = (float(np.mean(1. - 2. * np.abs(observed - .5)))
                     if observed.size else .5)
        score = ((.55 * unknown + .45 * max(0., ambiguity)) /
                 (1. + .12 * float(distances[cell])))
        scored.append((score, cell))
    for _, cell in sorted(scored, key=lambda item: (-item[0], item[1])):
        if any(np.linalg.norm(np.asarray(cell) - np.asarray(old)) < 4
               for old in chosen):
            continue
        chosen.append(cell)
        if len(chosen) >= region_limit:
            break
    result = []
    for cell in chosen:
        near = (slice(max(0, cell[0] - 5), min(GRID_SIZE, cell[0] + 6)),
                slice(max(0, cell[1] - 5), min(GRID_SIZE, cell[1] + 6)))
        for modality in ("geometry", "traction"):
            pose = _acquisition_pose(cell, modality, distances, goal_distances,
                                     observation, config, agent_name, path)
            if pose is None:
                continue
            grid = (observation["known_surface"] if modality == "geometry"
                    else observation["known_traction"])
            uncertainty = float(np.isnan(grid[near]).mean())
            travel = float(distances[pose])
            travel_time = travel / speed
            dwell_time = config.dt * (config.scan_dwell_steps if modality == "geometry"
                                      else config.probe_dwell_steps)
            sensing_cost = config.scan_cost if modality == "geometry" else config.probe_cost
            deadline_slack = (observation["remaining_steps"] * config.dt
                              - travel_time - dwell_time
                              - config.communication_delay_steps * config.dt)
            if deadline_slack <= 0:
                continue
            pose_route_distance = min(np.linalg.norm(
                (np.asarray(pose) - np.asarray(r)) * RESOLUTION) for r in route)
            result.append(SensingCandidate(modality, pose, travel, travel_time,
                                           dwell_time, sensing_cost, uncertainty,
                                           float(pose_route_distance),
                                           float(deadline_slack), cell))
    return tuple(result[:max_candidates])
