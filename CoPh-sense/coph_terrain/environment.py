"""Causal VMAS E2 environment with private terrain, real sensing, and packets."""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from vmas import make_env

from .generator import GRID_SIZE, RESOLUTION, generate_map, realize_map


AGENTS = ("scout", "carrier")


@dataclass(frozen=True)
class TerrainConfig:
    dt: float = .05
    horizon_steps: int = 1300  # provisional 65 s from excluded full-map calibration
    default_range: float = .6
    scan_range_scout: float = 1.5
    scan_range_carrier: float = 1.0
    probe_range: float = .25
    scan_dwell_steps: int = 4
    probe_dwell_steps: int = 8
    scan_cost: float = .02
    probe_cost: float = .04
    max_measurements_per_agent: int = 4
    max_team_packets: int = 4
    communication_range: float = 2.0
    communication_delay_steps: int = 3
    packet_drop_probability: float = 0.
    communication_attempt_cost: float = .01
    byte_cost: float = .00005
    header_bytes: int = 32
    ack_bytes: int = 40
    time_cost_per_step: float = .002
    motion_cost: float = .001
    collision_cost: float = .1
    failure_cost: float = 10.  # provisional pre-freeze correction: failure must not be cheaper than delay
    goal_radius: float = .40
    stuck_speed: float = .02
    stuck_command: float = .3
    stuck_steps: int = 40
    appearance_noise_std: float = .18
    # Optional-scout missions require carrier delivery.  A scout that was
    # dispatched must also be recovered before success can terminate the
    # episode, so an excursion cannot disappear from the team ledger.
    carrier_primary: bool = False
    scout_recovery_radius: float = .35


@dataclass(frozen=True)
class TerrainAction:
    motion: tuple = (0., 0.)
    waypoint: Optional[Tuple[float, float]] = None
    sense: Optional[Tuple] = None  # (modality, grid_x, grid_y)
    send_evidence_id: Optional[str] = None

    def validate(self):
        if len(self.motion) != 2 or not all(np.isfinite(self.motion)) \
                or max(abs(float(v)) for v in self.motion) > 1:
            raise ValueError("motion must be finite and in [-1,1]^2")
        if self.sense is not None:
            if len(self.sense) != 3 or self.sense[0] not in ("geometry", "traction"):
                raise ValueError("sense must be (geometry|traction, grid_x, grid_y)")
            if not all(isinstance(v, int) and 0 <= v < GRID_SIZE for v in self.sense[1:]):
                raise ValueError("sensing cell is outside the map")
        if self.waypoint is not None and (len(self.waypoint) != 2 or
                                          not all(np.isfinite(self.waypoint))):
            raise ValueError("waypoint must be a finite 2D position")


@dataclass(frozen=True)
class TerrainEvidence:
    evidence_id: str
    source: str
    modality: str
    cells: tuple
    values: tuple
    acquired_step: int
    location: tuple


@dataclass
class TerrainPacket:
    evidence_id: str
    sender: str
    receiver: str
    delivery_step: int
    dropped: bool
    payload_bytes: int
    kind: str = "evidence"


@dataclass
class TerrainLedger:
    time: float = 0.
    motion: float = 0.
    sensing: float = 0.
    communication: float = 0.
    collision: float = 0.
    failure: float = 0.
    material_exposure: float = 0.

    @property
    def mission_cost(self):
        return self.time + self.motion + self.sensing + self.communication + self.collision + self.failure

    def add(self, other):
        for name, value in asdict(other).items():
            setattr(self, name, getattr(self, name) + value)


def position_to_cell(position):
    return tuple(max(0, min(GRID_SIZE - 1,
                             math.floor((float(position[axis]) + 6.) / RESOLUTION)))
                 for axis in (0, 1))


def cell_to_position(cell):
    return np.asarray((-6. + (cell[0] + .5) * RESOLUTION,
                       -6. + (cell[1] + .5) * RESOLUTION), dtype=np.float32)


class CoPHTerrainEnv:
    """One replayable world; no actor method returns the hidden map."""

    def __init__(self, family="open", parent_seed=100000, realization_seed=0,
                 seed=1701, config=None, executor="vmas", device="cpu"):
        self.family = family
        self.parent_seed = int(parent_seed)
        self.realization_seed = int(realization_seed)
        self.seed = int(seed)
        self.config = config or TerrainConfig()
        if executor not in ("vmas", "ph", "material_ph"):
            raise ValueError(executor)
        self.executor = executor
        self.device = str(device)
        self.parent_map = generate_map(family, self.parent_seed)
        self._truth = realize_map(self.parent_map, self.realization_seed)
        self.physics = make_env(
            scenario_name=str(Path(__file__).with_name("scenario.py")),
            num_envs=1, device=self.device, continuous_actions=True,
            max_steps=self.config.horizon_steps, seed=self.seed,
            obstacle_boxes=self._truth.obstacle_boxes, dt=self.config.dt)
        self.reset()

    @property
    def positions(self):
        return {name: self.physics.world.agents[i].state.pos[0].detach().cpu().numpy().copy()
                for i, name in enumerate(AGENTS)}

    def reset(self):
        self.physics.reset(seed=self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.appearance = np.clip(.86 - .65 * self._truth.risk
                                  + self.rng.normal(0, self.config.appearance_noise_std,
                                                    self._truth.traction.shape), 0, 1)
        self.step_index = 0
        self.done = False
        self.success = False
        self.failure_reason = None
        self.ledger = TerrainLedger()
        self.agent_risk = {name: 0. for name in AGENTS}
        self.known_geometry = {name: self._truth.occupancy.astype(np.float32).copy()
                               for name in AGENTS}
        self.known_surface = {name: np.full((GRID_SIZE, GRID_SIZE), np.nan, np.float32)
                              for name in AGENTS}
        self.appearance_belief = {name: np.full((GRID_SIZE, GRID_SIZE), np.nan, np.float32)
                                  for name in AGENTS}
        self.known_traction = {name: np.full((GRID_SIZE, GRID_SIZE), np.nan, np.float32)
                               for name in AGENTS}
        self.owned = {name: {} for name in AGENTS}
        self.received = {name: {} for name in AGENTS}
        self.acknowledged = {name: set() for name in AGENTS}
        self.pending = []
        self.active_sense = {name: None for name in AGENTS}
        self.measurement_count = {name: 0 for name in AGENTS}
        self.packet_attempts = 0
        self.stuck_count = {name: 0 for name in AGENTS}
        self.last_passive_cell = {name: None for name in AGENTS}
        self.passive_visibility_cache = {}
        self.evidence_counter = 0
        self.scout_ever_dispatched = False
        self.events = []
        self.action_trace = []
        self._default_observe()
        self.state_trace = [self._state_record()]
        return self.observations()

    def _state_record(self):
        return {"step": self.step_index,
                "positions": {name: self.positions[name].tolist() for name in AGENTS},
                "velocities": {name: self.physics.world.agents[i].state.vel[0]
                               .detach().cpu().numpy().tolist()
                               for i, name in enumerate(AGENTS)},
                "mission_cost": self.ledger.mission_cost,
                "material_exposure": self.ledger.material_exposure,
                "agent_material_exposure": dict(self.agent_risk),
                "measurement_count": dict(self.measurement_count),
                "packet_attempts": self.packet_attempts}

    def _visible(self, source, destination):
        start = np.asarray(source, dtype=float)
        end = np.asarray(destination, dtype=float)
        distance = float(np.linalg.norm(end - start))
        destination_cell = position_to_cell(end)
        for fraction in np.linspace(0, 1, max(2, int(distance / .05)), endpoint=False)[1:]:
            cell = position_to_cell(start + fraction * (end - start))
            if self._truth.occupancy[cell] and cell != destination_cell:
                return False
        return True

    def _cells_in_range(self, source, radius):
        x0, y0 = position_to_cell(source)
        reach = int(np.ceil(radius / RESOLUTION)) + 1
        for x in range(max(0, x0 - reach), min(GRID_SIZE, x0 + reach + 1)):
            for y in range(max(0, y0 - reach), min(GRID_SIZE, y0 + reach + 1)):
                center = cell_to_position((x, y))
                if np.linalg.norm(center - source) <= radius and self._visible(source, center):
                    yield (x, y)

    def _default_observe(self):
        for name, position in self.positions.items():
            source_cell = position_to_cell(position)
            if self.last_passive_cell[name] == source_cell:
                continue
            self.last_passive_cell[name] = source_cell
            if source_cell not in self.passive_visibility_cache:
                self.passive_visibility_cache[source_cell] = tuple(self._cells_in_range(
                    cell_to_position(source_cell), self.config.default_range))
            for cell in self.passive_visibility_cache[source_cell]:
                self.known_geometry[name][cell] = float(self._truth.occupancy[cell])
                self.appearance_belief[name][cell] = float(self.appearance[cell])
            cells = self.passive_visibility_cache[source_cell]
            self.events.append({"type": "passive_observation", "step": self.step_index,
                                "agent": name, "source_cell": source_cell,
                                "cells": cells,
                                "geometry": [int(self._truth.occupancy[cell]) for cell in cells],
                                "appearance": [float(self.appearance[cell]) for cell in cells]})

    def _complete_sense(self, name, request, incremental):
        modality, x, y = request["sense"]
        location = cell_to_position((x, y))
        position = self.positions[name]
        footprint_radius = (self.config.probe_range if modality == "traction" else
                            self.config.scan_range_scout if name == "scout" else
                            self.config.scan_range_carrier)
        valid = bool(np.linalg.norm(position - location) <= self.config.probe_range
                     and self._visible(position, location))
        if not valid:
            self.events.append({"type": "acquisition", "step": self.step_index,
                                "agent": name, "modality": modality,
                                "cell": (x, y), "valid": False})
            return
        cells = ((x, y),) if modality == "traction" else tuple(
            self._cells_in_range(position, footprint_radius))
        values = tuple(float(self._truth.traction[cell]) if modality == "traction"
                       else float(self._truth.surface[cell]) for cell in cells)
        evidence_id = f"obs-{self.evidence_counter:06d}"
        self.evidence_counter += 1
        evidence = TerrainEvidence(evidence_id, name, modality, cells, values,
                                   self.step_index, tuple(float(v) for v in position))
        self.owned[name][evidence_id] = evidence
        self._apply_evidence(name, evidence)
        self.events.append({"type": "acquisition", "step": self.step_index,
                            "agent": name, "modality": modality,
                            "cell": (x, y), "valid": True,
                            "evidence_id": evidence_id, "cell_count": len(cells),
                            "cells": cells, "values": values})

    def _apply_evidence(self, name, evidence):
        grid = (self.known_surface[name] if evidence.modality == "geometry"
                else self.known_traction[name])
        for cell, value in zip(evidence.cells, evidence.values):
            grid[cell] = float(value)

    def _deliver_packets(self, incremental):
        future = []
        for packet in self.pending:
            if packet.delivery_step > self.step_index:
                future.append(packet)
                continue
            delivered = not packet.dropped
            if delivered:
                if packet.kind == "ack":
                    self.acknowledged[packet.receiver].add(packet.evidence_id)
                else:
                    evidence = self.owned[packet.sender][packet.evidence_id]
                    self.received[packet.receiver][packet.evidence_id] = evidence
                    self._apply_evidence(packet.receiver, evidence)
                    if np.linalg.norm(self.positions[packet.sender]
                                      - self.positions[packet.receiver]) <= self.config.communication_range:
                        incremental.communication += (self.config.communication_attempt_cost
                                                      + self.config.byte_cost * self.config.ack_bytes)
                        future.append(TerrainPacket(
                            packet.evidence_id, packet.receiver, packet.sender,
                            self.step_index + max(1, self.config.communication_delay_steps),
                            bool(self.rng.random() < self.config.packet_drop_probability),
                            8, "ack"))
            self.events.append({"type": "packet_delivery", "step": self.step_index,
                                "evidence_id": packet.evidence_id,
                                "receiver": packet.receiver, "delivered": delivered,
                                "kind": packet.kind})
        self.pending = future

    def observations(self):
        result = {}
        for i, name in enumerate(AGENTS):
            result[name] = {
                "kinematics": self.physics.scenario.observation(
                    self.physics.world.agents[i])[0].detach().cpu().numpy().copy(),
                "time_step": self.step_index,
                "remaining_steps": self.config.horizon_steps - self.step_index,
                "known_geometry": self.known_geometry[name].copy(),
                "known_surface": self.known_surface[name].copy(),
                "appearance": self.appearance_belief[name].copy(),
                "known_traction": self.known_traction[name].copy(),
                "owned_evidence_ids": tuple(sorted(self.owned[name])),
                "received_evidence_ids": tuple(sorted(self.received[name])),
                "acknowledged_evidence_ids": tuple(sorted(self.acknowledged[name])),
                "evidence": tuple(asdict(e) for e in
                                  list(self.owned[name].values()) + list(self.received[name].values())),
                "measurements_remaining": self.config.max_measurements_per_agent
                                          - self.measurement_count[name],
                "team_packets_remaining": self.config.max_team_packets
                                          - self.packet_attempts,
                "active_sense": None if self.active_sense[name] is None else
                                dict(self.active_sense[name]),
            }
        return result

    def step(self, actions):
        if self.done:
            raise RuntimeError("episode is done")
        if set(actions) != set(AGENTS):
            raise ValueError("actions must contain scout and carrier")
        for action in actions.values():
            action.validate()
        self.action_trace.append({name: asdict(actions[name]) for name in AGENTS})
        incremental = TerrainLedger(time=self.config.time_cost_per_step)
        self._deliver_packets(incremental)
        for name in AGENTS:
            action = actions[name]
            if action.sense is not None:
                if self.active_sense[name] is not None or self.measurement_count[name] >= self.config.max_measurements_per_agent:
                    self.events.append({"type": "rejected_sense", "agent": name,
                                        "step": self.step_index, "reason": "busy_or_budget"})
                else:
                    modality = action.sense[0]
                    self.measurement_count[name] += 1
                    incremental.sensing += (self.config.scan_cost if modality == "geometry"
                                            else self.config.probe_cost)
                    self.active_sense[name] = {
                        "sense": action.sense,
                        "remaining": (self.config.scan_dwell_steps if modality == "geometry"
                                      else self.config.probe_dwell_steps),
                    }
                    self.events.append({"type": "sensing_start", "agent": name,
                                        "step": self.step_index, "sense": action.sense})
        for name in AGENTS:
            evidence_id = actions[name].send_evidence_id
            if evidence_id is None:
                continue
            receiver = "carrier" if name == "scout" else "scout"
            owns = evidence_id in self.owned[name]
            in_range = np.linalg.norm(self.positions[name] - self.positions[receiver]) <= self.config.communication_range
            budget = self.packet_attempts < self.config.max_team_packets
            evidence = self.owned[name].get(evidence_id)
            payload_bytes = (16 + 4 * len(evidence.cells)) if evidence is not None else 0
            if budget:
                self.packet_attempts += 1
                incremental.communication += (self.config.communication_attempt_cost
                                              + self.config.byte_cost
                                              * (self.config.header_bytes + payload_bytes))
            accepted = bool(owns and in_range and budget)
            self.events.append({"type": "transmission", "sender": name,
                                "step": self.step_index, "evidence_id": evidence_id,
                                "accepted": accepted, "in_range": bool(in_range),
                                "charged_bytes": self.config.header_bytes + payload_bytes if budget else 0})
            if not accepted:
                continue
            self.pending.append(TerrainPacket(
                evidence_id, name, receiver,
                self.step_index + self.config.communication_delay_steps,
                bool(self.rng.random() < self.config.packet_drop_probability),
                payload_bytes))
        self._deliver_packets(incremental)  # supports zero-delay packets
        pre = self.positions
        physical = []
        physical_effort = {}
        for name in AGENTS:
            command = np.asarray(actions[name].motion, dtype=np.float32)
            physical_effort[name] = float(np.linalg.norm(command))
            if self.executor == "vmas":
                incremental.motion += (self.config.motion_cost
                                       * float(command @ command) * self.config.dt)
            cell = position_to_cell(pre[name])
            traction = float(self._truth.traction[cell]
                             * (1. - .2 * self._truth.surface[cell]))
            physical.append(torch.tensor(command * traction,
                                         dtype=torch.float32,
                                         device=self.physics.world.device).unsqueeze(0))
        if self.executor == "vmas":
            self.physics.step(physical)
        else:
            from .execution import ph_step
            from coph_fork.ph_factorial_controller import box_clearance
            pre_observation = self.observations()
            proposed = {}
            for i, name in enumerate(AGENTS):
                actor = self.physics.world.agents[i]
                velocity = actor.state.vel[0].detach().cpu().numpy().copy()
                target = actions[name].waypoint
                if target is None:
                    target = tuple(float(v) for v in pre[name])
                stepper = ph_step
                if self.executor == "material_ph":
                    from .material_executor import MaterialHamiltonianExecutor
                    if not hasattr(self, "_material_executor"):
                        self._material_executor = MaterialHamiltonianExecutor(device=self.device)
                    stepper = self._material_executor.step
                q_next, v_next, diagnostic = stepper(
                    pre[name], velocity, target, pre_observation[name],
                    float(actor.mass), float(actor.shape.radius),
                    float(actor.max_speed),
                    float(self._truth.traction[position_to_cell(pre[name])]
                          * (1. - .2 * self._truth.surface[position_to_cell(pre[name])])),
                    self.config.dt)
                incremental.motion += (self.config.motion_cost
                                       * diagnostic["force_norm"] ** 2
                                       * self.config.dt)
                physical_effort[name] = diagnostic["force_norm"]
                proposed[name] = [q_next, v_next]
            for i, name in enumerate(AGENTS):
                q_next, _ = proposed[name]
                known = pre_observation[name]["known_geometry"]
                ix, iy = position_to_cell(q_next)
                radius = float(self.physics.world.agents[i].shape.radius)
                collision_predicted = False
                for x in range(max(0, ix - 2), min(GRID_SIZE, ix + 3)):
                    for y in range(max(0, iy - 2), min(GRID_SIZE, iy + 3)):
                        if known[x, y] != 1.:
                            continue
                        clearance, _ = box_clearance(
                            q_next, cell_to_position((x, y)), (.1, .1), radius)
                        if clearance < .015:
                            collision_predicted = True
                            break
                    if collision_predicted:
                        break
                if collision_predicted:
                    proposed[name] = [pre[name].astype(float), np.zeros(2)]
                    self.events.append({"type": "shield_intervention", "agent": name,
                                        "step": self.step_index, "reason": "observed_geometry"})
            separation = float(np.linalg.norm(proposed["scout"][0]
                                              - proposed["carrier"][0]))
            if separation < .065 + .10 + .02:
                for name in AGENTS:
                    proposed[name] = [pre[name].astype(float), np.zeros(2)]
                    self.events.append({"type": "shield_intervention", "agent": name,
                                        "step": self.step_index, "reason": "teammate_clearance"})
            for i, name in enumerate(AGENTS):
                actor = self.physics.world.agents[i]
                actor.set_pos(torch.tensor(proposed[name][0], dtype=torch.float32,
                                           device=self.physics.world.device),
                              batch_index=0)
                actor.set_vel(torch.tensor(proposed[name][1], dtype=torch.float32,
                                           device=self.physics.world.device),
                              batch_index=0)
        self.step_index += 1
        self._default_observe()
        for name in AGENTS:
            request = self.active_sense[name]
            if request is None:
                continue
            request["remaining"] -= 1
            if request["remaining"] == 0:
                self._complete_sense(name, request, incremental)
                self.active_sense[name] = None
        positions = self.positions
        risk_by_agent = {name: float(self._truth.risk[position_to_cell(positions[name])])
                         for name in AGENTS}
        risk_scale = 1. / self.config.horizon_steps
        incremental.material_exposure += risk_scale * sum(risk_by_agent.values())
        for name in AGENTS:
            self.agent_risk[name] += risk_scale * risk_by_agent[name]
        from coph_fork.ph_factorial_controller import box_clearance
        collisions = 0
        for i, name in enumerate(AGENTS):
            radius = float(self.physics.world.agents[i].shape.radius)
            q = positions[name]
            for x0, x1, y0, y1 in self._truth.obstacle_boxes:
                center = (-6. + (x0 + x1) * .1, -6. + (y0 + y1) * .1)
                half = ((x1 - x0) * .1, (y1 - y0) * .1)
                clearance, _ = box_clearance(q, center, half, radius)
                collisions += int(clearance < 0.)
        collisions += int(np.linalg.norm(positions["scout"] - positions["carrier"])
                          < .065 + .10)
        if collisions:
            self.failure_reason = "collision"
            incremental.collision += self.config.collision_cost * collisions
            self.events.append({"type": "collision", "step": self.step_index,
                                "positions": {name: position.tolist()
                                              for name, position in positions.items()},
                                "count": collisions})
        for i, name in enumerate(AGENTS):
            actor = self.physics.world.agents[i]
            speed = float(torch.linalg.vector_norm(actor.state.vel[0]).item())
            effort = physical_effort[name]
            self.stuck_count[name] = (self.stuck_count[name] + 1 if
                                      effort >= self.config.stuck_command
                                      and speed < self.config.stuck_speed else 0)
            if self.stuck_count[name] >= self.config.stuck_steps:
                if name == "carrier" or not self.config.carrier_primary:
                    self.failure_reason = "immobilization"
                else:
                    self.events.append({"type": "scout_immobilized",
                                        "step": self.step_index})
        goal = np.asarray((5., 0.))
        if self.config.carrier_primary:
            carrier_arrived = (np.linalg.norm(positions["carrier"] - goal)
                               <= self.config.goal_radius)
            scout_recovered = (not self.scout_ever_dispatched or
                               np.linalg.norm(positions["scout"] -
                                              positions["carrier"])
                               <= self.config.scout_recovery_radius)
            self.success = (carrier_arrived and scout_recovered and
                            self.failure_reason is None)
        else:
            self.success = all(
                np.linalg.norm(positions[name] - goal) <= self.config.goal_radius
                for name in AGENTS) and self.failure_reason is None
        if self.success or self.failure_reason or self.step_index >= self.config.horizon_steps:
            self.done = True
            if not self.success:
                self.failure_reason = self.failure_reason or "deadline"
                incremental.failure += self.config.failure_cost
        self.ledger.add(incremental)
        self.state_trace.append(self._state_record())
        info = {"step": self.step_index, "done": self.done,
                "success": self.success, "failure_reason": self.failure_reason,
                "incremental": asdict(incremental),
                "mission_cost": self.ledger.mission_cost,
                "material_exposure": self.ledger.material_exposure,
                "risk_by_agent": risk_by_agent, "collisions": collisions}
        return self.observations(), -incremental.mission_cost, self.done, info

    def replay_payload(self):
        payload = {"family": self.family, "parent_seed": self.parent_seed,
                   "realization_seed": self.realization_seed, "seed": self.seed,
                   "config": asdict(self.config), "executor": self.executor,
                   "device": self.device,
                   "actions": self.action_trace,
                   "map_sha256": self._truth.digest(),
                   "ledger": asdict(self.ledger),
                   "agent_material_exposure": dict(self.agent_risk),
                   "success": self.success,
                   "failure_reason": self.failure_reason}
        payload["sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True).encode()).hexdigest()
        return payload

    def save_artifact(self, directory):
        """Evaluator-only replay bundle; no truth layer enters observations."""
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps(self.replay_payload(), indent=2) + "\n")
        (directory / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in self.events))
        np.savez_compressed(
            directory / "states.npz",
            positions=np.asarray([[record["positions"][name] for name in AGENTS]
                                  for record in self.state_trace], dtype=np.float32),
            velocities=np.asarray([[record["velocities"][name] for name in AGENTS]
                                   for record in self.state_trace], dtype=np.float32),
            mission_cost=np.asarray([record["mission_cost"] for record in self.state_trace]),
            material_exposure=np.asarray([record["material_exposure"]
                                          for record in self.state_trace]),
            agent_material_exposure=np.asarray(
                [[record["agent_material_exposure"][name] for name in AGENTS]
                 for record in self.state_trace], dtype=np.float64),
        )
        np.savez_compressed(directory / "truth_map.npz",
                            occupancy=self._truth.occupancy,
                            surface=self._truth.surface,
                            traction=self._truth.traction,
                            risk=self._truth.risk)
        np.savez_compressed(
            directory / "belief_final.npz",
            **{f"{name}_{kind}": grid for name in AGENTS for kind, grid in (
                ("geometry", self.known_geometry[name]),
                ("surface", self.known_surface[name]),
                ("appearance", self.appearance_belief[name]),
                ("traction", self.known_traction[name]))})

    def clone(self):
        """Exact simulator/RNG snapshot for nonanticipative paired rollouts."""
        return copy.deepcopy(self)

    @classmethod
    def replay(cls, payload):
        env = cls(payload["family"], payload["parent_seed"],
                  payload["realization_seed"], payload["seed"],
                  TerrainConfig(**payload["config"]),
                  executor=payload["executor"], device=payload.get("device", "cpu"))
        for row in payload["actions"]:
            env.step({name: TerrainAction(**row[name]) for name in AGENTS})
        if (env._truth.digest() != payload["map_sha256"]
                or asdict(env.ledger) != payload["ledger"]
                or env.agent_risk != payload["agent_material_exposure"]):
            raise RuntimeError("deterministic replay mismatch")
        return env
