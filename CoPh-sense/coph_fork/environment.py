"""Causal sensing, communication, terrain dynamics, and accounting for CoPH-Fork."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from vmas import make_env


ROUTES = ("top", "bottom")
MODALITIES = ("geometry", "traction")
AGENTS = ("scout", "carrier")


@dataclass(frozen=True)
class ForkWorld:
    top_geometry: bool = True
    top_traction: float = 0.90
    bottom_geometry: bool = True
    bottom_traction: float = 0.28

    def value(self, route: str, modality: str):
        if route not in ROUTES or modality not in MODALITIES:
            raise ValueError("invalid route or modality")
        if modality == "geometry":
            return bool(getattr(self, route + "_geometry"))
        return float(getattr(self, route + "_traction"))


@dataclass(frozen=True)
class ForkConfig:
    horizon: int = 240
    dt: float = 0.05
    sensing_range_scout: float = 0.32
    sensing_range_carrier: float = 0.20
    communication_range: float = 0.90
    communication_delay_steps: int = 3
    packet_drop_probability: float = 0.0
    decision_x: float = -0.18
    scout_start_y: float = 0.13
    carrier_start_y: float = -0.13
    goal_y: float = 0.0
    carrier_goal_y: float = -0.08
    scout_goal_y: float = 0.12
    bottom_route_y: float = -0.43
    carrier_top_y: float = 0.43
    direct_top_exit: bool = False
    scout_bottom_y: float = -0.64
    scout_detour_x: float = -0.70
    traction_safe_threshold: float = 0.55
    backup_traction_prior_mean: float = 0.28
    goal_radius: float = 0.18
    sensing_cost: float = 0.04
    communication_attempt_cost: float = 0.01
    byte_cost: float = 0.00005
    header_bytes: int = 32
    payload_bytes: int = 16
    ack_payload_bytes: int = 8
    time_cost: float = 0.002
    motion_cost: float = 0.001
    risk_cost: float = 0.12
    collision_cost: float = 0.10
    invalid_action_cost: float = 0.02
    terminal_failure_cost: float = 3.0


@dataclass
class ForkAction:
    motion: Tuple[float, float] = (0.0, 0.0)
    sense_route: Optional[str] = None
    sense_modality: Optional[str] = None
    send_observation_id: Optional[str] = None

    def validate(self):
        if len(self.motion) != 2 or max(abs(float(x)) for x in self.motion) > 1.0:
            raise ValueError("motion must contain two values in [-1, 1]")
        if (self.sense_route is None) != (self.sense_modality is None):
            raise ValueError("sense route and modality must be specified together")
        if self.sense_route is not None and self.sense_route not in ROUTES:
            raise ValueError("unknown sensing route")
        if self.sense_modality is not None and self.sense_modality not in MODALITIES:
            raise ValueError("unknown sensing modality")


@dataclass
class Evidence:
    observation_id: str
    source: str
    route: str
    modality: str
    value: object
    acquisition_step: int
    acquisition_position: Tuple[float, float]


@dataclass
class Packet:
    packet_id: str
    sender: str
    receiver: str
    observation_id: str
    transmission_step: int
    delivery_step: int
    payload_bytes: int
    header_bytes: int
    dropped: bool
    kind: str = "evidence"


@dataclass
class CostLedger:
    time: float = 0.0
    motion: float = 0.0
    risk: float = 0.0
    sensing: float = 0.0
    communication: float = 0.0
    collision: float = 0.0
    invalid_action: float = 0.0
    terminal_failure: float = 0.0

    @property
    def total(self):
        return float(sum(asdict(self).values()))

    def add(self, other):
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


class CoPHForkEnv:
    """Single-world A1 environment with a structured causal action interface."""

    def __init__(
        self,
        config: ForkConfig = ForkConfig(),
        world: Optional[ForkWorld] = None,
        seed: int = 0,
    ):
        self.config = config
        self.world_spec = world or ForkWorld()
        self.seed = int(seed)
        self.scenario_path = Path(__file__).with_name("scenario.py")
        self._make_physics()
        self.reset(seed=self.seed)

    def _make_physics(self):
        self.physics = make_env(
            scenario_name=str(self.scenario_path),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            max_steps=self.config.horizon,
            seed=self.seed,
            top_geometry=self.world_spec.top_geometry,
            bottom_geometry=self.world_spec.bottom_geometry,
            dt=self.config.dt,
            scout_start_y=self.config.scout_start_y,
            carrier_start_y=self.config.carrier_start_y,
            goal_y=self.config.goal_y,
        )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.physics.reset(seed=self.seed)
        self.step_index = 0
        self.done = False
        self.success = False
        self.route_commitment = None
        self.evidence = {name: {} for name in AGENTS}  # type: Dict[str, Dict[str, Evidence]]
        self.received = {name: {} for name in AGENTS}  # type: Dict[str, Dict[str, Evidence]]
        self.acknowledged = {name: set() for name in AGENTS}
        self.packet_queue = []  # type: List[Packet]
        self.ledger = CostLedger()
        self.events = []  # type: List[dict]
        self.action_trace = []  # type: List[dict]
        self.state_trace = [self._state_record()]
        self._observation_counter = 0
        self._packet_counter = 0
        self.forced_evidence_drops = None
        self._forced_evidence_drop_index = 0
        return self.observations()

    @property
    def agent_positions(self):
        return {
            name: self.physics.world.agents[i].state.pos[0].detach().cpu().numpy().copy()
            for i, name in enumerate(AGENTS)
        }

    def _route_at(self, position):
        if -0.25 <= float(position[0]) <= 0.55 and abs(float(position[1])) >= 0.20:
            return "top" if float(position[1]) >= 0 else "bottom"
        return None

    def _traction_at(self, position):
        route = self._route_at(position)
        return 1.0 if route is None else float(self.world_spec.value(route, "traction"))

    @staticmethod
    def _probe_position(route):
        return np.array([-0.34, 0.43 if route == "top" else -0.43], dtype=np.float32)

    def _deliver_due_packets(self, decision_open, incremental):
        retained = []
        pending = list(self.packet_queue)
        while pending:
            packet = pending.pop(0)
            if packet.delivery_step > self.step_index:
                retained.append(packet)
                continue
            event = {
                "type": "packet_delivery",
                "step": self.step_index,
                **asdict(packet),
                "delivered": not packet.dropped,
                "usable_before_decision": bool(decision_open and not packet.dropped),
            }
            if not packet.dropped:
                if packet.kind == "ack":
                    self.acknowledged[packet.receiver].add(packet.observation_id)
                else:
                    source = self.evidence[packet.sender][packet.observation_id]
                    self.received[packet.receiver][packet.observation_id] = source
                    sender_position = self.agent_positions[packet.receiver]
                    receiver_position = self.agent_positions[packet.sender]
                    ack_in_range = float(np.linalg.norm(sender_position - receiver_position)) <= self.config.communication_range
                    ack_bytes = self.config.header_bytes + self.config.ack_payload_bytes
                    # The protocol sends a charged ACK attempt only if radio
                    # contact exists when the evidence is delivered.
                    if ack_in_range:
                        incremental.communication += (
                            self.config.communication_attempt_cost
                            + self.config.byte_cost * ack_bytes
                        )
                        ack = Packet(
                            packet_id="pkt-{:05d}".format(self._packet_counter),
                            sender=packet.receiver,
                            receiver=packet.sender,
                            observation_id=packet.observation_id,
                            transmission_step=self.step_index,
                            delivery_step=self.step_index + self.config.communication_delay_steps,
                            payload_bytes=self.config.ack_payload_bytes,
                            header_bytes=self.config.header_bytes,
                            dropped=bool(self.rng.rand() < self.config.packet_drop_probability),
                            kind="ack",
                        )
                        self._packet_counter += 1
                        if ack.delivery_step <= self.step_index:
                            pending.append(ack)
                        else:
                            retained.append(ack)
                        self.events.append({
                            "type": "ack_transmission", "step": self.step_index,
                            **asdict(ack), "charged_bytes": ack_bytes,
                        })
            self.events.append(event)
        self.packet_queue = retained

    def _sense(self, name, action, incremental):
        if action.sense_route is None:
            return
        incremental.sensing += self.config.sensing_cost
        position = self.agent_positions[name]
        radius = (
            self.config.sensing_range_scout
            if name == "scout"
            else self.config.sensing_range_carrier
        )
        distance = float(np.linalg.norm(position - self._probe_position(action.sense_route)))
        valid = distance <= radius
        event = {
            "type": "acquisition",
            "step": self.step_index,
            "agent": name,
            "route": action.sense_route,
            "modality": action.sense_modality,
            "valid": valid,
            "distance": distance,
            "cost": self.config.sensing_cost,
        }
        if valid:
            observation_id = "obs-{:05d}".format(self._observation_counter)
            self._observation_counter += 1
            evidence = Evidence(
                observation_id=observation_id,
                source=name,
                route=action.sense_route,
                modality=action.sense_modality,
                value=self.world_spec.value(action.sense_route, action.sense_modality),
                acquisition_step=self.step_index,
                acquisition_position=(float(position[0]), float(position[1])),
            )
            self.evidence[name][observation_id] = evidence
            event["observation_id"] = observation_id
            event["value"] = evidence.value
        else:
            incremental.invalid_action += self.config.invalid_action_cost
        self.events.append(event)

    def _send(self, name, action, incremental):
        if action.send_observation_id is None:
            return
        receiver = "carrier" if name == "scout" else "scout"
        position = self.agent_positions
        distance = float(np.linalg.norm(position[name] - position[receiver]))
        bytes_sent = self.config.header_bytes + self.config.payload_bytes
        incremental.communication += (
            self.config.communication_attempt_cost + self.config.byte_cost * bytes_sent
        )
        owns_evidence = action.send_observation_id in self.evidence[name] or action.send_observation_id in self.received[name]
        in_range = distance <= self.config.communication_range
        accepted = owns_evidence and in_range
        event = {
            "type": "transmission",
            "step": self.step_index,
            "sender": name,
            "receiver": receiver,
            "observation_id": action.send_observation_id,
            "distance": distance,
            "in_range": in_range,
            "owns_evidence": owns_evidence,
            "accepted": accepted,
            "charged_bytes": bytes_sent,
        }
        if accepted:
            # Relays retain original evidence provenance.
            if action.send_observation_id in self.evidence[name]:
                source_evidence = self.evidence[name][action.send_observation_id]
            else:
                source_evidence = self.received[name][action.send_observation_id]
                self.evidence[name][action.send_observation_id] = source_evidence
            if self.forced_evidence_drops is None:
                dropped = bool(self.rng.rand() < self.config.packet_drop_probability)
            else:
                if self._forced_evidence_drop_index >= len(self.forced_evidence_drops):
                    raise RuntimeError("forced evidence-drop mask is exhausted")
                dropped = bool(self.forced_evidence_drops[self._forced_evidence_drop_index])
                self._forced_evidence_drop_index += 1
            packet = Packet(
                packet_id="pkt-{:05d}".format(self._packet_counter),
                sender=name,
                receiver=receiver,
                observation_id=action.send_observation_id,
                transmission_step=self.step_index,
                delivery_step=self.step_index + self.config.communication_delay_steps,
                payload_bytes=self.config.payload_bytes,
                header_bytes=self.config.header_bytes,
                dropped=dropped,
                kind="evidence",
            )
            self._packet_counter += 1
            self.packet_queue.append(packet)
            event["packet_id"] = packet.packet_id
            event["delivery_step"] = packet.delivery_step
            event["dropped"] = packet.dropped
        else:
            incremental.invalid_action += self.config.invalid_action_cost
        self.events.append(event)

    def _collision_count(self):
        count = 0
        agents = self.physics.world.agents
        for agent in agents:
            for obstacle in self.physics.world.landmarks[1:]:
                count += int(self.physics.world.is_overlapping(agent, obstacle)[0].item())
        count += int(self.physics.world.is_overlapping(agents[0], agents[1])[0].item())
        return count

    def _goal_reached(self):
        goal = self.physics.world.landmarks[0].state.pos[0]
        distances = [torch.linalg.vector_norm(agent.state.pos[0] - goal).item() for agent in self.physics.world.agents]
        return all(distance <= self.config.goal_radius for distance in distances)

    def step(self, actions: Dict[str, ForkAction]):
        if self.done:
            raise RuntimeError("episode is done; call reset")
        if set(actions) != set(AGENTS):
            raise ValueError("actions must contain scout and carrier")
        for action in actions.values():
            action.validate()
        self.action_trace.append({name: asdict(actions[name]) for name in AGENTS})

        incremental = CostLedger(time=self.config.time_cost)
        carrier_x = float(self.agent_positions["carrier"][0])
        decision_open = self.route_commitment is None and carrier_x < self.config.decision_x
        self._deliver_due_packets(decision_open, incremental)
        for name in AGENTS:
            self._sense(name, actions[name], incremental)
        for name in AGENTS:
            self._send(name, actions[name], incremental)
        # Zero-delay messages are available to the physical decision at this step.
        self._deliver_due_packets(decision_open, incremental)

        physical_actions = []
        commanded_effort = 0.0
        pre_positions = self.agent_positions
        for name in AGENTS:
            command = np.asarray(actions[name].motion, dtype=np.float32)
            commanded_effort += float(np.dot(command, command))
            traction = self._traction_at(pre_positions[name])
            applied = np.clip(command * traction, -1.0, 1.0)
            physical_actions.append(torch.tensor(applied, dtype=torch.float32).unsqueeze(0))
        self.physics.step(physical_actions)

        self.step_index += 1
        positions = self.agent_positions
        if self.route_commitment is None and float(positions["carrier"][0]) >= self.config.decision_x:
            self.route_commitment = "top" if float(positions["carrier"][1]) >= 0 else "bottom"
            self.events.append(
                {"type": "route_commitment", "step": self.step_index, "route": self.route_commitment}
            )
        incremental.motion += self.config.motion_cost * commanded_effort * self.config.dt
        carrier = self.physics.world.agents[1]
        carrier_speed = float(torch.linalg.vector_norm(carrier.state.vel[0]).item())
        carrier_traction = self._traction_at(positions["carrier"])
        incremental.risk += (
            self.config.risk_cost
            * max(0.0, self.config.traction_safe_threshold - carrier_traction)
            * carrier_speed
            * self.config.dt
        )
        collisions = self._collision_count()
        incremental.collision += self.config.collision_cost * collisions

        self.success = self._goal_reached()
        if self.success:
            self.done = True
        elif self.step_index >= self.config.horizon:
            self.done = True
            incremental.terminal_failure += self.config.terminal_failure_cost
        self.ledger.add(incremental)
        self.state_trace.append(self._state_record())
        info = {
            "step": self.step_index,
            "success": self.success,
            "done": self.done,
            "route_commitment": self.route_commitment,
            "incremental_cost": asdict(incremental),
            "incremental_total": incremental.total,
            "episode_ledger": asdict(self.ledger),
            "episode_total": self.ledger.total,
            "collisions": collisions,
        }
        reward = -incremental.total
        return self.observations(), reward, self.done, info

    def _evidence_summary(self, name):
        summary = {route: {modality: None for modality in MODALITIES} for route in ROUTES}
        for collection in (self.evidence[name], self.received[name]):
            for item in collection.values():
                summary[item.route][item.modality] = item.value
        return summary

    def observations(self):
        kinematics = self.physics.scenario.observation
        result = {}
        for i, name in enumerate(AGENTS):
            result[name] = {
                "kinematics": kinematics(self.physics.world.agents[i])[0].detach().cpu().numpy().tolist(),
                "time_step": self.step_index,
                "remaining_steps": self.config.horizon - self.step_index,
                "task_metadata": {
                    "backup_traction_prior_mean": self.config.backup_traction_prior_mean,
                    "carrier_top_y": self.config.carrier_top_y,
                    "bottom_route_y": self.config.bottom_route_y,
                    "goal_y": self.config.goal_y,
                },
                "decision_open": self.route_commitment is None,
                "route_commitment": self.route_commitment,
                "evidence": self._evidence_summary(name),
                "owned_observation_ids": sorted(self.evidence[name]),
                "received_observation_ids": sorted(self.received[name]),
                "acknowledged_observation_ids": sorted(self.acknowledged[name]),
                "queued_outgoing_packets": sum(packet.sender == name for packet in self.packet_queue),
            }
        return result

    def flat_observation(self, name):
        obs = self.observations()[name]
        values = list(obs["kinematics"])
        for route in ROUTES:
            for modality in MODALITIES:
                value = obs["evidence"][route][modality]
                values.extend((0.0 if value is None else 1.0, 0.0 if value is None else float(value)))
        values.extend(
            (
                self.step_index / self.config.horizon,
                float(obs["decision_open"]),
                float(obs["queued_outgoing_packets"]),
            )
        )
        return np.asarray(values, dtype=np.float32)

    def _state_record(self):
        positions = self.agent_positions
        return {
            "step": getattr(self, "step_index", 0),
            "scout_position": positions["scout"].round(8).tolist(),
            "carrier_position": positions["carrier"].round(8).tolist(),
        }

    def replay_payload(self):
        payload = {
            "format": "coph-fork-replay-v1",
            "seed": self.seed,
            "config": asdict(self.config),
            "world": asdict(self.world_spec),
            "actions": self.action_trace,
            "states": self.state_trace,
            "events": self.events,
            "ledger": asdict(self.ledger),
            "success": self.success,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    def save_replay(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.replay_payload(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def replay(cls, payload):
        env = cls(
            config=ForkConfig(**payload["config"]),
            world=ForkWorld(**payload["world"]),
            seed=payload["seed"],
        )
        for joint_action in payload["actions"]:
            actions = {}
            for name in AGENTS:
                data = joint_action[name].copy()
                data["motion"] = tuple(data["motion"])
                actions[name] = ForkAction(**data)
            env.step(actions)
        return env
