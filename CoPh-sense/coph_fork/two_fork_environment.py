"""Two-fork extension of the audited VMAS packet and cost ledger."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vmas import make_env

from .environment import CoPHForkEnv, Evidence, ForkAction, ForkConfig


@dataclass(frozen=True)
class TwoForkWorld:
    geometry_1: bool = True
    traction_1: float = .9
    geometry_2: bool = True
    traction_2: float = .9
    backup_traction_1: float = .3
    backup_traction_2: float = .3

    def value(self, region, route, modality):
        if region not in (1, 2) or route not in ("top", "bottom") \
                or modality not in ("geometry", "traction"):
            raise ValueError("invalid two-fork terrain query")
        if route == "bottom":
            return True if modality == "geometry" else getattr(self, f"backup_traction_{region}")
        return getattr(self, f"{modality}_{region}")


@dataclass(frozen=True)
class TwoForkConfig(ForkConfig):
    horizon: int = 1900
    decision_x: float = -.64
    decision_x2: float = .20
    scout_start_y: float = .4
    carrier_start_y: float = .0
    goal_radius: float = .22
    sensing_range_carrier: float = .24
    backup_y1: float = -.43
    backup_y2: float = -.43


@dataclass
class TwoForkAction(ForkAction):
    sense_region: int = 1

    def validate(self):
        super().validate()
        if self.sense_region not in (1, 2):
            raise ValueError("invalid sensing region")


@dataclass
class TwoForkEvidence(Evidence):
    region: int = 1


class TwoForkEnv(CoPHForkEnv):
    def __init__(self, config=None, world=None, seed=0):
        self.route_commitments = {1: None, 2: None}
        super().__init__(config or TwoForkConfig(), world or TwoForkWorld(), seed)

    def _make_physics(self):
        self.scenario_path = Path(__file__).with_name("two_fork_scenario.py")
        self.physics = make_env(
            scenario_name=str(self.scenario_path), num_envs=1,
            device="cpu", continuous_actions=True,
            max_steps=self.config.horizon, seed=self.seed,
            top_geometry=(self.world_spec.geometry_1, self.world_spec.geometry_2),
            backup_waypoints=(self.config.backup_y1, self.config.backup_y2),
            dt=self.config.dt, scout_start_y=self.config.scout_start_y,
            carrier_start_y=self.config.carrier_start_y,
        )

    def reset(self, seed=None):
        self.route_commitments = {1: None, 2: None}
        result = super().reset(seed)
        self.route_commitments = {1: None, 2: None}
        return result

    def _route_at(self, position):
        x, y = float(position[0]), float(position[1])
        region = 1 if -.60 <= x <= -.20 else 2 if .24 <= x <= .62 else None
        if region is None or abs(y) < .18:
            return None
        return region, "top" if y > 0 else "bottom"

    def _traction_at(self, position):
        region_route = self._route_at(position)
        if region_route is None:
            return 1.
        return float(self.world_spec.value(*region_route, "traction"))

    @staticmethod
    def probe_position(region, route):
        # Both pre-fork viewpoints are physically reachable from the staging zone.
        # Region 2 is a remote geometry/material reading, not a contact probe.
        if route == "top":
            return np.asarray((-1.0, .20 if region == 1 else -.20),
                              dtype=np.float32)
        return np.asarray((-1.0, -.20), dtype=np.float32)

    def _sense(self, name, action, incremental):
        if action.sense_route is None:
            return
        region = getattr(action, "sense_region", 1)
        incremental.sensing += self.config.sensing_cost
        position = self.agent_positions[name]
        radius = (self.config.sensing_range_scout if name == "scout"
                  else self.config.sensing_range_carrier)
        distance = float(np.linalg.norm(position - self.probe_position(
            region, action.sense_route)))
        valid = distance <= radius
        event = {"type": "acquisition", "step": self.step_index,
                 "agent": name, "region": region, "route": action.sense_route,
                 "modality": action.sense_modality, "valid": valid,
                 "distance": distance, "cost": self.config.sensing_cost}
        if valid:
            observation_id = f"obs-{self._observation_counter:05d}"
            self._observation_counter += 1
            evidence = TwoForkEvidence(
                observation_id=observation_id, source=name,
                route=action.sense_route, modality=action.sense_modality,
                value=self.world_spec.value(region, action.sense_route,
                                            action.sense_modality),
                acquisition_step=self.step_index,
                acquisition_position=tuple(float(v) for v in position),
                region=region,
            )
            self.evidence[name][observation_id] = evidence
            event.update(observation_id=observation_id, value=evidence.value)
        else:
            incremental.invalid_action += self.config.invalid_action_cost
        self.events.append(event)

    def _deliver_due_packets(self, decision_open, incremental):
        # The inherited packet/ACK semantics are unchanged; a packet is usable
        # while either physical route decision remains open.
        open_any = (self.route_commitment is None
                    or self.route_commitments[2] is None)
        super()._deliver_due_packets(open_any, incremental)

    def step(self, actions):
        observation, reward, done, info = super().step(actions)
        self.route_commitments[1] = self.route_commitment
        if self.route_commitments[2] is None \
                and float(self.agent_positions["carrier"][0]) >= self.config.decision_x2:
            route = "top" if float(self.agent_positions["carrier"][1]) >= 0 else "bottom"
            self.route_commitments[2] = route
            self.events.append({"type": "route_commitment", "region": 2,
                                "step": self.step_index, "route": route})
        info["route_commitments"] = dict(self.route_commitments)
        return self.observations(), reward, done, info

    def _evidence_summary(self, name):
        summary = {region: {route: {modality: None for modality in ("geometry", "traction")}
                            for route in ("top", "bottom")}
                   for region in (1, 2)}
        for collection in (self.evidence[name], self.received[name]):
            for item in collection.values():
                summary[item.region][item.route][item.modality] = item.value
        return summary

    def observations(self):
        result = super().observations()
        for item in result.values():
            item["route_commitments"] = dict(self.route_commitments)
            item["decision_open_by_region"] = {
                region: self.route_commitments[region] is None for region in (1, 2)
            }
        return result

    @classmethod
    def replay(cls, payload):
        env = cls(config=TwoForkConfig(**payload["config"]),
                  world=TwoForkWorld(**payload["world"]), seed=payload["seed"])
        for joint in payload["actions"]:
            env.step({name: TwoForkAction(**{**joint[name],
                                             "motion": tuple(joint[name]["motion"])})
                      for name in ("scout", "carrier")})
        return env
