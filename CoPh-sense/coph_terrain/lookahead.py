"""Static risk look-ahead with local Gaussian beliefs and compact innovations.

This is a separate mechanism experiment. It reuses the E2 VMAS plant, pH
executor, sensing actions, packet timing, range, and ledger. It does not alter
E2-Natural or assert that region-constant terrain models natural maps.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from vmas import make_env

from .environment import (CoPHTerrainEnv, TerrainAction, TerrainConfig,
                          TerrainEvidence, cell_to_position)
from .generator import GRID_SIZE, TerrainMap


AGENTS = ("scout", "carrier")
REGIONS = ((18, 28, 24, 29), (18, 28, 32, 37))
SUMMARY_PAYLOAD_BYTES = 48
SUMMARY_ACCOUNTING_CELLS = 8  # existing radio: 16 + 4 * 8 = 48 bytes
WIRE_FORMAT = "<BBfffffH12sB11x"
assert struct.calcsize(WIRE_FORMAT) == SUMMARY_PAYLOAD_BYTES


@dataclass(frozen=True)
class LookaheadSpec:
    parent_seed: int
    surface: tuple = (.82, .06)
    traction: tuple = (.65, .90)
    scout_start: tuple = (12, 27)
    carrier_start: tuple = (5, 28)


@dataclass
class GaussianRegionBelief:
    precision: float
    weighted_mean: float

    @property
    def mean(self):
        return self.weighted_mean / self.precision

    @property
    def variance(self):
        return 1. / self.precision

    def add(self, precision, weighted_mean):
        self.precision += float(precision)
        self.weighted_mean += float(weighted_mean)


@dataclass(frozen=True)
class InnovationPacket:
    evidence_id: str
    source: str
    region: int
    modality: str
    delta_precision: float
    delta_weighted_mean: float
    prior_mean: float
    posterior_mean: float
    posterior_variance: float
    acquired_step: int
    provenance: str
    support_cells: tuple
    payload_bytes: int = SUMMARY_PAYLOAD_BYTES

    def to_wire_bytes(self):
        """Fixed 48-byte binary payload; packet ID is in the radio header."""
        provenance = self.provenance.encode("ascii")
        if len(provenance) > 12:
            raise ValueError("provenance is too long for v1 wire format")
        return struct.pack(
            WIRE_FORMAT, self.region,
            int(self.modality == "traction"),
            self.delta_precision, self.delta_weighted_mean,
            self.prior_mean, self.posterior_mean,
            self.posterior_variance, self.acquired_step,
            provenance, AGENTS.index(self.source))

    def wire_factor(self):
        fields = struct.unpack(WIRE_FORMAT, self.to_wire_bytes())
        return float(fields[2]), float(fields[3])


def gaussian_kl(after, before):
    """KL(N_after || N_before), a recipient-conditioned novelty proxy."""
    return max(0., .5 * (math.log(before.variance / after.variance)
                          + (after.variance + (after.mean - before.mean) ** 2)
                          / before.variance - 1.))


def region_index(cell):
    for index, (x0, x1, y0, y1) in enumerate(REGIONS):
        if x0 <= cell[0] < x1 and y0 <= cell[1] < y1:
            return index
    return None


def region_cells(index):
    x0, x1, y0, y1 = REGIONS[index]
    return {(x, y) for x in range(x0, x1)
            for y in range(y0, y1)}


class LookaheadEnv(CoPHTerrainEnv):
    """Two moving pH agents with separate beliefs and compressed radio packets."""

    def __init__(self, spec, seed=None, device="cpu", config=None):
        self.lookahead_spec = spec
        self.family = "lookahead_static"
        self.parent_seed = int(spec.parent_seed)
        self.realization_seed = 0
        self.seed = spec.parent_seed + 9 if seed is None else seed
        self.config = config or TerrainConfig()
        self.executor = "ph"
        self.device = str(device)
        occupancy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        occupancy[16:42, 30:31] = True
        boxes = ((16, 42, 30, 31),)
        surface = np.full((GRID_SIZE, GRID_SIZE), .04, np.float32)
        traction = np.full((GRID_SIZE, GRID_SIZE), .90, np.float32)
        for region, value_surface, value_traction in zip(
                REGIONS, spec.surface, spec.traction):
            x0, x1, y0, y1 = region
            surface[x0:x1, y0:y1] = value_surface
            traction[x0:x1, y0:y1] = value_traction
        self._truth = TerrainMap(self.family, spec.parent_seed, occupancy,
                                 traction, boxes, (), surface)
        self.parent_map = self._truth
        self.physics = make_env(
            scenario_name=str(Path(__file__).with_name("scenario.py")),
            num_envs=1, device=self.device, continuous_actions=True,
            max_steps=self.config.horizon_steps, seed=self.seed,
            obstacle_boxes=boxes, dt=self.config.dt)
        self.reset()

    def reset(self):
        self.region_beliefs = {
            name: {(index, modality): GaussianRegionBelief(4., 4. * prior)
                   for index in range(len(REGIONS))
                   for modality, prior in (("geometry", .35),
                                           ("traction", .70))}
            for name in AGENTS}
        self.seen_provenance = {name: set() for name in AGENTS}
        self.local_support = {name: set() for name in AGENTS}
        self.passive_support = {name: set() for name in AGENTS}
        self.received_support = {name: set() for name in AGENTS}
        self.innovation_packets = {}
        self.innovation_drafts = {}
        observation = super().reset()
        for index, cell in enumerate((self.lookahead_spec.scout_start,
                                      self.lookahead_spec.carrier_start)):
            actor = self.physics.world.agents[index]
            actor.state.pos[0] = actor.state.pos.new_tensor(cell_to_position(cell))
            actor.state.vel[0].zero_()
        for name in AGENTS:
            self.appearance_belief[name].fill(np.nan)
            self.last_passive_cell[name] = None
            self.passive_support[name].clear()
        self.events.clear()
        self._default_observe()
        unknown = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for x0, x1, y0, y1 in REGIONS:
            unknown[x0:x1, y0:y1] = True
        for name in AGENTS:
            # In this static diagnostic only the marked regions are latent.
            # Broadcasting ordinary background cells cannot masquerade as a
            # benefit of the raw-packet baseline.
            self.known_surface[name][~unknown] = .04
            self.known_traction[name][~unknown] = .90
        self.state_trace = [self._state_record()]
        return self.observations()

    def _default_observe(self):
        start = len(self.events)
        super()._default_observe()
        for event in self.events[start:]:
            if event["type"] == "passive_observation":
                self.passive_support[event["agent"]].update(event["cells"])

    def observations(self):
        result = super().observations()
        positions = self.positions
        progress = {name: float(positions[name][0]) for name in AGENTS}
        for name in AGENTS:
            other = "carrier" if name == "scout" else "scout"
            local = self.local_support[name] | self.passive_support[name]
            received = self.received_support[name]
            ahead = lambda cells: max(
                [0.] + [float(cell_to_position(cell)[0] - positions[name][0])
                         for cell in cells])
            result[name]["lookahead"] = {
                "tau": progress[name],
                "delta_tau": progress[name] - progress[other],
                "local_support_count": len(local),
                "communicated_support_count": len(received - local),
                "local_lookahead": ahead(local),
                "effective_lookahead": ahead(local | received),
                "local_region_beliefs": {
                    f"{index}:{modality}": (
                        self.region_beliefs[name][(index, modality)].mean,
                        self.region_beliefs[name][(index, modality)].variance)
                    for index in range(len(REGIONS))
                    for modality in ("geometry", "traction")},
                "received_innovations": tuple(sorted(
                    evidence_id for evidence_id in self.received[name]
                    if evidence_id in self.innovation_packets)),
            }
            result[name]["public_material_regions"] = REGIONS
            result[name]["evidence"] = tuple(
                item for item in result[name]["evidence"]
                if item["evidence_id"] not in self.innovation_packets)
        return result

    def _write_region_belief(self, name, region, modality):
        x0, x1, y0, y1 = REGIONS[region]
        grid = (self.known_surface[name] if modality == "geometry"
                else self.known_traction[name])
        grid[x0:x1, y0:y1] = self.region_beliefs[name][
            (region, modality)].mean

    def _apply_evidence(self, name, evidence):
        packet = self.innovation_packets.get(evidence.evidence_id)
        if packet is not None:
            if packet.provenance in self.seen_provenance[name]:
                return
            self.seen_provenance[name].add(packet.provenance)
            belief = self.region_beliefs[name][(packet.region, packet.modality)]
            belief.add(*packet.wire_factor())
            self._write_region_belief(name, packet.region, packet.modality)
            # Region membership is public. Under the declared constant-region
            # prior, a sufficient statistic updates the entire region without
            # transmitting a cell list.
            self.received_support[name].update(region_cells(packet.region))
            return
        super()._apply_evidence(name, evidence)
        if evidence.evidence_id in self.seen_provenance[name]:
            return
        self.seen_provenance[name].add(evidence.evidence_id)
        grouped = {}
        for cell, value in zip(evidence.cells, evidence.values):
            index = region_index(cell)
            if index is not None:
                grouped.setdefault(index, []).append((cell, float(value)))
        drafts = []
        for index, items in grouped.items():
            values = np.asarray([value for _, value in items], dtype=float)
            # One correlated footprint is one region-level observation; its
            # precision is capped so a broad scan is not counted as many
            # independent measurements.
            variance = .05 ** 2 + float(values.var())
            delta_precision = 1. / variance
            delta_weighted_mean = float(values.mean()) / variance
            belief = self.region_beliefs[name][(index, evidence.modality)]
            prior_mean = belief.mean
            belief.add(delta_precision, delta_weighted_mean)
            self._write_region_belief(name, index, evidence.modality)
            support = tuple(cell for cell, _ in items)
            if evidence.evidence_id in self.owned[name]:
                self.local_support[name].update(support)
            else:
                self.received_support[name].update(support)
            drafts.append((index, evidence.modality, delta_precision,
                           delta_weighted_mean, prior_mean, belief.mean,
                           belief.variance, support))
        self.innovation_drafts[evidence.evidence_id] = drafts

    def _complete_sense(self, name, request, incremental):
        before = set(self.owned[name])
        super()._complete_sense(name, request, incremental)
        new_raw_ids = set(self.owned[name]) - before
        for raw_id in new_raw_ids:
            raw = self.owned[name][raw_id]
            for (region, modality, dp, dw, before_mean, after_mean,
                 after_var, support) in self.innovation_drafts.pop(raw_id, []):
                packet_id = f"innovation-{self.evidence_counter:06d}"
                self.evidence_counter += 1
                message = InnovationPacket(
                    packet_id, name, region, modality, dp, dw, before_mean,
                    after_mean, after_var, self.step_index, raw_id, support)
                if len(message.to_wire_bytes()) != SUMMARY_PAYLOAD_BYTES:
                    raise RuntimeError("innovation wire size changed")
                self.innovation_packets[packet_id] = message
                # The placeholder occupies the declared fixed payload size in
                # the unchanged E2 radio ledger. Its zero values are never
                # interpreted as terrain measurements.
                anchor = support[0]
                self.owned[name][packet_id] = TerrainEvidence(
                    packet_id, name, modality,
                    (anchor,) * SUMMARY_ACCOUNTING_CELLS,
                    (0.,) * SUMMARY_ACCOUNTING_CELLS,
                    self.step_index, raw.location)
                self.events.append({"type": "innovation_created",
                                    "step": self.step_index, "sender": name,
                                    "message_id": packet_id,
                                    "region": region, "modality": modality,
                                    "provenance": raw_id,
                                    "payload_bytes": SUMMARY_PAYLOAD_BYTES})

    def conditional_information_gain(self, recipient, packet_id):
        packet = self.innovation_packets[packet_id]
        if packet.provenance in self.seen_provenance[recipient]:
            return 0.
        before = self.region_beliefs[recipient][
            (packet.region, packet.modality)]
        after = GaussianRegionBelief(before.precision,
                                     before.weighted_mean)
        after.add(*packet.wire_factor())
        return gaussian_kl(after, before)

    def message_payload(self, packet_id):
        packet = self.innovation_packets[packet_id]
        result = asdict(packet)
        result.pop("support_cells")  # evaluator-only acquisition footprint
        return result

    def forget_delivered_evidence(self, name, evidence, prior):
        """Remove one message's belief update at an otherwise matched state."""
        packet = self.innovation_packets[evidence.evidence_id]
        key = (packet.region, packet.modality)
        old = prior.region_beliefs[name][key]
        self.region_beliefs[name][key] = GaussianRegionBelief(
            old.precision, old.weighted_mean)
        x0, x1, y0, y1 = REGIONS[packet.region]
        grid = (self.known_surface[name] if packet.modality == "geometry"
                else self.known_traction[name])
        old_grid = (prior.known_surface[name] if packet.modality == "geometry"
                    else prior.known_traction[name])
        grid[x0:x1, y0:y1] = old_grid[x0:x1, y0:y1]
        self.seen_provenance[name].discard(packet.provenance)
        self.received_support[name].clear()
        for evidence_id, retained in self.received[name].items():
            other = self.innovation_packets.get(evidence_id)
            self.received_support[name].update(
                region_cells(other.region) if other is not None
                else retained.cells)

    def support_metrics(self):
        local_a = self.local_support["scout"] | self.passive_support["scout"]
        local_b = self.local_support["carrier"] | self.passive_support["carrier"]
        return {"scout_local_cells": len(local_a),
                "carrier_local_cells": len(local_b),
                "overlap_cells": len(local_a & local_b),
                "disjoint_cells": len(local_a ^ local_b),
                "carrier_new_received_cells": len(
                    self.received_support["carrier"] - local_b)}

    def replay_payload(self):
        payload = super().replay_payload()
        payload["lookahead_spec"] = asdict(self.lookahead_spec)
        payload.pop("sha256")
        payload["sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True).encode()).hexdigest()
        return payload

    @classmethod
    def replay(cls, payload):
        spec = dict(payload["lookahead_spec"])
        for key in ("surface", "traction", "scout_start", "carrier_start"):
            spec[key] = tuple(spec[key])
        env = cls(LookaheadSpec(**spec), seed=payload["seed"],
                  config=TerrainConfig(**payload["config"]),
                  device=payload.get("device", "cpu"))
        for row in payload["actions"]:
            env.step({name: TerrainAction(**row[name]) for name in AGENTS})
        if (env._truth.digest() != payload["map_sha256"] or
                asdict(env.ledger) != payload["ledger"] or
                env.agent_risk != payload["agent_material_exposure"]):
            raise RuntimeError("lookahead deterministic replay mismatch")
        return env
