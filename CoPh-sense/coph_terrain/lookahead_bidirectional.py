"""Predeclared bidirectional need/innovation protocol for identifiable A4.

The receiver can announce a compact description of an information need.  The
packet exposes no hidden terrain, future randomness, or evaluator value.  This
module fixes the wire representation before the admission dataset is made.
"""

from dataclasses import dataclass
import struct

from .environment import TerrainEvidence
from .lookahead import LookaheadEnv


NEED_PAYLOAD_BYTES = 32
NEED_WIRE_FORMAT = "<BBffffH12x"
assert struct.calcsize(NEED_WIRE_FORMAT) == NEED_PAYLOAD_BYTES


@dataclass(frozen=True)
class NeedPacket:
    """Receiver-authored, legally observable information-need summary."""

    region: int
    modality: str
    uncertainty: float
    task_relevance: float
    latest_useful_time: float
    receiver_progress: float
    request_id: int

    def to_wire_bytes(self):
        if self.modality not in ("geometry", "traction"):
            raise ValueError("unknown modality")
        if self.region < 0 or self.region > 255:
            raise ValueError("region must fit in one byte")
        if self.request_id < 0 or self.request_id > 65535:
            raise ValueError("request_id must fit in uint16")
        return struct.pack(
            NEED_WIRE_FORMAT, self.region,
            int(self.modality == "traction"), self.uncertainty,
            self.task_relevance, self.latest_useful_time,
            self.receiver_progress, self.request_id)

    @classmethod
    def from_wire_bytes(cls, payload):
        if len(payload) != NEED_PAYLOAD_BYTES:
            raise ValueError("invalid need-packet length")
        region, traction, uncertainty, relevance, deadline, progress, request = (
            struct.unpack(NEED_WIRE_FORMAT, payload))
        return cls(region, "traction" if traction else "geometry",
                   uncertainty, relevance, deadline, progress, request)


def compatible_need(innovation, need, now):
    """Causal compatibility gate; task value is still evaluated downstream."""
    return bool(innovation.region == need.region and
                innovation.modality == need.modality and
                need.uncertainty > 0. and need.task_relevance > 0. and
                now < need.latest_useful_time)


class BidirectionalLookaheadEnv(LookaheadEnv):
    """Lookahead task extension whose NEED payloads do not alter terrain belief."""

    def reset(self):
        self.need_packets = {}
        return super().reset()

    def _apply_evidence(self, name, evidence):
        if evidence.evidence_id in self.need_packets:
            return
        super()._apply_evidence(name, evidence)

    def create_need(self, receiver, need):
        """Create a locally owned 32-byte request for normal charged delivery."""
        evidence_id = f"need-{need.request_id:05d}-{self.evidence_counter:06d}"
        self.evidence_counter += 1
        self.need_packets[evidence_id] = need
        # The unchanged radio ledger charges 16 + 4 bytes per placeholder cell.
        anchor = (0, 0)
        self.owned[receiver][evidence_id] = TerrainEvidence(
            evidence_id, receiver, "geometry", (anchor,) * 4, (0.,) * 4,
            self.step_index, tuple(float(v) for v in self.positions[receiver]))
        self.events.append({"type": "need_created", "step": self.step_index,
                            "sender": receiver, "message_id": evidence_id,
                            "payload_bytes": NEED_PAYLOAD_BYTES,
                            "region": need.region,
                            "modality": need.modality,
                            "request_id": need.request_id})
        return evidence_id
