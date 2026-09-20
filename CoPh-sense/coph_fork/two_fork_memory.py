"""Causal adapter from delivered VMAS evidence to frozen finite A5-v2."""

from pathlib import Path

import numpy as np
import torch

from .memory_a5 import EvidenceLedger, EvidenceRecord
from .memory_critic_a5 import MemorySetCritic, model_action, posterior_for
from .oracle import ExactOracleConfig


HERE = Path(__file__).resolve().parent


def _delivery_step(env, observation_id):
    arrivals = [event["step"] for event in env.events
                if event.get("type") == "packet_delivery"
                and event.get("observation_id") == observation_id
                and event.get("delivered")]
    return min(arrivals) if arrivals else None


def delivered_ledger(env):
    """Use only carrier-own and actually received evidence; never world_spec."""
    records = []
    for source, collection in (("carrier", env.evidence["carrier"]),
                               ("scout", env.received["carrier"])):
        for observation_id, item in collection.items():
            records.append(EvidenceRecord(
                evidence_id=observation_id, region=item.region,
                modality=item.modality,
                value=(bool(item.value) if item.modality == "geometry"
                       else bool(float(item.value) >= env.config.traction_safe_threshold)),
                uncertainty=0., acquisition_time=item.acquisition_step,
                delivery_time=(item.acquisition_step if source == "carrier"
                               else _delivery_step(env, observation_id)),
                source=source,
            ))
    return EvidenceLedger(records)


def actor_ledger(delivered, condition):
    if condition in ("independent", "shared_reset"):
        return EvidenceLedger()
    if condition == "shuffled":
        return delivered.shuffled_region()
    if condition in ("persistent", "oracle_memory"):
        return delivered.copy()
    raise ValueError("unknown memory condition")


class CarrierA5Adapter:
    def __init__(self, seed=1701, kind="pairwise", config=None):
        self.config = config or ExactOracleConfig()
        self.model = MemorySetCritic(kind)
        self.model.load_state_dict(torch.load(
            HERE / "results" / "a5_memory_v2" / f"{kind}_{seed}.pth",
            map_location="cpu", weights_only=True))
        self.model.eval()

    def choose(self, memory, posterior=None):
        if posterior is None:
            posterior = posterior_for(memory)
        return model_action(self.model, memory, self.config,
                            posterior=np.asarray(posterior, dtype=np.float64))
