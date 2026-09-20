"""Two-region finite cooperative-memory diagnostic using frozen A3/A4 critics.

This extends the normalized CoPH-Fork information task, not VMAS geometry.
Scout sensing and evidence delivery precede the carrier's later acquisition
choice. A scripted terminal route executor is fixed across all conditions.
"""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .critic import SUBSETS
from .nested_oracle import learned_message_set
from .oracle import ExactOracleConfig, default_prior
from .share_oracle import ExactShareOracle, ShareState
from .train_critic import public_features


REGIONS = (1, 2)
MODALITIES = ("geometry", "traction")
CANDIDATES = (
    (),
    ((1, "geometry"),), ((1, "traction"),),
    ((1, "geometry"), (1, "traction")),
    ((2, "geometry"),), ((2, "traction"),),
    ((2, "geometry"), (2, "traction")),
)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    region: int
    modality: str
    value: bool
    uncertainty: float
    acquisition_time: int
    delivery_time: Optional[int]
    source: str


class EvidenceLedger:
    def __init__(self, records=()):
        self.records = list(records)

    def add(self, record):
        self.records.append(record)

    def value(self, region, modality):
        matches = [record for record in self.records
                   if record.region == region and record.modality == modality]
        return matches[-1].value if matches else None

    def copy(self):
        return EvidenceLedger(self.records)

    def shuffled_region(self):
        return EvidenceLedger(
            EvidenceRecord(
                evidence_id=record.evidence_id,
                region=2 if record.region == 1 else 1,
                modality=record.modality,
                value=record.value,
                uncertainty=record.uncertainty,
                acquisition_time=record.acquisition_time,
                delivery_time=record.delivery_time,
                source=record.source,
            )
            for record in self.records
        )


def world_value(world, region, modality):
    index = 2 * (region - 1) + (0 if modality == "geometry" else 1)
    return bool(world[index])


def local_probabilities(memory, region):
    return tuple(
        0.5 if memory.value(region, modality) is None
        else float(memory.value(region, modality))
        for modality in MODALITIES
    )


def route_for(memory, region):
    return "top" if all(memory.value(region, modality) is True for modality in MODALITIES) else "bottom"


def actual_route_cost(route, world, region, config):
    if route == "bottom":
        return config.bottom_safe_cost
    return config.top_safe_cost if all(world_value(world, region, m) for m in MODALITIES) else config.unsafe_route_cost


def expected_route_cost(memory, region, config):
    route = route_for(memory, region)
    if route == "bottom":
        return config.bottom_safe_cost
    return config.top_safe_cost


def exact_acquisition_value(memory, action, config):
    """Value of one batch under the receiver's current, possibly wrong belief."""
    baseline = sum(expected_route_cost(memory, region, config) for region in REGIONS)
    if not action:
        return 0.0
    region = action[0][0]
    p_g, p_t = local_probabilities(memory, region)
    continuation = 0.0
    for geometry, traction in product((False, True), repeat=2):
        probability = (p_g if geometry else 1 - p_g) * (p_t if traction else 1 - p_t)
        if probability == 0:
            continue
        updated = memory.copy()
        for _, modality in action:
            updated.add(EvidenceRecord(
                evidence_id="counterfactual", region=region, modality=modality,
                value=geometry if modality == "geometry" else traction,
                uncertainty=0.0, acquisition_time=2, delivery_time=2,
                source="carrier",
            ))
        continuation += probability * sum(
            expected_route_cost(updated, target, config) for target in REGIONS
        )
    q = len(action) * config.sensing_cost + continuation
    return baseline - q


def oracle_carrier_action(memory, config):
    return max(CANDIDATES, key=lambda action: (exact_acquisition_value(memory, action, config), -len(action), -CANDIDATES.index(action)))


def a3_region_scores(model, memory, region, config):
    p_g, p_t = local_probabilities(memory, region)
    prior = {
        (geometry, traction, True, True):
            (p_g if geometry else 1 - p_g) * (p_t if traction else 1 - p_t)
        for geometry, traction in product((False, True), repeat=2)
    }
    state, candidates = public_features(config, prior, p_g, p_t)
    with torch.no_grad():
        return model(
            torch.tensor(state)[None], torch.tensor(candidates)[None]
        )[0].sum(-1).numpy()


def frozen_a3_action(model, memory, config):
    scores = {region: a3_region_scores(model, memory, region, config) for region in REGIONS}
    def gain(action):
        if not action:
            return 0.0
        region = action[0][0]
        subset = tuple("top_" + modality for _, modality in action)
        return float(scores[region][0] - scores[region][SUBSETS.index(subset)])
    return max(CANDIDATES, key=lambda action: (gain(action), -len(action), -CANDIDATES.index(action)))


def share_set(a4_model, world, config):
    acquired = (
        ("top_geometry", world_value(world, 1, "geometry")),
        ("top_traction", world_value(world, 1, "traction")),
    )
    oracle = ExactShareOracle(
        ShareState(acquired=acquired, timely_probability=config.timely_delivery_probability),
        config=config, prior=default_prior(),
    )
    return learned_message_set(a4_model, oracle)


def episode_branches(world, condition, a3_model, a4_model, config, a5_model=None,
                     a5_posteriors=None):
    if condition not in (
        "independent", "shared_reset", "persistent", "oracle_memory", "shuffled",
        "learned_memory", "learned_memory_v2", "oracle_memory_v2"
    ):
        raise ValueError("unknown A5 condition")
    sent = () if condition == "independent" else share_set(a4_model, world, config)
    packet_cost = config.packet_cost
    ack_cost = config.communication_attempt_cost + config.byte_cost * (
        config.header_bytes + config.ack_payload_bytes
    )
    for delivered_mask in product((False, True), repeat=len(sent)):
        probability = 1.0
        for delivered in delivered_mask:
            probability *= (
                config.timely_delivery_probability if delivered
                else 1 - config.timely_delivery_probability
            )
        if probability == 0:
            continue
        delivered = EvidenceLedger()
        acked = []
        for name, arrived in zip(sent, delivered_mask):
            if not arrived:
                continue
            modality = "geometry" if name.endswith("geometry") else "traction"
            record = EvidenceRecord(
                evidence_id="r1-" + modality, region=1, modality=modality,
                value=world_value(world, 1, modality), uncertainty=0.0,
                acquisition_time=0, delivery_time=1, source="scout",
            )
            delivered.add(record)
            acked.append(record.evidence_id)
        if condition == "shared_reset":
            acquisition_memory = EvidenceLedger()
            execution_memory = delivered.copy()
        elif condition == "shuffled":
            acquisition_memory = delivered.shuffled_region()
            execution_memory = acquisition_memory.copy()
        else:
            acquisition_memory = delivered.copy()
            execution_memory = delivered.copy()
        if condition == "oracle_memory":
            action = oracle_carrier_action(acquisition_memory, config)
        elif condition in ("learned_memory_v2", "oracle_memory_v2"):
            if a5_posteriors is None:
                raise ValueError("A5-v2 requires A4-consistent delivery posteriors")
            from .memory_critic_a5 import action_costs, delivery_signature, model_action
            posterior = a5_posteriors[delivery_signature(delivered)]
            exact_costs = action_costs(posterior, acquisition_memory, config)
            if condition == "oracle_memory_v2":
                action = CANDIDATES[int(exact_costs.argmin())]
            else:
                if a5_model is None:
                    raise ValueError("learned_memory_v2 requires an A5-v2 critic")
                action = model_action(a5_model, acquisition_memory, config,
                                      posterior=posterior)
        elif condition == "learned_memory":
            if a5_model is None:
                raise ValueError("learned_memory requires an A5-v2 critic")
            from .memory_critic_a5 import model_action
            action = model_action(a5_model, acquisition_memory, config)
        else:
            action = frozen_a3_action(a3_model, acquisition_memory, config)
        policy_acquisition_value = exact_acquisition_value(acquisition_memory, action, config)
        # Scientific redundancy is conditioned on evidence actually delivered,
        # even when an ablation discards or corrupts the actor's memory.
        if a5_posteriors is not None:
            from .memory_critic_a5 import action_costs, delivery_signature, posterior_for
            valid_posterior = (
                posterior_for(delivered) if condition == "independent"
                else a5_posteriors[delivery_signature(delivered)]
            )
            exact_costs = action_costs(valid_posterior, delivered, config)
            acquisition_value = float(exact_costs[0] - exact_costs[CANDIDATES.index(action)])
            best_acquisition_value = float(exact_costs[0] - exact_costs.min())
            if condition in ("learned_memory_v2", "oracle_memory_v2"):
                policy_acquisition_value = acquisition_value
        else:
            acquisition_value = exact_acquisition_value(delivered, action, config)
            best_acquisition_value = max(
                exact_acquisition_value(delivered, option, config) for option in CANDIDATES
            )
        for region, modality in action:
            record = EvidenceRecord(
                evidence_id="carrier-r{}-{}".format(region, modality),
                region=region, modality=modality,
                value=world_value(world, region, modality), uncertainty=0.0,
                acquisition_time=2, delivery_time=2, source="carrier",
            )
            acquisition_memory.add(record)
            execution_memory.add(record)
        routes = {region: route_for(execution_memory, region) for region in REGIONS}
        route_costs = {
            region: actual_route_cost(routes[region], world, region, config)
            for region in REGIONS
        }
        cost = (
            2 * config.sensing_cost  # fixed scout inspection of Region 1
            + len(sent) * packet_cost
            + sum(delivered_mask) * ack_cost
            + len(action) * config.sensing_cost
            + sum(route_costs.values())
        )
        yield probability, {
            "world": list(world),
            "sent": list(sent),
            "delivered_ids": [record.evidence_id for record in delivered.records],
            "acknowledged_ids": acked,
            "carrier_memory_before_sensing": [asdict(record) for record in acquisition_memory.records
                                              if record.acquisition_time < 2],
            "valid_delivered_memory": [asdict(record) for record in delivered.records],
            "carrier_action": [list(item) for item in action],
            "exact_acquisition_value": acquisition_value,
            "best_acquisition_value": best_acquisition_value,
            "policy_belief_acquisition_value": policy_acquisition_value,
            "redundant_batch": bool(action and acquisition_value <= 1e-10),
            "duplicate_measurements": sum(
                any(record.region == region and record.modality == modality
                    for record in delivered.records)
                for region, modality in action
            ),
            "region2_useful": bool(action and action[0][0] == 2 and acquisition_value > 1e-10),
            "routes": routes,
            "unsafe_executions": sum(
                routes[region] == "top" and route_costs[region] == config.unsafe_route_cost
                for region in REGIONS
            ),
            "scout_sensing_cost": 2 * config.sensing_cost,
            "communication_cost": len(sent) * packet_cost + sum(delivered_mask) * ack_cost,
            "carrier_sensing_cost": len(action) * config.sensing_cost,
            "route_cost": sum(route_costs.values()),
            "total_team_cost": cost,
        }


def exact_evaluation(condition, a3_model, a4_model, config=ExactOracleConfig(),
                     a5_model=None, a5_posteriors=None):
    rows = []
    for world in product((False, True), repeat=4):
        for delivery_probability, event in episode_branches(
            world, condition, a3_model, a4_model, config, a5_model, a5_posteriors
        ):
            rows.append({"probability": delivery_probability / 16, **event})
    if abs(sum(row["probability"] for row in rows) - 1) > 1e-10:
        raise RuntimeError("world and packet branch probabilities do not sum to one")
    expected = lambda key: sum(row["probability"] * row[key] for row in rows)
    acquisitions = expected("carrier_sensing_cost") / config.sensing_cost
    return {
        "condition": condition,
        "expected_total_team_cost": expected("total_team_cost"),
        "expected_scout_sensing_cost": expected("scout_sensing_cost"),
        "expected_communication_cost": expected("communication_cost"),
        "expected_carrier_sensing_cost": expected("carrier_sensing_cost"),
        "expected_route_cost": expected("route_cost"),
        "expected_carrier_measurements": acquisitions,
        "redundant_batch_rate": expected("redundant_batch") /
            sum(row["probability"] * bool(row["carrier_action"]) for row in rows),
        "duplicate_measurement_rate": expected("duplicate_measurements") / acquisitions,
        "region2_useful_acquisition_rate": expected("region2_useful"),
        "unsafe_execution_rate": expected("unsafe_executions"),
        "expected_nested_acquisition_regret": sum(
            row["probability"] * (
                row["best_acquisition_value"] - row["exact_acquisition_value"]
            )
            for row in rows
        ),
        "branches": rows,
    }
