"""Exact finite A6 timing/range diagnostic with frozen A4 and A5-v2.

The scout may inspect Region 1 or skip. If it inspects, A4 chooses messages
after seeing the readings. A message affects the carrier only when its arrival
time is strictly before the common route-commitment deadline. This finite task
uses one commitment time for both route choices, not continuous VMAS motion.
"""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .memory_a5 import (CANDIDATES, EvidenceLedger, EvidenceRecord,
                        actual_route_cost, route_for, share_set)
from .memory_critic_a5 import (WORLDS, action_costs, delivery_posteriors,
                               delivery_signature, model_action, posterior_for)
from .oracle import ExactOracleConfig


@dataclass(frozen=True)
class TimingContext:
    scout_position: float = 1.0
    carrier_position: float = 0.0
    communication_range: float = 1.0
    network_delay: float = 0.1
    delivery_probability: float = 0.9
    decision_slack: float = 0.4
    acquisition_duration: float = 0.05
    return_speed: float = 1.0
    return_effort_per_distance: float = 0.12
    waiting_cost_per_second: float = 0.03

    def __post_init__(self):
        if self.communication_range < 0 or self.network_delay < 0 or self.decision_slack < 0:
            raise ValueError("range, delay, and slack must be nonnegative")
        if not 0 <= self.delivery_probability <= 1:
            raise ValueError("delivery probability must be in [0,1]")
        if self.return_speed <= 0:
            raise ValueError("return speed must be positive")

    @property
    def return_distance(self):
        return max(0.0, abs(self.scout_position - self.carrier_position)
                   - self.communication_range)

    @property
    def return_duration(self):
        return self.return_distance / self.return_speed

    @property
    def delivery_time(self):
        return self.acquisition_duration + self.return_duration + self.network_delay

    @property
    def timely(self):
        # A packet arriving exactly at commitment cannot change that decision.
        return self.delivery_time < self.decision_slack - 1e-12

    @property
    def temporal_margin(self):
        return self.decision_slack - self.delivery_time


def carrier_result(world, memory, posterior, a5_model, config, exact_carrier=False):
    if exact_carrier:
        action = CANDIDATES[int(action_costs(posterior, memory, config).argmin())]
    else:
        action = model_action(a5_model, memory, config, posterior=posterior)
    acquired = memory.copy()
    for region, modality in action:
        j = 2 * (region - 1) + (modality == "traction")
        acquired.add(EvidenceRecord(
            evidence_id="carrier-r{}-{}".format(region, modality),
            region=region, modality=modality, value=bool(world[j]),
            uncertainty=0.0, acquisition_time=2, delivery_time=2,
            source="carrier",
        ))
    routes = {region: route_for(acquired, region) for region in (1, 2)}
    route_cost = sum(actual_route_cost(routes[region], world, region, config)
                     for region in (1, 2))
    return action, len(action) * config.sensing_cost, route_cost


class ExactTimingEvaluator:
    def __init__(self, a4_model, a5_model, config=ExactOracleConfig()):
        self.a4_model = a4_model
        self.a5_model = a5_model
        self.config = config
        self._send_cache = {}
        self._posterior_cache = {}

    def _sharing(self, timely_probability):
        if timely_probability not in self._send_cache:
            config = replace(self.config,
                             timely_delivery_probability=timely_probability)
            self._send_cache[timely_probability] = [
                share_set(self.a4_model, world, config) for world in WORLDS
            ]
            self._posterior_cache[timely_probability] = delivery_posteriors(
                self.a4_model, config
            )[0]
        return self._send_cache[timely_probability]

    def evaluate(self, context, scout_inspects, exact_carrier=False,
                 return_branches=False):
        if not scout_inspects:
            send_sets = [()] * 16
            timely_probability = 0.0
        else:
            timely_probability = context.delivery_probability if context.timely else 0.0
            send_sets = self._sharing(timely_probability)
        ack_cost = self.config.communication_attempt_cost + self.config.byte_cost * (
            self.config.header_bytes + self.config.ack_payload_bytes
        )
        rows = []
        for wi, world in enumerate(WORLDS):
            sent = send_sets[wi]
            for mask in product((False, True), repeat=len(sent)):
                probability = 1 / 16
                for arrived in mask:
                    probability *= (context.delivery_probability if arrived else
                                    1 - context.delivery_probability)
                if probability == 0:
                    continue
                delivered = EvidenceLedger()
                if context.timely:
                    for name, arrived in zip(sent, mask):
                        if arrived:
                            modality = "geometry" if name.endswith("geometry") else "traction"
                            delivered.add(EvidenceRecord(
                                evidence_id="r1-" + modality, region=1,
                                modality=modality,
                                value=bool(world[0 if modality == "geometry" else 1]),
                                uncertainty=0.0, acquisition_time=0,
                                delivery_time=1, source="scout",
                            ))
                if scout_inspects and context.timely:
                    posterior = self._posterior_cache[timely_probability][
                        delivery_signature(delivered)
                    ]
                else:
                    posterior = posterior_for(delivered)
                action, carrier_sensing, route_cost = carrier_result(
                    world, delivered, posterior, self.a5_model, self.config,
                    exact_carrier=exact_carrier,
                )
                return_effort = (context.return_distance *
                                 context.return_effort_per_distance if sent else 0.0)
                # The scout acts only after it knows A4's send choice; it need
                # not return to range when A4 sends nothing.
                waiting = (context.waiting_cost_per_second *
                           (min(context.delivery_time, context.decision_slack)
                            if sent else context.acquisition_duration)
                           if scout_inspects else 0.0)
                scout_sensing = 2 * self.config.sensing_cost if scout_inspects else 0.0
                communication = len(sent) * self.config.packet_cost + sum(mask) * ack_cost
                total = (scout_sensing + return_effort + waiting + communication
                         + carrier_sensing + route_cost)
                rows.append({
                    "probability": probability, "world": list(world),
                    "sent": list(sent), "delivered_before_commit": [
                        record.evidence_id for record in delivered.records
                    ], "delivered_after_commit": [
                        name for name, arrived in zip(sent, mask) if arrived
                    ] if not context.timely else [],
                    "carrier_action": [list(atom) for atom in action],
                    "scout_sensing_cost": scout_sensing,
                    "return_effort_cost": return_effort,
                    "waiting_cost": waiting,
                    "communication_cost": communication,
                    "carrier_sensing_cost": carrier_sensing,
                    "route_cost": route_cost,
                    "total_cost": total,
                    "deadline_missed_inspection": bool(scout_inspects and not context.timely),
                    "sent_packets": len(sent),
                    "late_delivered_packets": sum(mask) if not context.timely else 0,
                })
        if not np.isclose(sum(row["probability"] for row in rows), 1):
            raise RuntimeError("A6 branch probabilities do not sum to one")
        expected = lambda key: float(sum(row["probability"] * row[key] for row in rows))
        report = {
            "expected_team_cost": expected("total_cost"),
            "expected_scout_sensing_cost": expected("scout_sensing_cost"),
            "expected_return_effort_cost": expected("return_effort_cost"),
            "expected_waiting_cost": expected("waiting_cost"),
            "expected_communication_cost": expected("communication_cost"),
            "expected_carrier_sensing_cost": expected("carrier_sensing_cost"),
            "expected_route_cost": expected("route_cost"),
            "deadline_missed_inspection_rate": expected("deadline_missed_inspection"),
            "expected_sent_packets": expected("sent_packets"),
            "expected_late_delivered_packets": expected("late_delivered_packets"),
        }
        if return_branches:
            report["branches"] = rows
        return report

    def gate_oracle(self, context, exact_carrier=False):
        skip = self.evaluate(context, False, exact_carrier)
        inspect = self.evaluate(context, True, exact_carrier)
        return {
            "skip": skip, "inspect": inspect,
            "oracle_choice": "inspect" if inspect["expected_team_cost"] <
                             skip["expected_team_cost"] - 1e-12 else "skip",
            "oracle_cost": min(skip["expected_team_cost"], inspect["expected_team_cost"]),
            "frozen_always_inspect_regret": max(0.0, inspect["expected_team_cost"] -
                                               skip["expected_team_cost"]),
        }
