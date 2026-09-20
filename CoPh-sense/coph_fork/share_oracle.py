"""Exact post-observation message-set oracle for A4.

The sender's realized readings are private until a packet arrives. Silence is
not an observation: NO_SEND and DROP both produce the same receiver history.
"""

from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Mapping, Tuple

from .oracle import ExactAcquisitionOracle, ExactOracleConfig, ROUTES, VARIABLES, default_prior


@dataclass(frozen=True)
class ShareState:
    # Immutable (variable, realized binary value) pairs. All items in acquired
    # were actually measured by the sender; acknowledged items are receiver-held.
    acquired: Tuple[Tuple[str, bool], ...]
    acknowledged: Tuple[Tuple[str, bool], ...] = ()
    timely_probability: float = 0.9
    deadline_open: bool = True

    def __post_init__(self):
        for collection in (self.acquired, self.acknowledged):
            names = [name for name, _ in collection]
            if len(names) != len(set(names)) or any(name not in VARIABLES for name in names):
                raise ValueError("evidence identifiers must be unique known variables")
        if not 0 <= self.timely_probability <= 1:
            raise ValueError("timely probability must lie in [0,1]")


@dataclass(frozen=True)
class ShareValue:
    message_set: Tuple[str, ...]
    expected_communication_cost: float
    expected_execution_cost: float
    total_cost: float
    expected_bytes: float
    receiver_decisions: Mapping[str, str]
    delivery_probabilities: Mapping[str, float]


class ExactShareOracle:
    def __init__(self, state: ShareState, config=ExactOracleConfig(), prior=None):
        self.state = state
        self.config = config
        self.prior = dict(default_prior() if prior is None else prior)
        self.route_oracle = ExactAcquisitionOracle(config=config, prior=self.prior)
        acquired = dict(state.acquired)
        acknowledged = dict(state.acknowledged)
        if any(name in acknowledged and acknowledged[name] != value for name, value in acquired.items()):
            raise ValueError("acknowledged and acquired evidence conflict")
        self.acquired = acquired
        self.acknowledged = acknowledged
        self._require_positive_mass(acquired)
        self._require_positive_mass(acknowledged)

    def _consistent_mass(self, evidence):
        return {
            world: probability
            for world, probability in self.prior.items()
            if all(self.route_oracle.world_dict(world)[name] == value for name, value in evidence.items())
        }

    def _require_positive_mass(self, evidence):
        if sum(self._consistent_mass(evidence).values()) <= 0:
            raise ValueError("evidence has zero probability under the declared prior")

    def candidate_sets(self):
        if not self.state.deadline_open:
            yield ()
            return
        names = tuple(name for name, _ in self.state.acquired)
        for size in range(len(names) + 1):
            for subset in combinations(names, size):
                yield subset

    def _receiver_decision(self, delivered):
        knowledge = dict(self.acknowledged)
        knowledge.update((name, self.acquired[name]) for name in delivered)
        worlds = self._consistent_mass(knowledge)
        candidates = list(ROUTES)
        if self.config.decision_mode == "certified":
            candidates = [
                route for route in ROUTES
                if all(
                    self.route_oracle.execution_cost(route, world)
                    < self.config.unsafe_route_cost
                    for world in worlds
                )
            ]
            if not candidates:
                raise RuntimeError("no certified route or backup")
        costs = {
            route: sum(
                probability * self.route_oracle.execution_cost(route, world)
                for world, probability in worlds.items()
            ) / sum(worlds.values())
            for route in candidates
        }
        return min(candidates, key=lambda route: (costs[route], ROUTES.index(route)))

    def evaluate(self, message_set):
        message_set = tuple(message_set)
        if len(set(message_set)) != len(message_set) or any(name not in self.acquired for name in message_set):
            raise ValueError("message set must contain distinct acquired evidence")
        if not self.state.deadline_open and message_set:
            # Packets may still be transmitted by an external system, but they
            # cannot affect this irreversible decision. This A4 menu excludes them.
            raise ValueError("no messages are feasible after the decision deadline")
        timely = self.state.timely_probability if self.state.deadline_open else 0.0
        sender_knowledge = dict(self.acknowledged)
        sender_knowledge.update(self.acquired)
        sender_worlds = self._consistent_mass(sender_knowledge)
        normalizer = sum(sender_worlds.values())
        expected_execution = 0.0
        decisions = {}
        probabilities = {}
        for mask in product((False, True), repeat=len(message_set)):
            delivered = tuple(name for name, success in zip(message_set, mask) if success)
            probability = 1.0
            for success in mask:
                probability *= timely if success else 1 - timely
            if probability == 0:
                continue
            decision = self._receiver_decision(delivered)
            key = "none" if not delivered else ",".join(delivered)
            decisions[key] = decision
            probabilities[key] = probabilities.get(key, 0.0) + probability
            expected_execution += probability * sum(
                world_probability * self.route_oracle.execution_cost(decision, world)
                for world, world_probability in sender_worlds.items()
            ) / normalizer
        # The A1 protocol attempts a charged ACK for each delivered evidence
        # packet. This finite abstraction assumes the return link remains in
        # range, so ACK attempts occur with the timely-delivery probability.
        ack_bytes = self.config.header_bytes + self.config.ack_payload_bytes
        communication = len(message_set) * (
            self.config.packet_cost
            + timely * (self.config.communication_attempt_cost
                        + self.config.byte_cost * ack_bytes)
        )
        bytes_sent = len(message_set) * (
            self.config.header_bytes + self.config.payload_bytes + timely * ack_bytes
        )
        return ShareValue(
            message_set=message_set,
            expected_communication_cost=communication,
            expected_execution_cost=expected_execution,
            total_cost=communication + expected_execution,
            expected_bytes=float(bytes_sent),
            receiver_decisions=decisions,
            delivery_probabilities=probabilities,
        )

    def solve(self):
        values = [self.evaluate(subset) for subset in self.candidate_sets()]
        optimum = min(values, key=lambda row: (row.total_cost, len(row.message_set), row.message_set))
        return optimum, values

    def report(self):
        optimum, values = self.solve()
        return {
            "state": asdict(self.state),
            "config": asdict(self.config),
            "semantics": {
                "silence": "no inference; NO_SEND and dropped packet both yield no delivered evidence",
                "acknowledgement": "exact receiver-held evidence in this finite diagnostic",
                "delivery": "independent Bernoulli per attempted packet",
                "ack_cost": "charged ACK attempt on each timely delivered packet; return link assumed in range",
            },
            "optimum": asdict(optimum),
            "q_share": [asdict(value) for value in values],
        }
