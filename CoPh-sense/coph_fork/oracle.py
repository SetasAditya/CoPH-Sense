"""Exact finite acquisition and sharing oracle for CoPH-Fork A2.

The oracle enumerates executable sensing subsets and decentralized sharing
rules. The receiver chooses a route using only the messages actually delivered.
All expectations are finite sums; no rollout sampling is used.
"""

from dataclasses import asdict, dataclass
from itertools import combinations, product
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


VARIABLES = (
    "top_geometry",
    "top_traction",
    "bottom_geometry",
    "bottom_traction",
)
ROUTES = ("bottom", "top")  # deterministic tie break uses the known fallback first
SHARE_RULES = ("never", "always", "if_bad", "if_good")
World = Tuple[bool, bool, bool, bool]
DeliveredHistory = Tuple[Tuple[str, bool], ...]


@dataclass(frozen=True)
class ExactOracleConfig:
    sensing_cost: float = 0.04
    communication_attempt_cost: float = 0.01
    byte_cost: float = 0.00005
    header_bytes: int = 32
    payload_bytes: int = 16
    ack_payload_bytes: int = 8
    timely_delivery_probability: float = 0.90
    top_safe_cost: float = 1.0
    bottom_safe_cost: float = 3.0
    unsafe_route_cost: float = 5.0
    decision_mode: str = "certified"

    @property
    def packet_cost(self):
        return self.communication_attempt_cost + self.byte_cost * (
            self.header_bytes + self.payload_bytes
        )


@dataclass(frozen=True)
class PolicyValue:
    sense_subset: Tuple[str, ...]
    share_rules: Tuple[str, ...]
    sensing_cost: float
    expected_communication_cost: float
    expected_execution_cost: float
    total_cost: float
    delivered_history_probability: Mapping[str, float]
    receiver_decisions: Mapping[str, str]

    @property
    def share_q(self):
        return self.expected_communication_cost + self.expected_execution_cost


def default_prior():
    """Complementarity diagnostic: top G/T unknown, bottom known safe."""
    return {
        (geometry, traction, True, True): 0.25
        for geometry, traction in product((False, True), repeat=2)
    }


def powerset(items: Sequence[str]):
    for size in range(len(items) + 1):
        for subset in combinations(items, size):
            yield subset


def history_key(history: DeliveredHistory):
    if not history:
        return "none"
    return ",".join("{}={}".format(name, int(value)) for name, value in history)


class ExactAcquisitionOracle:
    """Brute-force exact solver over a declared finite decentralized class."""

    def __init__(self, config=ExactOracleConfig(), prior=None):
        self.config = config
        self.prior = dict(default_prior() if prior is None else prior)
        self._validate_prior()

    def _validate_prior(self):
        if not self.prior:
            raise ValueError("prior must contain at least one world")
        if any(len(world) != len(VARIABLES) for world in self.prior):
            raise ValueError("world tuples must align with VARIABLES")
        if any(probability < 0 for probability in self.prior.values()):
            raise ValueError("world probabilities must be nonnegative")
        if abs(sum(self.prior.values()) - 1.0) > 1e-12:
            raise ValueError("world probabilities must sum to one")
        if not 0.0 <= self.config.timely_delivery_probability <= 1.0:
            raise ValueError("delivery probability must lie in [0,1]")
        if self.config.decision_mode not in ("certified", "expected_cost"):
            raise ValueError("decision_mode must be certified or expected_cost")

    @staticmethod
    def world_dict(world: World):
        return dict(zip(VARIABLES, world))

    def execution_cost(self, route: str, world: World):
        values = self.world_dict(world)
        if not values[route + "_geometry"] or not values[route + "_traction"]:
            return self.config.unsafe_route_cost
        return (
            self.config.top_safe_cost
            if route == "top"
            else self.config.bottom_safe_cost
        )

    @staticmethod
    def rule_attempts(rule: str, value: bool):
        if rule == "never":
            return False
        if rule == "always":
            return True
        if rule == "if_bad":
            return not value
        if rule == "if_good":
            return value
        raise ValueError("unknown share rule {}".format(rule))

    def _delivery_branches(self, attempts: Sequence[Tuple[str, bool]]):
        probability = self.config.timely_delivery_probability
        for mask in product((False, True), repeat=len(attempts)):
            delivered = tuple(
                attempts[index] for index, is_delivered in enumerate(mask) if is_delivered
            )
            branch_probability = 1.0
            for is_delivered in mask:
                branch_probability *= probability if is_delivered else (1.0 - probability)
            yield tuple(sorted(delivered)), branch_probability

    def evaluate_policy(
        self, sense_subset: Sequence[str], share_rules: Sequence[str]
    ) -> PolicyValue:
        sense_subset = tuple(sense_subset)
        share_rules = tuple(share_rules)
        if any(name not in VARIABLES for name in sense_subset):
            raise ValueError("unknown sensing variable")
        if len(set(sense_subset)) != len(sense_subset):
            raise ValueError("sensing subset contains duplicates")
        if len(share_rules) != len(sense_subset):
            raise ValueError("one sharing rule is required per sensed variable")
        if any(rule not in SHARE_RULES for rule in share_rules):
            raise ValueError("unknown sharing rule")

        # joint_mass[h][w] = P(delivered history h, world w). This is the
        # receiver's exact information state; undelivered private readings are
        # never supplied to its route decision.
        joint_mass = {}  # type: Dict[DeliveredHistory, Dict[World, float]]
        expected_attempts = 0.0
        for world, world_probability in self.prior.items():
            values = self.world_dict(world)
            attempts = tuple(
                (variable, values[variable])
                for variable, rule in zip(sense_subset, share_rules)
                if self.rule_attempts(rule, values[variable])
            )
            expected_attempts += world_probability * len(attempts)
            for delivered, delivery_probability in self._delivery_branches(attempts):
                if delivery_probability == 0:
                    continue
                by_world = joint_mass.setdefault(delivered, {})
                by_world[world] = by_world.get(world, 0.0) + (
                    world_probability * delivery_probability
                )

        history_probabilities = {}
        receiver_decisions = {}
        expected_execution_cost = 0.0
        for history, world_mass in sorted(joint_mass.items()):
            probability = sum(world_mass.values())
            route_costs = {
                route: sum(
                    mass * self.execution_cost(route, world)
                    for world, mass in world_mass.items()
                )
                for route in ROUTES
            }
            candidates = list(ROUTES)
            if self.config.decision_mode == "certified":
                candidates = [
                    route
                    for route in ROUTES
                    if all(
                        mass <= 0.0
                        or self.execution_cost(route, world) < self.config.unsafe_route_cost
                        for world, mass in world_mass.items()
                    )
                ]
                if not candidates:
                    raise RuntimeError("no certified route or backup is available")
            decision = min(
                candidates, key=lambda route: (route_costs[route], ROUTES.index(route))
            )
            key = history_key(history)
            history_probabilities[key] = probability
            receiver_decisions[key] = decision
            expected_execution_cost += route_costs[decision]

        sensing_cost = len(sense_subset) * self.config.sensing_cost
        communication_cost = expected_attempts * self.config.packet_cost
        total = sensing_cost + communication_cost + expected_execution_cost
        return PolicyValue(
            sense_subset=sense_subset,
            share_rules=share_rules,
            sensing_cost=sensing_cost,
            expected_communication_cost=communication_cost,
            expected_execution_cost=expected_execution_cost,
            total_cost=total,
            delivered_history_probability=history_probabilities,
            receiver_decisions=receiver_decisions,
        )

    def share_q_table(self, sense_subset: Sequence[str]):
        sense_subset = tuple(sense_subset)
        rows = []
        for rules in product(SHARE_RULES, repeat=len(sense_subset)):
            value = self.evaluate_policy(sense_subset, rules)
            rows.append(value)
        return rows

    def best_share_policy(self, sense_subset: Sequence[str]):
        rows = self.share_q_table(sense_subset)
        return min(rows, key=lambda value: (value.share_q, value.share_rules))

    def solve(self):
        sense_rows = []
        all_share_rows = []
        for subset in powerset(VARIABLES):
            candidates = self.share_q_table(subset)
            best = min(candidates, key=lambda value: (value.share_q, value.share_rules))
            sense_rows.append(best)
            all_share_rows.extend(candidates)
        optimum = min(sense_rows, key=lambda value: (value.total_cost, value.sense_subset))
        return optimum, sense_rows, all_share_rows

    def report(self):
        optimum, sense_rows, all_share_rows = self.solve()

        def serialize(value):
            row = asdict(value)
            row["share_q"] = value.share_q
            return row

        serialized_best = [serialize(value) for value in sense_rows]
        return {
            "version": "coph-fork-exact-oracle-a2-v1",
            "scope": {
                "worlds": len(self.prior),
                "sensing_subsets": 2 ** len(VARIABLES),
                "sharing_policies": sum(
                    len(SHARE_RULES) ** len(subset) for subset in powerset(VARIABLES)
                ),
                "sampling_used": False,
                "policy_class": "fixed sensing subset plus value-dependent per-reading send rules",
                "receiver_information": "delivered messages only",
            },
            "config": asdict(self.config),
            "variables": list(VARIABLES),
            "prior": [
                {"world": self.world_dict(world), "probability": probability}
                for world, probability in sorted(self.prior.items())
            ],
            "q_sense_star": serialized_best,
            "q_share_star": [
                {
                    "sense_subset": row["sense_subset"],
                    "share_rules": row["share_rules"],
                    "share_q": row["share_q"],
                    "expected_communication_cost": row["expected_communication_cost"],
                    "expected_execution_cost": row["expected_execution_cost"],
                    "delivered_history_probability": row["delivered_history_probability"],
                    "receiver_decisions": row["receiver_decisions"],
                }
                for row in serialized_best
            ],
            "q_share": [serialize(value) for value in all_share_rows],
            "optimum": serialize(optimum),
        }

    def save_report(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2, sort_keys=True) + "\n")
