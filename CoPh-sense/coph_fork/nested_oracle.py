"""Exact finite A3 sensing values under an A4 post-observation sharing policy.

All continuations use the A4 protocol: charged evidence/ACK attempts, packet
delivery before commitment, and no inference from silence. The route-cost
model remains the normalized finite diagnostic, not a VMAS rollout.
"""

from collections import defaultdict

import numpy as np
import torch

from .critic import SUBSETS
from .oracle import ExactOracleConfig, VARIABLES
from .share_oracle import ExactShareOracle, ShareState
from .train_share_critic import build_features_v2


def learned_message_set(model, oracle):
    state, candidates = build_features_v2(oracle.config, oracle.prior, oracle.state)
    feasible = set(oracle.candidate_sets())
    with torch.no_grad():
        predicted = model(
            torch.tensor(state)[None], torch.tensor(candidates)[None]
        )[0].sum(-1).numpy()
    return min(
        (subset for subset in SUBSETS if subset in feasible),
        key=lambda subset: (predicted[SUBSETS.index(subset)], len(subset), subset),
    )


def outcome_distribution(prior, sense_subset):
    groups = defaultdict(float)
    for world, probability in prior.items():
        if probability <= 0:
            continue
        reading = tuple(
            (name, bool(world[VARIABLES.index(name)])) for name in sense_subset
        )
        groups[reading] += probability
    return dict(groups)


def nested_values(config: ExactOracleConfig, prior, model=None, return_components=False):
    """Return 16 exact oracle values and, if supplied, learned-A4 values."""
    oracle_values = []
    learned_values = []
    oracle_components = []
    learned_components = []
    for subset in SUBSETS:
        sense_cost = len(subset) * config.sensing_cost
        oracle_execution = oracle_communication = 0.0
        learned_execution = learned_communication = 0.0
        for reading, mass in outcome_distribution(prior, subset).items():
            state = ShareState(
                acquired=reading,
                timely_probability=config.timely_delivery_probability,
            )
            oracle = ExactShareOracle(state, config=config, prior=prior)
            optimum, _ = oracle.solve()
            oracle_execution += mass * optimum.expected_execution_cost
            oracle_communication += mass * optimum.expected_communication_cost
            if model is not None:
                message_set = learned_message_set(model, oracle)
                selected = oracle.evaluate(message_set)
                learned_execution += mass * selected.expected_execution_cost
                learned_communication += mass * selected.expected_communication_cost
        q_oracle = sense_cost + oracle_execution + oracle_communication
        oracle_values.append(q_oracle)
        oracle_components.append((oracle_execution, sense_cost, oracle_communication))
        if model is not None:
            q_learned = sense_cost + learned_execution + learned_communication
            learned_values.append(q_learned)
            learned_components.append((learned_execution, sense_cost, learned_communication))
    result = (
        np.asarray(oracle_values),
        np.asarray(learned_values) if model is not None else None,
    )
    if return_components:
        return result + (
            np.asarray(oracle_components),
            np.asarray(learned_components) if model is not None else None,
        )
    return result


def pair_ordering(values, tolerance=1e-9):
    none = values[SUBSETS.index(())]
    geometry = values[SUBSETS.index(("top_geometry",))]
    traction = values[SUBSETS.index(("top_traction",))]
    both = values[SUBSETS.index(("top_geometry", "top_traction"))]
    return bool(
        both < none - tolerance
        and none < geometry - tolerance
        and none < traction - tolerance
    )
