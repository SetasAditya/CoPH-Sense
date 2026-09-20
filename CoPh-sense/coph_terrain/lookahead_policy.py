"""Locally executable, first-pass task-value rule for innovation messages."""

from dataclasses import dataclass

import numpy as np

from .environment import cell_to_position
from .lookahead import GaussianRegionBelief, gaussian_kl
from .physical_audit import AuditChoice


FEATURE_NAMES = ("bias", "posterior_mean", "posterior_std",
                 "belief_shift", "sender_estimated_kl", "delta_tau",
                 "teammate_distance", "time_fraction", "region_index",
                 "traction_message", "payload_kib")

LOOKAHEAD_VIEWS = (((15, 27), (18, 27)),
                   ((15, 34), (18, 34)))


def region_acquisition_priorities(env, agent):
    """Public-cost/recipient-belief heuristic for the A5 mechanism probe.

    This is not the learned A3/A5 critic. It provides an explicitly local
    intervention to check that delivered memory reallocates sensing.
    """
    obs = env.observations()[agent]
    position = np.asarray(obs["kinematics"][:2], dtype=float)
    beliefs = obs["lookahead"]["local_region_beliefs"]
    rows = []
    for region, (view, target) in enumerate(LOOKAHEAD_VIEWS):
        variance = float(beliefs[f"{region}:geometry"][1])
        travel = float(np.linalg.norm(cell_to_position(view) - position))
        score = variance / (.25 + travel)
        rows.append({"region": region, "viewpoint": view,
                     "target": target, "variance": variance,
                     "travel_distance": travel, "priority": score})
    return rows


def choose_region_to_inspect(env, agent):
    ranked = sorted(region_acquisition_priorities(env, agent),
                    key=lambda row: (-row["priority"], row["region"]))
    if not ranked or ranked[0]["priority"] < .01:
        return None
    row = ranked[0]
    return AuditChoice(agent, "geometry", row["viewpoint"], row["target"])


def sender_estimated_novelty(env, packet_id):
    """Uses only the sender's local state and the declared public prior.

    True receiver-conditioned KL is evaluator-only: the recipient may have
    private observations that the sender has not received. The estimate is
    zero after an ACK for this packet and otherwise assumes the public prior.
    """
    packet = env.innovation_packets[packet_id]
    if packet_id in env.acknowledged[packet.source]:
        return 0.
    prior_mean = .35 if packet.modality == "geometry" else .70
    prior = GaussianRegionBelief(4., 4. * prior_mean)
    after = GaussianRegionBelief(prior.precision, prior.weighted_mean)
    after.add(*packet.wire_factor())
    return gaussian_kl(after, prior)


def message_features(env, packet_id):
    """Feature vector available to the sender at the SEND/HOLD decision."""
    packet = env.innovation_packets[packet_id]
    local = env.observations()[packet.source]
    kin = np.asarray(local["kinematics"], dtype=float)
    teammate_distance = float(np.linalg.norm(kin[4:6]))
    lookahead = local["lookahead"]
    values = (1., packet.posterior_mean,
              float(np.sqrt(packet.posterior_variance)),
              packet.posterior_mean - packet.prior_mean,
              sender_estimated_novelty(env, packet_id),
              lookahead["delta_tau"], teammate_distance,
              env.step_index / env.config.horizon_steps,
              float(packet.region), float(packet.modality == "traction"),
              packet.payload_bytes / 1024.)
    return np.asarray(values, dtype=np.float64)


@dataclass
class RidgeMessageValue:
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, examples, ridge=.05):
        x = np.asarray([row["actor_features"] for row in examples],
                       dtype=np.float64)
        y = np.asarray([row["decision_value"] for row in examples],
                       dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("at least two labeled messages are required")
        mean = x[:, 1:].mean(0)
        scale = np.maximum(x[:, 1:].std(0), 1e-4)
        z = x.copy()
        z[:, 1:] = (x[:, 1:] - mean) / scale
        penalty = ridge * np.eye(z.shape[1])
        penalty[0, 0] = 0.
        coefficients = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        return cls(coefficients, mean, scale)

    def predict(self, features):
        z = np.asarray(features, dtype=np.float64).copy()
        z[1:] = (z[1:] - self.mean) / self.scale
        return float(z @ self.coefficients)

    def action(self, env, packet_id):
        if sender_estimated_novelty(env, packet_id) <= 0.:
            return "hold"
        return "send" if self.predict(message_features(env, packet_id)) > 0. \
            else "hold"
