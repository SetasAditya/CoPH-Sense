"""Variable-candidate, memory-conditioned expected-cost and risk critics.

The same architecture scores acquisition sets before sensing and message
subsets after readings; the two phases use separately fitted parameters.
"""

from itertools import combinations

import numpy as np
import torch
from torch import nn


RISK_QUANTILES = torch.tensor(
    [i / 20 for i in range(1, 17)]
    + [.85, .90, .925, .95, .96, .97, .98, .99, .995],
    dtype=torch.float32,
)
OUTPUTS = 1 + len(RISK_QUANTILES)


def enumerate_pairs(candidate_count):
    """Empty, singles, pairs; (−1,−1) is empty and (i,−1) is a singleton."""
    return torch.tensor([(-1, -1)] + [(i, -1) for i in range(candidate_count)]
                        + list(combinations(range(candidate_count), 2)),
                        dtype=torch.long)


class VariableSetValueNet(nn.Module):
    def __init__(self, candidate_dim=12, memory_dim=12, state_dim=12):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(8, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.SiLU())
        self.memory_encoder = nn.Sequential(nn.Linear(memory_dim, 64), nn.SiLU(),
                                            nn.Linear(64, 64), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(192, 128), nn.SiLU(),
                                     nn.Linear(128, 64), nn.SiLU())
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_dim, 64), nn.SiLU(),
                                               nn.Linear(64, 64), nn.SiLU())
        self.base = nn.Linear(64, 64)
        self.singleton = nn.Sequential(nn.Linear(128, 128), nn.SiLU(),
                                       nn.Linear(128, 64))
        self.pair = nn.Sequential(nn.Linear(256, 128), nn.SiLU(),
                                  nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.SiLU(),
                                     nn.Linear(128, OUTPUTS))

    def forward(self, map_tensor, state, memory, memory_mask,
                candidates, sets):
        """Return mission mean [B,S] and risk quantiles [B,S,Q].

        `sets` has shape [S,2] and contains only public, feasible candidate
        indices. There are no true-map or undelivered-evidence inputs.
        """
        batch, count, _ = candidates.shape
        if sets.ndim != 2 or sets.shape[1] != 2 or torch.any(sets >= count):
            raise ValueError("invalid set index table")
        map_embedding = self.map_encoder(map_tensor)
        memory_embedding = self.memory_encoder(memory)
        mask = memory_mask[..., None].to(memory_embedding.dtype)
        pooled = (memory_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.)
        context = self.context(torch.cat((map_embedding,
                                          self.state_encoder(state), pooled), -1))
        candidate = self.candidate_encoder(candidates)
        joined = torch.cat((context[:, None, :].expand(-1, count, -1), candidate), -1)
        singles = self.singleton(joined)
        index_a = sets[:, 0].clamp_min(0)
        index_b = sets[:, 1].clamp_min(0)
        present_a = (sets[:, 0] >= 0).to(singles.dtype)[None, :, None]
        present_b = (sets[:, 1] >= 0).to(singles.dtype)[None, :, None]
        latent = self.base(context)[:, None, :]
        latent = latent + singles[:, index_a] * present_a + singles[:, index_b] * present_b
        a, b = candidate[:, index_a], candidate[:, index_b]
        pair_input = torch.cat((context[:, None, :].expand(-1, len(sets), -1),
                                a + b, torch.abs(a - b), a * b), -1)
        latent = latent + self.pair(pair_input) * present_a * present_b
        # Quantiles do not add across information items. Decode the complete
        # set embedding into one bounded, ordered return distribution.
        raw = self.decoder(latent)
        mission = torch.nn.functional.softplus(raw[..., 0])
        # An extra terminal interval allows a nearly zero or nearly constant
        # bounded risk distribution while preserving ordered quantiles.
        intervals = torch.cat((raw[..., 1:], torch.zeros_like(raw[..., :1])), -1)
        quantiles = 2. * torch.cumsum(torch.softmax(intervals, -1), -1)[..., :-1]
        return mission, quantiles


def cvar_from_quantiles(quantiles, alpha=.95):
    """Piecewise-linear upper-tail integral; hold the .999 quantile to 1."""
    levels = RISK_QUANTILES.to(quantiles.device)
    keep = levels >= alpha
    if not bool(keep.any()):
        raise ValueError("quantile grid does not cover requested tail")
    grid = torch.cat((levels[keep], levels.new_tensor([1.])))
    values = torch.cat((quantiles[..., keep], quantiles[..., -1:]), -1)
    return torch.trapz(values, grid, dim=-1) / (1 - alpha)


def decision_score(mission, quantiles):
    """Predeclared E2 decision score: E[C0] + CVaR_0.95(C_R)."""
    return mission + cvar_from_quantiles(quantiles, .95)


def quantile_huber_loss(predicted_quantiles, rollout_risk, kappa=.05):
    """Train on individual continuation returns, never a tiny-sample CVaR.

    predicted_quantiles: [batch, set, quantile]; rollout_risk: [batch, set,
    rollout]. The rollout dimension can contain common-random-number paired
    continuations across candidate sets.
    """
    if (predicted_quantiles.ndim != 3 or rollout_risk.ndim != 3
            or predicted_quantiles.shape[:2] != rollout_risk.shape[:2]
            or predicted_quantiles.shape[-1] != len(RISK_QUANTILES)):
        raise ValueError("expected [batch,set,quantile] and [batch,set,rollout]")
    if not torch.isfinite(rollout_risk).all() or torch.any(rollout_risk < 0) \
            or torch.any(rollout_risk > 2):
        raise ValueError("risk returns must be finite and in [0,2]")
    residual = rollout_risk[..., None] - predicted_quantiles[..., None, :]
    absolute = residual.abs()
    huber = torch.where(absolute <= kappa, .5 * residual.square(),
                        kappa * (absolute - .5 * kappa))
    levels = RISK_QUANTILES.to(predicted_quantiles.device)
    pinball = (levels - (residual.detach() < 0).to(levels.dtype)).abs()
    return (pinball * huber / kappa).mean()


def observation_map_tensor(observation):
    """Eight channels; latent terrain unknowns have explicit masks."""
    channels = []
    for key, fill in (("known_geometry", 0.), ("appearance", .5),
                      ("known_surface", .5), ("known_traction", .5)):
        array = np.asarray(observation[key], dtype=np.float32)
        mask = np.isfinite(array)
        channels.extend((np.where(mask, array, fill), mask.astype(np.float32)))
    return torch.from_numpy(np.stack(channels).astype(np.float32))
