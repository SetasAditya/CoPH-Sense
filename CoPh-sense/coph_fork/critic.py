"""Small permutation-invariant A3 acquisition critics."""

from itertools import combinations

import torch
from torch import nn

from .oracle import VARIABLES, powerset


SUBSETS = tuple(powerset(VARIABLES))
SUBSET_MASK = torch.tensor(
    [[float(name in subset) for name in VARIABLES] for subset in SUBSETS],
    dtype=torch.float32,
)
PAIRS = tuple(combinations(range(len(VARIABLES)), 2))
PAIR_MASK = torch.tensor(
    [[float(VARIABLES[i] in subset and VARIABLES[j] in subset) for i, j in PAIRS] for subset in SUBSETS],
    dtype=torch.float32,
)


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


class SetCostCritic(nn.Module):
    """q0 + sum q1 + sum q2, or additive/DeepSets matched-input controls.

    Context and candidate features contain only pre-acquisition public data.
    The three output components are execution, sensing, and communication cost.
    """

    def __init__(self, state_dim, candidate_dim, kind="pairwise"):
        super().__init__()
        if kind not in ("additive", "pairwise", "deepsets"):
            raise ValueError("unknown critic kind")
        self.kind = kind
        self.state_encoder = mlp(state_dim, 128, 128)
        self.candidate_encoder = mlp(candidate_dim, 64, 64)
        if kind == "deepsets":
            self.phi = mlp(192, 128, 64)
            self.rho = nn.Sequential(
                nn.Linear(192, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 3),
            )
        else:
            self.baseline = mlp(128, 64, 3)
            self.singleton = nn.Sequential(
                nn.Linear(192, 128), nn.SiLU(),
                nn.Linear(128, 64), nn.SiLU(),
                nn.Linear(64, 3),
            )
            if kind == "pairwise":
                self.pair = nn.Sequential(
                    nn.Linear(320, 128), nn.SiLU(),
                    nn.Linear(128, 64), nn.SiLU(),
                    nn.Linear(64, 3),
                )

    def forward(self, state, candidates):
        batch = state.shape[0]
        context = self.state_encoder(state)
        candidate = self.candidate_encoder(candidates)
        joined = torch.cat((context[:, None, :].expand(-1, 4, -1), candidate), dim=-1)
        subset_mask = SUBSET_MASK.to(state.device)
        if self.kind == "deepsets":
            atoms = self.phi(joined)
            pooled = torch.einsum("sa,bad->bsd", subset_mask, atoms)
            return self.rho(torch.cat((context[:, None, :].expand(-1, len(SUBSETS), -1), pooled), dim=-1))

        singleton = self.singleton(joined)
        result = self.baseline(context)[:, None, :] + torch.einsum(
            "sa,bac->bsc", subset_mask, singleton
        )
        if self.kind == "pairwise":
            pair_inputs = []
            for i, j in PAIRS:
                first, second = candidate[:, i], candidate[:, j]
                pair_inputs.append(
                    torch.cat(
                        (context, first + second, torch.abs(first - second), first * second),
                        dim=-1,
                    )
                )
            pair_values = self.pair(torch.stack(pair_inputs, dim=1))
            result = result + torch.einsum(
                "sp,bpc->bsc", PAIR_MASK.to(state.device), pair_values
            )
        return result


def contrast(values):
    """Cost complementarity Q(G)+Q(T)-Q(GT)-Q(empty), per batch."""
    indices = [SUBSETS.index(tuple(names)) for names in (
        (), ("top_geometry",), ("top_traction",), ("top_geometry", "top_traction")
    )]
    none, geometry, traction, both = indices
    return values[:, geometry] + values[:, traction] - values[:, both] - values[:, none]


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
