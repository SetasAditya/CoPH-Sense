"""Exact posterior labels and small set critics for A5-v2.

The carrier acts after the frozen scout/A4 message stage. There is no second
sharing stage after this carrier action in the finite A5 protocol.
"""

from itertools import product

import numpy as np
import torch
from torch import nn

from .memory_a5 import CANDIDATES, MODALITIES, REGIONS


WORLDS = tuple(product((False, True), repeat=4))
ATOMS = tuple((region, modality) for region in REGIONS for modality in MODALITIES)
ACTION_MASK = torch.tensor(
    [[float(atom in action) for atom in ATOMS] for action in CANDIDATES],
    dtype=torch.float32,
)
PAIR_MASK = torch.tensor(
    [float(len(action) == 2) for action in CANDIDATES], dtype=torch.float32
)


def posterior_for(memory, prior=None):
    """Condition the declared world prior on exactly delivered, noiseless records."""
    base = np.full(16, 1 / 16, dtype=np.float64) if prior is None else np.asarray(prior, dtype=np.float64)
    if base.shape != (16,) or np.any(base < 0) or not np.isclose(base.sum(), 1):
        raise ValueError("prior must be a normalized 16-world distribution")
    keep = np.asarray([
        all(memory.value(region, modality) is None or
            bool(world[2 * (region - 1) + (modality == "traction")]) == memory.value(region, modality)
            for region, modality in ATOMS)
        for world in WORLDS
    ], dtype=bool)
    posterior = base * keep
    if posterior.sum() <= 0:
        raise ValueError("memory is impossible under the prior")
    return posterior / posterior.sum()


def delivery_signature(memory):
    return tuple(sorted((record.modality, bool(record.value)) for record in memory.records))


def delivery_posteriors(a4_model, config):
    """Bayesian posteriors from the frozen A4 send rule and packet channel.

    The carrier observes delivered content and public decision time, but not
    failed send attempts. Silence is included in the finite posterior.
    """
    from .memory_a5 import EvidenceLedger, EvidenceRecord, share_set
    masses = {}
    examples = {}
    delivery_probability = config.timely_delivery_probability
    for wi, world in enumerate(WORLDS):
        sent = share_set(a4_model, world, config)
        for delivered_mask in product((False, True), repeat=len(sent)):
            weight = 1 / 16
            ledger = EvidenceLedger()
            for name, arrived in zip(sent, delivered_mask):
                weight *= delivery_probability if arrived else 1 - delivery_probability
                if arrived:
                    modality = "geometry" if name.endswith("geometry") else "traction"
                    ledger.add(EvidenceRecord(
                        evidence_id="r1-" + modality, region=1, modality=modality,
                        value=world[0 if modality == "geometry" else 1],
                        uncertainty=0.0, acquisition_time=0, delivery_time=1,
                        source="scout",
                    ))
            if weight <= 0:
                continue
            key = delivery_signature(ledger)
            if key not in masses:
                masses[key] = np.zeros(16, dtype=np.float64)
                examples[key] = ledger
            masses[key][wi] += weight
    total = sum(mass.sum() for mass in masses.values())
    if not np.isclose(total, 1):
        raise RuntimeError("A4 delivery branches do not sum to one")
    return {key: mass / mass.sum() for key, mass in masses.items()}, examples, {
        key: float(mass.sum()) for key, mass in masses.items()
    }


def action_costs(posterior, memory, config):
    """Exact expected cost under the *fixed evidence-based* route executor."""
    known = [memory.value(*atom) for atom in ATOMS]
    result = []
    for action in CANDIDATES:
        q = len(action) * config.sensing_cost
        for wi, world in enumerate(WORLDS):
            probability = float(posterior[wi])
            if probability <= 0:
                continue
            evidence = list(known)
            for atom in action:
                j = ATOMS.index(atom)
                evidence[j] = bool(world[j])
            q += probability * sum(
                config.top_safe_cost if evidence[2 * (region - 1)] is True
                    and evidence[2 * (region - 1) + 1] is True
                else config.bottom_safe_cost
                for region in REGIONS
            )
        result.append(q)
    return np.asarray(result, dtype=np.float32)


def encode_memory(memory, config, prior=None, posterior=None):
    posterior = posterior_for(memory, prior) if posterior is None else np.asarray(posterior, dtype=np.float64)
    if posterior.shape != (16,) or np.any(posterior < 0) or not np.isclose(posterior.sum(), 1):
        raise ValueError("posterior must be a normalized 16-world distribution")
    known = np.asarray([memory.value(*atom) is not None for atom in ATOMS], dtype=np.float32)
    values = np.asarray([float(memory.value(*atom) or False) for atom in ATOMS], dtype=np.float32)
    marginal = np.asarray([
        sum(posterior[wi] for wi, world in enumerate(WORLDS) if world[j])
        for j in range(4)
    ], dtype=np.float32)
    state = np.asarray([
        *posterior, *known, *values,
        config.sensing_cost, config.top_safe_cost, config.bottom_safe_cost,
        config.unsafe_route_cost,
    ], dtype=np.float32)
    candidates = np.asarray([
        [float(region == 1), float(region == 2),
         float(modality == "geometry"), float(modality == "traction"),
         config.sensing_cost, known[j], marginal[j], 1 - known[j],
         config.bottom_safe_cost - config.top_safe_cost]
        for j, (region, modality) in enumerate(ATOMS)
    ], dtype=np.float32)
    return state, candidates, posterior


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 96), nn.SiLU(), nn.Linear(96, outputs))


class MemorySetCritic(nn.Module):
    """Matched-input additive, DeepSets, and within-region pairwise critics."""

    def __init__(self, kind):
        super().__init__()
        if kind not in ("additive", "deepsets", "pairwise"):
            raise ValueError("unknown critic kind")
        self.kind = kind
        self.state = mlp(28, 64)
        self.atom = mlp(9, 32)
        self.base = mlp(64, 1)
        self.singleton = mlp(96, 1)
        if kind == "deepsets":
            self.phi = mlp(96, 32)
            self.rho = mlp(96, 1)
        if kind == "pairwise":
            self.pair = mlp(128, 1)

    def forward(self, state, candidates):
        context = self.state(state)
        atom = self.atom(candidates)
        joined = torch.cat((context[:, None, :].expand(-1, 4, -1), atom), -1)
        mask = ACTION_MASK.to(state.device)
        if self.kind == "deepsets":
            pooled = torch.einsum("sa,bad->bsd", mask, self.phi(joined))
            return self.rho(torch.cat((context[:, None, :].expand(-1, 7, -1), pooled), -1)).squeeze(-1)
        values = self.base(context) + torch.einsum("sa,bac->bsc", mask, self.singleton(joined)).squeeze(-1)
        if self.kind == "pairwise":
            pairs = []
            for i, j in ((0, 1), (2, 3)):
                pairs.append(torch.cat((context, atom[:, i] + atom[:, j],
                                        torch.abs(atom[:, i] - atom[:, j])), -1))
            pair_values = self.pair(torch.stack(pairs, 1)).squeeze(-1)
            values = values + torch.stack((torch.zeros_like(values[:, 0]),
                torch.zeros_like(values[:, 0]), torch.zeros_like(values[:, 0]), pair_values[:, 0],
                torch.zeros_like(values[:, 0]), torch.zeros_like(values[:, 0]), pair_values[:, 1]), 1)
        return values


def model_action(model, memory, config, prior=None, posterior=None):
    state, candidates, _ = encode_memory(memory, config, prior, posterior)
    with torch.no_grad():
        values = model(torch.from_numpy(state)[None], torch.from_numpy(candidates)[None])[0].numpy()
    return CANDIDATES[int(values.argmin())]
