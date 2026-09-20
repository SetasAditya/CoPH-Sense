"""Predeclared E2 tail-risk and paired hierarchical-bootstrap calculations."""

import numpy as np


def empirical_cvar(costs, alpha=.95):
    """Variational empirical CVaR with fractional upper-tail mass."""
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("CVaR needs at least one finite observation")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    upper = np.sort(values)[::-1]
    mass = (1 - alpha) * values.size
    whole = int(np.floor(mass))
    fraction = mass - whole
    tail_sum = upper[:whole].sum()
    if fraction > 1e-12:
        tail_sum += fraction * upper[whole]
    return float(tail_sum / mass)


def paired_hierarchical_bootstrap(risk, success, replicates=10000, seed=1701):
    """Inputs have shape [training seed, map group, realization, method].

    Method 0 is CoPH; method 1 is the validation-selected composite rival.
    Each resample preserves method pairing and all realizations of a selected
    parent map group. CVaR is recomputed inside every bootstrap replicate.
    """
    risk = np.asarray(risk, dtype=np.float64)
    success = np.asarray(success, dtype=np.float64)
    if risk.shape != success.shape or risk.ndim != 4 or risk.shape[-1] != 2:
        raise ValueError("expected matching [seed, group, realization, 2] arrays")
    if not np.isfinite(risk).all() or not np.isin(success, (0, 1)).all():
        raise ValueError("invalid risk or success data")
    if np.any(risk < 0):
        raise ValueError("material exposure must be nonnegative")
    nseed, ngroup = risk.shape[:2]
    if nseed < 2 or ngroup < 2:
        raise ValueError("hierarchical inference requires multiple seeds and groups")

    def differences(seed_indices, group_indices):
        risks = []
        successes = []
        for s, groups in zip(seed_indices, group_indices):
            chosen_risk = risk[s, groups]
            chosen_success = success[s, groups]
            risks.append(empirical_cvar(chosen_risk[..., 0])
                         - empirical_cvar(chosen_risk[..., 1]))
            successes.append(float(chosen_success[..., 0].mean()
                                   - chosen_success[..., 1].mean()))
        return np.mean(risks), np.mean(successes)

    original = differences(np.arange(nseed),
                           np.broadcast_to(np.arange(ngroup), (nseed, ngroup)))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 2), dtype=np.float64)
    for k in range(replicates):
        selected_seeds = rng.integers(nseed, size=nseed)
        # Test parent maps are shared across independently trained seeds, so
        # resample a common group vector to retain cross-seed map difficulty.
        selected_groups = np.broadcast_to(rng.integers(ngroup, size=ngroup),
                                          (nseed, ngroup))
        draws[k] = differences(selected_seeds, selected_groups)
    risk_interval = np.quantile(draws[:, 0], (.025, .975))
    success_lower = float(np.quantile(draws[:, 1], .05))
    return {
        "risk_difference": float(original[0]),
        "risk_two_sided_95_interval": risk_interval.tolist(),
        "success_difference": float(original[1]),
        "success_one_sided_95_lower": success_lower,
        "completion_noninferior_2pp": bool(success_lower > -.02),
        "risk_superior_if_completion_noninferior": bool(
            success_lower > -.02 and risk_interval[1] < 0),
        "bootstrap_replicates": int(replicates),
        "training_seeds": int(nseed), "map_groups": int(ngroup),
        "realizations_per_group": int(risk.shape[2]),
    }
