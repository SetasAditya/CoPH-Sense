"""Deterministic engineering adapter used only to test campaign plumbing."""

import hashlib
import numpy as np

from .confirmatory_campaign import METHODS


def evaluate(key, method):
    seed = int.from_bytes(hashlib.sha256(
        f"{key.episode_seed}:{method}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    index = METHODS.index(method)
    # Explicitly synthetic: these values verify pairing, persistence, plots,
    # and report generation; they are not scientific results.
    return {"success": bool(rng.random() > .08),
            "team_cost": 1.2-.025*index+rng.uniform(0, .04),
            "material_risk": .3-.008*index+rng.uniform(0, .02),
            "bytes": (0, 900, 180, 100, 80, 70)[index],
            "duplicate_sensing": max(0., 3-.45*index),
            "disjoint_support": 10+1.2*index,
            "lookahead_gain": .2+.12*index}
