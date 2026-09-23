from dataclasses import replace
import unittest

import numpy as np

from coph_terrain.conditional_scout_material import (
    _replace_compatible_truth, dispatch_candidates)
from coph_terrain.environment import CoPHTerrainEnv, TerrainAction, TerrainConfig


def _env(seed=7101):
    return CoPHTerrainEnv(
        "open", seed, 0, seed=seed+1,
        config=TerrainConfig(carrier_primary=True, horizon_steps=120),
        executor="vmas", device="cpu")


class ConditionalScoutMaterialTests(unittest.TestCase):
    def test_dispatch_features_do_not_depend_on_hidden_material(self):
        env = _env()
        before = [(c.candidate_id, c.features) for c in dispatch_candidates(env)]
        altered = _replace_compatible_truth(env, 99991)
        after = [(c.candidate_id, c.features) for c in dispatch_candidates(altered)]
        self.assertEqual(before, after)
        self.assertFalse(np.array_equal(env._truth.traction,
                                        altered._truth.traction))

    def test_carrier_arrival_does_not_erase_outstanding_scout(self):
        env = _env(7201)
        env.scout_ever_dispatched = True
        carrier = env.physics.world.agents[1]
        scout = env.physics.world.agents[0]
        carrier.state.pos[0] = carrier.state.pos.new_tensor([4.9, .1])
        scout.state.pos[0] = scout.state.pos.new_tensor([-4.9, -.5])
        self.assertGreater(np.linalg.norm(env.positions["carrier"]-
                                          env.positions["scout"]), 1.)
        env.step({name: TerrainAction(waypoint=tuple(env.positions[name]))
                  for name in ("scout", "carrier")})
        self.assertFalse(env.success)

    def test_compatible_world_preserves_observed_cells(self):
        env = _env(7301)
        env.known_traction["scout"][10, 10] = env._truth.traction[10, 10]
        branch = _replace_compatible_truth(env, 991)
        self.assertEqual(branch._truth.traction[10, 10],
                         env._truth.traction[10, 10])
