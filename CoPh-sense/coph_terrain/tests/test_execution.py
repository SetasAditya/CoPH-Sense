import unittest

import numpy as np

from coph_terrain.environment import CoPHTerrainEnv, TerrainAction
from coph_terrain.execution import ph_step


class ExecutionTests(unittest.TestCase):
    def test_ph_step_is_finite_and_observation_only(self):
        a = CoPHTerrainEnv("open", 100000, 0, 44)
        b = CoPHTerrainEnv("open", 100000, 0, 44)
        b._truth.traction[40:50, 40:50] = .15
        observation_a = a.observations()["carrier"]
        observation_b = b.observations()["carrier"]
        kwargs = dict(mass=1.8, radius=.10, max_speed=.55,
                      traction=.8, dt=.05)
        first = ph_step((-5., -.25), (0., 0.), (-4., -.25),
                        observation_a, **kwargs)
        second = ph_step((-5., -.25), (0., 0.), (-4., -.25),
                         observation_b, **kwargs)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertTrue(np.isfinite(first[2]["storage_after"]))

    def test_ph_environment_replays(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44, executor="ph")
        for _ in range(5):
            env.step({"scout": TerrainAction(waypoint=(-4., .25)),
                      "carrier": TerrainAction(waypoint=(-4., -.25))})
        replay = CoPHTerrainEnv.replay(env.replay_payload())
        self.assertEqual(replay.ledger, env.ledger)
        self.assertEqual(replay.step_index, env.step_index)


if __name__ == "__main__":
    unittest.main()
