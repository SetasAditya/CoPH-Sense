import copy
import unittest

import numpy as np

from coph_terrain.environment import CoPHTerrainEnv, TerrainAction
from coph_terrain.material_executor import MaterialHamiltonianExecutor


class MaterialExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = MaterialHamiltonianExecutor()

    def test_force_cannot_depend_on_hidden_truth(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44)
        observation = env.observations()["carrier"]
        before, _ = self.executor.proposal(
            (-5., -.25), (0., 0.), (-4., -.25), observation)
        env._truth.surface[:] = 1.
        env._truth.traction[:] = .05
        after, _ = self.executor.proposal(
            (-5., -.25), (0., 0.), (-4., -.25), observation)
        np.testing.assert_allclose(before, after, atol=0., rtol=0.)

    def test_delivered_belief_can_change_material_proposal(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44)
        observation = env.observations()["carrier"]
        changed = copy.deepcopy(observation)
        changed["known_surface"][4:12, 25:35] = 1.
        changed["known_traction"][4:12, 25:35] = .15
        first, _ = self.executor.proposal(
            (-5., -.25), (0., 0.), (-4., -.25), observation)
        second, _ = self.executor.proposal(
            (-5., -.25), (0., 0.), (-4., -.25), changed)
        self.assertGreater(float(np.linalg.norm(first-second)), 1e-7)

    def test_zero_risk_belief_has_zero_soft_material_force(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44)
        observation = env.observations()["carrier"]
        observation["appearance"][:] = .86
        observation["known_surface"][:] = 0.
        observation["known_traction"][:] = .55
        components, _ = self.executor.force_components(
            (-5., -.25), (0., 0.), (-4., -.25), observation)
        np.testing.assert_allclose(components["soft"], 0., atol=1e-7)

    def test_same_state_and_belief_is_deterministic(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44)
        observation = env.observations()["carrier"]
        first, _ = self.executor.proposal(
            (-5., -.25), (.1, 0.), (-4., -.25), observation)
        second, _ = self.executor.proposal(
            (-5., -.25), (.1, 0.), (-4., -.25), observation)
        np.testing.assert_array_equal(first, second)

    def test_material_environment_smoke(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44,
                             executor="material_ph")
        for _ in range(3):
            env.step({"scout": TerrainAction(waypoint=(-4., .25)),
                      "carrier": TerrainAction(waypoint=(-4., -.25))})
        for value in env.positions.values():
            self.assertTrue(np.all(np.isfinite(value)))

    def test_source_rollout_integrator_is_exposed(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44)
        result = self.executor.rollout(
            (-5., -.25), (0., 0.), (-4., -.25),
            env.observations()["carrier"], horizon_steps=3, dt=.05,
            mass=1.8, radius=.1)
        self.assertTrue(np.all(np.isfinite(result["position"])))
        self.assertGreaterEqual(result["path_length"], 0.)


if __name__ == "__main__":
    unittest.main()
