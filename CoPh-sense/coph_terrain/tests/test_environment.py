import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from coph_terrain.environment import (
    CoPHTerrainEnv, TerrainAction, TerrainConfig, position_to_cell,
)
from coph_fork.ph_factorial_controller import box_clearance
from coph_terrain.pilot_environment import run_one


class TerrainEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = CoPHTerrainEnv("open", 100000, 0, 17,
                                  TerrainConfig(horizon_steps=100))

    def _step(self, scout=None, carrier=None):
        return self.env.step({"scout": scout or TerrainAction(),
                              "carrier": carrier or TerrainAction()})

    def test_static_topology_public_terrain_private_and_replay_exact(self):
        obs = self.env.observations()["carrier"]
        np.testing.assert_array_equal(obs["known_geometry"],
                                      self.env._truth.occupancy.astype(np.float32))
        self.assertTrue(np.isnan(obs["known_surface"]).all())
        self.assertTrue(np.isnan(obs["known_traction"]).all())
        self.assertFalse(any("truth" in key for key in obs))
        for _ in range(3):
            self._step()
        replay = CoPHTerrainEnv.replay(self.env.replay_payload())
        self.assertEqual(replay.step_index, self.env.step_index)
        self.assertEqual(replay.ledger, self.env.ledger)

    def test_clone_preserves_counterfactual_prefix_and_rng(self):
        clone = self.env.clone()
        action = {"scout": TerrainAction(motion=(.2, 0.)),
                  "carrier": TerrainAction(motion=(.2, 0.))}
        self.env.step(action)
        clone.step(action)
        self.assertEqual(self.env.ledger, clone.ledger)
        for name in ("scout", "carrier"):
            np.testing.assert_array_equal(self.env.positions[name], clone.positions[name])

    def test_material_exposure_uses_declared_horizon(self):
        env = CoPHTerrainEnv("open", 100000, 0, 17,
                             TerrainConfig(horizon_steps=100))
        expected = 0.
        for _ in range(3):
            _, _, _, info = env.step({"scout": TerrainAction(),
                                      "carrier": TerrainAction()})
            expected += sum(info["risk_by_agent"].values()) / 100.
        self.assertAlmostEqual(env.ledger.material_exposure, expected)
        self.assertLessEqual(env.ledger.material_exposure, 2.)

    def test_terminal_waypoint_reaches_shared_goal_region(self):
        # This excluded bottleneck parent previously timed out with both
        # agents near the goal because the scout's target was at its edge.
        result = run_one("bottleneck", 700001, full_map=True,
                         config=TerrainConfig(horizon_steps=1300))
        self.assertTrue(result["success"])

    def test_artifact_contains_replay_and_evaluator_truth(self):
        self._step()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode"
            self.env.save_artifact(path)
            self.assertTrue((path / "metadata.json").exists())
            states = np.load(path / "states.npz")
            self.assertEqual(tuple(states["positions"].shape), (2, 2, 2))
            self.assertEqual(tuple(states["agent_material_exposure"].shape), (2, 2))
            truth = np.load(path / "truth_map.npz")
            self.assertEqual(tuple(truth["traction"].shape), (60, 60))
            self.assertEqual(tuple(truth["surface"].shape), (60, 60))

    def test_probe_requires_dwell_and_delivery(self):
        cell = position_to_cell(self.env.positions["scout"])
        self._step(scout=TerrainAction(sense=("traction", *cell)))
        self.assertFalse(self.env.owned["scout"])
        for _ in range(self.env.config.probe_dwell_steps - 1):
            self._step()
        self.assertEqual(len(self.env.owned["scout"]), 1)
        evidence_id = next(iter(self.env.owned["scout"]))
        self.assertTrue(np.isnan(self.env.known_traction["carrier"]).all())
        self._step(scout=TerrainAction(send_evidence_id=evidence_id))
        self.assertTrue(np.isnan(self.env.known_traction["carrier"]).all())
        for _ in range(self.env.config.communication_delay_steps):
            self._step()
        self.assertIn(evidence_id, self.env.received["carrier"])
        self.assertTrue(np.isfinite(self.env.known_traction["carrier"][cell]))
        self.assertNotIn(evidence_id, self.env.acknowledged["scout"])
        for _ in range(self.env.config.communication_delay_steps):
            self._step()
        self.assertIn(evidence_id, self.env.acknowledged["scout"])
        self.assertGreater(self.env.ledger.sensing, 0.)
        self.assertGreater(self.env.ledger.communication, 0.)

    def test_geometry_scan_measures_surface_not_static_occupancy(self):
        cell = position_to_cell(self.env.positions["scout"])
        before = self.env.known_geometry["scout"].copy()
        self._step(scout=TerrainAction(sense=("geometry", *cell)))
        for _ in range(self.env.config.scan_dwell_steps - 1):
            self._step()
        evidence = next(iter(self.env.owned["scout"].values()))
        self.assertEqual(evidence.modality, "geometry")
        self.assertAlmostEqual(self.env.known_surface["scout"][cell],
                               self.env._truth.surface[cell])
        self.assertTrue(np.isnan(self.env.known_surface["carrier"][cell]))
        np.testing.assert_array_equal(self.env.known_geometry["scout"], before)

    def test_budget_caps_and_invalid_reading(self):
        far = (40, 40)
        self._step(scout=TerrainAction(sense=("traction", *far)))
        for _ in range(self.env.config.probe_dwell_steps - 1):
            self._step()
        self.assertEqual(len(self.env.owned["scout"]), 0)
        self.assertEqual(self.env.measurement_count["scout"], 1)
        self.assertAlmostEqual(self.env.ledger.sensing, self.env.config.probe_cost)

    def test_static_obstacle_is_public_before_motion(self):
        env = CoPHTerrainEnv("bottleneck", 600004, 0, 1)
        self.assertTrue(env._truth.occupancy[20, 39])
        self.assertEqual(env.known_geometry["carrier"][20, 39], 1.)

    def test_analytic_disc_box_collision_matches_vmas(self):
        env = CoPHTerrainEnv("bottleneck", 600001, 0, 3)
        agent = env.physics.world.agents[1]
        rng = np.random.default_rng(42)
        for _ in range(100):
            box_index = int(rng.integers(len(env._truth.obstacle_boxes)))
            x0, x1, y0, y1 = env._truth.obstacle_boxes[box_index]
            center = np.asarray((-6. + (x0 + x1) * .1,
                                 -6. + (y0 + y1) * .1))
            half = ((x1 - x0) * .1, (y1 - y0) * .1)
            position = center + rng.uniform(-1., 1., size=2)
            agent.set_pos(torch.tensor(position, dtype=torch.float32), batch_index=0)
            analytic = box_clearance(position, center, half,
                                     float(agent.shape.radius))[0] < 0.
            vmas = bool(env.physics.world.is_overlapping(
                agent, env.physics.world.landmarks[box_index + 1])[0].item())
            self.assertEqual(analytic, vmas)


if __name__ == "__main__":
    unittest.main()
