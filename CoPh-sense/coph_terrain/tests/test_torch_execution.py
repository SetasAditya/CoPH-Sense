import unittest

import numpy as np
import torch

from coph_terrain.environment import CoPHTerrainEnv
from coph_terrain.execution import belief_risk_grid, ph_step
from coph_terrain.torch_execution import ph_step_batch


class TorchExecutionTests(unittest.TestCase):
    def test_batched_kernel_matches_existing_ph_step(self):
        env = CoPHTerrainEnv("open", 100000, 0, 44)
        observation = env.observations()["carrier"]
        # Exercise obstacle barriers and a nonflat material field, not only
        # the free-space goal term used at the spawn point.
        observation["known_geometry"][6, 28] = 1.
        observation["known_geometry"][7, 29] = 1.
        observation["appearance"][4:10, 27:33] = np.linspace(
            .25, .85, 6, dtype=np.float32)[:, None]
        starts = np.asarray(((-5., -.25), (-4.8, -.2), (-4.6, -.3)), dtype=np.float64)
        velocities = np.asarray(((0., 0.), (.1, .02), (.2, -.01)))
        targets = np.asarray(((-4., -.25), (-3.8, -.2), (-3.6, -.3)))
        mass = np.full(3, 1.8)
        radius = np.full(3, .10)
        speed = np.full(3, .55)
        traction = np.full(3, .8)
        geometry = torch.tensor(np.repeat(observation["known_geometry"][None], 3, 0),
                                dtype=torch.float64)
        risk = torch.tensor(np.repeat(belief_risk_grid(observation)[None], 3, 0),
                            dtype=torch.float64)
        teammate = np.repeat((observation["kinematics"][:2]
                              + observation["kinematics"][4:6])[None], 3, axis=0)
        with torch.no_grad():
            q, v, diagnostics = ph_step_batch(
                torch.tensor(starts), torch.tensor(velocities),
                torch.tensor(targets), geometry, risk,
                torch.tensor(teammate, dtype=torch.float64),
                torch.tensor(mass), torch.tensor(radius), torch.tensor(speed),
                torch.tensor(traction), .05)
        for index in range(3):
            expected_q, expected_v, expected_d = ph_step(
                starts[index], velocities[index], targets[index], observation,
                mass[index], radius[index], speed[index], traction[index], .05)
            np.testing.assert_allclose(q[index].numpy(), expected_q, atol=1e-9)
            np.testing.assert_allclose(v[index].numpy(), expected_v, atol=1e-9)
            self.assertAlmostEqual(diagnostics["storage_after"][index].item(),
                                   expected_d["storage_after"], places=8)


if __name__ == "__main__":
    unittest.main()
