import unittest

import numpy as np

from coph_fork.ph_factorial_controller import (
    ForceConfig, conservative_field, execute_step,
)
from coph_fork.ph_factorial_environment import FactorialEnv
from coph_fork.two_fork_environment import TwoForkConfig, TwoForkWorld


class _PublicEnv:
    class Config:
        dt = .05
        backup_y1 = -.72
        backup_y2 = -.50
        traction_safe_threshold = .55

    config = Config()

    def observations(self):
        evidence = {r: {"top": {"geometry": None, "traction": None}}
                    for r in (1, 2)}
        return {"carrier": {"evidence": evidence}}


class PHFactorialTests(unittest.TestCase):
    def test_public_field_uses_only_delivered_information(self):
        env = _PublicEnv()
        cfg = ForceConfig()
        q = np.array([-.8, .4])
        first = conservative_field(q, [-.1, .4], env, "carrier", cfg, .10)
        second = conservative_field(q, [-.1, .4], env, "carrier", cfg, .10)
        self.assertEqual(first[0], second[0])
        for key in first[1]:
            np.testing.assert_allclose(first[1][key], second[1][key])

    def test_hidden_world_does_not_change_uninformed_force(self):
        config = TwoForkConfig(horizon=5)
        clear = FactorialEnv(config=config, world=TwoForkWorld(geometry_2=True), seed=1)
        blocked = FactorialEnv(config=config, world=TwoForkWorld(geometry_2=False), seed=1)
        q = np.asarray([-.4, .43])
        a = conservative_field(q, [.4, .43], clear, "carrier", ForceConfig(), .10)
        b = conservative_field(q, [.4, .43], blocked, "carrier", ForceConfig(), .10)
        self.assertAlmostEqual(a[0], b[0])
        for key in a[1]:
            np.testing.assert_allclose(a[1][key], b[1][key])

    def test_ph_and_direct_obey_identical_speed_limit(self):
        env = _PublicEnv()
        for scheme in ("direct", "ph"):
            q, v, diagnostic = execute_step(
                [-1., -.6], [0., 0.], [1., -.6], env, "carrier",
                scheme, ForceConfig(), 1.8, .10, .55, 1.)
            self.assertLessEqual(np.linalg.norm(v), .55 + 1e-9)
            self.assertTrue(np.isfinite(q).all())
            self.assertTrue(np.isfinite(diagnostic["energy_after"]))

    def test_ipc_force_matches_potential_gradient_away_from_caps(self):
        env = _PublicEnv()
        cfg = ForceConfig()
        q = np.array([-.42, -.59])
        eps = 1e-5
        _, components = conservative_field(q, [-.4, -.4], env, "carrier", cfg, .10)
        for axis in range(2):
            shift = np.zeros(2)
            shift[axis] = eps
            plus = conservative_field(q + shift, [-.4, -.4], env, "carrier", cfg, .10)[0]
            minus = conservative_field(q - shift, [-.4, -.4], env, "carrier", cfg, .10)[0]
            numerical = -(plus - minus) / (2 * eps)
            predicted = sum(force[axis] for force in components.values())
            self.assertAlmostEqual(numerical, predicted, places=3)


if __name__ == "__main__":
    unittest.main()
