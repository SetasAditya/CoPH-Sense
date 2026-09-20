import unittest

import numpy as np

from coph_terrain.evaluation import empirical_cvar, paired_hierarchical_bootstrap


class EvaluationTests(unittest.TestCase):
    def test_fractional_tail_mass(self):
        self.assertAlmostEqual(empirical_cvar([1, 2, 3], .5), (3 + .5 * 2) / 1.5)
        self.assertAlmostEqual(empirical_cvar([1, 2, 3], .95), 3.)

    def test_paired_bootstrap_recomputes_cvar_and_noninferiority(self):
        risk = np.ones((3, 10, 5, 2), dtype=float)
        risk[..., 1] = 2.
        success = np.ones_like(risk)
        result = paired_hierarchical_bootstrap(risk, success, 200, 1)
        self.assertAlmostEqual(result["risk_difference"], -1.)
        self.assertTrue(result["completion_noninferior_2pp"])
        self.assertTrue(result["risk_superior_if_completion_noninferior"])

    def test_completion_gate_blocks_safe_failure(self):
        risk = np.zeros((3, 10, 5, 2), dtype=float)
        risk[..., 1] = 1.
        success = np.ones_like(risk)
        success[..., 0] = 0.
        result = paired_hierarchical_bootstrap(risk, success, 200, 1)
        self.assertFalse(result["completion_noninferior_2pp"])
        self.assertFalse(result["risk_superior_if_completion_noninferior"])


if __name__ == "__main__":
    unittest.main()
