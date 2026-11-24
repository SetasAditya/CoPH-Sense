import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('baseline_primitives',Path(__file__).with_name('primitive_checks.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PrimitiveChecks(unittest.TestCase):
    def test_frozen_small_checks_pass_and_do_not_claim_native_reproduction(self):
        r=m.run()
        self.assertTrue(r['passed']);self.assertFalse(r['native_reproduction']);self.assertFalse(r['sota_comparison'])

    def test_tree_gbp_matches_dense_graph_marginals(self):
        cfg=json.loads((m.ROOT/'primitive_inputs.json').read_text())['distgp']
        r=m.gaussian_checks(cfg)
        self.assertTrue(r['converged']);self.assertLess(r['tree_gbp_mean_max_error'],1e-9)
        self.assertLess(r['tree_gbp_marginal_covariance_max_error'],1e-9)

    def test_astar_matches_dijkstra_and_objective_component_sum(self):
        cfg=json.loads((m.ROOT/'primitive_inputs.json').read_text())['exposure']
        r=m.exposure_checks(cfg)
        self.assertAlmostEqual(r['astar']['objective'],r['dijkstra']['objective'],places=9)
        self.assertAlmostEqual(r['astar']['objective'],r['astar']['component_sum'],places=9)
        self.assertLess(r['moments_quadrature_max_error'],1e-12)


if __name__=='__main__':unittest.main()
