import unittest

import numpy as np

from coph_fork.memory_a5 import CANDIDATES
from coph_fork.memory_critic_a5 import WORLDS, action_costs, encode_memory, posterior_for
from coph_fork.oracle import ExactOracleConfig
from coph_fork.train_memory_a5 import ledger_from_pattern, sufficiency_audit


class MemoryCriticA5Tests(unittest.TestCase):
    def test_full_posterior_conditions_only_on_delivered_evidence(self):
        empty = posterior_for(ledger_from_pattern((-1, -1, -1, -1)))
        delivered = posterior_for(ledger_from_pattern((1, -1, -1, -1)))
        self.assertTrue(np.allclose(empty, 1 / 16))
        self.assertEqual(np.count_nonzero(delivered), 8)
        self.assertTrue(np.allclose(delivered[delivered > 0], 1 / 8))

    def test_known_region_one_pair_is_costly_and_region_two_is_best(self):
        config = ExactOracleConfig()
        memory = ledger_from_pattern((1, 1, -1, -1))
        state, candidates, posterior = encode_memory(memory, config)
        self.assertEqual(state.shape, (28,))
        self.assertEqual(candidates.shape, (4, 9))
        costs = action_costs(posterior, memory, config)
        self.assertEqual(CANDIDATES[int(costs.argmin())],
                         ((2, "geometry"), (2, "traction")))
        self.assertAlmostEqual(float(costs[3] - costs[0]), 0.08, places=6)
        self.assertAlmostEqual(float(costs[6] - costs[0]), -0.42, places=6)

    def test_no_conflicting_exact_labels_for_finite_memory_states(self):
        audit = sufficiency_audit()
        self.assertEqual(audit["patterns_checked"], 81)
        self.assertEqual(audit["conflicting_exact_labels"], 0)

    def test_same_marginals_but_different_joint_prior_change_value(self):
        memory = ledger_from_pattern((1, 1, 1, -1))
        independent = posterior_for(memory)
        correlated_prior = np.asarray([
            1.0 if world[2] == world[3] else 0.1 for world in WORLDS
        ])
        correlated_prior /= correlated_prior.sum()
        correlated = posterior_for(memory, correlated_prior)
        config = ExactOracleConfig()
        self.assertAlmostEqual(sum(correlated_prior[i] for i, w in enumerate(WORLDS) if w[2])
                               / correlated_prior.sum(), 0.5)
        self.assertNotAlmostEqual(float(action_costs(independent, memory, config)[5]),
                                  float(action_costs(correlated, memory, config)[5]))


if __name__ == "__main__":
    unittest.main()
