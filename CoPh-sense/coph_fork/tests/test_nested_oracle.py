import unittest

import numpy as np
import torch

from coph_fork.critic import SUBSETS
from coph_fork.nested_oracle import nested_values, pair_ordering
from coph_fork.oracle import ExactOracleConfig, default_prior


class NeverSend(torch.nn.Module):
    def forward(self, state, candidates):
        values = torch.ones((len(state), len(SUBSETS), 3))
        values[:, 0, :] = 0
        return values


class NestedA3A4Tests(unittest.TestCase):
    def test_exact_continuation_gap_is_nonnegative(self):
        q_star, q_never = nested_values(ExactOracleConfig(), default_prior(), NeverSend())
        self.assertTrue(np.all(q_never >= q_star - 1e-10))
        self.assertAlmostEqual(q_star[0], 3.0)

    def test_canonical_pair_ordering_requires_useful_sharing(self):
        q_star, q_never = nested_values(ExactOracleConfig(), default_prior(), NeverSend())
        self.assertTrue(pair_ordering(q_star))
        self.assertFalse(pair_ordering(q_never))
        self.assertEqual(int(q_never.argmin()), SUBSETS.index(()))

    def test_nested_components_close_to_total(self):
        q_star, q_never, star_parts, never_parts = nested_values(
            ExactOracleConfig(), default_prior(), NeverSend(), return_components=True
        )
        self.assertTrue(np.allclose(q_star, star_parts.sum(-1)))
        self.assertTrue(np.allclose(q_never, never_parts.sum(-1)))


if __name__ == "__main__":
    unittest.main()
