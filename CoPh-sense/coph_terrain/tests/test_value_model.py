import unittest

import torch

from coph_terrain.value_model import (
    RISK_QUANTILES, VariableSetValueNet, decision_score, enumerate_pairs,
    quantile_huber_loss,
)


class ValueModelTests(unittest.TestCase):
    def test_variable_candidates_and_permutation_invariance(self):
        torch.manual_seed(1)
        model = VariableSetValueNet()
        model.eval()
        maps = torch.zeros(1, 8, 60, 60)
        state = torch.zeros(1, 12)
        memory = torch.zeros(1, 3, 12)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        candidates = torch.randn(1, 4, 12)
        sets = enumerate_pairs(4)
        with torch.no_grad():
            mission, risk = model(maps, state, memory, mask, candidates, sets)
            score = decision_score(mission, risk)
            swapped = candidates[:, [1, 0, 2, 3]]
            changed, changed_risk = model(maps, state, memory, mask, swapped, sets)
        self.assertEqual(tuple(score.shape), (1, 11))
        self.assertTrue(torch.all(risk[..., 1:] >= risk[..., :-1]))
        self.assertTrue(torch.all((risk >= 0) & (risk <= 2)))
        # Set {0,1} has the same value when candidate order is swapped.
        self.assertTrue(torch.allclose(mission[:, 5], changed[:, 5], atol=1e-6))
        self.assertTrue(torch.allclose(risk[:, 5], changed_risk[:, 5], atol=1e-6))

    def test_distributional_loss_uses_individual_bounded_returns(self):
        count = len(RISK_QUANTILES)
        predicted = torch.linspace(.1, 1.9, count).reshape(1, 1, count).requires_grad_()
        returns = torch.tensor([[[.0, .3, 1.8, 2.0]]])
        loss = quantile_huber_loss(predicted, returns)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        with self.assertRaises(ValueError):
            quantile_huber_loss(predicted, torch.tensor([[[2.01]]]))


if __name__ == "__main__":
    unittest.main()
