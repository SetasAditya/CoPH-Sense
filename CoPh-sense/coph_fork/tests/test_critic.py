import unittest

import torch

from coph_fork.critic import SetCostCritic, contrast, parameter_count
from coph_fork.train_critic import canonical_a2_state


class SetCriticA3Tests(unittest.TestCase):
    def setUp(self):
        state, candidates, _, _ = canonical_a2_state()
        self.state = torch.tensor(state[None, :])
        self.candidates = torch.tensor(candidates[None, :, :])

    def test_all_critics_predict_all_subsets_and_three_components(self):
        for kind in ("additive", "deepsets", "pairwise"):
            model = SetCostCritic(self.state.shape[-1], self.candidates.shape[-1], kind)
            prediction = model(self.state, self.candidates)
            self.assertEqual(tuple(prediction.shape), (1, 16, 3))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertLess(parameter_count(model), 200_000)

    def test_additive_model_cannot_represent_pair_contrast(self):
        model = SetCostCritic(self.state.shape[-1], self.candidates.shape[-1], "additive")
        predicted_cost = model(self.state, self.candidates).sum(-1)
        self.assertAlmostEqual(float(contrast(predicted_cost)[0]), 0.0, places=5)

    def test_pair_head_inputs_are_symmetric(self):
        torch.manual_seed(14)
        model = SetCostCritic(self.state.shape[-1], self.candidates.shape[-1], "pairwise")
        context = model.state_encoder(self.state)
        candidates = model.candidate_encoder(self.candidates)
        first, second = candidates[:, 0], candidates[:, 1]

        def pair_features(a, b):
            return torch.cat((context, a + b, torch.abs(a - b), a * b), dim=-1)

        direct = model.pair(pair_features(first, second))
        swapped = model.pair(pair_features(second, first))
        self.assertTrue(torch.equal(direct, swapped))


if __name__ == "__main__":
    unittest.main()
