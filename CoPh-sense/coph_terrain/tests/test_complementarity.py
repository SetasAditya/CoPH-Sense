import unittest
from unittest.mock import patch

import numpy as np

from coph_terrain.algorithm import delivery_memory_pair
from coph_terrain.complementarity import (ChallengeSpec, ComplementarityEnv,
                                          structural_acquisition_values)
from coph_terrain.environment import TerrainAction, TerrainEvidence


class ComplementarityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.env = ComplementarityEnv(
            ChallengeSpec(970000, (True, False), (False, False)))

    def test_region_inference_and_withheld_memory(self):
        before = self.env.clone()
        evidence = TerrainEvidence("obs-000000", "scout", "geometry",
                                   ((10, 30),), (.95,), 0, (0., 0.))
        self.env.owned["scout"][evidence.evidence_id] = evidence
        delivered = self.env.clone()
        delivered.received["carrier"][evidence.evidence_id] = evidence
        delivered._apply_evidence("carrier", evidence)
        self.assertTrue(np.all(delivered.known_surface["carrier"][9:27, 23:37]
                               == .95))
        retained, withheld, _ = delivery_memory_pair(
            self.env, delivered, evidence.evidence_id)
        self.assertTrue(np.all(np.isnan(
            withheld.known_surface["carrier"][9:27, 23:37])))
        self.assertTrue(np.all(retained.known_surface["carrier"][9:27, 23:37]
                               == .95))
        self.assertEqual(retained.ledger, withheld.ledger)
        self.assertTrue(np.array_equal(
            before.positions["carrier"], withheld.positions["carrier"]))

    def test_challenge_replay_preserves_spec_and_physics(self):
        self.env.step({"scout": TerrainAction(),
                       "carrier": TerrainAction()})
        copy = ComplementarityEnv.replay(self.env.replay_payload())
        self.assertEqual(copy.challenge_spec, self.env.challenge_spec)
        self.assertEqual(copy.ledger, self.env.ledger)
        self.assertTrue(all(np.array_equal(copy.positions[name],
                                           self.env.positions[name])
                            for name in ("scout", "carrier")))

    def test_admission_averages_hidden_worlds_not_realized_outcome(self):
        costs = ((1., 1.1, 1.1, 1.2),
                 (1., 1.1, 1.1, 1.2),
                 (1., 1.1, 1.1, 1.2),
                 (2., 1.9, 1.9, 1.1))
        calls = 0

        def fake_run(_env, _choices, selected, **kwargs):
            nonlocal calls
            self.assertTrue(kwargs["hold_carrier_until_delivery"])
            world, action = divmod(calls, 4)
            calls += 1
            self.assertEqual(selected, ((), (0,), (1,), (0, 1))[action])
            return {"score_single_world": costs[world][action],
                    "success": True,
                    "target_region_covered": [True, True] if action == 3 else [],
                    "delivered": 2 if action == 3 else action,
                    "carrier_cells_at_delivery": [(5, 30), (6, 30)]
                    if action == 3 else []}

        with patch("coph_terrain.complementarity.run_continuation", fake_run):
            result = structural_acquisition_values(self.env)
        self.assertEqual(calls, 16)
        self.assertTrue(result["common_initial_information"])
        self.assertAlmostEqual(result["costs"]["skip"], 1.25)
        self.assertAlmostEqual(result["costs"]["GT"], 1.175)
        self.assertTrue(result["admitted"])


if __name__ == "__main__":
    unittest.main()
