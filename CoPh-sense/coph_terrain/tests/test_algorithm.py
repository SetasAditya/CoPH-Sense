import unittest

import numpy as np

from coph_terrain.algorithm import (a4_teacher, acquire_to_evidence,
                                    delivery_memory_pair)
from coph_terrain.e2_dev_stack import acquisition_menu, execute_snapshot
from coph_terrain.environment import CoPHTerrainEnv
from coph_terrain.physical_audit import AuditChoice, advance_without_acquisition
from coph_terrain.planning import candidate_viewpoints
from coph_terrain.value_model import VariableSetValueNet


class AlgorithmBoundaryTests(unittest.TestCase):
    def test_post_observation_teacher_and_delivery_memory(self):
        seed = 100000
        env = CoPHTerrainEnv("open", seed, 0, seed + 9, executor="ph")
        advance_without_acquisition(env, 120)
        candidate = candidate_viewpoints(env.observations()["carrier"],
                                         env.config, "carrier",
                                         max_candidates=4)[0]
        reading = acquire_to_evidence(
            env, AuditChoice("carrier", candidate.modality,
                             candidate.viewpoint, candidate.target_region))
        self.assertIsNotNone(reading)
        self.assertIn(reading.evidence_id,
                      reading.env.observations()["carrier"]["owned_evidence_ids"])
        self.assertNotIn(reading.evidence_id,
                         reading.env.observations()["scout"]["received_evidence_ids"])
        teacher = a4_teacher(reading.env, reading.evidence_id)
        delivered = teacher["delivered_snapshot"]
        self.assertIn(reading.evidence_id,
                      delivered.observations()["scout"]["received_evidence_ids"])
        retained, withheld, recipient = delivery_memory_pair(
            reading.env, delivered, reading.evidence_id)
        self.assertEqual(recipient, "scout")
        self.assertEqual(retained.step_index, withheld.step_index)
        self.assertEqual(retained.ledger, withheld.ledger)
        self.assertTrue(all(np.array_equal(retained.positions[name],
                                           withheld.positions[name])
                            for name in ("scout", "carrier")))
        self.assertNotIn(reading.evidence_id,
                         withheld.observations()["scout"]["received_evidence_ids"])

    def test_pair_executes_both_measurements_before_sharing(self):
        seed = 100003
        env = CoPHTerrainEnv("open", seed, 0, seed + 9, executor="ph")
        advance_without_acquisition(env, 120)
        _, _, sets = acquisition_menu(env, "scout")
        pair_index = next(i for i, pair in enumerate(sets) if pair[1] >= 0)
        result = execute_snapshot(env, VariableSetValueNet(),
                                  VariableSetValueNet(),
                                  selected_index=pair_index,
                                  force_share_send=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["a4"]), 2)
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
