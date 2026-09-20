import unittest

import numpy as np

from coph_terrain.algorithm import (acquire_to_evidence,
                                    delivery_memory_pair, post_reading_branch)
from coph_terrain.execution import belief_risk_grid
from coph_terrain.lookahead import (LookaheadEnv, LookaheadSpec,
                                    SUMMARY_PAYLOAD_BYTES)
from coph_terrain.environment import TerrainAction
from coph_terrain.lookahead_policy import (choose_region_to_inspect,
                                           message_features,
                                           sender_estimated_novelty)
from coph_terrain.lookahead_recipient_a4 import (recipient_features,
                                                  response_features)
from coph_terrain.lookahead_a4_downstream import acquisition_continuation
from coph_terrain.lookahead_bidirectional import (NEED_PAYLOAD_BYTES,
                                                   NeedPacket,
                                                   compatible_need)
from coph_terrain.physical_audit import AuditChoice


class LookaheadMechanismTests(unittest.TestCase):
    def test_bidirectional_need_packet_is_fixed_and_causal(self):
        need = NeedPacket(1, "geometry", .2, .8, 4.5, 2.0, 7)
        wire = need.to_wire_bytes()
        self.assertEqual(len(wire), NEED_PAYLOAD_BYTES)
        decoded = NeedPacket.from_wire_bytes(wire)
        self.assertEqual((decoded.region, decoded.modality,
                          decoded.request_id), (1, "geometry", 7))
        innovation = type("Packet", (), {"region": 1,
                                          "modality": "geometry"})()
        self.assertTrue(compatible_need(innovation, decoded, 4.0))
        self.assertFalse(compatible_need(innovation, decoded, 4.5))

    def test_summary_is_distinct_from_raw_and_updates_only_receiver(self):
        env = LookaheadEnv(LookaheadSpec(980000))
        reading = acquire_to_evidence(
            env, AuditChoice("scout", "geometry", (15, 27), (18, 27)))
        self.assertIsNotNone(reading)
        before = reading.env
        packet_id = next(iter(before.innovation_packets))
        packet = before.innovation_packets[packet_id]
        self.assertEqual(packet.payload_bytes, SUMMARY_PAYLOAD_BYTES)
        self.assertEqual(len(packet.to_wire_bytes()), SUMMARY_PAYLOAD_BYTES)
        self.assertNotIn("support_cells", before.message_payload(packet_id))
        self.assertNotEqual(packet.evidence_id, reading.evidence_id)
        self.assertGreater(before.conditional_information_gain(
            "carrier", packet_id), 0.)
        self.assertNotIn(packet_id, before.received["carrier"])
        risk_before = belief_risk_grid(before.observations()["carrier"])
        scout_before = before.region_beliefs["scout"][(0, "geometry")].mean

        _, delivered = post_reading_branch(before, packet_id, True)
        self.assertIn(packet_id, delivered.received["carrier"])
        self.assertEqual(delivered.conditional_information_gain(
            "carrier", packet_id), 0.)
        self.assertAlmostEqual(delivered.region_beliefs["scout"]
                               [(0, "geometry")].mean, scout_before)
        self.assertGreater(delivered.region_beliefs["carrier"]
                           [(0, "geometry")].mean, .35)
        self.assertGreater(float(np.abs(
            belief_risk_grid(delivered.observations()["carrier"])[18, 27]
            - risk_before[18, 27])), .01)
        self.assertGreater(delivered.observations()["carrier"]
                           ["lookahead"]["effective_lookahead"],
                           delivered.observations()["carrier"]
                           ["lookahead"]["local_lookahead"])
        self.assertGreater(delivered.support_metrics()
                           ["carrier_new_received_cells"], 0)

    def test_duplicate_provenance_does_not_double_count(self):
        env = LookaheadEnv(LookaheadSpec(980000))
        reading = acquire_to_evidence(
            env, AuditChoice("scout", "geometry", (15, 27), (18, 27)))
        packet_id = next(iter(reading.env.innovation_packets))
        recipient = reading.env.clone()
        evidence = recipient.owned["scout"][packet_id]
        recipient._apply_evidence("carrier", evidence)
        once = recipient.region_beliefs["carrier"][(0, "geometry")].precision
        recipient._apply_evidence("carrier", evidence)
        self.assertEqual(once, recipient.region_beliefs["carrier"]
                         [(0, "geometry")].precision)
        self.assertEqual(recipient.conditional_information_gain(
            "carrier", packet_id), 0.)

    def test_replay_preserves_local_beliefs_and_motion(self):
        env = LookaheadEnv(LookaheadSpec(980000))
        env.step({"scout": TerrainAction(), "carrier": TerrainAction()})
        copy = LookaheadEnv.replay(env.replay_payload())
        self.assertEqual(copy.lookahead_spec, env.lookahead_spec)
        self.assertEqual(copy.ledger, env.ledger)
        for agent in ("scout", "carrier"):
            self.assertTrue(np.array_equal(copy.positions[agent],
                                           env.positions[agent]))

    def test_actor_features_do_not_read_private_receiver_belief(self):
        env = LookaheadEnv(LookaheadSpec(980000))
        reading = acquire_to_evidence(
            env, AuditChoice("scout", "geometry", (15, 27), (18, 27)))
        packet_id = next(iter(reading.env.innovation_packets))
        altered = reading.env.clone()
        altered.region_beliefs["carrier"][(0, "geometry")].add(999., 999.)
        self.assertGreater(abs(
            reading.env.conditional_information_gain("carrier", packet_id)
            - altered.conditional_information_gain("carrier", packet_id)), .01)
        np.testing.assert_array_equal(message_features(reading.env, packet_id),
                                      message_features(altered, packet_id))
        self.assertEqual(sender_estimated_novelty(reading.env, packet_id),
                         sender_estimated_novelty(altered, packet_id))
        np.testing.assert_array_equal(recipient_features(reading.env, packet_id),
                                      recipient_features(altered, packet_id))
        np.testing.assert_array_equal(response_features(reading.env, packet_id),
                                      response_features(altered, packet_id))
        response = response_features(reading.env, packet_id)
        self.assertEqual(response[-8], 0.)  # estimated HOLD: Region 0
        self.assertEqual(response[-7], 1.)  # estimated SEND: Region 1

    def test_delivered_memory_reallocates_next_physical_scan(self):
        env = LookaheadEnv(LookaheadSpec(
            980000, surface=(.95, .06), traction=(.20, .90)))
        reading = acquire_to_evidence(
            env, AuditChoice("scout", "geometry", (15, 27), (18, 27)))
        packet_id = next(iter(reading.env.innovation_packets))
        _, delivered = post_reading_branch(reading.env, packet_id, True)
        retained, withheld, recipient = delivery_memory_pair(
            reading.env, delivered, packet_id)
        self.assertEqual(recipient, "carrier")
        self.assertEqual(retained.ledger, withheld.ledger)
        self.assertEqual(choose_region_to_inspect(
            withheld, "carrier").target_region, (18, 27))
        next_choice = choose_region_to_inspect(retained, "carrier")
        self.assertEqual(next_choice.target_region, (18, 34))
        second = acquire_to_evidence(retained, next_choice)
        self.assertIsNotNone(second)
        self.assertTrue(any(packet.source == "carrier" and packet.region == 1
                            for packet in second.env.innovation_packets.values()))

    def test_a4_downstream_executes_both_receiver_choices(self):
        env = LookaheadEnv(LookaheadSpec(
            980000, surface=(.95, .06), traction=(.20, .90)))
        reading = acquire_to_evidence(
            env, AuditChoice("scout", "geometry", (15, 27), (18, 27)))
        packet_id = next(iter(reading.env.innovation_packets))
        held = acquisition_continuation(reading.env, packet_id, False)
        sent = acquisition_continuation(reading.env, packet_id, True)
        self.assertEqual(held["choice_region"], 0)
        self.assertEqual(sent["choice_region"], 1)
        self.assertTrue(held["acquisition_executed"])
        self.assertTrue(sent["acquisition_executed"])
        self.assertGreater(sent["radio"], held["radio"])


if __name__ == "__main__":
    unittest.main()
