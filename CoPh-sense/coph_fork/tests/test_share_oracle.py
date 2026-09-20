import unittest

from coph_fork.oracle import ExactOracleConfig
from coph_fork.share_oracle import ExactShareOracle, ShareState
from coph_fork.train_share_critic import build_features, build_features_v2

import numpy as np


G = "top_geometry"
T = "top_traction"
B = "bottom_geometry"


class ShareOracleA4Tests(unittest.TestCase):
    def test_both_good_messages_are_jointly_useful(self):
        oracle = ExactShareOracle(ShareState(acquired=((G, True), (T, True))))
        optimum, rows = oracle.solve()
        by_set = {row.message_set: row for row in rows}
        self.assertEqual(optimum.message_set, (G, T))
        self.assertGreater(by_set[()].total_cost, optimum.total_cost)
        self.assertGreater(by_set[(G,)].total_cost, by_set[()].total_cost)
        self.assertGreater(by_set[(T,)].total_cost, by_set[()].total_cost)
        self.assertEqual(optimum.receiver_decisions[G + "," + T], "top")

    def test_one_message_completes_acknowledged_complement(self):
        oracle = ExactShareOracle(
            ShareState(acquired=((T, True),), acknowledged=((G, True),))
        )
        self.assertEqual(oracle.solve()[0].message_set, (T,))

    def test_duplicate_acknowledged_evidence_is_not_sent(self):
        oracle = ExactShareOracle(
            ShareState(acquired=((G, True),), acknowledged=((G, True),))
        )
        self.assertEqual(oracle.solve()[0].message_set, ())

    def test_bad_evidence_with_safe_fallback_is_not_sent(self):
        oracle = ExactShareOracle(ShareState(acquired=((G, False), (T, True))))
        self.assertEqual(oracle.solve()[0].message_set, ())

    def test_stale_link_disables_message_value(self):
        oracle = ExactShareOracle(
            ShareState(acquired=((G, True), (T, True)), timely_probability=0.0)
        )
        self.assertEqual(oracle.solve()[0].message_set, ())
        expired = ExactShareOracle(
            ShareState(acquired=((G, True), (T, True)), deadline_open=False)
        )
        self.assertEqual([row.message_set for row in expired.solve()[1]], [()])

    def test_expensive_radio_suppresses_even_useful_pair(self):
        expensive = ExactOracleConfig(communication_attempt_cost=3.0)
        oracle = ExactShareOracle(ShareState(acquired=((G, True), (T, True))), expensive)
        self.assertEqual(oracle.solve()[0].message_set, ())

    def test_irrelevant_bottom_reading_is_not_sent(self):
        oracle = ExactShareOracle(ShareState(acquired=((B, True),)))
        self.assertEqual(oracle.solve()[0].message_set, ())

    def test_bad_warning_can_be_useful_under_declared_expected_cost_decision(self):
        config = ExactOracleConfig(
            decision_mode="expected_cost", top_safe_cost=1.0,
            bottom_safe_cost=3.0, unsafe_route_cost=5.0,
        )
        prior = {
            (False, True, True, True): 0.1,
            (True, True, True, True): 0.9,
        }
        oracle = ExactShareOracle(
            ShareState(acquired=((G, False),), timely_probability=1.0),
            config=config, prior=prior,
        )
        self.assertEqual(oracle.solve()[0].message_set, (G,))
        self.assertEqual(oracle.evaluate(()).receiver_decisions["none"], "top")

    def test_no_send_and_drop_have_same_receiver_history(self):
        state = ShareState(acquired=((G, True), (T, True)), timely_probability=0.0)
        oracle = ExactShareOracle(state)
        no_send = oracle.evaluate(())
        both_dropped = oracle.evaluate((G, T))
        self.assertEqual(no_send.receiver_decisions, both_dropped.receiver_decisions)
        self.assertEqual(no_send.expected_execution_cost, both_dropped.expected_execution_cost)
        self.assertGreater(both_dropped.expected_communication_cost, 0)

    def test_delivered_message_charges_evidence_and_ack_attempt(self):
        config = ExactOracleConfig()
        state = ShareState(acquired=((G, True),), timely_probability=1.0)
        value = ExactShareOracle(state, config).evaluate((G,))
        ack_bytes = config.header_bytes + config.ack_payload_bytes
        self.assertAlmostEqual(
            value.expected_communication_cost,
            config.packet_cost + config.communication_attempt_cost
            + config.byte_cost * ack_bytes,
        )
        self.assertEqual(
            value.expected_bytes,
            config.header_bytes + config.payload_bytes + ack_bytes,
        )

    def test_joint_prior_collision_is_resolved_by_v2_input(self):
        config = ExactOracleConfig(decision_mode="expected_cost")
        state = ShareState(acquired=((G, False),), timely_probability=1.0)
        low_joint = {
            (True, True, True, True): 0.5,
            (True, False, True, True): 0.25,
            (False, True, True, True): 0.25,
        }
        high_joint = {
            (True, True, True, True): 0.75,
            (False, False, True, True): 0.25,
        }
        self.assertTrue(np.array_equal(
            build_features(config, low_joint, state)[0],
            build_features(config, high_joint, state)[0],
        ))
        self.assertFalse(np.array_equal(
            build_features_v2(config, low_joint, state)[0],
            build_features_v2(config, high_joint, state)[0],
        ))
        self.assertNotEqual(
            ExactShareOracle(state, config, low_joint).solve()[0].message_set,
            ExactShareOracle(state, config, high_joint).solve()[0].message_set,
        )

    def test_v2_input_distinguishes_ack_price_with_same_evidence_price(self):
        state = ShareState(acquired=((G, True),), timely_probability=0.9)
        first = ExactOracleConfig(communication_attempt_cost=0.05, byte_cost=0.0)
        second = ExactOracleConfig(communication_attempt_cost=0.026, byte_cost=0.0005)
        self.assertAlmostEqual(first.packet_cost, second.packet_cost)
        self.assertTrue(np.array_equal(
            build_features(first, ExactShareOracle(state).prior, state)[0],
            build_features(second, ExactShareOracle(state).prior, state)[0],
        ))
        self.assertFalse(np.array_equal(
            build_features_v2(first, ExactShareOracle(state).prior, state)[0],
            build_features_v2(second, ExactShareOracle(state).prior, state)[0],
        ))


if __name__ == "__main__":
    unittest.main()
