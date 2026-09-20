import unittest

from coph_fork.oracle import (
    ExactAcquisitionOracle,
    ExactOracleConfig,
    VARIABLES,
)


class ExactOracleA2Tests(unittest.TestCase):
    def test_enumeration_counts_are_complete(self):
        report = ExactAcquisitionOracle().report()
        self.assertEqual(report["scope"]["worlds"], 4)
        self.assertEqual(report["scope"]["sensing_subsets"], 16)
        self.assertEqual(report["scope"]["sharing_policies"], 625)
        self.assertEqual(len(report["q_sense_star"]), 16)
        self.assertEqual(len(report["q_share_star"]), 16)
        self.assertEqual(len(report["q_share"]), 625)

    def test_no_sensing_value_is_known_safe_route(self):
        value = ExactAcquisitionOracle().evaluate_policy((), ())
        self.assertAlmostEqual(value.total_cost, 3.0)
        self.assertEqual(value.receiver_decisions, {"none": "bottom"})

    def test_joint_geometry_traction_is_valuable_but_singletons_are_not(self):
        oracle = ExactAcquisitionOracle()
        no_sense = oracle.best_share_policy(()).total_cost
        geometry = oracle.best_share_policy(("top_geometry",)).total_cost
        traction = oracle.best_share_policy(("top_traction",)).total_cost
        joint = oracle.best_share_policy(("top_geometry", "top_traction")).total_cost
        self.assertGreaterEqual(geometry, no_sense)
        self.assertGreaterEqual(traction, no_sense)
        self.assertLess(joint, no_sense)

    def test_named_value_categories(self):
        oracle = ExactAcquisitionOracle()
        baseline = oracle.best_share_policy(()).total_cost
        q_geometry = oracle.best_share_policy(("top_geometry",)).total_cost
        q_traction = oracle.best_share_policy(("top_traction",)).total_cost
        q_joint = oracle.best_share_policy(("top_geometry", "top_traction")).total_cost
        q_with_redundant = oracle.best_share_policy(
            ("top_geometry", "top_traction", "bottom_geometry")
        ).total_cost
        self.assertLess(q_joint, baseline, "joint sensing is beneficial")
        self.assertGreater(q_geometry, baseline, "geometry alone is harmful")
        self.assertGreater(q_traction, baseline, "traction alone is harmful")
        self.assertGreater(q_geometry + q_traction - q_joint - baseline, 0, "pair is complementary")
        self.assertAlmostEqual(
            q_with_redundant - q_joint,
            oracle.config.sensing_cost,
            places=10,
            msg="known-safe bottom geometry is decision-redundant but costs one probe",
        )

    def test_optimum_avoids_known_bottom_measurements(self):
        optimum, _, _ = ExactAcquisitionOracle().solve()
        self.assertEqual(set(optimum.sense_subset), {"top_geometry", "top_traction"})
        self.assertNotIn("bottom_geometry", optimum.sense_subset)
        self.assertNotIn("bottom_traction", optimum.sense_subset)

    def test_receiver_decision_depends_only_on_delivered_history(self):
        optimum, _, _ = ExactAcquisitionOracle().solve()
        for history in optimum.receiver_decisions:
            self.assertIsInstance(history, str)
        self.assertEqual(set(optimum.receiver_decisions.values()), {"bottom", "top"})

    def test_certified_oracle_requires_both_good_top_cues(self):
        optimum, _, _ = ExactAcquisitionOracle().solve()
        self.assertEqual(optimum.share_rules, ("if_good", "if_good"))
        for history, route in optimum.receiver_decisions.items():
            if route == "top":
                self.assertIn("top_geometry=1", history)
                self.assertIn("top_traction=1", history)

    def test_expected_cost_mode_is_labeled_and_can_use_selective_silence(self):
        config = ExactOracleConfig(decision_mode="expected_cost")
        optimum, _, _ = ExactAcquisitionOracle(config).solve()
        self.assertEqual(config.decision_mode, "expected_cost")
        self.assertNotEqual(optimum.share_rules, ("if_good", "if_good"))

    def test_history_probabilities_sum_to_one(self):
        oracle = ExactAcquisitionOracle()
        for value in oracle.share_q_table(VARIABLES):
            self.assertAlmostEqual(sum(value.delivered_history_probability.values()), 1.0)

    def test_worse_channel_cannot_improve_optimized_value(self):
        perfect = ExactAcquisitionOracle(
            ExactOracleConfig(timely_delivery_probability=1.0)
        ).solve()[0]
        lossy = ExactAcquisitionOracle(
            ExactOracleConfig(timely_delivery_probability=0.5)
        ).solve()[0]
        self.assertGreaterEqual(lossy.total_cost + 1e-12, perfect.total_cost)

    def test_repeated_solution_is_exactly_deterministic(self):
        first = ExactAcquisitionOracle().report()
        second = ExactAcquisitionOracle().report()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
