import unittest

from coph_terrain.conditional_scout_benchmark import (
    default_cases, evaluate_case, run_suite,
)


class ConditionalScoutBenchmarkTests(unittest.TestCase):
    def test_selective_dispatch_pattern(self):
        rows = {row["name"]: row for row in run_suite()}
        self.assertTrue(rows["static_hidden_block_useful"]["dispatch"])
        self.assertFalse(rows["static_local_known_no_need"]["dispatch"])
        self.assertFalse(rows["static_hidden_block_too_late"]["dispatch"])
        self.assertTrue(rows["moving_blocker_prediction"]["dispatch"])
        self.assertTrue(rows["moving_cost_prediction"]["dispatch"])
        self.assertFalse(rows["moving_blocker_stale_report"]["dispatch"])

    def test_useful_cases_require_parallel_remote_access(self):
        for case in default_cases():
            row = evaluate_case(case)
            if row["dispatch"]:
                self.assertFalse(row["single_agent_can_query"])
                self.assertTrue(row["timely"])
                self.assertLess(row["conditional_cost"], row["idle_cost"])

    def test_unnecessary_or_late_dispatch_is_not_rewarded(self):
        rows = {row["name"]: row for row in run_suite()}
        self.assertLessEqual(rows["static_local_known_no_need"]["scout_value"], 0.)
        self.assertLessEqual(rows["static_hidden_block_too_late"]["scout_value"], 0.)
        self.assertLessEqual(rows["moving_blocker_stale_report"]["scout_value"], 0.)


if __name__ == "__main__":
    unittest.main()
