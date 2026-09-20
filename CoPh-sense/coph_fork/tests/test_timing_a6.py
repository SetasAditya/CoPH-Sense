import unittest
from pathlib import Path

import torch

from coph_fork.memory_critic_a5 import MemorySetCritic
from coph_fork.run_memory_a5 import load_models
from coph_fork.timing_a6 import ExactTimingEvaluator, TimingContext
from coph_fork.train_timing_gate_a6 import ScoutTimingGate, features


class TimingA6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, a4 = load_models(1701, 1801)
        a5 = MemorySetCritic("additive")
        a5.load_state_dict(torch.load(
            Path(__file__).resolve().parents[1] / "results" / "a5_memory_v2" /
            "additive_1701.pth",
            map_location="cpu", weights_only=True
        ))
        a5.eval()
        cls.evaluator = ExactTimingEvaluator(a4, a5)
        gate = ScoutTimingGate()
        gate.load_state_dict(torch.load(
            Path(__file__).resolve().parents[1] / "results" / "a6_timing" /
            "scout_timing_gate.pth", map_location="cpu", weights_only=True
        ))
        gate.eval()
        cls.gate = gate

    def test_packet_at_commitment_cannot_change_route(self):
        near = TimingContext(network_delay=0.34, decision_slack=0.4,
                             delivery_probability=1.0)
        equal = TimingContext(network_delay=0.35, decision_slack=0.4,
                              delivery_probability=1.0)
        self.assertTrue(near.timely)
        self.assertFalse(equal.timely)
        before = self.evaluator.evaluate(near, True, return_branches=True)
        after = self.evaluator.evaluate(equal, True, return_branches=True)
        early_world = [row for row in before["branches"] if row["world"] == [True] * 4][0]
        late_world = [row for row in after["branches"] if row["world"] == [True] * 4][0]
        self.assertEqual(len(early_world["delivered_before_commit"]), 2)
        self.assertEqual(late_world["delivered_before_commit"], [])
        self.assertLess(early_world["route_cost"], late_world["route_cost"])

    def test_exact_gate_switches_from_inspect_to_skip(self):
        early = TimingContext(network_delay=0.1, decision_slack=0.4)
        late = TimingContext(network_delay=0.5, decision_slack=0.4)
        self.assertEqual(self.evaluator.gate_oracle(early)["oracle_choice"], "inspect")
        self.assertEqual(self.evaluator.gate_oracle(late)["oracle_choice"], "skip")
        self.assertTrue(self.gate.chooses_inspect(early))
        self.assertFalse(self.gate.chooses_inspect(late))

    def test_learned_gate_uses_public_timing_features(self):
        early = TimingContext(network_delay=0.1, decision_slack=0.4)
        late = TimingContext(network_delay=0.5, decision_slack=0.4)
        self.assertEqual(features(early).shape, (12,))
        self.assertNotEqual(float(features(early)[9]), float(features(late)[9]))

    def test_range_return_consumes_slack(self):
        short = TimingContext(communication_range=0.6, network_delay=0.0,
                              decision_slack=0.4)
        long = TimingContext(communication_range=1.0, network_delay=0.0,
                             decision_slack=0.4)
        self.assertAlmostEqual(short.return_distance, 0.4)
        self.assertFalse(short.timely)
        self.assertTrue(long.timely)

    def test_cost_ledger_and_zero_delivery_probability(self):
        context = TimingContext(network_delay=0.1, delivery_probability=0.0)
        report = self.evaluator.evaluate(context, True)
        components = sum(report[name] for name in (
            "expected_scout_sensing_cost", "expected_return_effort_cost",
            "expected_waiting_cost", "expected_communication_cost",
            "expected_carrier_sensing_cost", "expected_route_cost"
        ))
        self.assertAlmostEqual(report["expected_team_cost"], components)
        self.assertEqual(report["expected_late_delivered_packets"], 0)


if __name__ == "__main__":
    unittest.main()
