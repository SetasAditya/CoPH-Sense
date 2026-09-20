"""Information-boundary checks for the VMAS timing bridge."""

import unittest
from itertools import product

import numpy as np

import torch

from coph_fork.environment import ForkConfig, ForkWorld
from coph_fork.run_moving_bridge_a65 import (
    forecast_commit_slack, gate_snapshot, load_actors, run_episode,
)
from coph_fork.vmas_gate_data_a65 import GateContext, public_features
from coph_fork.vmas_gate_trajectory_a65 import actionability_features, reconnect_features


class MovingBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.a4, cls.gate = load_actors()

    def test_rollout_forecast_is_public_and_matches_uninformed_commit(self):
        config = ForkConfig(horizon=900, communication_delay_steps=100,
                            packet_drop_probability=0.0)
        safe = ForkWorld(top_geometry=True, top_traction=0.9,
                         bottom_geometry=True, bottom_traction=0.5)
        blocked = ForkWorld(top_geometry=False, top_traction=0.3,
                            bottom_geometry=True, bottom_traction=0.5)
        safe_state = gate_snapshot(safe, config, 1701)
        blocked_state = gate_snapshot(blocked, config, 1701)
        self.assertEqual(safe_state.observations(), blocked_state.observations())
        safe_forecast = forecast_commit_slack(safe_state)
        self.assertAlmostEqual(safe_forecast, forecast_commit_slack(blocked_state))
        uninformed = run_episode(safe, config, "never", 1701, self.a4, self.gate,
                                 starting_env=safe_state, rollout_forecast=True)
        self.assertAlmostEqual(safe_forecast,
                               uninformed["actual_commit_slack_from_gate"])

    def test_corrected_forecast_allows_precommit_delivery(self):
        config = ForkConfig(horizon=900, communication_delay_steps=100,
                            packet_drop_probability=0.0)
        world = ForkWorld(top_geometry=True, top_traction=0.9,
                          bottom_geometry=True, bottom_traction=0.5)
        state = gate_snapshot(world, config, 1701)
        result = run_episode(world, config, "always", 1701, self.a4, self.gate,
                             starting_env=state, rollout_forecast=True)
        self.assertEqual(result["delivered_before_commit"], 2)
        self.assertTrue(all(event["step"] < result["actual_commit_step"]
                            for event in result["evidence_deliveries"]
                            if event["usable_before_decision"]))

    def test_gate_features_do_not_reveal_hidden_top_world(self):
        context = GateContext(
            key="boundary", split="test", bottom_traction_mean=0.3,
            sensing_cost=0.08, delay_steps=32,
            delivery_probability=0.65, communication_range=0.7,
            safe_top_prior=0.4, carrier_start_y=0.04,
            scout_start_y=0.4, bottom_route_y=-0.65,
            carrier_top_y=0.48,
        )
        features = []
        reconnect = []
        observations = []
        for geometry, traction in product((False, True), (0.3, 0.9)):
            world = ForkWorld(top_geometry=geometry, top_traction=traction,
                              bottom_geometry=True, bottom_traction=0.3)
            snapshot = gate_snapshot(world, context.config(), 1701)
            features.append(public_features(context, snapshot))
            reconnect.append(reconnect_features(snapshot))
            observations.append(snapshot.observations()["scout"])
        for item in observations[1:]:
            self.assertEqual(item, observations[0])
        for item in features[1:]:
            np.testing.assert_allclose(item, features[0], atol=1e-6)
        for item in reconnect[1:]:
            np.testing.assert_allclose(item, reconnect[0], atol=1e-6)

    def test_actionability_deadline_is_public_and_precedes_commitment(self):
        config = ForkConfig(horizon=900, communication_delay_steps=74,
                            packet_drop_probability=0.0,
                            carrier_start_y=.05, scout_start_y=.4,
                            bottom_route_y=-.65, carrier_top_y=.32)
        decisions = []
        for geometry, traction in ((True, .9), (False, .3)):
            world = ForkWorld(top_geometry=geometry, top_traction=traction,
                              bottom_geometry=True, bottom_traction=.15)
            snapshot = gate_snapshot(world, config, 1701)
            decisions.append(actionability_features(snapshot))
        np.testing.assert_allclose(decisions[0], decisions[1], atol=1e-6)
        latest, lead, margin, feasible = decisions[0]
        self.assertEqual(feasible, 1.)
        self.assertGreater(lead, 0.)
        self.assertLess(latest, forecast_commit_slack(snapshot))
        self.assertLess(margin, 0.)


if __name__ == "__main__":
    unittest.main()
