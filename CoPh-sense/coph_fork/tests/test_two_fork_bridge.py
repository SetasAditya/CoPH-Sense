"""Causal checks for the embodied two-fork information boundary."""

import unittest

import numpy as np

from coph_fork.two_fork_environment import (
    TwoForkAction, TwoForkEnv, TwoForkWorld,
)
from coph_fork.two_fork_memory import delivered_ledger
from coph_fork.run_moving_bridge_a65 import control, state
from coph_fork.run_memory_a5 import load_models
from coph_fork.run_two_fork_memory_bridge import (
    a4_message_ids, protocol_posterior, step,
)
from coph_fork.memory_a5 import EvidenceLedger
from coph_fork.two_fork_environment import TwoForkConfig
from coph_fork.search_two_fork_physical_family import Layout, staged_snapshot
import copy


class TwoForkBridgeTests(unittest.TestCase):
    def test_hidden_terrain_does_not_enter_initial_observation_or_memory(self):
        a = TwoForkEnv(world=TwoForkWorld(True, .9, True, .9), seed=19)
        b = TwoForkEnv(world=TwoForkWorld(False, .3, False, .3), seed=19)
        self.assertEqual(a.observations(), b.observations())
        self.assertEqual(delivered_ledger(a).records, [])
        self.assertEqual(delivered_ledger(b).records, [])
        for region in (1, 2):
            self.assertLess(float(np.linalg.norm(
                TwoForkEnv.probe_position(region, "top") -
                a.agent_positions["carrier"])), .5)

    def test_region_two_reading_is_owned_locally_after_real_acquisition(self):
        env = TwoForkEnv(world=TwoForkWorld(geometry_2=False), seed=19)
        viewpoint = TwoForkEnv.probe_position(2, "top")
        for _ in range(100):
            position, velocity = state(env, "carrier")
            if np.linalg.norm(position - viewpoint) < .15:
                break
            env.step({
                "scout": TwoForkAction(),
                "carrier": TwoForkAction(motion=control(
                    position, velocity, viewpoint, gain=2.8, damping=1.5)),
            })
        self.assertIsNone(env.route_commitments[1])
        _, _, _, _ = env.step({
            "scout": TwoForkAction(),
            "carrier": TwoForkAction(sense_region=2, sense_route="top",
                                     sense_modality="geometry"),
        })
        records = delivered_ledger(env).records
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0].region, records[0].modality,
                          records[0].value, records[0].source),
                         (2, "geometry", False, "carrier"))
        self.assertGreater(env.ledger.sensing, 0.)

    def test_public_a4_silence_is_included_in_posterior(self):
        _, a4 = load_models(1701, 1801)
        empty = EvidenceLedger()
        config = TwoForkConfig(packet_drop_probability=.5)
        posterior = protocol_posterior(empty, "persistent", a4, config)
        self.assertAlmostEqual(float(posterior.sum()), 1.)
        self.assertGreater(float(posterior.max() - posterior.min()), .01)
        np.testing.assert_allclose(
            protocol_posterior(empty, "independent", a4, config),
            np.full(16, 1 / 16))
        nonuniform = np.arange(1, 17, dtype=np.float64)
        nonuniform /= nonuniform.sum()
        np.testing.assert_allclose(
            protocol_posterior(empty, "independent", a4, config,
                               nonuniform), nonuniform)

    def test_unobserved_second_region_cannot_change_carrier_decision_input(self):
        _, a4 = load_models(1701, 1801)
        states = []
        for second in ((True, .9), (False, .3)):
            env = TwoForkEnv(world=TwoForkWorld(
                geometry_1=True, traction_1=.9,
                geometry_2=second[0], traction_2=second[1]), seed=19)
            viewpoint = TwoForkEnv.probe_position(1, "top")
            while np.linalg.norm(state(env, "scout")[0] - viewpoint) > .10:
                step(env, "probe_1", "hold")
            for modality in ("geometry", "traction"):
                step(env, "probe_1", "hold",
                     scout_sense=(1, "top", modality))
            for observation_id in a4_message_ids(env, a4, 1.):
                step(env, "probe_1", "hold", scout_send=observation_id)
            for _ in range(env.config.communication_delay_steps + 2):
                step(env, "probe_1", "hold")
            memory = delivered_ledger(env)
            states.append((env.observations()["carrier"], memory.records,
                           protocol_posterior(memory, "persistent", a4,
                                              env.config)))
        self.assertEqual(states[0][0], states[1][0])
        self.assertEqual(states[0][1], states[1][1])
        np.testing.assert_allclose(states[0][2], states[1][2])

    def test_matched_memory_intervention_preserves_physical_snapshot(self):
        env = staged_snapshot((True, True, False, True),
                              Layout(-.43, -.43, .3, .3), seed=19)
        no_memory = copy.deepcopy(env)
        no_memory.received["carrier"].clear()
        self.assertEqual(env.step_index, no_memory.step_index)
        self.assertEqual(env.ledger, no_memory.ledger)
        self.assertEqual(env.acknowledged, no_memory.acknowledged)
        self.assertEqual(env.route_commitments, no_memory.route_commitments)
        for agent in ("carrier", "scout"):
            np.testing.assert_allclose(env.agent_positions[agent],
                                       no_memory.agent_positions[agent])
        self.assertEqual(len(delivered_ledger(env).records), 2)
        self.assertEqual(len(delivered_ledger(no_memory).records), 0)


if __name__ == "__main__":
    unittest.main()
