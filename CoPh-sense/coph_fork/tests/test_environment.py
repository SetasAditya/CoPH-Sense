import json
import unittest

import numpy as np
import torch

from coph_fork import CoPHForkEnv, ForkAction, ForkConfig, ForkWorld


NAMES = ("scout", "carrier")


def idle():
    return {name: ForkAction() for name in NAMES}


class CoPHForkA1Tests(unittest.TestCase):
    def test_latent_terrain_is_not_in_actor_observation(self):
        first = CoPHForkEnv(world=ForkWorld(top_traction=0.9), seed=12)
        second = CoPHForkEnv(world=ForkWorld(top_traction=0.2), seed=12)
        self.assertEqual(first.observations(), second.observations())
        serialized = json.dumps(first.observations())
        self.assertNotIn("top_traction", serialized)
        self.assertNotIn("bottom_traction", serialized)

    def test_sensing_is_charged_and_location_gated(self):
        env = CoPHForkEnv(seed=2)
        action = idle()
        action["scout"] = ForkAction(sense_route="top", sense_modality="geometry")
        _, _, _, info = env.step(action)
        self.assertEqual(len(env.evidence["scout"]), 0)
        self.assertAlmostEqual(info["incremental_cost"]["sensing"], env.config.sensing_cost)
        self.assertAlmostEqual(
            info["incremental_cost"]["invalid_action"], env.config.invalid_action_cost
        )

        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        env.step(action)
        evidence = list(env.evidence["scout"].values())
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].value, env.world_spec.top_geometry)

    def test_packet_requires_owned_evidence_and_arrives_after_delay(self):
        env = CoPHForkEnv(seed=3)
        invalid = idle()
        invalid["scout"] = ForkAction(send_observation_id="future-reading")
        env.step(invalid)
        self.assertFalse(env.packet_queue)

        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        acquire = idle()
        acquire["scout"] = ForkAction(sense_route="top", sense_modality="traction")
        env.step(acquire)
        observation_id = next(iter(env.evidence["scout"]))
        send = idle()
        send["scout"] = ForkAction(send_observation_id=observation_id)
        env.step(send)
        self.assertNotIn(observation_id, env.received["carrier"])
        for _ in range(env.config.communication_delay_steps - 1):
            env.step(idle())
            self.assertNotIn(observation_id, env.received["carrier"])
        env.step(idle())
        self.assertIn(observation_id, env.received["carrier"])

    def test_acknowledgement_is_charged_and_arrives_only_after_its_delay(self):
        env = CoPHForkEnv(config=ForkConfig(communication_delay_steps=1), seed=33)
        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        action = idle()
        action["scout"] = ForkAction(sense_route="top", sense_modality="geometry")
        env.step(action)
        observation_id = next(iter(env.evidence["scout"]))
        action["scout"] = ForkAction(send_observation_id=observation_id)
        env.step(action)
        self.assertNotIn(observation_id, env.observations()["scout"]["acknowledged_observation_ids"])
        _, _, _, info = env.step(idle())
        self.assertIn(observation_id, env.received["carrier"])
        self.assertNotIn(observation_id, env.acknowledged["scout"])
        self.assertAlmostEqual(
            info["incremental_cost"]["communication"],
            env.config.communication_attempt_cost
            + env.config.byte_cost * (env.config.header_bytes + env.config.ack_payload_bytes),
        )
        env.step(idle())
        self.assertIn(observation_id, env.acknowledged["scout"])

    def test_zero_delay_acknowledgement_is_visible_in_same_transition(self):
        env = CoPHForkEnv(config=ForkConfig(communication_delay_steps=0), seed=34)
        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        action = idle()
        action["scout"] = ForkAction(sense_route="top", sense_modality="geometry")
        env.step(action)
        observation_id = next(iter(env.evidence["scout"]))
        action["scout"] = ForkAction(send_observation_id=observation_id)
        env.step(action)
        self.assertIn(observation_id, env.received["carrier"])
        self.assertIn(observation_id, env.acknowledged["scout"])

    def test_forced_evidence_drop_mask_only_controls_audit_packets(self):
        env = CoPHForkEnv(config=ForkConfig(communication_delay_steps=0,
                                           packet_drop_probability=0.5), seed=34)
        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        ids = []
        for modality in ("geometry", "traction"):
            action = idle()
            action["scout"] = ForkAction(sense_route="top", sense_modality=modality)
            env.step(action)
            ids.append(next(item for item in env.evidence["scout"] if item not in ids))
        env.forced_evidence_drops = (True, False)
        for observation_id in ids:
            action = idle()
            action["scout"] = ForkAction(send_observation_id=observation_id)
            env.step(action)
        self.assertNotIn(ids[0], env.received["carrier"])
        self.assertIn(ids[1], env.received["carrier"])

    def test_out_of_range_transmission_is_charged_but_not_queued(self):
        env = CoPHForkEnv(seed=31)
        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        acquire = idle()
        acquire["scout"] = ForkAction(sense_route="top", sense_modality="geometry")
        env.step(acquire)
        observation_id = next(iter(env.evidence["scout"]))
        env.physics.world.agents[1].set_pos(torch.tensor([0.90, -0.75]), batch_index=0)
        send = idle()
        send["scout"] = ForkAction(send_observation_id=observation_id)
        _, _, _, info = env.step(send)
        self.assertFalse(env.packet_queue)
        self.assertGreater(info["incremental_cost"]["communication"], 0)
        self.assertGreater(info["incremental_cost"]["invalid_action"], 0)

    def test_late_packet_is_delivered_but_marked_unusable(self):
        env = CoPHForkEnv(config=ForkConfig(communication_delay_steps=1), seed=4)
        env.physics.world.agents[0].set_pos(torch.tensor([-0.34, 0.43]), batch_index=0)
        acquire = idle()
        acquire["scout"] = ForkAction(sense_route="top", sense_modality="geometry")
        env.step(acquire)
        observation_id = next(iter(env.evidence["scout"]))
        env.physics.world.agents[1].set_pos(torch.tensor([-0.10, 0.30]), batch_index=0)
        send = idle()
        send["scout"] = ForkAction(send_observation_id=observation_id)
        env.step(send)
        env.step(idle())
        deliveries = [event for event in env.events if event["type"] == "packet_delivery"]
        self.assertTrue(deliveries[-1]["delivered"])
        self.assertFalse(deliveries[-1]["usable_before_decision"])

    def test_true_traction_changes_executed_progress(self):
        high = CoPHForkEnv(world=ForkWorld(top_traction=0.9), seed=5)
        low = CoPHForkEnv(world=ForkWorld(top_traction=0.2), seed=5)
        for env in (high, low):
            carrier = env.physics.world.agents[1]
            carrier.set_pos(torch.tensor([-0.05, 0.45]), batch_index=0)
            carrier.set_vel(torch.zeros(2), batch_index=0)
        action = idle()
        action["carrier"] = ForkAction(motion=(1.0, 0.0))
        for _ in range(5):
            high.step(action)
            low.step(action)
        high_x = high.agent_positions["carrier"][0]
        low_x = low.agent_positions["carrier"][0]
        self.assertGreater(high_x - low_x, 0.002)
        self.assertGreater(low.ledger.risk, high.ledger.risk)

    def test_hidden_geometry_changes_physical_collision(self):
        open_route = CoPHForkEnv(world=ForkWorld(top_geometry=True), seed=41)
        blocked_route = CoPHForkEnv(world=ForkWorld(top_geometry=False), seed=41)
        for env in (open_route, blocked_route):
            env.physics.world.agents[1].set_pos(torch.tensor([0.12, 0.47]), batch_index=0)
        _, _, _, open_info = open_route.step(idle())
        _, _, _, blocked_info = blocked_route.step(idle())
        self.assertEqual(open_info["collisions"], 0)
        self.assertGreater(blocked_info["collisions"], 0)

    def test_ledger_closes_and_reward_is_negative_increment(self):
        env = CoPHForkEnv(seed=6)
        _, reward, _, info = env.step(idle())
        self.assertAlmostEqual(info["incremental_total"], sum(info["incremental_cost"].values()))
        self.assertAlmostEqual(info["episode_total"], sum(info["episode_ledger"].values()))
        self.assertAlmostEqual(reward, -info["incremental_total"])

    def test_deadline_failure_is_terminal_and_charged(self):
        env = CoPHForkEnv(config=ForkConfig(horizon=1), seed=8)
        _, _, done, info = env.step(idle())
        self.assertTrue(done)
        self.assertFalse(env.success)
        self.assertEqual(
            info["incremental_cost"]["terminal_failure"], env.config.terminal_failure_cost
        )

    def test_replay_reproduces_states_events_and_ledger(self):
        env = CoPHForkEnv(
            config=ForkConfig(packet_drop_probability=0.25, horizon=20), seed=23
        )
        actions = []
        for step in range(12):
            actions.append(
                {
                    "scout": ForkAction(motion=(0.4, 0.15 if step < 5 else -0.1)),
                    "carrier": ForkAction(motion=(0.25, -0.05)),
                }
            )
        for action in actions:
            env.step(action)
        replayed = CoPHForkEnv.replay(env.replay_payload())
        self.assertEqual(env.state_trace, replayed.state_trace)
        self.assertEqual(env.events, replayed.events)
        self.assertEqual(env.replay_payload()["ledger"], replayed.replay_payload()["ledger"])
        self.assertEqual(env.replay_payload()["sha256"], replayed.replay_payload()["sha256"])


if __name__ == "__main__":
    unittest.main()
