import unittest
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from coph_terrain.conditional_scout_material import (
    DispatchHistory, FiniteTerrainPrior, candidate_scores,
    dispatch_candidates, posterior_support)
from coph_terrain.environment import AGENTS, CoPHTerrainEnv, TerrainAction, TerrainConfig
from coph_terrain.environment import position_to_cell
from coph_terrain.generator import realize_map
from coph_terrain.run_conditional_scout_material import gate_decision


def _env(seed=7101, horizon=1300):
    return CoPHTerrainEnv(
        "open", seed, 0, seed=seed+1,
        config=TerrainConfig(carrier_primary=True, horizon_steps=horizon),
        executor="vmas", device="cpu")


class ConstantModel(nn.Module):
    def forward(self, features):
        return torch.ones(features.shape[0])


class ConditionalScoutMaterialTests(unittest.TestCase):
    def test_dispatch_features_ignore_hidden_and_private_scout_material(self):
        env = _env()
        before = [(c.candidate_id, c.features) for c in dispatch_candidates(env)]
        alternative = realize_map(env.parent_map, 99991)
        env._truth = replace(env._truth, surface=alternative.surface,
                             traction=alternative.traction)
        env.known_traction["scout"][10, 10] = .123
        env.known_surface["scout"][11, 11] = .987
        after = [(c.candidate_id, c.features) for c in dispatch_candidates(env)]
        self.assertEqual(before, after)

    def test_finite_prior_replays_actual_hypothesis_and_counts_appearance_once(self):
        env = _env(7301)
        history = DispatchHistory.from_env(env)
        prior = FiniteTerrainPrior.build(env, [env.realization_seed, 991])
        support, audit = posterior_support(history, prior)
        seeds = [row["hypothesis"].realization_seed for row in support]
        self.assertIn(env.realization_seed, seeds)
        expected = int(np.isfinite(history.carrier_observation["appearance"]).sum())
        for row in support:
            self.assertEqual(row["audit"]["appearance_cell_count"], expected)
        self.assertGreater(audit["effective_sample_size"], 0.)

    def test_incompatible_exact_carrier_measurement_is_rejected(self):
        env = _env(7351)
        cell = position_to_cell(env.positions["carrier"])
        env.step({"scout": TerrainAction(),
                  "carrier": TerrainAction(sense=("traction", *cell))})
        for _ in range(env.config.probe_dwell_steps-1):
            env.step({name: TerrainAction() for name in AGENTS})
        history = DispatchHistory.from_env(env)
        prior = FiniteTerrainPrior.build(env, [env.realization_seed, 99881])
        support, audit = posterior_support(history, prior)
        self.assertIn(env.realization_seed,
                      [row["hypothesis"].realization_seed for row in support])
        self.assertNotIn(99881,
                         [row["hypothesis"].realization_seed for row in support])
        rejected_reasons = {row["reason"] for row in audit["rejected"]}
        self.assertTrue(rejected_reasons <= {"carrier_measurement", "public_motion"})

    def test_carrier_arrival_does_not_erase_outstanding_scout(self):
        env = _env(7201, 120); env.scout_ever_dispatched = True
        env.physics.world.agents[1].state.pos[0] = env.physics.world.agents[1].state.pos.new_tensor([4.9, .1])
        env.physics.world.agents[0].state.pos[0] = env.physics.world.agents[0].state.pos.new_tensor([-4.9, -.5])
        env.step({name: TerrainAction(waypoint=tuple(env.positions[name])) for name in AGENTS})
        self.assertFalse(env.success)

    def test_idle_score_is_exactly_zero(self):
        candidates = dispatch_candidates(_env(7401))
        scores = candidate_scores(ConstantModel(), candidates)
        self.assertEqual(scores[0], 0.)
        for candidate in candidates[1:]:
            self.assertGreaterEqual(candidate.task.acquisition_dwell_steps, 1)
            self.assertGreaterEqual(candidate.task.estimated_delivery_cost, 0.)
            self.assertEqual(candidate.task.provenance, "carrier_public_belief")

    def test_common_packet_schedule_is_deterministic_and_exhaustion_fails(self):
        env = _env(7501); clone = env.clone()
        env.set_packet_random_schedule([.2]); clone.set_packet_random_schedule([.2])
        self.assertEqual(env._packet_dropped(), clone._packet_dropped())
        with self.assertRaises(RuntimeError): env._packet_dropped()

    def test_complete_gate_cannot_pass_on_metrics_without_causality(self):
        metrics = {"mean_regret": 0., "exact_rate": 1., "useful_recall": 1.,
                   "useful_denominator": 10, "unnecessary_rejection": 1.,
                   "unnecessary_denominator": 10}
        gate = gate_decision(metrics, {"useful": 2, "unnecessary": 2},
                             {"useful": True, "known": True, "too_late": True},
                             {"passed": False, "delivered": {"recovered": True},
                              "withheld": {"recovered": True}}, True)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["causal_delivery"])
