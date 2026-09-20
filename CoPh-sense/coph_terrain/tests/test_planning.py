import unittest

import numpy as np

from coph_terrain.environment import CoPHTerrainEnv, cell_to_position
from coph_terrain.environment import position_to_cell
from coph_terrain.planning import candidate_viewpoints, plan_path
from coph_terrain.physical_audit import AuditChoice, advance_without_acquisition, run_continuation
from coph_terrain.rich_resource_audit import public_choices
from coph_terrain.value_decomposition_audit import _menu_coverage


class PlanningTests(unittest.TestCase):
    def test_candidates_and_features_ignore_hidden_world(self):
        first = CoPHTerrainEnv("open", 100000, 0, 99)
        second = CoPHTerrainEnv("open", 100000, 0, 99)
        # Change unseen terrain in one simulator while preserving the actor's
        # current local history. Candidate generation has no simulator input.
        second._truth.traction[40:50, 40:50] = .15
        second._truth.surface[40:45, 40:45] = 1.
        a = candidate_viewpoints(first.observations()["carrier"], first.config,
                                 "carrier")
        b = candidate_viewpoints(second.observations()["carrier"], second.config,
                                 "carrier")
        self.assertEqual(a, b)
        self.assertEqual([x.features() for x in a], [x.features() for x in b])
        self.assertGreaterEqual(len(a), 8)
        self.assertLessEqual(len(a), 16)

    def test_planner_uses_observed_blockers(self):
        env = CoPHTerrainEnv("open", 100000, 0, 99)
        obs = env.observations()["carrier"]
        path, _ = plan_path(obs)
        self.assertTrue(path)
        blocked = path[len(path) // 2]
        obs["known_geometry"][blocked] = 1.
        alternate, _ = plan_path(obs)
        self.assertNotIn(blocked, alternate)

    def test_mission_integrated_poses_are_on_each_agents_route(self):
        env = CoPHTerrainEnv("bottleneck", 780001, 0, 99)
        obs = env.observations()
        for agent in ("scout", "carrier"):
            path, _ = plan_path(obs[agent])
            order = {cell: index for index, cell in enumerate(path)}
            candidates = candidate_viewpoints(obs[agent], env.config, agent,
                                              max_candidates=8)
            self.assertTrue(candidates)
            for item in candidates:
                self.assertIn(item.viewpoint, order)
                self.assertIn(item.target_region, order)
                if item.modality == "geometry":
                    self.assertLess(order[item.viewpoint],
                                    order[item.target_region])

    def test_region_proposals_offer_colocated_modalities(self):
        env = CoPHTerrainEnv("bottleneck", 780001, 0, 99)
        candidates = candidate_viewpoints(env.observations()["carrier"],
                                          env.config, "carrier", max_candidates=8)
        by_region = {}
        for candidate in candidates:
            by_region.setdefault(candidate.target_region, set()).add(candidate.modality)
        self.assertLessEqual(len(by_region), 4)
        self.assertTrue(by_region)
        self.assertTrue(all(modalities == {"geometry", "traction"}
                            for modalities in by_region.values()))
        for candidate in candidates:
            if candidate.modality == "geometry":
                footprint = env._cells_in_range(
                    cell_to_position(candidate.viewpoint), env.config.scan_range_carrier)
                self.assertIn(candidate.target_region, footprint)
            else:
                self.assertEqual(candidate.viewpoint, candidate.target_region)

    def test_planner_can_exit_inflated_margin(self):
        env = CoPHTerrainEnv("open", 100000, 0, 99)
        obs = env.observations()["carrier"]
        x, y = position_to_cell(obs["kinematics"][:2])
        obs["known_geometry"][x + 1, y] = 1.
        path, _ = plan_path(obs)
        self.assertGreater(len(path), 1)

    def test_sensing_detour_routes_around_public_walls(self):
        # The old straight-line detour on this excluded map stalled before
        # acquisition despite a public corridor route to the viewpoint.
        seed = 840034
        env = CoPHTerrainEnv("bottleneck", seed, 0, seed + 9,
                             executor="ph")
        advance_without_acquisition(env, 120)
        result = run_continuation(
            env, (AuditChoice("scout", "geometry", (26, 30)),), (0,))
        self.assertTrue(result["success"])
        self.assertEqual(result["measurements"], 1)
        self.assertEqual(result["delivered"], 1)

    def test_modality_specific_pair_physically_covers_target_region(self):
        seed = 860001
        env = CoPHTerrainEnv("bottleneck", seed, 0, seed + 9,
                             executor="ph")
        choices = public_choices(env)
        pair = next((i, j) for i, first in enumerate(choices)
                    for j, second in enumerate(choices) if i < j and
                    first.agent == second.agent == "carrier" and
                    first.target_region == second.target_region)
        self.assertNotEqual(choices[pair[0]].viewpoint,
                            choices[pair[1]].viewpoint)
        result = run_continuation(env, choices, pair)
        self.assertTrue(result["success"])
        self.assertEqual(result["measurements"], 2)
        self.assertEqual(result["target_region_covered"], [True, True])

    def test_point_probe_does_not_cover_neighbor_cell(self):
        env = CoPHTerrainEnv("open", 100000, 0, 99)
        choices = candidate_viewpoints(env.observations()["carrier"],
                                      env.config, "carrier", max_candidates=8)
        probe = next(item for item in choices if item.modality == "traction")
        neighbor = (probe.viewpoint[0] + 1, probe.viewpoint[1])
        self.assertFalse(_menu_coverage((probe,), neighbor, "traction", env))

    def test_scan_settles_before_dwell_and_covers_advertised_region(self):
        # Previously the scan began at speed and failed its completion-time
        # probe-radius check while the robot crossed the sensing pose.
        seed = 880000
        env = CoPHTerrainEnv("open", seed, 0, seed + 9, executor="ph")
        advance_without_acquisition(env, 120)
        choices = public_choices(env)
        result = run_continuation(env, choices, (0, 1))
        self.assertTrue(result["success"])
        self.assertEqual(result["measurements"], 2)
        self.assertEqual(result["target_region_covered"], [True, True])


if __name__ == "__main__":
    unittest.main()
