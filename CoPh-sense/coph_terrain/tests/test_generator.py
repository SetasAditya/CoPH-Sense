import unittest

import numpy as np

from coph_terrain.generator import FAMILIES, generate_map, realize_map, _reachable
from coph_terrain.manifests import ALLOCATIONS


class TerrainGeneratorTests(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        a = generate_map("open", 123)
        b = generate_map("open", 123)
        c = generate_map("open", 124)
        self.assertEqual(a.digest(), b.digest())
        self.assertNotEqual(a.digest(), c.digest())

    def test_all_families_reachable_and_boxes_match_grid(self):
        for family in FAMILIES:
            for seed in (100, 101, 102):
                with self.subTest(family=family, seed=seed):
                    world = generate_map(family, seed)
                    self.assertTrue(_reachable(world.occupancy))
                    reconstructed = np.zeros_like(world.occupancy)
                    for x0, x1, y0, y1 in world.obstacle_boxes:
                        reconstructed[x0:x1, y0:y1] = True
                    np.testing.assert_array_equal(world.occupancy, reconstructed)
                    self.assertGreaterEqual(len(world.patch_metadata), 5)
                    self.assertLessEqual(len(world.patch_metadata), 10)
                    self.assertTrue(np.all((0 <= world.risk) & (world.risk <= 1)))

    def test_split_family_separation(self):
        training = set().union(*(set(ALLOCATIONS[k]) for k in
                                 ("train", "validation", "calibration", "test")))
        self.assertFalse(training & set(ALLOCATIONS["ood_test"]))

    def test_realizations_share_parent_geometry_but_change_terrain(self):
        parent = generate_map("labyrinth", 77)
        a = realize_map(parent, 0)
        b = realize_map(parent, 1)
        np.testing.assert_array_equal(a.occupancy, b.occupancy)
        self.assertEqual(a.obstacle_boxes, b.obstacle_boxes)
        self.assertFalse(np.array_equal(a.traction, b.traction))
        self.assertEqual(a.digest(), realize_map(parent, 0).digest())


if __name__ == "__main__":
    unittest.main()
