import unittest
import numpy as np

from coph_terrain.conditional_scout_benchmark import default_cases
from coph_terrain.conditional_scout_long_horizon import (
    AGENT_RADIUS, SAFETY_MARGIN, START, DOCK_OFFSET,
    DiskObstacle, KinematicState, _case_obstacles, _docked_pose,
    _generate_scout_site, _project_feasible,
)


class ConditionalScoutGeometryTests(unittest.TestCase):
    def test_scout_starts_at_carrier_dock(self):
        q = _docked_pose(START)
        self.assertTrue(np.allclose(q, START + DOCK_OFFSET))
        self.assertLess(np.linalg.norm(q - START), 0.30)

    def test_hard_projection_respects_inflated_disk(self):
        obs = DiskObstacle(np.asarray([0.0, 0.0]), 0.5, "test")
        q, v, count = _project_feasible(
            np.asarray([0.0, 0.0]), np.asarray([-1.0, 0.0]), [obs]
        )
        required = obs.radius + AGENT_RADIUS + SAFETY_MARGIN
        self.assertGreaterEqual(np.linalg.norm(q - obs.center), required)
        self.assertGreater(count, 0)

    def test_generated_scout_site_is_ahead_and_clear(self):
        case = next(c for c in default_cases() if c.name == "moving_blocker_prediction")
        mode = next(m for m in case.modes if m.name == "clearing")
        q = np.asarray([-4.45, 0.0])
        v = np.asarray([0.55, 0.0])
        obs = _case_obstacles(case, mode, t=0.8)
        site, info = _generate_scout_site(q, v, "bottom", case, mode, 0.8, obs)
        fwd = v / np.linalg.norm(v)
        self.assertGreater(float(np.dot(site - q, fwd)), 0.75)
        self.assertGreater(info["forward_distance"], 0.75)
        self.assertGreater(info["clearance"], 0.0)

    def test_moving_blocker_is_physical_obstacle(self):
        case = next(c for c in default_cases() if c.name == "moving_blocker_prediction")
        mode = next(m for m in case.modes if m.name == "crossing")
        obs = _case_obstacles(case, mode, t=4.0)
        names = {o.name for o in obs}
        self.assertIn("moving_blocker", names)
        self.assertIn("patrol", names)
        self.assertIn("central_island", names)


if __name__ == "__main__":
    unittest.main()
