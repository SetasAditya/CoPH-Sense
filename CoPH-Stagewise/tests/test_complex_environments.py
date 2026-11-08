import unittest
import numpy as np
from coph_stagewise.complex_environments import scene,SCENES,ARMS

class ComplexEnvironmentTests(unittest.TestCase):
    def test_frozen_scenes_horizons_and_initial_clearance(self):
        for name in SCENES:
            for seed in (61000,61001):
                s=scene(name,seed)
                self.assertEqual(s,scene(name,seed))
                self.assertFalse(s['dynamic']);self.assertEqual(s['channel'],'normal')
                self.assertEqual(s['steps'],int(np.ceil(180*s['goals'][0][0]/6)))
                for q in [x[:2] for x in s['initials']]+s['goals']:
                    self.assertTrue(all(np.linalg.norm(np.array(q)-[x,y])>r+.12 for x,y,r in s['obstacles']))
        self.assertGreaterEqual(scene('long_obstacle_corridor',61000)['goals'][0][0]/2.1,8)
        self.assertEqual(ARMS['disabled'][1],ARMS['reciprocal_fixed'][1])

    def test_each_layout_has_geometrically_free_routes(self):
        # Evaluator-only occupancy connectivity, not a route supplied to the learner.
        from collections import deque
        for name in SCENES:
            s=scene(name,61000);lo,hi,bot,top=s['bounds']
            xs=np.arange(lo,hi,.1);ys=np.arange(bot,top,.1)
            free=np.ones((len(xs),len(ys)),dtype=bool)
            for x,y,r in s['obstacles']:free&=(xs[:,None]-x)**2+(ys[None,:]-y)**2>(r+.17)**2
            for start,goal in zip(s['initials'],s['goals']):
                index=lambda q:(int(np.argmin(abs(xs-q[0]))),int(np.argmin(abs(ys-q[1]))))
                source=index(start);dest=index(goal);seen={source};queue=deque([source])
                while queue and dest not in seen:
                    a,b=queue.popleft()
                    for i,j in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                        if 0<=i<len(xs) and 0<=j<len(ys) and free[i,j] and (i,j) not in seen:seen.add((i,j));queue.append((i,j))
                self.assertIn(dest,seen)
