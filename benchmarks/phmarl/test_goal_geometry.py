import unittest
import numpy as np
import torch
from goal_geometry import static_targets,assignment_consensus,permute_goal_slots
from protection import Protection
from test_assignment import observations


class GoalGeometryTests(unittest.TestCase):
    def test_pair_splits_deficit_symmetrically(self):
        g=np.array([[0.,0.],[.17,0.],[2.,0.],[0.,2.]])
        t,r=static_targets(g)
        self.assertTrue(r['feasible']);np.testing.assert_allclose(t[:2]-g[:2],[[-.075,0],[.075,0]],atol=1e-8)
        np.testing.assert_allclose(t[2:],g[2:],atol=1e-8)

    def test_joint_cluster_feasibility(self):
        g=np.array([[0.,0.],[.12,0.],[.06,.11],[2.,2.]])
        t,r=static_targets(g)
        self.assertTrue(r['feasible'])
        self.assertLessEqual(np.linalg.norm(t-g,axis=1).max(),.15)
        self.assertGreaterEqual(min(np.linalg.norm(t[i]-t[j]) for i in range(4) for j in range(i+1,4)),.32-1e-9)

    def test_pair_impossibility_and_cluster_qualification(self):
        _,r=static_targets(np.array([[0.,0.],[.001,0.],[2.,0.],[0.,2.]]))
        self.assertFalse(r['feasible']);self.assertTrue(r['true_pairwise_infeasibility_certificate'])
        _,r=static_targets(np.array([[0.,0.],[.021,0.],[.0105,.0182],[2.,2.]]))
        self.assertFalse(r['feasible']);self.assertFalse(r['true_pairwise_infeasibility_certificate'])

    def test_independent_assignment_and_permutation(self):
        q=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        g=q[[2,3,0,1]];obs=observations(q,g);before=np.asarray(obs).copy()
        c=assignment_consensus(obs);self.assertTrue(c['agreed'])
        mapped=permute_goal_slots(obs,c['assignments'][0])
        np.testing.assert_array_equal(np.asarray(obs),before)
        np.testing.assert_array_equal(mapped.reshape(4,18)[:,:4],before.reshape(4,18)[:,:4])
        np.testing.assert_array_equal(mapped.reshape(4,18)[:,12:],before.reshape(4,18)[:,12:])
        for i in range(4):np.testing.assert_allclose(mapped.reshape(4,18)[i,4+2*i:6+2*i],0.)

    def test_static_target_arrival_order_and_motion_constraints(self):
        g=np.array([[0.,0.],[.17,0.],[2.,0.],[0.,2.]])
        targets,r=static_targets(g);positions=np.array([[0.,.02],[.35,.02],[2.,0.],[0.,2.]])
        obs=observations(positions,g);copy=np.asarray(obs).copy()
        static=Protection(fixed_targets=targets);legacy=Protection()
        static.prepare(obs);legacy.prepare(obs)
        for a,b in zip(static.context,legacy.context):
            np.testing.assert_array_equal(a['normals'],b['normals']);np.testing.assert_array_equal(a['bounds'],b['bounds'])
        np.testing.assert_allclose(static.context[0]['target'],targets[0])
        np.testing.assert_allclose(static.context[1]['target'],targets[1])
        np.testing.assert_array_equal(np.asarray(obs),copy)
        moved=positions.copy();moved[0]=targets[0];moved[1]=targets[1]
        static.prepare(observations(moved,g))
        for i in range(4):np.testing.assert_allclose(static.context[i]['target'],targets[i])

    def test_legacy_default_and_explicit_none(self):
        g=np.array([[0.,0.],[.17,0.],[2.,0.],[0.,2.]])
        obs=observations(g,g);a=Protection();b=Protection(fixed_targets=None)
        for x,y in zip(a.prepare(obs),b.prepare(obs)):np.testing.assert_array_equal(x,y)
        actions=[torch.zeros((1,2)) for _ in range(4)]
        aa,al=a.apply(actions);bb,bl=b.apply(actions)
        self.assertEqual(al,bl)
        for x,y in zip(aa,bb):self.assertTrue(torch.equal(x,y))

if __name__=='__main__':unittest.main()
