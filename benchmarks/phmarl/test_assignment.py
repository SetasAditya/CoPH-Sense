"""Goal allocation is explicit and never modifies native observations."""
import itertools
import unittest
import numpy as np
import torch
from assignment_campaign import choose_assignment
from adapter import NativeCoPH, unpack
from protection import Protection


def observations(positions,goals):
    result=[]
    for i,q in enumerate(positions):
        peers=[positions[j]-q for j in range(4) if j!=i]
        result.append(np.r_[q,[0.,0.],(goals-q).reshape(-1),np.asarray(peers).reshape(-1)].reshape(1,18).astype(np.float32))
    return result


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.positions=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        self.goals=self.positions[[2,3,0,1]].copy()
        self.obs=observations(self.positions,self.goals)

    def test_optimal_assignment_and_tie(self):
        p,info=choose_assignment(self.obs,'initial_min_distance')
        self.assertEqual(p,(2,3,0,1));self.assertEqual(info['assignment_squared_distance'],0.)
        tied=observations(self.positions,np.zeros((4,2)))
        self.assertEqual(choose_assignment(tied,'initial_min_distance')[0],(0,1,2,3))
        costs=[sum(np.linalg.norm(self.positions[i]-self.goals[p[i]])**2 for i in range(4)) for p in itertools.permutations(range(4))]
        self.assertEqual(info['assignment_squared_distance'],min(costs))

    def test_protection_and_controller_same_target_observations_immutable(self):
        old=[o.copy() for o in self.obs];p=choose_assignment(self.obs,'initial_min_distance')[0]
        protect=Protection(p);prepared=protect.prepare(self.obs);controller=NativeCoPH('known_native',p)
        controller.action(prepared)
        for i in range(4):
            np.testing.assert_array_equal(old[i],self.obs[i])
            np.testing.assert_allclose(protect.context[i]['goal'],self.goals[p[i]])
            np.testing.assert_allclose(controller.last[i]['target'],self.goals[p[i]])
        moved=observations(self.positions+.2,self.goals);protect.prepare(moved)
        self.assertEqual(protect.goal_indices,p);self.assertEqual(controller.goal_indices,p)

    def test_explicit_index_is_default(self):
        a=Protection();b=Protection(range(4))
        aa=a.prepare(self.obs);bb=b.prepare(self.obs)
        for x,y in zip(aa,bb):np.testing.assert_array_equal(x,y)
        c=NativeCoPH('online');d=NativeCoPH('online',range(4))
        for x,y in zip(c.action(aa),d.action(bb)):self.assertTrue(torch.equal(x,y))
        for i in range(4):np.testing.assert_array_equal(unpack(self.obs[i].reshape(-1),i)[1],unpack(self.obs[i].reshape(-1),i,i)[1])

    def test_invalid_permutation(self):
        for cls,args in [(Protection,()),(NativeCoPH,('online',))]:
            with self.assertRaises(ValueError):cls(*args,goal_indices=(0,0,2,3))

if __name__=='__main__':unittest.main()
