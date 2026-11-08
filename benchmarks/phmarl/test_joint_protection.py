import copy
import unittest
import numpy as np
import torch
from joint_protection import JointProtection,constraints,solve
from protection import Protection


def observations(q,v=None):
    q=np.asarray(q,float);v=np.zeros_like(q) if v is None else np.asarray(v,float);goals=np.array([[-.8,0.],[.8,0.],[0.,.8],[0.,-.8]])
    return [np.r_[q[i],v[i],(goals-q[i]).reshape(-1),(q[[j for j in range(4) if j!=i]]-q[i]).reshape(-1)] for i in range(4)]


class JointProtectionTests(unittest.TestCase):
    def test_legacy_class_unchanged_and_goal_preparation_parity(self):
        obs=observations([[-.2,0],[.2,0],[0,.6],[0,-.6]])
        old=Protection();new=JointProtection();a=old.prepare(obs);b=new.prepare(obs)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertIsNot(Protection.apply,JointProtection.apply)
    def test_bounds_residual_and_joint_kkt(self):
        obs=observations([[-.19,0],[.19,0],[0,.8],[0,-.8]],[[.1,0],[-.1,0],[0,0],[0,0]])
        p=JointProtection();p.prepare(obs);u,logs=p.apply([torch.tensor([[1.,0]]),torch.tensor([[-1.,0]]),torch.zeros(1,2),torch.zeros(1,2)])
        self.assertTrue(logs[0]['feasible']);self.assertGreaterEqual(min(logs[0]['constraint_residuals']),-2e-7)
        self.assertLessEqual(logs[0]['joint']['attempts'][-1]['certificate']['stationarity'],1e-6)
        self.assertTrue(all(bool((x.abs()<=1).all()) for x in u))
    def test_all_pairs_and_third_party_are_constrained(self):
        q=np.array([[0,0],[.24,0],[.55,0],[0,1.]])
        A,b,pairs=constraints(q,np.zeros_like(q));self.assertEqual(A.shape,(6,8))
        self.assertEqual(pairs[0]['target'],.24);self.assertTrue(pairs[0]['initially_contacting'])
        self.assertTrue(any(p['agents']==[1,2] for p in pairs))
        result,d=solve(np.zeros(8),A,b);self.assertIsNotNone(result);self.assertTrue(d['accepted'])
    def test_permutation_equivariance_away_from_coincident_normals(self):
        q=np.array([[-.2,0],[.2,0],[0,.65],[0,-.65]]);v=np.array([[.1,0],[-.1,0],[0,-.1],[0,.1]])
        raw=np.array([[.8,.3],[-.6,-.4],[.2,-.8],[-.3,.6]])
        A,b,_=constraints(q,v);first,d=solve(raw.reshape(-1),A,b);self.assertTrue(d['accepted'])
        permutation=np.array([2,0,3,1]);A,b,_=constraints(q[permutation],v[permutation]);second,e=solve(raw[permutation].reshape(-1),A,b)
        self.assertTrue(e['accepted']);np.testing.assert_allclose(second.reshape(4,2)[np.argsort(permutation)],first.reshape(4,2),atol=1e-6)
    def test_infeasible_is_affine_only_and_braking_is_explicit(self):
        q=np.array([[-.16,0],[.16,0],[0,1],[0,-1.]])
        p=JointProtection();p.prepare(observations(q,[[2,0],[-2,0],[0,0],[0,0]]));_,logs=p.apply([torch.zeros(1,2)]*4)
        self.assertFalse(logs[0]['feasible']);self.assertFalse(logs[0]['joint']['geometric_infeasibility_certified'])
        self.assertEqual(logs[0]['fallback'],'bounded_braking_uncertified')
    def test_overlap_cluster_is_finite_and_reported(self):
        p=JointProtection();obs=observations([[0,0],[.1,0],[.05,.09],[1,1]]);before=copy.deepcopy(obs);p.prepare(obs)
        u,logs=p.apply([torch.zeros(1,2)]*4)
        self.assertGreaterEqual(len(logs[0]['joint']['initial_unsafe_pairs']),3)
        self.assertTrue(all(torch.isfinite(x).all() for x in u))
        for a,b in zip(obs,before):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
