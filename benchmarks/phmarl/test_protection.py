import copy
import unittest
import numpy as np
import torch
from scipy.optimize import minimize
from protection import Protection,project,horizon_coefficients
from test_adapter import observation

class ProtectionTests(unittest.TestCase):
    def test_projection_reference(self):
        rng=np.random.RandomState(81)
        for _ in range(30):
            A=rng.normal(size=(3,2));b=-rng.uniform(.1,1,size=3);p=rng.normal(size=2)
            u,_=project(p,A,b,True)
            ref=minimize(lambda x:np.sum((x-p)**2),np.zeros(2),jac=lambda x:2*(x-p),
                bounds=[(-1,1)]*2,constraints={'type':'ineq','fun':lambda x:A@x-b,'jac':lambda x:A},
                method='SLSQP',options={'ftol':1e-12,'maxiter':100})
            self.assertTrue(ref.success);np.testing.assert_allclose(u,ref.x,atol=2e-6)
        u,_=project([0,0],[[1,0],[-1,0]],[2,2],True);self.assertIsNone(u)

    def test_native_coefficients(self):
        for h in [1,2,3,4]:
            q=0.;v=.4;u=-.3
            for _ in range(h):v=.75*v+.1*u;q+=.1*v
            a,b=horizon_coefficients(h);self.assertAlmostEqual(q,a*.4+b*u)

    def test_locality_history_and_bounds(self):
        obs=[observation() for _ in range(4)];p=Protection();before=copy.deepcopy(obs)
        adjusted=p.prepare(obs);u,logs=p.apply([torch.tensor([[.2,-.3]])]*4)
        for a,b in zip(obs,before):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(u[0],[[.2,-.3]],atol=1e-7)
        other=copy.deepcopy(obs);other[1][:4]=[10,10,2,3]
        a=Protection();a.prepare(other);v,_=a.apply([torch.tensor([[.2,-.3]])]*4)
        np.testing.assert_array_equal(u[0],v[0])
        self.assertTrue(all((x.abs()<=1).all() for x in u))
        clone=copy.deepcopy(p);clone.step=99;self.assertNotEqual(p.step,clone.step)

    def test_close_goal_offset_and_infeasible_braking(self):
        p=Protection();obs=[observation(q=(0,0),target=(0,0),peers=((.23,0),(2,0),(2,1))) for _ in range(4)]
        adjusted=p.prepare(obs)
        self.assertLess(adjusted[0][0,4],0)
        _,logs=p.apply([torch.zeros(1,2)]*4)
        self.assertGreater(logs[0]['goal_shift'],0)
        self.assertGreater(logs[0]['unknown_peer_velocities'],0)
        p.prepare(obs);_,logs=p.apply([torch.zeros(1,2)]*4)
        self.assertEqual(logs[0]['unknown_peer_velocities'],0)
        p.context[0]['bounds']=[100.]
        _,logs=p.apply([torch.zeros(1,2)]*4)
        self.assertFalse(logs[0]['feasible']);self.assertEqual(logs[0]['fallback'],'bounded_braking')

    def test_reciprocal_separation(self):
        # Known velocities and feasible projected inputs enforce each plane.
        p=Protection();obs=[observation() for _ in range(4)]
        obs[0]=observation(q=(-.18,0),v=(.1,0),target=(1,0),peers=((.18,0),(2,0),(2,1)))
        obs[1]=observation(q=(.18,0),v=(-.1,0),target=(-1,0),peers=((- .18,0),(2,0),(2,1)))
        p.history[0][1]=np.array([.19,0]);p.history[1][0]=np.array([-.19,0])
        p.prepare(obs);u,logs=p.apply([torch.tensor([[1.,0]]),torch.tensor([[-1.,0]]),torch.zeros(1,2),torch.zeros(1,2)])
        self.assertTrue(logs[0]['feasible'] and logs[1]['feasible'])
        for row in logs[:2]:self.assertGreaterEqual(min(row['constraint_residuals']),-1e-6)

if __name__=='__main__':unittest.main()
