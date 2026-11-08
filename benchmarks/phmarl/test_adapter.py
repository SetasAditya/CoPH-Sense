import copy
import unittest
import numpy as np
import torch
from adapter import NativeCoPH, native_flow, unpack, model


def observation(q=(0,0),v=(0,0),target=(1,0),peers=((2,0),(2,1),(2,-1))):
    return np.array([*q,*v,*sum(([target[0]-q[0],target[1]-q[1]] for _ in range(4)),[]),
        *sum(([p[0]-q[0],p[1]-q[1]] for p in peers),[])],dtype=np.float32)

class AdapterTests(unittest.TestCase):
    def test_native_flow_matches_simulator(self):
        from vmas import make_env
        env=make_env(scenario_name='simple_spread',num_envs=1,device='cpu',continuous_actions=True,n_agents=1,seed=7)
        obs=env.reset(seed=7)
        for k in range(12):
            x=torch.tensor(obs[0][0,:4],dtype=torch.float32)
            u=torch.tensor([[.2,-.3]])
            expected=native_flow(x,u[0],.25)
            obs,_,_,_=env.step([u])
            torch.testing.assert_close(obs[0][0,:4],expected,atol=1e-7,rtol=1e-6)

    def test_mask_goal_and_no_mutation(self):
        a=NativeCoPH('online');o=observation(); obs=[o.copy() for _ in range(4)]
        before=copy.deepcopy(obs);actions=a.action(obs)
        for x,y in zip(obs,before):np.testing.assert_array_equal(x,y)
        remote=o.copy();remote[-6:]=[20,0,20,1,20,-1]
        b=NativeCoPH('online')
        torch.testing.assert_close(actions[0],b.action([remote]*4)[0],rtol=0,atol=0)
        self.assertTrue(all(bool((u.abs()<=1).all()) for u in actions))
        x,g,p=unpack(o,0);np.testing.assert_array_equal(g,[1,0]);self.assertEqual(len(p),0)

    def test_potential_gradient(self):
        a=NativeCoPH('frozen');q=model.tensor([.1,.1]).requires_grad_();goal=model.tensor([1.,0.])
        weights=model.tensor([]);centers=torch.empty((0,2),dtype=torch.float64);policy=model.tensor([0.,2.])
        obs=((.35,.1,.18),)
        def f(z):return model.potential(z,goal,weights,centers,.8,policy,obs,a.cfg)
        g=torch.autograd.grad(f(q),q)[0]
        h=1e-5;fd=[]
        for i in range(2):
            d=torch.zeros(2,dtype=torch.float64);d[i]=h
            fd.append(float((f(q.detach()+d)-f(q.detach()-d))/(2*h)))
        np.testing.assert_allclose(g.numpy(),fd,rtol=1e-6,atol=1e-7)

    def test_identification_memory_and_isolation(self):
        a=NativeCoPH('online')
        for k in range(160):
            x=np.array([0,0,.1,.2]);u=np.array([.3,-.2]);y=native_flow(model.tensor(x),model.tensor(u),.25).numpy()
            a.records[0].append(dict(uid=k,x=tuple(x),u=tuple(u),after=tuple(y),eligible=True,version=0))
            a.records[0]=a.records[0][-120:]
        original=copy.deepcopy(a.records)
        a.update(0);a.update(0)
        self.assertAlmostEqual(a.drag[0],.25,places=12)
        self.assertEqual(a.records,original);self.assertEqual(len(a.records[0]),120)
        b=copy.deepcopy(a);b.drag[0]=.9;self.assertAlmostEqual(a.drag[0],.25)
        # Force inconsistent reserved records to verify rollback.
        for r in a.records[0]:
            r['after']=tuple(native_flow(model.tensor(r['x']),model.tensor(r['u']),.25 if r['uid']%5==0 else .8).numpy())
        version=a.versions[0];row=a.update(0)
        self.assertFalse(row['accepted']);self.assertEqual(a.versions[0],version)

    def test_contact_exclusion(self):
        a=NativeCoPH('online');o=observation(peers=((.2,0),(2,0),(2,1)))
        actions=a.action([o]*4);a.observe([o]*4,actions,[o]*4)
        self.assertFalse(a.records[0][0]['eligible'])

if __name__=='__main__':unittest.main()
