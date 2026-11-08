import copy
from dataclasses import asdict,replace
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from coph_stagewise.model import tensor, risk, compact_features, shaping_parameters
from coph_stagewise.localized_receiver import vector, validate, RADIUS
from coph_stagewise.cooperative import CooperativeLearner, CommunicationConfig
from coph_stagewise.cooperative_evaluation import CooperativeWorld, specification
from coph_stagewise.receiver_summary import summarize_receiver, reconstruct_receiver
from coph_stagewise.records import Config


class LocalizedReceiverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=Config(replay_objective='mission',replay_steps=3)
        self.comm=CommunicationConfig(ttl=300.)
    def pair(self):
        world=CooperativeWorld(specification('complementary',34,'smoke'),self.cfg)
        agents=[]
        for i in range(2):
            x=np.array([.1,.2*i,0.,0.])
            a=CooperativeLearner(self.cfg,x,(6.,0.),mode='evidence',agent_id=str(i),communication=self.comm,
                receiver_fit_mode='precision_bounded',receiver_validation_mode='relative_local',
                receiver_solver_mode='active_set',receiver_search_mode='diagnostic',receiver_localization_mode='compact')
            for t in range(10):a.observe(world.sense(x,t))
            agents.append(a)
        return agents
    def residual(self,coefficient=.2):
        return dict(schema=1,radius=RADIUS,centers=[[.1,.2]],coefficients=[coefficient],packets=['test'],source_ids=['0:test'])
    def test_exact_locality_zero_and_smooth_center(self):
        stage=self.pair()[0].stages[0]
        q=tensor([[2.,2.],[.1,.2]]).requires_grad_(True)
        base=risk(q,stage.weights,stage.centers,self.cfg.rbf_width)
        zero=risk(q,stage.weights,stage.centers,self.cfg.rbf_width,self.residual(0))
        self.assertTrue(torch.equal(base,zero))
        value=risk(q,stage.weights,stage.centers,self.cfg.rbf_width,self.residual())
        self.assertEqual(float(base[0]),float(value[0]))
        gradient=torch.autograd.grad(value.sum(),q,create_graph=True)[0]
        second=torch.autograd.grad(gradient.sum(),q)[0]
        self.assertTrue(torch.isfinite(gradient).all() and torch.isfinite(second).all())
    def test_vector_and_objective_derivatives(self):
        s=self.pair()[0].stages[0]
        view=SimpleNamespace(centers=s.centers,receiver_base_weights=s.weights,receiver_compact_centers=tensor([[.1,.2],[.4,-.1]]))
        q=tensor([[.2,.1],[.3,.25]]);c=tensor([.1,-.2]).requires_grad_(True)
        v=vector(c,q,view,self.cfg.rbf_width)
        H=tensor([[2.,.1],[.1,3.]])
        objective=.5*c@H@c+.5*(v-tensor([.1,.02,.03,.2,-.01,.04])).square().sum()
        gradient=torch.autograd.grad(objective,c)[0]
        def f(z):return .5*z@H@z+.5*(vector(z,q,view,self.cfg.rbf_width)-tensor([.1,.02,.03,.2,-.01,.04])).square().sum()
        for k in range(2):
            d=torch.zeros_like(c);d[k]=1e-6
            self.assertAlmostEqual(float(gradient[k]),float((f(c.detach()+d)-f(c.detach()-d))/2e-6),places=8)
        for i in range(len(q)):
            for k in range(2):
                d=torch.zeros_like(q);d[i,k]=1e-6
                vp=vector(c.detach(),q+d,view,self.cfg.rbf_width).reshape(-1,3)[i,0]
                vm=vector(c.detach(),q-d,view,self.cfg.rbf_width).reshape(-1,3)[i,0]
                self.assertAlmostEqual(float(v.reshape(-1,3)[i,k+1]),float((vp-vm)/2e-6),places=8)
    def test_no_message_equivalence(self):
        a=self.pair()[1];b=copy.deepcopy(a);b.receiver_localization_mode='global'
        x=np.array([.1,.2,0.,0.]);da=a.update(x);db=b.update(x)
        self.assertEqual(da['accepted'],db['accepted'])
        self.assertEqual(da['reason'],db['reason'])
        for s,t in zip(a.stages,b.stages):
            self.assertTrue(torch.equal(s.weights,t.weights));self.assertIsNone(getattr(s,'local_residual',None))
        np.testing.assert_array_equal(a.action(x)[0],b.action(x)[0])
    def test_transactional_rollback_measurements_and_export_isolation(self):
        a,b=self.pair();x=np.array([.1,.2,0.,0.]);packet=a.export(np.array([.1,0.,0.,0.]),'1')
        original_export=b.export(x,'0');old=b.stages[0].weights.clone();oldversion=b.version
        b.stages[0].local_residual=self.residual(.05);before=copy.deepcopy(b.stages[0].local_residual)
        readings=copy.deepcopy(b.stages[0].readings)
        self.assertEqual(b.assimilate(packet)['reason'],'eligible')
        d=b.update(x,force_invalid=True)
        self.assertFalse(d['accepted']);self.assertEqual(oldversion,b.version)
        self.assertTrue(torch.equal(old,b.stages[0].weights))
        self.assertEqual(before,b.stages[0].local_residual);self.assertEqual(readings,b.stages[0].readings)
        after=b.export(x,'0')
        self.assertEqual(original_export.anchors,after.anchors)
        self.assertEqual(original_export.sources,after.sources)
        self.assertEqual(asdict(original_export),asdict(after))
    def test_bounds_and_memory(self):
        validate(self.residual(),12)
        with self.assertRaises(ValueError):validate(self.residual(.50001),12)
        with self.assertRaises(ValueError):validate(self.residual(),0)
    def test_actual_solve_keeps_base_and_certifies_bounds(self):
        a,b=self.pair();baseline=copy.deepcopy(b);baseline.fit_received=False
        x=np.array([.1,.2,0.,0.]);packet=a.export(np.array([.1,0.,0.,0.]),'1')
        b.assimilate(packet);baseline.assimilate(packet)
        baseline.update(x);d=b.update(x)
        self.assertTrue(d['accepted'])
        for local,corrected in zip(baseline.stages,b.stages):
            self.assertTrue(torch.equal(local.weights,corrected.weights))
            self.assertGreater(max(abs(c) for c in corrected.local_residual['coefficients']),0)
            self.assertLessEqual(max(abs(c) for c in corrected.local_residual['coefficients']),.5)
        for stage in d['receiver_fit']['stages']:
            self.assertEqual(stage['status'],'converged')
            self.assertLessEqual(stage['stationarity'],1e-6)
        self.assertTrue(d['receiver_fit']['relative_gate_passed'])
    def test_correction_does_not_certify_support(self):
        a=self.pair()[0];q=np.array([1.45,1.7])
        before=a.stages[0].supported(q,a.cfg)
        a.stages[0].local_residual=dict(self.residual(),centers=[q.tolist()])
        self.assertEqual(before,a.stages[0].supported(q,a.cfg))
    def test_summary_roundtrip_carries_committed_residual(self):
        a=self.pair()[0];a.stages[0].local_residual=self.residual()
        summary=summarize_receiver(a,(.1,0.,0.,0.),'1')
        body=summary.validate('1',a.now)
        self.assertEqual(body['receiver_localization_mode'],'compact')
        b=reconstruct_receiver(summary,'1',a.now)
        self.assertEqual(a.stages[0].local_residual,b.stages[0].local_residual)
        self.assertEqual(b.receiver_localization_mode,'compact')
    def test_uniform_gradient_bound(self):
        c=self.residual(.4)
        points=tensor([[.1+x,.2] for x in np.linspace(-.7,.7,301)]).requires_grad_(True)
        value=compact_features(points,tensor(c['centers']),RADIUS)@tensor(c['coefficients'])
        derivative=torch.autograd.grad(value.sum(),points)[0]
        self.assertLessEqual(float(torch.linalg.vector_norm(derivative,dim=1).max()),.4*135/(64*RADIUS)+1e-12)
    def test_frozen_candidate_context_and_multistep_gradients(self):
        from coph_stagewise.stage_prediction import FrozenStages
        a=self.pair()[0];a.cfg=replace(a.cfg,replay_domain='retained')
        a.stages[0].local_residual=self.residual(.1)
        a.receiver_localized_proposal=[self.residual(.2)]
        self.assertEqual(FrozenStages.freeze(a).stages[0].local_residual,self.residual(.1))
        self.assertEqual(FrozenStages.freeze(a,[a.stages[0].weights]).stages[0].local_residual,self.residual(.2))
        a.receiver_localized_proposal=None
        policy=a.policy.clone().requires_grad_(True)
        cost,paths=a.replay(policy,[a.stages[0].weights],a.damping)
        self.assertIsNotNone(cost)
        gradient=torch.autograd.grad(cost,policy)[0]
        eps=tensor([1e-5,0.])
        plus,_=a.replay(policy.detach()+eps,[a.stages[0].weights],a.damping)
        minus,_=a.replay(policy.detach()-eps,[a.stages[0].weights],a.damping)
        self.assertAlmostEqual(float(gradient[0]),float((plus-minus)/2e-5),places=7)


if __name__=='__main__':unittest.main()
