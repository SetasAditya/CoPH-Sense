import copy
from dataclasses import asdict,replace
import inspect,json,unittest
import numpy as np
import torch
from coph_stagewise.records import Config,Reading,new_stage
from coph_stagewise.model import tensor,flow,balance
from coph_stagewise.cooperative import CooperativeLearner,CommunicationConfig,advertise
from coph_stagewise.receiver_summary import ReceiverSummary,SummaryCache,summarize_receiver,reconstruct_receiver
from coph_stagewise.stage_prediction import FrozenStages,DomainExit
from coph_stagewise.model_prediction import variant_prediction
from coph_stagewise.cooperative_evaluation import specification,CooperativeWorld,execute
from coph_stagewise.diagnostics import snapshot

class StagePredictionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def fixture(self):
        cfg=Config(replay_objective='mission',replay_domain='retained',replay_steps=6)
        x=np.array([.85,0.,.8,0.]);a=CooperativeLearner(cfg,x,(3.,0.),agent_id='1')
        s0=new_stage(0,(0.,0.),x,a.goal,cfg);s1=new_stage(1,(2.1,0.),(1.1,0.,.8,0.),a.goal,cfg)
        for s in (s0,s1):
            points=[(x,y) for x in np.linspace(s.center[0]-1.5,s.center[0]+1.5,9) for y in np.linspace(-2,2,9)]
            s.readings=[Reading(f'1:fixture:{k*5+1}',0.,q,.1) for k,q in enumerate(points)]
        a.stages=[s0,s1];return a,x

    def test_crossing_execution_gradient_and_frontier(self):
        a,x=self.fixture();p=a.policy.clone().requires_grad_(True);weights=[s.weights for s in a.stages]
        c,paths=a.replay(p,weights,a.damping)
        self.assertIsNotNone(c);self.assertTrue(a.replay_stage_events)
        self.assertTrue(all(e['physical_jump']==0 for e in a.replay_stage_events))
        grad=torch.autograd.grad(c,p)[0];eps=tensor([1e-5,0.])
        plus,_=a.replay(p.detach()+eps,weights,a.damping);minus,_=a.replay(p.detach()-eps,weights,a.damping)
        self.assertAlmostEqual(float(grad[0]),float((plus-minus)/2e-5),places=6)
        live=copy.deepcopy(a);live.stages=[copy.deepcopy(a.stages[0])];state=x.copy()
        for k,expected in enumerate(paths[0][1][1:]):
            live.advance(state)
            u,_=live.action(state);account=balance(tensor(state),tensor(u),live.damping,live.cfg.dt)
            self.assertLess(abs(account['residual']),1e-12)
            state=flow(tensor(state),tensor(u),live.damping,live.cfg.dt).numpy()
            np.testing.assert_allclose(state,expected,atol=1e-12)
        a.stages=a.stages[:1]
        failed,_=a.replay(a.policy,[a.stages[0].weights],a.damping)
        self.assertIsNone(failed);self.assertEqual(a.replay_eligibility[-1]['reason'],'unavailable_stage')

    def test_frozen_and_reserved_support(self):
        a,x=self.fixture();atlas=FrozenStages.freeze(a);before=atlas.stages[0].weights
        a.stages[0].weights+=1;self.assertEqual(atlas.stages[0].weights,before)
        a.stages[0].readings=[Reading('1:reserved:0',0.,(0.,0.),1.)]
        frozen=FrozenStages.freeze(a)
        self.assertFalse(frozen.stages[0].supported((0.,0.),a.cfg))
        with self.assertRaises(DomainExit):frozen.index_at((100.,0.))

    def test_summary_roundtrip_no_raw_history_and_cache(self):
        a,x=self.fixture();a.policy=tensor([1.3,2.1]);a.damping=.9;a.version=4
        a.stages[0].readings.append(Reading('1:reserved:0',0.,(.123,.456),123456.789))
        summary=summarize_receiver(a,x,'0');wire=json.loads(json.dumps(asdict(summary)))
        restored=ReceiverSummary.from_dict(wire);self.assertEqual(summary,restored)
        self.assertNotIn('123456.789',summary.model_json);self.assertNotIn('1:reserved:0',summary.model_json)
        body=json.loads(summary.model_json)
        self.assertTrue(all('readings' not in s and 'transitions' not in s for s in body['stages']))
        shadow=reconstruct_receiver(restored,'0',0.,True)
        np.testing.assert_array_equal(shadow.policy,a.policy);self.assertEqual(shadow.damping,a.damping)
        self.assertEqual([s.id for s in shadow.stages],[0,1]);self.assertEqual(shadow.version,4)
        self.assertTrue(all(r.uid.startswith('hypothetical') for s in shadow.stages for r in s.readings))
        self.assertTrue(all(len(s.readings)<=a.cfg.sample_cap and not s.transitions for s in shadow.stages))
        cache=SummaryCache();received,n,reason=cache.transfer(summary,'0',0.,10**6)
        self.assertEqual(n,summary.bytes());self.assertEqual(reason,'full_summary')
        _,smaller,reason=cache.transfer(summary,'0',0.,10**6)
        self.assertLess(smaller,n);self.assertEqual(reason,'cached_reference')
        with self.assertRaises(ValueError):cache.transfer(summary,'0',10.,10**6)
        with self.assertRaises(ValueError):cache.transfer(replace(summary,version=3),'0',0.,10**6)
        self.assertEqual(cache.transfer(summary,'0',0.,0)[2],'summary_budget')
        for version in range(10):
            a.version=version+5;a.now=version+1
            cache.transfer(summarize_receiver(a,x,'0'),'0',a.now,10**6)
        self.assertLessEqual(len(cache.models),8)

    def test_prediction_isolation_and_budget(self):
        cfg=Config(replay_objective='mission',replay_steps=3)
        spec=specification('complementary',60000,'smoke');world=CooperativeWorld(spec,cfg)
        states=[np.array(z) for z in spec['initials']]
        agents=[CooperativeLearner(cfg,states[i],spec['goals'][i],agent_id=str(i)) for i in range(2)]
        for i,a in enumerate(agents):a.observe(world.sense(states[i],0))
        packet=agents[0].export(states[0],'1');status=advertise(agents[1],states[1],'0',[z[:2] for z in packet.anchors])
        summary=summarize_receiver(agents[1],states[1],'0');before=copy.deepcopy(snapshot(agents[1]))
        draws=[]
        for variant in ('baseline','compact','cross_stage','both'):
            result=variant_prediction(packet,status,summary,cfg,agents[1].communication,variant,horizons=(1,2))
            draws.append(result['sampled_anchors'])
            for branch in result['branches'].values():self.assertEqual(branch['path'][0],states[1].tolist())
        self.assertTrue(all(draw==draws[0] for draw in draws))
        self.assertEqual(before,snapshot(agents[1]))
        result=execute(dict(spec,steps=9),'reciprocal_fixed',CommunicationConfig(prediction_variant='both',byte_budget=50000),audits=False,collect_training=True)
        self.assertLessEqual(result['summary']['bytes'],100000)
        self.assertGreater(result['summary']['summary_bytes'],0)
        self.assertTrue(any(d['receiver_summary'] for d in result['decisions']))
        self.assertTrue(result['training'])
        self.assertTrue(all(r['label']['horizon']==16 for r in result['training']))

    def test_no_evaluator_imports(self):
        from coph_stagewise import stage_prediction,receiver_summary,model_prediction,continuations
        for module in (stage_prediction,receiver_summary,model_prediction,continuations):
            self.assertNotIn('from .cooperative_evaluation',inspect.getsource(module))
            self.assertNotIn('from .recipient_diagnostics',inspect.getsource(module))

    def test_gate_requires_coverage_and_matching_cases(self):
        from coph_stagewise.stagewise_diagnostics import gate
        rows=[]
        for i in range(12):
            for v,error in [('baseline',.2),('both',.1)]:
                rows.append(dict(scene='s',seed=i,packet=str(i),variant=v,horizon=16,valid=True,error=error))
        self.assertTrue(gate(rows)['passed'])
        rows[1]['valid']=False
        self.assertFalse(gate(rows)['passed'])
        rows[1]['valid']=True
        self.assertFalse(gate(rows,complete=False)['passed'])

    def test_summary_rejects_bad_provenance_and_support_bounds(self):
        import hashlib
        from coph_stagewise.receiver_summary import canonical
        a,x=self.fixture();summary=summarize_receiver(a,x,'0')
        body=json.loads(summary.model_json);body['local_source_ids']=['other:1']
        payload=canonical(body)
        corrupted=replace(summary,model_json=payload,model_digest=hashlib.sha256(payload.encode()).hexdigest())
        with self.assertRaises(ValueError):corrupted.validate('0',0.)
        body=json.loads(summary.model_json);body['stages'][0]['support']=[[0.,0.]]*241
        payload=canonical(body)
        corrupted=replace(summary,model_json=payload,model_digest=hashlib.sha256(payload.encode()).hexdigest())
        with self.assertRaises(ValueError):corrupted.validate('0',0.)
