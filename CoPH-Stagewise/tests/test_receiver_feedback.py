import copy,inspect,json,unittest
from dataclasses import replace
import numpy as np
import torch
import test_stage_prediction as fixtures
from coph_stagewise.records import Config
from coph_stagewise.cooperative import CooperativeLearner,CommunicationConfig,advertise
from coph_stagewise.cooperative_evaluation import specification,CooperativeWorld,execute
from coph_stagewise.replay_budget import prepare
from coph_stagewise.diagnostics import snapshot
from coph_stagewise import packet_feedback

class ReceiverFeedbackTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_boundary_plan_gradient_and_budget(self):
        a,x=fixtures.StagePredictionTests().fixture()
        a.cfg=replace(a.cfg,replay_steps=8,replay_horizon_mode='supported_boundary',replay_update_cap=1024)
        # Slow entry makes the supported handoff require more than eight steps.
        a.stages[0].entry=(.2,0.,.1,0.)
        weights=[s.weights for s in a.stages];prepare(a,weights,a.damping)
        plan=copy.deepcopy(a.replay_plan)
        self.assertEqual(plan[0]['reason'],'supported_boundary')
        self.assertGreater(plan[0]['horizon'],8)
        p=a.policy.clone().requires_grad_(True);cost,_=a.replay(p,weights,a.damping)
        self.assertIsNotNone(cost)
        grad=torch.autograd.grad(cost,p)[0];eps=torch.tensor([1e-5,0.],dtype=torch.float64)
        plus,_=a.replay(p.detach()+eps,weights,a.damping);minus,_=a.replay(p.detach()-eps,weights,a.damping)
        self.assertAlmostEqual(float(grad[0]),float((plus-minus)/2e-5),places=5)
        self.assertEqual(plan,a.replay_plan)
        self.assertTrue(a.replay_stage_events)
        self.assertTrue(all(e['physical_jump']==0 for e in a.replay_stage_events))
        a.replay_plan_active=False
        r=a.update(x);self.assertLessEqual(r['counterfactual_steps'],1024)
        self.assertGreater(r['replay_pilot_steps'],0)
        # Reserve baseline and gradient before spending any work; field update may proceed.
        a.cfg=replace(a.cfg,replay_update_cap=1)
        r=a.update(x);self.assertTrue(r['replay_budget_exhausted']);self.assertEqual(r['counterfactual_steps'],0)
        self.assertEqual(r['policy_gradient_norm'],0)

    def test_pilot_failure_frontier_and_rollback(self):
        a,x=fixtures.StagePredictionTests().fixture();a.cfg=replace(a.cfg,replay_steps=8,replay_horizon_mode='supported_boundary')
        a.stages[0].entry=(-1.4,0.,0.,0.);a.policy*=0;a.damping=100.
        prepare(a,[s.weights for s in a.stages],a.damping)
        self.assertEqual(a.replay_plan[0]['reason'],'handoff_not_reached')
        self.assertEqual(a.replay_plan[0]['horizon'],8)
        a.damping=.6;a.stages=a.stages[:1];a.stages[0].entry=(.85,0.,.8,0.)
        prepare(a,[s.weights for s in a.stages],a.damping)
        self.assertEqual(a.replay_plan[0]['reason'],'no_known_successor')
        cost,_=a.replay(a.policy,[a.stages[0].weights],a.damping);self.assertIsNone(cost)
        a.replay_plan_active=False;before=snapshot(a);r=a.update(x,force_invalid=True)
        self.assertFalse(r['accepted']);self.assertEqual(before,snapshot(a))

    def fixture(self):
        spec=specification('complementary',60000,'smoke');cfg=Config(replay_objective='mission',replay_domain='retained',replay_update_cap=1024)
        world=CooperativeWorld(spec,cfg);states=[np.array(x) for x in spec['initials']]
        agents=[CooperativeLearner(cfg,states[i],spec['goals'][i],agent_id=str(i)) for i in range(2)]
        for a,x in zip(agents,states):a.observe(world.sense(x,0))
        return agents[1],agents[0].export(states[0],'1')

    def test_receiver_isolation_shared_draws_actual_validation(self):
        a,p=self.fixture();before=copy.deepcopy(snapshot(a));counts=(a.counterfactual_steps,a.identification_steps,len(a.packet_events))
        result=a.evaluate_packet(p,seed=12)
        self.assertEqual(before,snapshot(a));self.assertEqual(counts,(a.counterfactual_steps,a.identification_steps,len(a.packet_events)))
        self.assertEqual(len(result['draws']),4)
        self.assertEqual(len({d['seed'] for d in result['draws']}),4)
        for deliver in (False,True):
            reference=copy.deepcopy(a)
            if deliver:reference.assimilate(p)
            expected=reference.update(np.array(a.current_state))
            for draw in result['draws']:
                branch=draw['branches']['deliver' if deliver else 'withhold']
                self.assertEqual(branch['initial_update']['accepted'],expected['accepted'])
                self.assertEqual(branch['initial_update']['field_validation_after'],expected['field_validation_after'])
                np.testing.assert_array_equal(branch['initial_control'],reference.action(np.array(a.current_state))[0])
                self.assertEqual(branch['path'][0],list(a.current_state))
        repeat=a.evaluate_packet(p,seed=12)
        self.assertEqual(result['feedback'],repeat['feedback']);self.assertEqual(result['wire_bytes'],repeat['wire_bytes'])
        self.assertNotIn('cooperative_evaluation',inspect.getsource(packet_feedback))

    def test_operational_once_and_feedback_accounting(self):
        spec=dict(specification('complementary',60000,'smoke'),steps=9)
        r=execute(spec,'reciprocal_fixed',audits=False,replay_domain='retained',feedback_mode='receiver',replay_cap=1024,feedback_audits=True)
        self.assertTrue(r['feedback_records'])
        keys=[(f['receiver'],f['stage']) for f in r['feedback_records']];self.assertEqual(len(keys),len(set(keys)))
        for f in r['feedback_records']:
            initial=f['assessment']['draws'][0]['branches']['deliver']['initial_update']
            self.assertEqual(initial['accepted'],f['actual_accepted'])
            self.assertEqual(initial['version_after'],f['committed_version'])
            events=[e for e in r['packet_events'][f['receiver']] if e.get('packet')==f['packet']]
            self.assertEqual(len(events),1)
        self.assertEqual(r['summary']['feedback_bytes'],sum(f['feedback_bytes'] for f in r['feedback_records']))
        for a in r['agents']:
            self.assertEqual(a['bytes'],a['payload_bytes']+a['status_bytes']+a['ack_bytes'])
            self.assertAlmostEqual(a['cost'],sum(a['components'].values()))
        self.assertTrue(all(u['diagnostic']['counterfactual_steps']<=1024 for us in r['updates'] for u in us))

    def test_excluded_draws_and_unsent_feedback(self):
        from unittest.mock import patch
        a,p=self.fixture()
        invalid=replace(p,recipient='wrong')
        result=a.evaluate_packet(invalid)
        self.assertTrue(all(r['valid_draws']==0 and r['benefit'] is None for r in result['rows']))
        self.assertFalse(a.packet_events)
        spec=dict(specification('complementary',60000,'smoke'),steps=1)
        with patch.object(packet_feedback.PacketFeedback,'bytes',return_value=10**9):
            r=execute(spec,'reciprocal_fixed',audits=False,replay_domain='retained',feedback_mode='receiver',replay_cap=1024)
        self.assertTrue(r['feedback_records'])
        self.assertTrue(all(not f['feedback_sent'] and f['feedback_reason']=='byte_budget' for f in r['feedback_records']))
        self.assertGreater(r['summary']['feedback_seconds'],0)
        self.assertEqual(r['summary']['feedback_bytes'],0)
