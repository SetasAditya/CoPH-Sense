import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from coph_stagewise.historical_policy_replay import SharedPolicyReplayLearner
from coph_stagewise.route_decision import RouteDecisionLearner,RouteDecisionOptions,evaluate_routes
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.records import new_stage,Reading
from coph_stagewise.model import tensor


def learner(cls=SharedPolicyReplayLearner,cap=1024,actual='graph'):
    cfg=replace(configuration(64),replay_horizon_mode='fixed',replay_update_cap=cap)
    nodes=tuple((float(k),0.) for k in range(5));edges=tuple((k,k+1) for k in range(4))
    a=cls(cfg,(.4,0.,0.,0.),(4.,0.),agent_id='0',mode='full',network=(nodes,edges),
        risk_weight=1.,unknown_risk=.02,route_options=RouteDecisionOptions(mode=actual,starts='current'))
    second=new_stage(1,(2.5,0.),(2.5,0.,0.,0.),(4.,0.),cfg)
    a.stages.append(second)
    for stage in a.stages:
        stage.entry_navigation=copy.deepcopy(a.navigation)
        stage.readings=[Reading(f'0:fixture:{k}',k*.01,(float(q),0.),.04+.008*np.sin(q))
            for k,q in enumerate(np.linspace(stage.center[0]-1.4,stage.center[0]+1.4,41))]
    a.current_state=(2.7,0.,0.,0.)
    # Valid current-state route identities must not leak into earlier starts.
    a.navigation.update(route_candidate_schema=1,route_candidate_nodes=[3,4])
    return a


class HistoricalPolicyReplayTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_existing_policy_replay_parity(self):
        original=learner(RouteDecisionLearner);corrected=learner()
        x=np.array([2.7,0.,0.,0.]);d0=original.update(x.copy());d1=corrected.update(x.copy())
        self.assertEqual(d0['accepted'],d1['accepted'])
        self.assertEqual(d0['reason'],d1['reason'])
        self.assertEqual(d0['counterfactual_steps'],d1['counterfactual_steps'])
        self.assertTrue(torch.equal(original.policy,corrected.policy))
        self.assertEqual(original.damping,corrected.damping)
        for a,b in zip(original.stages,corrected.stages):self.assertTrue(torch.equal(a.weights,b.weights))
        np.testing.assert_array_equal(original.action(x.copy())[0],corrected.action(x.copy())[0])
    def test_each_entry_replans_from_own_state(self):
        a=learner();weights=[s.weights for s in a.stages]
        cost,paths=a.replay(a.policy.clone().requires_grad_(True),weights,a.damping)
        self.assertIsNotNone(cost)
        first=a.shared_policy_calls[-1]['first_controls']
        self.assertEqual([p[1][0] for p in paths],[tuple(s.entry) for s in a.stages])
        self.assertNotEqual(first[0]['route_nodes'],first[1]['route_nodes'])
        self.assertNotEqual(first[0]['route_nodes'],a.navigation['route_candidate_nodes'])
        self.assertEqual(first[0]['route_nodes'][-1],4)
        self.assertEqual(first[1]['route_nodes'][-1],4)
        self.assertEqual(a.navigation['route_candidate_nodes'],[3,4])
    def test_multistep_gradients_and_fixed_entries(self):
        a=learner();p=a.policy.clone().requires_grad_(True);weights=[s.weights for s in a.stages]
        old_entries=[s.entry for s in a.stages];old_contexts=copy.deepcopy([s.entry_navigation for s in a.stages])
        value,_=a.replay(p,weights,a.damping);gradient=torch.autograd.grad(value,p)[0]
        step=tensor([1e-5,0.]);plus,_=a.replay(p.detach()+step,weights,a.damping);minus,_=a.replay(p.detach()-step,weights,a.damping)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual(float(gradient[0]),float((plus-minus)/2e-5),places=7)
        self.assertEqual(old_entries,[s.entry for s in a.stages]);self.assertEqual(old_contexts,[s.entry_navigation for s in a.stages])
        self.assertEqual(a.shared_policy_calls[0]['batch'],a.shared_policy_calls[1]['batch'])
    def test_actual_route_selection_current_only(self):
        a=learner(actual='prefix');x=np.array([2.7,0.,0.,0.])
        with patch('coph_stagewise.route_decision.evaluate_routes',wraps=evaluate_routes) as actual:
            d=a.update(x.copy())
        self.assertEqual(actual.call_count,1)
        self.assertEqual(actual.call_args.args[2].starts,'current')
        np.testing.assert_array_equal(actual.call_args.args[1],x)
        self.assertEqual(len(d['route_decision']['starts']),1)
        self.assertEqual(d['route_decision']['starts'][0]['label'],'current')
        with self.assertRaises(ValueError):
            a.route_options=replace(a.route_options,starts='historical');a.update(x)
    def test_transactional_rejection_and_flow_ceiling(self):
        a=learner();x=np.array([2.7,0.,0.,0.]);old=(a.version,a.policy.clone(),a.damping,[s.weights.clone() for s in a.stages])
        with patch.object(a,'_receiver_acceptance',return_value=(False,'test_rejection',dict(success=False))):
            d=a.update(x.copy())
        self.assertFalse(d['accepted']);self.assertEqual(a.version,old[0]);self.assertTrue(torch.equal(a.policy,old[1]));self.assertEqual(a.damping,old[2])
        for stage,w in zip(a.stages,old[3]):self.assertTrue(torch.equal(stage.weights,w))
        self.assertLessEqual(d['counterfactual_steps'],1024)
        tiny=learner(cap=64);dd=tiny.update(x.copy());self.assertLessEqual(dd['counterfactual_steps'],64)
        self.assertTrue(dd.get('replay_budget_exhausted',False))
    def test_batch_context_change_is_detected(self):
        a=learner();a._shared_active=True;a._shared_signature=None
        a.replay(a.policy,[s.weights for s in a.stages],a.damping)
        a.stages[0].entry_navigation['unknown_risk']=.7
        with self.assertRaises(RuntimeError):a.replay(a.policy,[s.weights for s in a.stages],a.damping)


if __name__=='__main__':unittest.main()
