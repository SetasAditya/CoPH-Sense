import copy
import unittest
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.attribution import AttributionLearner,ARMS
from coph_stagewise.decision_trace import DecisionLearner,route_ranking,path_reach
from coph_stagewise.decision_audits import branch_observer,decision_packet_audit
from coph_stagewise.decision_suite import specification,ForkWorld,NODES,EDGES,compare_operational
from coph_stagewise.maze_demo import configuration,specification as maze_spec,MazeWorld
from coph_stagewise.staggered_demo import execute,sense
from coph_stagewise.attribution_audits import digest
from coph_stagewise.model import tensor

class DecisionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def agent(self,arm='historical_replay',q=(0.,0.,0.,0.),agent_id='0'):
        cfg=configuration(40);spec=specification('direct_risk',78000);w=ForkWorld(spec,cfg)
        a=DecisionLearner(cfg,np.array(q),[8.,0.],attribution=ARMS[arm],layout=(NODES,EDGES),agent_id=agent_id)
        a.observe(sense(w,np.array(q),1,0.));return a,w

    def test_rank_accounting_and_ties(self):
        ctx={'network':(NODES,EDGES)};r=route_ranking([0.,0.],[8.,0.],ctx)
        self.assertAlmostEqual(r['best']['cost'],40.)
        self.assertAlmostEqual(r['second']['cost'],57.)
        self.assertAlmostEqual(r['margin'],17.)
        self.assertEqual(r,route_ranking([0.,0.],[8.,0.],ctx))
        for e in r['quadrature']:self.assertAlmostEqual(e['cost'],sum(l*(1+4*v) for l,v in zip(e['lengths'],e['values'])))

    def test_ranking_matches_operational_route(self):
        for q in ((0.,0.,0.,0.),(.4,0.,0.,0.),(1.5,0.,0.,0.),(3.,0.,0.,0.)):
            a,w=self.agent('reconstruction',q);a.update(np.array(q));a.action(np.array(q))
            ids=list(a.last_control['route_nodes']);nodes=np.asarray(NODES)
            while ids and np.linalg.norm(nodes[ids[0]]-np.asarray(q[:2]))<1e-9:ids.pop(0)
            self.assertEqual(ids,a.last_control['decision_route']['best']['nodes'])

    def test_legacy_missing_fields_are_explicit(self):
        from coph_stagewise.decision_reporting import episode_summary
        import json
        s=maze_spec(75000);s['steps']=2
        r=execute(s,'disabled',audit=False,learner_factory=AttributionLearner,world_factory=MazeWorld,config=configuration(2))
        summary=episode_summary(r)
        self.assertIsNone(summary['counts']['backtracking_accepted']);self.assertTrue(summary['unavailable'])
        json.dumps(summary,allow_nan=False)

    def test_off_center_departure_not_a_second_route(self):
        ctx={'network':(NODES,EDGES)}
        r=route_ranking([.4,.05],[8.,0.],ctx)
        self.assertEqual(r['best']['nodes'],[1,2,3])
        self.assertEqual(r['second']['nodes'],[1,4,5,2,3])
        self.assertGreater(r['margin'],16.)

    def test_truth_support_does_not_expand(self):
        a,w=self.agent();before=digest(a)
        r=route_ranking([0.,0.],a.goal,a.navigation,w.risk_truth,'supported_truth')
        full=route_ranking([0.,0.],a.goal,a.navigation,w.risk_truth,'full_truth')
        self.assertAlmostEqual(r['best']['cost'],40.)
        self.assertNotEqual(r['best']['nodes'],full['best']['nodes'])
        self.assertEqual(before,digest(a))

    def test_fork_geometry_and_static_scenes(self):
        for case in ('clear_fork','direct_risk','bypass_risk'):
            s=specification(case,78000);self.assertFalse(s['obstacles']);self.assertEqual(s['start_seconds'],[0.,5.])
            w=ForkWorld(s,configuration(500));self.assertTrue(w.step(np.array([4.,4.,0.,0.]),np.zeros(2))[1])
        self.assertEqual(specification('bypass_risk',78000)['patches'][0],[3.6,1.7,.5,.38,4.])

    def test_branch_entry_requires_junction_visit(self):
        ctx={'network':(NODES,EDGES)}
        self.assertTrue(path_reach([[1.4,0.],[1.5,0.],[2.,0.]],ctx)['reaches_decision'])
        self.assertFalse(path_reach([[3.,0.],[4.,0.]],ctx)['reaches_decision'])
        self.assertFalse(path_reach([[1.4,0.],[1.55,0.]],ctx)['reaches_decision'])

    def test_instrumentation_keeps_operational_results(self):
        s=maze_spec(75000);s['steps']=56;cfg=configuration(56)
        left=execute(s,'reciprocal_fixed',audit=False,learner_factory=AttributionLearner,world_factory=MazeWorld,config=cfg)
        right=execute(s,'reciprocal_fixed',audit=False,learner_factory=DecisionLearner,world_factory=MazeWorld,config=cfg)
        self.assertGreater(len(left['packets']),0)
        self.assertTrue(compare_operational(left,right)['passed'],compare_operational(left,right))

    def test_policy_bookkeeping_and_rollback(self):
        a,w=self.agent('reconstruction');before=a.policy.clone();r=a.update(np.array(a.current_state))
        self.assertFalse(r['policy_trace']['backtracking_accepted']);self.assertFalse(r['policy_trace']['policy_changed'])
        self.assertTrue(torch.equal(before,a.policy));self.assertLessEqual(r['counterfactual_steps'],1024)
        a,w=self.agent();calls=[]
        def validation(_):calls.append(1);return 0. if len(calls)==1 else float('inf')
        a._field_validation=validation;before=a.policy.clone();version=a.version;r=a.update(np.array(a.current_state))
        self.assertFalse(r['accepted']);self.assertEqual(a.version,version);self.assertTrue(torch.equal(before,a.policy))
        self.assertEqual(r['policy_trace']['committed_delta'],[0.,0.])

    def test_sensitivity_and_oracle_isolation(self):
        a,w=self.agent('reconstruction');a.update(np.array(a.current_state));a.policy[0]=a.cfg.policy_bounds[0]
        before=digest(a);wb=digest(w);d=branch_observer(a,np.array(a.current_state),w)
        self.assertEqual(before,digest(a));self.assertEqual(wb,digest(w));self.assertEqual(d['sensitivity'][0]['scheme'],'one_sided')
        for s in d['sensitivity']:self.assertTrue(np.isfinite(s['applied_derivative']).all())
        self.assertEqual(d['control_evaluations'],5)

    def test_packet_audit_isolated_and_force_sum(self):
        sender,w=self.agent('reconstruction',(1.,0.,0.,0.),'0');receiver,_=self.agent('reconstruction',(1.,0.,0.,0.),'1')
        p=sender.export(np.array(sender.current_state),'1');before=digest(receiver)
        r=decision_packet_audit(receiver,np.array(receiver.current_state),p,w,1)
        self.assertEqual(before,digest(receiver));self.assertTrue(r['operational_state_unchanged'])
        for b in r['branches'].values():
            f=b['decision_audit']['forces'];np.testing.assert_allclose(np.array(f['attraction'])+f['soft_shaping']+np.array(f['damping']),f['nominal'],atol=1e-12)
            np.testing.assert_allclose(np.array(f['nominal'])+f['clipping']+np.array(f['safety']),f['applied'],atol=1e-12)

if __name__=='__main__':unittest.main()
