import copy
import pickle
import unittest
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.attribution import ARMS, AttributionLearner
from coph_stagewise.maze_demo import configuration, specification, MazeWorld
from coph_stagewise.staggered_demo import sense
from coph_stagewise.stage_prediction import FrozenStages, replay_entries
from coph_stagewise.attribution_audits import packet_audit


class AttributionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=configuration(40)

    def agent(self, arm, agent_id='1', x=(3.,0.,0.,0.)):
        w=MazeWorld(specification(75000),self.cfg)
        a=AttributionLearner(self.cfg,np.array(x),[12.,0.],agent_id=agent_id,attribution=ARMS[arm])
        a.observe(sense(w,np.array(x),1,10.))
        return a,w

    def packet(self):
        a,_=self.agent('historical_replay','0',(4.,0.,0.,0.))
        return a.export(np.array(a.current_state),'1')

    def test_channel_isolation_and_provenance(self):
        packet=self.packet()
        for arm,cfg in ARMS.items():
            with self.subTest(arm=arm):
                a,_=self.agent(arm)
                measurements=pickle.dumps([(s.readings,s.transitions) for s in a.stages])
                physical=a.current_state
                e=a.assimilate(packet)
                self.assertEqual(e['reason'],'eligible')
                self.assertEqual(bool(a.remote_geometry_reports),cfg.geometry)
                self.assertEqual(bool(a.remote_cells),cfg.route_anchors)
                self.assertEqual(bool(a.received),cfg.field_assimilation)
                self.assertEqual(measurements,pickle.dumps([(s.readings,s.transitions) for s in a.stages]))
                self.assertEqual(physical,a.current_state)
                self.assertEqual(a.assimilate(packet)['reason'],'duplicate_or_reordered')
                self.assertFalse(a.export(np.array(physical),'0').geometry)

    def test_disabled_components_still_validate(self):
        a,_=self.agent('local_evidence')
        m=replace(self.packet(),geometry_schema=9)
        self.assertEqual(a.assimilate(m)['reason'],'incompatible_geometry')
        self.assertFalse(a.received or a.remote_cells or a.remote_geometry_reports)

    def test_no_remote_support_certification(self):
        a,_=self.agent('historical_replay',x=(2.,0.,0.,0.))
        before=FrozenStages.freeze(a)
        event=a.assimilate(self.packet())
        self.assertEqual(event['reason'],'eligible')
        self.assertFalse(event['components']['field_assimilation']['eligible'])
        self.assertTrue(event['components']['route_anchors']['used'])
        self.assertEqual([s.support for s in before.stages],[s.support for s in FrozenStages.freeze(a).stages])

    def test_rejected_reconstruction_rolls_back(self):
        a,_=self.agent('reconstruction')
        a.assimilate(self.packet())
        weights=[s.weights.clone() for s in a.stages];policy=a.policy.clone();version=a.version
        calls=[]
        def validation(_):
            calls.append(1)
            return 0. if len(calls)==1 else float('inf')
        a._field_validation=validation
        report=a.update(np.array(a.current_state))
        self.assertFalse(report['accepted'])
        self.assertEqual(a.version,version)
        self.assertTrue(torch.equal(policy,a.policy))
        for old,s in zip(weights,a.stages):self.assertTrue(torch.equal(old,s.weights))

    def test_scenarios_frozen_and_noise_varies(self):
        from coph_stagewise.attribution_suite import specification as spec
        blocked=spec('blocked_maze',75000)
        self.assertEqual(blocked,specification(75000))
        risk=spec('soft_risk_maze',75000)
        self.assertFalse(risk['obstacles'] or risk['gate_groups'])
        self.assertEqual(risk['network_edges'],blocked['network_edges'])
        self.assertEqual(risk['patches'],[[x,y,.5,.38,4.] for x,y,_ in blocked['gate_groups']])
        w=MazeWorld(risk,self.cfg);v=MazeWorld(spec('soft_risk_maze',75001),self.cfg)
        x=np.zeros(4)
        self.assertNotEqual(w.sense(x,0).readings[0].value,v.sense(x,0).readings[0].value)

    def test_report_masks_missing_predictions(self):
        from coph_stagewise.attribution_reporting import paired_probes
        a=dict(support=[True,False],coverage=.5,squared_error=[1.,None],gradient_squared_error=[2.,None])
        b=dict(support=[False,True],coverage=.5,squared_error=[None,3.],gradient_squared_error=[None,4.])
        left=[[dict(tick=0,evaluation=a)],[]];right=[[dict(tick=0,evaluation=b)],[]]
        row=paired_probes(left,right)[0]
        self.assertEqual(row['common_probes'],0)
        self.assertIsNone(row['squared_error_left'])
        self.assertFalse(paired_probes(left,[[],[]]))

    def test_full_attribution_preserves_legacy_maze_decisions(self):
        from coph_stagewise.maze_learning import MazeLearner
        a,w=self.agent('historical_replay')
        b=MazeLearner(self.cfg,np.array(a.current_state),[12.,0.],agent_id='1')
        b.observe(sense(w,np.array(a.current_state),1,10.))
        packet=self.packet()
        a.assimilate(packet);b.assimilate(packet)
        ra=a.update(np.array(a.current_state));rb=b.update(np.array(b.current_state))
        self.assertEqual(ra['accepted'],rb['accepted'])
        np.testing.assert_array_equal(a.action(np.array(a.current_state))[0],b.action(np.array(b.current_state))[0])
        for s,t in zip(a.stages,b.stages):self.assertTrue(torch.equal(s.weights,t.weights))

    def test_current_start_is_single_and_does_not_edit_entry(self):
        a,_=self.agent('current_replay')
        entries=pickle.dumps([s.entry for s in a.stages])
        x=np.array(a.current_state)+np.array([.03,0.,0.,0.])
        d=a.update(x)
        starts=replay_entries(a,FrozenStages.freeze(a))
        self.assertEqual(len(starts),1)
        self.assertEqual(starts[0][1].entry,tuple(x))
        self.assertEqual(entries,pickle.dumps([s.entry for s in a.stages]))
        self.assertLessEqual(d['counterfactual_steps'],1024)
        self.assertEqual(d['replay_plan'][0]['entry'],tuple(x))

    def test_audit_isolation_shared_draws_and_checkpoints(self):
        a,w=self.agent('reconstruction')
        before=pickle.dumps(a)
        r=packet_audit(a,np.array(a.current_state),self.packet(),w,1)
        self.assertEqual(before,pickle.dumps(a))
        self.assertTrue(r['randomness']['shared_between_branches'])
        for b in r['branches'].values():
            for h in ('8','16','32'):
                c=b['checkpoints'][h]
                self.assertEqual(c['path'][0],list(a.current_state))
                self.assertAlmostEqual(c['cost'],sum(c['components'].values()))
        self.assertFalse(a.received)
        a.assimilate(self.packet())
        self.assertEqual(a.assimilate(self.packet())['reason'],'duplicate_or_reordered')


if __name__=='__main__':unittest.main()
