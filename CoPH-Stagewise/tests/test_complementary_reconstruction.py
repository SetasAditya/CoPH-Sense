import copy,pickle,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise import complementary_reconstruction as cr
from coph_stagewise.staggered_demo import execute
from coph_stagewise.maze_family import NetworkWorld
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.hazard_demo import specification,communication
from coph_stagewise.receiver_reconstruction import compare
from coph_stagewise.reconstruction_diagnostics import receiver_audit,ranking
from coph_stagewise.evaluation import dump,load

class ComplementaryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_frozen_specification(self):
        for case in cr.CASES:
            s=cr.specification(case,79100)
            self.assertEqual(s['patches'],[[1.25,.45,.6,1.,.5]])
            self.assertEqual(s['start_seconds'],[0.,0.]);self.assertEqual(s['steps'],500)
            self.assertEqual(s['communication_start_seconds'],.96);self.assertEqual(s['obstacles'],[])
            self.assertEqual(s['risk_weights'],[.25,6.]);self.assertEqual(s['damping'],.85)
        self.assertEqual(cr.specification('reverse',79100)['initials'],cr.specification('complementary',79100)['initials'][::-1])
    def test_branch_marker(self):
        self.assertIsNone(cr.branch_entry((1.4,0.)))
        self.assertIsNone(cr.branch_entry((1.5,.24)))
        self.assertEqual(cr.branch_entry((1.8,0.)),[1,2])
        self.assertEqual(cr.branch_entry((1.5,.3)),[1,4])
        self.assertIsNone(cr.branch_entry((2.2,.8)))
    def test_prefix_roundtrip_and_changes(self):
        p={'observations':[{'q':(.2,.3),'value':1.}], 'navigation':{'route_field_mode':'anchors'}}
        import json
        self.assertTrue(cr.equivalent_prefix(p,json.loads(json.dumps(p)))['passed'])
        q=copy.deepcopy(p);q['observations'][0]['value']=1.1
        self.assertFalse(cr.equivalent_prefix(p,q)['passed'])
    def test_callbacks_and_acquisition_schedule(self):
        # Existing synthetic hazard fixture, not the official complementary outcome.
        s=dict(specification('mild'),steps=10,start_seconds=[0.,0.]);cfg=configuration(10);comm=communication('field')
        kw=dict(spec=s,arm='reciprocal_fixed',comm=comm,audit=False,learner_factory=cr.factory_for(s,False),world_factory=NetworkWorld,config=cfg)
        old=execute(**kw);seen=[]
        new=execute(**kw,boundary_callback=lambda c:seen.append(c['time']),delivery_callback=lambda *args:None)
        self.assertTrue(compare(new,old)['passed']);self.assertIn(.96,seen)
        delayed=dict(s,communication_start_seconds=.96);events=[]
        r=execute(**dict(kw,spec=delayed),delivery_callback=lambda *args:events.append(args[2].timestamp))
        self.assertTrue(events);self.assertEqual(min(events),.96)
        ctx=cr.acquire(delayed,cfg,comm)
        actual=dict(observations=[[o for o in z if o['time']<=.96] for z in r['observations']],
            samples=[[o for o in z if o['time']<=.96] for z in r['samples']],
            trajectories=[[o for o in z if o['end_time']<=.96] for z in r['trajectories']],
            updates=[[o for o in z if o['time']<.96] for z in r['updates']])
        self.assertTrue(cr.equivalent_prefix(cr.prefix(ctx),cr.prefix(actual))['passed'])
    def test_proposals_and_isolation(self):
        from test_receiver_reconstruction import ReceiverReconstructionTests
        helper=ReceiverReconstructionTests();helper.setUp()
        sender,xx,_=helper.agent(0,(6.,4.));a,x,w=helper.agent();m=sender.export(xx,'1');before=pickle.dumps((a,x,w))
        r=receiver_audit(a,x,m,w,2,capture_proposals=True)
        self.assertEqual(before,pickle.dumps((a,x,w)))
        self.assertTrue(r['branches']['fit']['proposal']['proposed_fields'])
        self.assertGreater(r['branches']['fit']['message_fit_work'],0)
        self.assertEqual(r['branches']['no_fit']['message_fit_work'],0)
        self.assertTrue(r['branches']['fit']['remote_anchors_identical'])
        for b in r['branches'].values():self.assertLessEqual(b['replay_work'],1024)
    def test_snapshot_selection(self):
        from types import SimpleNamespace
        c=cr.DeliverySnapshots({});a=SimpleNamespace(agent_id='1',stages=[SimpleNamespace(id=0)])
        m=SimpleNamespace(identity='first');x=np.array([.4,0.,0.,0.]);w={}
        c.delivery(a,x,m,w,0);m.identity='second';c.delivery(a,x,m,w,1)
        a.stages[0].id=8;x[0]=9.
        self.assertEqual(c.selected['first']['snapshot'][0].stages[0].id,0)
        c.boundary(dict(time=1.2,states=[np.zeros(4),np.array([1.8,0.,0.,0.])]))
        self.assertIn('second',c.selected);self.assertEqual(c.selected['second']['snapshot'][1][0],.4)
    def test_gate_failure_blocks_comparisons(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'result'
            fake=dict(cr.manifest(),specs={'synthetic':dict(scene='complementary',seed=1)})
            row=dict(required=True,passed=False,scene='complementary',seed=1,directions=[])
            with patch.object(cr,'manifest',return_value=fake),patch.object(cr,'archive_sources'),patch.object(cr,'acquire',return_value={}),patch.object(cr,'prefix',return_value={}),patch.object(cr,'gate_snapshot',return_value=(row,[])),patch.object(cr,'run_episode') as run,patch('coph_stagewise.complementary_reporting.report'),patch('coph_stagewise.complementary_reporting.render'):
                cr.suite(root);run.assert_not_called();self.assertFalse(load(root/'gate.json')['passed'])
    def test_independent_noise_at_redundant_coordinates(self):
        # Direct sensing only; no official acquisition or comparative outcomes.
        s=cr.specification('redundant',79100);cfg=configuration(500);x=np.array(s['initials'][0]);read=[]
        for i in (0,1):
            w=NetworkWorld(s,cfg);w.seed=2*s['seed']+i;read.append(w.sense(x,0).readings)
        self.assertEqual([r.q for r in read[0]],[r.q for r in read[1]])
        self.assertNotEqual([r.value for r in read[0]],[r.value for r in read[1]])

if __name__=='__main__':unittest.main()
