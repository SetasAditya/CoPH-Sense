import copy,pickle,unittest
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.staggered_demo import specification,execute,sense,packet_effect
from coph_stagewise.records import Config
from coph_stagewise.cooperative import CooperativeLearner
from coph_stagewise.cooperative_evaluation import CooperativeWorld

class StaggeredTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_exact_start_full_local_horizon_and_no_prestart_observations(self):
        s=specification('slalom_passage',73000);s['steps']=2
        r=execute(s,'reciprocal_fixed',audit=False)
        self.assertEqual([a['start_time'] for a in r['agents']],[0.,5.])
        self.assertEqual([a['finish_time'] for a in r['agents']],[.24,5.24])
        self.assertEqual([len(t) for t in r['trajectories']],[2,2]);self.assertFalse(r['packets'])
        np.testing.assert_array_equal(r['trajectories'][0][0]['x'],r['trajectories'][1][0]['x'])
        self.assertEqual(r['samples'][1][0]['time'],5.)
        self.assertTrue(all(o['time']>=5 and all(z['time']==o['time'] for z in o['readings']) for o in r['observations'][1]))
        self.assertEqual(r['agents'][1]['components']['time'],.24)
    def test_asynchronous_delivery_clock_and_matched_updates(self):
        s=specification('slalom_passage',73000);s.update(steps=12,start_seconds=[0.,.2])
        a=execute(s,'disabled',audit=False);b=execute(s,'reciprocal_fixed',audit=False)
        self.assertEqual([[u['time'] for u in seq] for seq in a['updates']],[[u['time'] for u in seq] for seq in b['updates']])
        self.assertTrue(b['packets'])
        for e in b['events']:
            if e['kind']=='delivered':
                self.assertGreaterEqual(e['age'],-1e-12);self.assertLessEqual(e['age'],.12+1e-9)
                self.assertIn(e['time'],[s['time'] for s in b['samples'][e['receiver']]])
                self.assertNotEqual(e['reason'],'invalid_age')
        self.assertTrue(all(a['max_update_work']<=1024 for a in b['agents']))
    def test_packet_audit_isolation(self):
        cfg=Config(controller_mode='routed',shaping_mode='progress',replay_domain='retained',replay_objective='mission',replay_horizon_mode='supported_boundary')
        s=specification('slalom_passage',73000);w=CooperativeWorld(s,cfg);x=np.array([0.,0.,0.,0.]);agents=[]
        for i in range(2):
            a=CooperativeLearner(cfg,x,s['goal'],agent_id=str(i));a.observe(sense(w,x,0,5.));agents.append(a)
        m=agents[0].export(x,'1');before=pickle.dumps(agents[1]);r=packet_effect(agents[1],x,m)
        self.assertEqual(before,pickle.dumps(agents[1]));self.assertTrue(r['operational_state_unchanged']);self.assertIn('withhold',r['branches'])
    def test_scenes_deterministic_and_shared_origin(self):
        for name in ('slalom_passage','long_corridor'):
            a=specification(name,73000);self.assertEqual(a,specification(name,73000));self.assertEqual(a['initials'][0],a['initials'][1]);self.assertEqual(a['goals'][0],a['goals'][1]);self.assertEqual(a['start_seconds'],[0.,5.])
