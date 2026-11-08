import copy
import unittest
import numpy as np
import torch
from coph_stagewise.records import Config
from coph_stagewise.cooperative import CooperativeLearner,CommunicationConfig
from coph_stagewise.cooperative_evaluation import specification,CooperativeWorld,execute
from coph_stagewise.recipient_diagnostics import diagnose,continuation,FrozenWorld,summarize,DomainExit
from coph_stagewise.diagnostics import snapshot

class RecipientDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=Config(steps=40,replay_steps=3,replay_objective='mission')
        self.spec=specification('complementary',60000,'development')
        self.world=CooperativeWorld(self.spec,self.cfg)
        self.x=np.array([0.,0.,0.,0.])
        self.agents=[CooperativeLearner(self.cfg,self.x,(6.,0.),agent_id=str(i),communication=CommunicationConfig()) for i in range(2)]
        for a in self.agents:
            for tick in range(3):a.observe(self.world.sense(self.x,tick))
        self.packet=self.agents[0].export(self.x,'1')

    def test_isolation_and_common_starts(self):
        a=self.agents[1];before=copy.deepcopy(snapshot(a));readings=copy.deepcopy(a.stages[0].readings)
        counters=(a.counterfactual_steps,a.identification_steps,a.export_work)
        out=diagnose(a,self.x,self.world,self.packet,2,(1,2))
        self.assertEqual(before,snapshot(a));self.assertEqual(readings,a.stages[0].readings)
        self.assertEqual(counters,(a.counterfactual_steps,a.identification_steps,a.export_work))
        self.assertEqual(len(out['cells']),4)
        for cell in out['cells'].values():
            self.assertEqual(cell['start'],self.x.tolist())
            for b in cell['branches'].values():
                self.assertEqual(b['path'][0],self.x.tolist())
                for c in b['checkpoints'].values():
                    if c['valid']:self.assertAlmostEqual(c['cost'],sum(c['components'].values()))

    def test_prefixes_and_unsupported_domain(self):
        a=self.agents[1]
        short=continuation(a,self.x,self.world,self.packet,2,(1,))
        long=continuation(a,self.x,self.world,self.packet,2,(1,3))
        for name in ('deliver','withhold'):
            self.assertEqual(short['branches'][name]['checkpoints']['1'],long['branches'][name]['checkpoints']['1'])
        frozen=FrozenWorld(a,self.world.seed)
        with self.assertRaises(DomainExit):frozen.risk_truth(np.array([100.,100.]))
        out=continuation(a,np.array([100.,100.,0.,0.]),frozen,self.packet,2,(1,))
        self.assertFalse(out['rows'][0]['valid']);self.assertIsNone(out['rows'][0]['benefit'])

    def test_callback_does_not_change_execution(self):
        spec=dict(self.spec,steps=9)
        baseline=execute(spec,'reciprocal_fixed')
        calls=[]
        def callback(a,x,w,m,t):
            calls.append(t);a.policy*=0;x[:]=99;w._damping=99
        audited=execute(spec,'reciprocal_fixed',diagnostic_callback=callback)
        self.assertTrue(calls)
        for key in ('trajectories','packet_events','packets','final_models'):
            if key in baseline:self.assertEqual(baseline[key],audited[key])
        self.assertEqual(baseline['summary']['team_cost'],audited['summary']['team_cost'])

    def test_common_mask_and_missing_advertisement(self):
        case=dict(scene='test',seed=0,packet='p',horizons=[8],cells={})
        for c in ('exact_true','exact_frozen','advertised_true'):
            case['cells'][c]=dict(rows=[dict(horizon=8,valid=True,benefit=1.)])
        rows,summary=summarize([case])
        self.assertEqual(len(rows),3)
        self.assertTrue(all(s['common_valid']==0 and s['mae'] is None for s in summary))
