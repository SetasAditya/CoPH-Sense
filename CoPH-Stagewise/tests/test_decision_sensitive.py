import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from coph_stagewise import decision_sensitive as d
from coph_stagewise.receiver_corrections import load_saved_snapshot
from coph_stagewise.attribution_audits import digest
from coph_stagewise.learner import Learner

@unittest.skipUnless((Path('results/receiver_corrections_v1/snapshots/complementary_79200_acquisition.pkl.gz')).exists(), 'Requires preserved experiment archives; see docs/DEVELOPMENT.md')
class DecisionSensitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ctx=load_saved_snapshot('results/receiver_corrections_v1/snapshots/complementary_79200_acquisition.pkl.gz')
    def test_grid_deterministic(self):
        self.assertEqual(d.grid(),d.grid());self.assertEqual(len(d.grid()),20)
        for name,g in d.grid().items():
            s=d.specification(name,79300);self.assertEqual(s['steps'],500);self.assertEqual(s['start_seconds'],[0.,0.]);self.assertFalse(s['obstacles'])
            if g['patches']:self.assertAlmostEqual(d.patch_integral(d.NODES[1],d.NODES[2],g['patches'][0]),g['ratio']*d.ESTAR,places=12)
    def test_integral_truth_and_bypass_tails(self):
        s=d.specification('x2.5_e1.6',79300);p=s['patches'][0];a=np.array(d.NODES[1]);b=np.array(d.NODES[2]);t=np.linspace(0,1,10001);q=a+t[:,None]*(b-a)
        y=p[4]*np.exp(-.5*((q[:,0]-p[0])/p[2])**2-.5*((q[:,1]-p[1])/p[3])**2)
        self.assertAlmostEqual(d.patch_integral(a,b,p),np.trapezoid(y,t)*np.linalg.norm(b-a),places=8)
        self.assertGreater(d.truth_costs(s)['bypass'],7.4*1.48)
        truth=[d.truth_costs(d.specification(f'x3_e{r:g}',79300))['margin'] for r in d.RATIOS]
        self.assertLess(min(truth),0);self.assertGreater(max(truth),0)
    def test_fixed_branch_score(self):
        a=self.ctx['agents'][1];x=self.ctx['states'][1];s=d.branch_scores(a,x)
        self.assertAlmostEqual(s['margin'],s['quadrature'][0]['cost']-sum(e['cost'] for e in s['quadrature'][1:]),places=12)
        self.assertTrue(all(-1 not in e['edge'] for e in s['quadrature']))
        self.assertAlmostEqual(s['margin'],d.context_margin(a.navigation,s['quadrature']),places=12)
    def test_design_parity_and_shared_blocks(self):
        a=copy.deepcopy(self.ctx['agents'][1]);s=copy.deepcopy(a.stages[0]);s.id+=1;s.anchors=tuple(r.q for r in s.readings if int(r.uid.split(':')[-1])%5)[:4];a.stages.append(s)
        design=d.bootstrap_design(a);weights,_=Learner._field_fit(a);w=d.bootstrap_fit(design,np.ones(len(design['blocks'])))
        torch.testing.assert_close(w,torch.cat(weights),rtol=1e-12,atol=1e-12)
        self.assertTrue(np.array_equal(design['blocks'][:len(design['blocks'])//2],design['blocks'][len(design['blocks'])//2:]))
        self.assertTrue(all(int(uid.split(':')[-1])%5 for uid in design['uids']));self.assertGreater(len(design['overlap']),0)
    def test_bootstrap_isolation_repeatability(self):
        a=self.ctx['agents'][1];x=self.ctx['states'][1];before=digest(a)
        u=d.bootstrap_range(a,x,123,4);v=d.bootstrap_range(a,x,123,4)
        for z in (u,v):z.pop('seconds')
        self.assertEqual(u,v);self.assertEqual(digest(a),before)
    def test_gate_packet_isolation(self):
        ctx=copy.deepcopy(self.ctx);sender=copy.deepcopy(ctx['agents'][0]);m=sender.export(ctx['states'][0],'1');before=digest(ctx)
        with patch.object(d,'bootstrap_range',return_value=dict(available=True,straddles=False,interval=[2,3])):r=d.inspect_snapshot(ctx,m,12)
        self.assertTrue(r['isolated']);self.assertTrue(r['valid_evidence']);self.assertEqual(digest(ctx),before)
    def test_gate_selection_controls_and_no_outcome_reads(self):
        scenes={'grid':dict(control=False),'control':dict(control=True)};rows=[dict(scene=s,passed=True) for s in scenes for _ in d.SEEDS]
        self.assertFalse(d.selected_scenes(rows,scenes)['stop']);rows[0]['passed']=False
        self.assertTrue(d.selected_scenes(rows,scenes)['stop']);self.assertEqual(d.selected_scenes(rows,scenes)['controls'],[])
        checks=dict(support=True,evidence=True,margin=False,uncertainty=False)
        self.assertTrue(d.pair_pass(checks,True,False));self.assertFalse(d.pair_pass(checks,False,True))
        import inspect
        self.assertNotIn('episodes',inspect.getsource(d.screen_job));self.assertNotIn('status',inspect.getsource(d.screen_job))
    def test_screen_shadow_exports_and_branch_entry(self):
        ctx=copy.deepcopy(self.ctx);capture=d.ScreenCapture(ctx['agents'][1].cfg);before=digest(ctx);capture.boundary(ctx)
        self.assertEqual(capture.cursor,1);self.assertEqual(digest(ctx),before)
        self.assertEqual(d.observed_branch([dict(time=1,x=[2,0,0,0])])['branch'],'direct')
        self.assertEqual(d.observed_branch([dict(time=1,x=[1.5,.5,0,0])])['branch'],'bypass')
        self.assertFalse(d.correct_branch(None,dict(optimal=['direct','bypass'])))
        self.assertTrue(d.correct_branch('direct',dict(optimal=['direct','bypass'])))
    def test_analysis(self):
        pairs=[dict(seed=s,local=0,corrected=1,scene='a') for s in d.SEEDS]
        a=d.primary_analysis(pairs,100);self.assertEqual(a,d.primary_analysis(pairs,100));self.assertEqual(a['corrected_only'],8);self.assertAlmostEqual(a['mcnemar_p'],2/256);self.assertEqual(a['interval'],[1.,1.])
        self.assertEqual(d.primary_analysis([dict(seed=s,local=1,corrected=1) for s in d.SEEDS],20)['mcnemar_p'],1.)
        self.assertFalse(d.primary_analysis([])['available'])
    def test_freeze_resume_and_treatment_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run'
            with patch.object(d,'make_manifest',return_value=dict(value=1)),patch.object(d,'archive_sources'):
                a=d.freeze(root);b=d.freeze(root);self.assertEqual(a,b)
            with patch.object(d,'make_manifest',return_value=dict(value=2)):
                with self.assertRaises(ValueError):d.freeze(root)
            with patch.object(d,'verify_frozen',return_value=dict()):
                with self.assertRaises(FileNotFoundError):d.verify_treatment_gate(root,'x')

if __name__=='__main__':unittest.main()
