import copy,unittest
from pathlib import Path
from coph_stagewise.routed_validation import readiness,check_specs,verify_sources,verify_episode,frame
from coph_stagewise.evaluation import load

ROOT=Path(__file__).resolve().parents[1]/'results/completion_first_v1'

@unittest.skipUnless((ROOT/'manifest.json').exists(), 'Requires preserved experiment archives; see docs/DEVELOPMENT.md')
class RoutedValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=load(ROOT/'manifest.json');cls.completed=load(ROOT/'development/completion.json')
    def test_gate_selects_routed_and_keeps_protected_failure(self):
        gate=readiness(self.completed,self.manifest['specs']['development'])
        self.assertTrue(gate['passed']);self.assertFalse(load(ROOT/'development_gate.json')['passed'])
        self.assertEqual([g['goals'] for g in gate['groups']],[64,64])
    def test_gate_rejects_missing_duplicate_wrong_seed_failure_incomplete(self):
        selected=copy.deepcopy([c for c in self.completed if c['mode']=='routed']);specs=self.manifest['specs']['development']
        for bad in (selected[:-1],selected+[selected[0]]):self.assertFalse(readiness(bad,specs)['passed'])
        for change in ('seed','failure','incomplete'):
            bad=copy.deepcopy(selected)
            if change=='seed':bad[0]['seed']=999
            if change=='failure':bad[0]['metrics'].update(goals=1,timeouts=1)
            if change=='incomplete':bad[0]['complete']=False
            self.assertFalse(readiness(bad,specs)['passed'])
    def test_frozen_sources_specs_and_runtime_config(self):
        self.assertTrue(verify_sources(ROOT)['verified']);check_specs(self.manifest['specs']['validation'])
        with self.assertRaises(AssertionError):check_specs(self.manifest['specs']['validation'][:-1])
        r=load(ROOT/'smoke/episodes/hazard_slalom_61000_routed_disabled.json.gz')
        verify_episode(r,r['spec'],'disabled',self.manifest['controllers']['routed'],self.manifest['communication'])
        bad=copy.deepcopy(r);bad['config']['shaping_eta']=.4
        with self.assertRaises(AssertionError):verify_episode(bad,r['spec'],'disabled',self.manifest['controllers']['routed'],self.manifest['communication'])
        bad=copy.deepcopy(r);bad['spec']['steps']+=1
        with self.assertRaises(AssertionError):verify_episode(bad,r['spec'],'disabled',self.manifest['controllers']['routed'],self.manifest['communication'])
    def test_synchronized_frame_uses_logs_without_mutation(self):
        rs={a:load(ROOT/f'smoke/episodes/hazard_slalom_61000_routed_{a}.json.gz') for a in ('disabled','reciprocal_fixed')}
        before=copy.deepcopy(rs);im=frame(rs,10**9);self.assertEqual(im.size,(1200,700));self.assertEqual(before,rs)
