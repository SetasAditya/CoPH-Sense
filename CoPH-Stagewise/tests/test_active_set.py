from pathlib import Path
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from coph_stagewise import receiver_fitting as rf
from coph_stagewise import solver_diagnosis as sd
from coph_stagewise.receiver_corrections import load_saved_snapshot,configure
from coph_stagewise.reconstruction_quality import _snapshot_jobs,SOURCE,_local

@unittest.skipUnless((Path('results/complementary_reconstruction_v1/audit_rows.json')).exists(), 'Requires preserved experiment archives; see docs/DEVELOPMENT.md')
class ActiveSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.snapshot=load_saved_snapshot(next(p for p,m in _snapshot_jobs(SOURCE)[0] if m['group']=='navigation'))
    def problem(self):
        a,x,m,_,_=copy.deepcopy(self.snapshot);a.assimilate(m);w,c=_local(a);info,_=rf.precision_information(a,w);s=a.stages[0];b,_=rf.blocks_for(a,s)
        return s,w[0],info[0],b,a.cfg.rbf_width
    def test_default_bitwise(self):
        args=self.problem();a,da=rf.solve(*args);b,db=rf._legacy_solve(*args);self.assertTrue(torch.equal(a,b));self.assertEqual(da,db)
    def test_candidate_port(self):
        s,w,H,b,width=self.problem();p=dict(stage=s,w0=w,H=H,blocks=b,width=width)
        g,*_=sd.gauss_newton(w,p);p['scale']=1+float(g.abs().max());p['initial']=float(sd.rf_objective(w,p))
        a,da=sd.active_set_gauss_newton(p);z,dz=rf.solve(s,w,H,b,width,solver_mode='active_set',search_mode='diagnostic')
        self.assertTrue(torch.equal(a,z))
        for k in ('status','iterations','objective_final','stationarity'):self.assertEqual(da[k],dz[k])
    def test_quadratic_boundary_and_free_system(self):
        s=SimpleNamespace(centers=None);w=torch.zeros(2,dtype=torch.float64);H=torch.tensor([[4.,1.],[1.,2.]],dtype=torch.float64);z=torch.tensor([5.,.1],dtype=torch.float64)
        b=[(None,z,torch.eye(2,dtype=torch.float64))]
        with patch.object(rf,'vector',side_effect=lambda v,*args:v):
            for search in ('legacy','diagnostic'):
                v,d=rf.solve(s,w,H,b,1.,solver_mode='active_set',search_mode=search)
                self.assertTrue(d['success']);np.testing.assert_allclose(v,[.5,-.4/3],atol=1e-9)
                self.assertLessEqual(d['total_correction'],.5);self.assertGreater(d['active_set_refinements'],0)
                self.assertTrue(all(t['halvings']<=12 for h in d['history'] for t in h['trials']) if search=='legacy' else True)
    def test_multiple_bounds_interior_and_budget(self):
        s=SimpleNamespace(centers=None);w=torch.zeros(2,dtype=torch.float64);H=torch.eye(2,dtype=torch.float64)
        with patch.object(rf,'vector',side_effect=lambda v,*args:v):
            for target in ([3.,-3.],[.1,-.1],[0.,0.]):
                b=[(None,torch.tensor(target,dtype=torch.float64),H)]
                v,d=rf.solve(s,w,H,b,1.,solver_mode='active_set');self.assertTrue(d['success']);self.assertLessEqual(d['stationarity'],1e-6)
            b=[(None,torch.ones(2,dtype=torch.float64),H)]
            v,d=rf.solve(s,w,H,b,1.,solver_mode='active_set',max_iterations=0);self.assertFalse(d['success']);self.assertEqual(d['status'],'iteration_exhausted')
    def test_rollback_and_isolation(self):
        from coph_stagewise.attribution_audits import digest
        a,x,m,world,t=copy.deepcopy(self.snapshot);a.assimilate(m);a=configure(a,'precision_iterative','relative_local');a.receiver_solver_mode='active_set';a.receiver_search_mode='legacy'
        version=a.version;fields=[s.weights.clone() for s in a.stages];measurement=digest([(s.readings,s.transitions) for s in a.stages]);physical=x.copy()
        with patch.dict(rf.SETTINGS,max_iterations=0):d=a.update(x)
        self.assertFalse(d['accepted']);self.assertEqual(d['reason'],'receiver_solver_failure');self.assertEqual(a.version,version)
        self.assertTrue(all(torch.equal(s.weights,w) for s,w in zip(a.stages,fields)));self.assertEqual(measurement,digest([(s.readings,s.transitions) for s in a.stages]));self.assertTrue(np.array_equal(x,physical))
    def test_nonfinite_and_near_threshold(self):
        args=self.problem();bad=args[2]*float('nan')
        v,d=rf.solve(args[0],args[1],bad,args[3],args[4],solver_mode='active_set');self.assertFalse(d['success']);self.assertEqual(d['status'],'nonfinite')
        v,d=rf.solve(*args,solver_mode='active_set');self.assertEqual(d['near_tolerance'],.5e-6<=d['stationarity']<=2e-6)

    def test_bound_release_and_fallback(self):
        H=torch.tensor([[.927232841833744,1.9927788723509388,-.8102783402309619],[1.9927788723509388,5.46900356775344,-1.0334163313755875],[-.8102783402309619,-1.0334163313755875,2.024934424859101]],dtype=torch.float64)
        z=torch.tensor([2.9962909659510677,-.06584365881114075,1.4876401993926651],dtype=torch.float64);w=torch.zeros(3,dtype=torch.float64);args=(SimpleNamespace(centers=None),w,H,[(None,z,torch.eye(3,dtype=torch.float64))],1.)
        with patch.object(rf,'vector',side_effect=lambda v,*a:v):
            v,d=rf.solve(*args,solver_mode='active_set');hs=d['history']
            self.assertTrue(d['success']);self.assertTrue(any(any(f and not b for f,b in zip(a['fixed'],c['binding'])) for a,c in zip(hs,hs[1:])))
            original=torch.linalg.solve
            with patch('torch.linalg.solve',side_effect=lambda *a,**k:-original(*a,**k)):
                v,d=rf.solve(*args,solver_mode='active_set')
            self.assertGreater(d['gradient_fallbacks'],0);self.assertLessEqual(d['total_correction'],.5)
    def test_search_exhaustion_and_option_validation(self):
        s=SimpleNamespace(centers=None);w=torch.zeros(2,dtype=torch.float64);H=torch.eye(2,dtype=torch.float64)
        with patch.object(rf,'vector',side_effect=lambda v,*a:v),patch.dict(rf.SETTINGS,armijo=1e6):
            v,d=rf.solve(s,w,H,[(None,torch.ones(2,dtype=torch.float64),H)],1.,solver_mode='active_set')
        self.assertEqual(d['status'],'line_search_failed');self.assertFalse(d['success']);self.assertEqual(len(d['history'][0]['trials']),13)
        from coph_stagewise.cooperative import CooperativeLearner
        a,x,*_=self.snapshot
        with self.assertRaises(ValueError):CooperativeLearner(a.cfg,x,a.goal,agent_id='0',receiver_solver_mode='active_set')

if __name__=='__main__':unittest.main()
