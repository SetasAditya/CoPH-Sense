from pathlib import Path
import copy,unittest
import numpy as np
import torch
from coph_stagewise import solver_diagnosis as sd
from coph_stagewise import receiver_fitting as rf


@unittest.skipUnless((Path('results/receiver_corrections_v1/audit_rows.json')).exists(), 'Requires preserved experiment archives; see docs/DEVELOPMENT.md')
class SolverDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        tasks=[t for t in sd.jobs() if t[1]['group']=='navigation']
        if not tasks:raise unittest.SkipTest('snapshots not present')
        cls.audits=[a for t in (tasks[0],tasks[200]) for a in sd._job(t)]  # each asserts its snapshot is unchanged

    def test_captured_problems_replay_and_trace_bitwise(self):
        for a in self.audits:
            for p in a['problems']:self.assertTrue(p['replay_exact']);self.assertTrue(p['trace_exact'])

    def test_reference_is_no_worse_than_converged_original(self):
        for a in self.audits:
            for p in a['problems']:
                if p['status']=='converged':self.assertGreaterEqual(p['gap_to_reference'],-1e-10)

    def test_exact_hessian_matches_finite_differences(self):
        torch.manual_seed(0);m=16;stage=sd.Stage(0,torch.rand(m,2,dtype=torch.float64)*3)
        L=torch.linalg.cholesky(torch.eye(3,dtype=torch.float64)*.02)
        p=dict(stage=stage,w0=torch.full((m,),-.06,dtype=torch.float64),H=torch.eye(m,dtype=torch.float64)*5,width=.8,
               blocks=[(torch.tensor([[1.,1.5]],dtype=torch.float64),torch.tensor([.2,.1,-.05],dtype=torch.float64),L)])
        w=p['w0']+.1;H=sd.hessian(w,p);h=1e-5;fd=torch.zeros_like(H)
        for i in range(m):
            e=torch.zeros(m,dtype=torch.float64);e[i]=h;fd[:,i]=(sd.grad(w+e,p)[1]-sd.grad(w-e,p)[1])/(2*h)
        np.testing.assert_allclose(H.numpy(),fd.numpy(),atol=1e-6)

    def test_candidates_keep_the_unchanged_tolerance(self):
        for a in self.audits:
            for p in a['problems']:
                for c in p['candidates'].values():
                    if c['status']=='converged':self.assertLessEqual(c['stationarity'],rf.SETTINGS['stationarity_tolerance'])


if __name__=='__main__':unittest.main()
