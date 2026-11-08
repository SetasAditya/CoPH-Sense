from pathlib import Path
import copy,gzip,pickle,unittest
from unittest.mock import patch
import numpy as np
import torch
from coph_stagewise import receiver_fitting as rf
from coph_stagewise.reconstruction_quality import _snapshot_jobs,SOURCE,_local
from coph_stagewise.reconstruction_diagnostics import field_queries
from coph_stagewise.model import tensor,features

@unittest.skipUnless((Path('results/complementary_reconstruction_v1/audit_rows.json')).exists(), 'Requires preserved experiment archives; see docs/DEVELOPMENT.md')
class ReceiverFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        jobs,_=_snapshot_jobs(SOURCE)
        path,_=next(j for j in jobs if j[1]['group']=='navigation')
        with gzip.open(path,'rb') as f:cls.snapshot=pickle.load(f)
    def agent(self):
        a,x,m,w,t=copy.deepcopy(self.snapshot);a.assimilate(m)
        a.receiver_fit_mode='legacy';a.receiver_validation_mode='legacy';a.receiver_diagnostic_variant=None
        return a,x
    def test_legacy_fit_bitwise(self):
        a,x=self.agent();b=copy.deepcopy(a);wa,_=a._field_fit();b.receiver_validation_mode='relative_local';wb,_=b._field_fit()
        for u,v in zip(wa,wb):self.assertTrue(torch.equal(u,v))
    def test_no_messages_no_change(self):
        a,x=self.agent();a.received={};w,c=_local(a);a.receiver_fit_mode='precision_bounded';z,_=rf.fit(a,w,c)
        for u,v in zip(w,z):self.assertTrue(torch.equal(u,v))
        self.assertEqual(a.receiver_fit_diagnostic['eligible_rows'],0)
    def test_overlap_precision_marginals(self):
        a,x=self.agent();a.stages=[copy.deepcopy(a.stages[0]),copy.deepcopy(a.stages[0])];a.stages[1].id+=100
        q=next(r.q for r in a.stages[0].readings if int(r.uid.split(':')[-1])%5);a.stages[1].anchors=[q]
        w=[s.weights.clone() for s in a.stages];original=pickle.dumps(a);hs,meta=rf.precision_information(a,w)
        self.assertEqual(meta['overlap_rows'],1);self.assertEqual(pickle.dumps(a),original)
        m=len(w[0]);A=[]
        for k,s in enumerate(a.stages):
            for r in s.readings:
                if int(r.uid.split(':')[-1])%5:
                    phi=features(tensor(r.q),s.centers,a.cfg.rbf_width);row=torch.zeros(2*m,dtype=torch.float64)
                    row[k*m:(k+1)*m]=phi*torch.sigmoid(phi@w[k])/(4*.003);A.append(row)
        phi=features(tensor(q),a.stages[0].centers,a.cfg.rbf_width);A.append(torch.cat((2*phi,-2*phi)))
        A=torch.stack(A);C=torch.linalg.inv(A.T@A+a.cfg.fit_ridge*torch.eye(2*m,dtype=torch.float64))
        self.assertTrue(torch.allclose(hs[0],torch.linalg.inv(C[:m,:m]),rtol=1e-10,atol=1e-10))
    def problem(self):
        a,x=self.agent();w,c=_local(a);s=a.stages[0];blocks,_=rf.blocks_for(a,s);H=torch.linalg.inv(c[0]);return a,s,w[0],H,blocks
    def test_total_bound_and_exhaustion(self):
        a,s,w,H,b=self.problem();z,d=rf.solve(s,w,H,b,a.cfg.rbf_width,max_iterations=0)
        self.assertFalse(d['success']);self.assertEqual(d['status'],'iteration_exhausted')
        z,d=rf.solve(s,w,H,b,a.cfg.rbf_width)
        self.assertLessEqual(float((z-w).abs().max()),.5+1e-12)
        if d['converged']:self.assertLessEqual(d['stationarity'],1e-6)
        self.assertLessEqual(d['iterations'],20)
    def test_exact_stationarity(self):
        a,s,w,H,b=self.problem();b=[(q,rf.vector(w,q,s,a.cfg.rbf_width).detach(),L) for q,z,L in b]
        z,d=rf.solve(s,w,H,b,a.cfg.rbf_width);self.assertTrue(d['converged']);self.assertEqual(d['iterations'],0);self.assertTrue(torch.equal(z,w))
    def test_objective_derivatives(self):
        a,s,w,H,b=self.problem();v=(w+.01).requires_grad_(True)
        def objective(v):
            d=v-w;out=.5*d@H@d
            for q,z,L in b:
                r=torch.linalg.solve_triangular(L,(rf.vector(v,q,s,a.cfg.rbf_width)-z)[:,None],upper=False).flatten();out+=.5*r@r
            return out
        g=torch.autograd.grad(objective(v),v)[0]
        for i in (0,5,15):
            e=torch.zeros_like(v);e[i]=1e-5;fd=(objective(v.detach()+e)-objective(v.detach()-e))/(2e-5)
            self.assertAlmostEqual(float(fd),float(g[i]),delta=1e-5)
    def test_relative_threshold_and_missing_validation(self):
        a,x=self.agent();a.receiver_validation_mode='relative_local';w,c=_local(a);rf.fit(a,w,c)
        with patch.object(a,'_field_validation',side_effect=[1e-4,1.05e-4]):self.assertTrue(rf.acceptance(a,w)[0])
        with patch.object(a,'_field_validation',side_effect=[1e-4,1.06e-4]):self.assertEqual(rf.acceptance(a,w)[1],'receiver_relative_validation')
        for s in a.stages:s.readings=[r for r in s.readings if int(r.uid.split(':')[-1])%5]
        self.assertEqual(rf.acceptance(a,w)[1],'receiver_no_reserved_measurements')
    def test_transactional_rejection(self):
        a,x=self.agent();a.receiver_validation_mode='relative_local';before=copy.deepcopy((a.version,a.damping,a.policy,a.routing_fields,[s.weights for s in a.stages]))
        measured=pickle.dumps([(s.readings,s.transitions) for s in a.stages]);physical=x.copy()
        with patch.object(a,'_field_validation',side_effect=[.001,.0005,.0001,.0005]):d=a.update(x)
        self.assertFalse(d['accepted']);self.assertEqual(d['reason'],'receiver_relative_validation')
        self.assertEqual(a.version,before[0]);self.assertEqual(a.damping,before[1]);self.assertTrue(torch.equal(a.policy,before[2]));self.assertEqual(a.routing_fields,before[3])
        for s,w in zip(a.stages,before[4]):self.assertTrue(torch.equal(s.weights,w))
        self.assertEqual(measured,pickle.dumps([(s.readings,s.transitions) for s in a.stages]));self.assertTrue(np.array_equal(x,physical));self.assertLessEqual(d['counterfactual_steps'],1024)
    def test_solver_failure_rejects_update(self):
        a,x=self.agent();a.receiver_fit_mode='precision_bounded';version=a.version
        with patch.dict(rf.SETTINGS,max_iterations=0):d=a.update(x)
        self.assertFalse(d['accepted']);self.assertEqual(d['reason'],'receiver_solver_failure');self.assertEqual(a.version,version)

    def test_nonfinite_information_rejects_without_committing(self):
        a,x=self.agent();a.receiver_fit_mode='precision_bounded';before=a.version
        with patch('coph_stagewise.receiver_fitting.precision_information',side_effect=RuntimeError('nonfinite information')):d=a.update(x)
        self.assertFalse(d['accepted']);self.assertEqual(d['reason'],'receiver_solver_failure');self.assertEqual(a.version,before)
    def test_matched_audit_proposals_and_isolation(self):
        from coph_stagewise.receiver_corrections import audit_snapshot
        original=pickle.dumps(self.snapshot)
        r=audit_snapshot(copy.deepcopy(self.snapshot),dict(case='test',seed=0,group='navigation'),variants=('no_fit','original','precision_iterative'))
        self.assertEqual(original,pickle.dumps(self.snapshot))
        for v in ('original','precision_iterative'):
            self.assertEqual(r['results'][v+'/legacy']['proposed_fields'],r['results'][v+'/relative_local']['proposed_fields'])
        self.assertTrue(r['state_unchanged']);self.assertTrue(r['measurements_unchanged'])

    def test_proposal_uses_published_overlap_selection(self):
        from coph_stagewise.receiver_corrections import audit_snapshot
        jobs,_=_snapshot_jobs(SOURCE)
        for path,meta in jobs:
            if meta['group']!='navigation':continue
            with gzip.open(path,'rb') as f:snapshot=pickle.load(f)
            if len(snapshot[0].stages)>1:break
        else:self.fail('missing retained-stage fixture')
        snapshot[0].routing_fields=snapshot[0].routing_fields[:1]
        r=audit_snapshot(snapshot,meta,variants=('no_fit','original'))
        self.assertTrue(any(b['accepted'] for b in r['results'].values()))
        for b in r['results'].values():
            if b['accepted']:
                for k in ('value_mae','gradient_mae'):
                    self.assertAlmostEqual(b['common_errors']['proposed'][k],b['common_errors']['committed'][k],places=12)

if __name__=='__main__':unittest.main()
