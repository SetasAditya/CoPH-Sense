import json,unittest
import torch
from coph_stagewise import reconstruction_quality as rq


class ReconstructionQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if not (rq.SOURCE/'audit_rows.json').exists():raise unittest.SkipTest('complementary snapshots not present')
        jobs,archived=rq._snapshot_jobs(rq.SOURCE)
        cls.index={(r['case'],r['seed'],r['method'],r['packet'],r['receiver']):r for r in archived}
        pick=[j for j in jobs if j[1]['group']=='acquisition'][:1]+[j for j in jobs if j[1]['group']=='navigation'][:2]
        cls.audits=[a for j in pick for a in rq._run(j)]  # each audit asserts the snapshot agent is unchanged

    def test_default_variant_is_the_operational_fit(self):
        for a in self.audits:self.assertTrue(a['default_matches_operational'])

    def test_reproduces_archived_fit_and_no_fit_errors(self):
        for a in self.audits:
            r=self.index[(a['case'],a['seed'],a.get('method','gate'),a['packet'],a['receiver'])]
            self.assertAlmostEqual(a['variants']['default']['value_mae'],r['errors']['fit']['value_mae'],places=12)
            self.assertAlmostEqual(a['variants']['no_fit']['value_mae'],r['errors']['no_fit']['value_mae'],places=12)

    def test_zero_message_scale_equals_no_fit(self):
        import copy,gzip,pickle,types
        jobs,_=rq._snapshot_jobs(rq.SOURCE);path,_=[j for j in jobs if j[1]['group']=='navigation'][0]
        with gzip.open(path,'rb') as f:agent,x,packet,world,tick=pickle.load(f)
        base=copy.deepcopy(agent);base.assimilate(packet);weights,cov=rq._local(copy.deepcopy(base))
        zero=rq.message_step(base,weights,cov,dict(lam=0.))
        for a,b in zip(zero,weights):self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
