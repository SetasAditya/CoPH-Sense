"""Validate logged contracts without training or modifying result episodes."""
import argparse
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load,dump


def validate(root):
    root=Path(root);manifest=load(root/'manifest.json');rows=load(root/'summary.json');checks=0;index={}
    for path in sorted((root/'episodes').glob('*.json.gz')):
        r=load(path);spec=r['spec'];budget=r['communication']['byte_budget'];arm=r['arm']
        assert spec in manifest['scenarios'];assert arm in manifest['arms'];assert budget in manifest['budgets']
        assert r['summary']['team_cost']==sum(a['cost'] for a in r['agents'])
        assert sum(a['bytes'] for a in r['agents'])==r['summary']['bytes']
        for i,a in enumerate(r['agents']):
            assert np.isclose(a['cost'],sum(a['components'].values()))
            assert a['bytes']<=budget;assert a['max_energy_residual']<1e-10
            assert a['peak_samples']<=r['config']['sample_cap']*(r['config']['previous']+1)
            tr=r['trajectories'][i]
            assert np.allclose(tr[0]['x'],spec['initials'][i])
            for k in ('time','risk','effort'):assert np.isclose(a['components'][k],sum(t['components'][k] for t in tr))
            for u in r['updates'][i]:
                assert u['diagnostic']['physical_jump']==0.
                if not u['diagnostic']['accepted']:
                    assert u['pre']['version']==u['post']['version']
                    assert u['pre']['policy']==u['post']['policy']
                    assert [s['weights'] for s in u['pre']['stages']]==[s['weights'] for s in u['post']['stages']]
                for s in u['post']['stages']:
                    assert all(reading['uid'].startswith(str(i)+':') for reading in s['readings'])
        for p in r['packets']:
            assert all(source.startswith(p['sender']+':') for source in p['sources'])
            assert all(np.isfinite(np.asarray(p[k])).all() for k in ('anchors','covariance','support_points'))
        for a in r['audits']:
            for branch in a['results'].values():
                assert branch['path'][0]==a['start'];assert len(branch['path'])<=a['horizon']+1
            if a['model_pair']['valid']:assert a['model_pair']['horizon']==a['horizon']
        key=(spec['name'],spec['seed'],budget)
        index.setdefault(key,{})[arm]=r
        checks+=1
    for arms in index.values():
        if 'disabled' not in arms:continue
        for r in arms.values():assert r['spec']==arms['disabled']['spec']
    result=dict(episodes_checked=checks,expected=manifest['expected'],complete=checks==manifest['expected'],summary_rows=len(rows))
    dump(root/'contract_checks.json',result);print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');validate(parser.parse_args().root)
