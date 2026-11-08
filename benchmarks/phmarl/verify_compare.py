"""Check saved paired experiment arrays, isolation and identification accounting."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'results/controller_transfer_v1'
manifest=json.loads((out/'manifest.json').read_text())
rows=json.loads((out/'metrics.json').read_text())['rows']
assert len(rows)==40
for name,value in manifest['sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
for row in rows:
    name=row['protocol'];i=row['episode'];prefix='%s_%02d'%(name,i)
    rec=np.load(out/(prefix+'.npz'))
    base=np.load(out.parent/'native_v1'/('corrected_400_%02d.npz'%i))
    assert np.array_equal(rec['positions'][0],base['positions'][0])
    assert np.array_equal(rec['landmarks'],base['landmarks'])
    assert rec['positions'].shape==(401,4,2)
    assert rec['actions'].shape==(400,4,1,2)
    assert np.isfinite(rec['actions']).all() and np.isfinite(rec['positions']).all()
    assert np.max(np.abs(rec['actions']))<=1
    rew=rec['rewards'].reshape(400,4)
    assert row['mean_agent_return']==sum(float(r.mean()) for r in rew)
    assert row['summed_agent_return']==sum(float(r.sum()) for r in rew)
    assert row['collision_pair_steps']==int(rec['collisions'].sum())
    if name=='phmarl':
        for key in base.files:assert np.array_equal(rec[key],base[key])
    else:
        control=json.loads((out/(prefix+'_controls.json')).read_text())
        assert len(control)==400
        for step,batch in enumerate(control):
            assert len(batch)==4
            for j,c in enumerate(batch):
                np.testing.assert_array_equal(c['applied'],rec['actions'][step,j,0])
                assert 0<=c['peer_count']<=3
                np.testing.assert_allclose(c['target'],rec['landmarks'][j],atol=2e-7,rtol=1e-6)
        updates=json.loads((out/(prefix+'_updates.json')).read_text())
        if name=='coph_online':
            assert len(updates)==200
            for u in updates:
                assert u['train']+u['validation']+u['excluded']<=120
                assert 0<=u['committed']<=.95
                assert abs(u['committed']-u['old'])<=.15+1e-12
                if not u['accepted']:
                    assert u['committed']==u['old'] and u['version_after']==u['version_before']
        else:assert not updates
with Image.open(out/'comparison_episode0.gif') as gif:assert gif.n_frames==101
checks=dict(episodes=40,native_phmarl_bitwise_parity=True,initial_pairing=True,
    unchanged_source_fingerprints=True,reward_recomputation=True,native_action_bounds=True,
    control_log_matches_execution=True,assigned_goals_match=True,bounded_memory=True,
    identification_rollback=True,gif_frames=101)
(out/'artifact_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
