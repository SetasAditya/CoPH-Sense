"""Independent checks of all saved native safety comparisons."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'results/native_safety_v2'
manifest=json.loads((out/'manifest.json').read_text());metrics=json.loads((out/'metrics.json').read_text());rows=metrics['rows']
assert len(rows)==132
assert len([r for r in rows if r['profile']=='fresh'])==120
for name,sha in manifest['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
for r in rows:
    prefix=r['protocol']+'_'+r['case'];a=np.load(out/(prefix+'.npz'))
    ref=np.load(out/('phmarl_'+r['case']+'.npz'))
    assert a['positions'].shape==(401,4,2);assert a['actions'].shape==(400,4,1,2)
    assert np.isfinite(a['positions']).all() and np.isfinite(a['actions']).all()
    assert np.max(np.abs(a['actions']))<=1
    assert np.array_equal(a['positions'][0],ref['positions'][0]) and np.array_equal(a['landmarks'],ref['landmarks'])
    assert np.array_equal(a['initial_contacts'],ref['initial_contacts'])
    assert np.array_equal(a['collisions'],a['contact_pairs'].sum(axis=1))
    rewards=a['rewards'].reshape(400,4)
    assert r['mean_agent_return']==sum(float(x.mean()) for x in rewards)
    assert r['collision_pair_steps']==int(a['contact_pairs'].sum())
    expected=-np.linalg.norm(a['positions'][1:,:,None,:]-a['landmarks'][None,None,:,:],axis=-1).min(axis=1).sum(axis=1)-.5*a['contact_pairs'].sum(axis=1)
    np.testing.assert_allclose(rewards.mean(axis=1),expected,rtol=2e-6,atol=2e-6)
    previous=np.vstack([a['initial_contacts'],a['contact_pairs'][:-1]])
    assert r['new_contact_onsets']==int((a['contact_pairs']&~previous).sum())
    with gzip.open(out/(prefix+'_trace.json.gz'),'rt') as f:trace=json.load(f)
    if r['protocol'].endswith('_protected'):
        flat=[x for batch in trace for x in batch];assert len(flat)==1600
        applied=np.asarray([x['applied'] for x in flat]).reshape(400,4,1,2)
        np.testing.assert_array_equal(applied,a['actions'])
        assert r['interventions']==sum(x['intervened'] for x in flat)
        assert r['infeasible_controls']==sum(not x['feasible'] for x in flat)
        for x in flat:
            if x['feasible'] and x['constraint_residuals']:assert min(x['constraint_residuals'])>=-2e-6
            assert all(p['distance']<=.45 for p in x['peers'])
            assert x['unknown_peer_velocities']==sum(not p['velocity_available'] for p in x['peers'])
    else:assert all(not t for t in trace)
    with gzip.open(out/(prefix+'_updates.json.gz'),'rt') as f:updates=json.load(f)
    for u in updates:
        assert u['train']+u['validation']+u['excluded']<=120
        if not u['accepted']:assert u['version_after']==u['version_before'] and u['committed']==u['old']
    if r['profile']=='development' and r['protocol']=='phmarl':
        old=np.load(out.parent/'native_v1'/('corrected_400_%02d.npz'%r['episode']))
        for key in old.files:assert np.array_equal(a[key],old[key])
with Image.open(out/'comparison_seed81000.gif') as gif:assert gif.n_frames==101
inventory=json.loads((out/'failure_gallery/inventory.json').read_text())
expected=sum(not(r['full_coverage'] and r['collision_free']) for r in rows)
assert len(inventory)==expected
assert json.loads((out/'incomplete.json').read_text())==[]
result=dict(episodes=132,fresh=120,development=12,source_hashes=True,all_initial_states_paired=True,
    archived_original_parity=True,native_reward_formula=True,controls_match_execution=True,
    action_bounds=True,finite_arrays=True,projection_residuals=True,receiver_local_visibility=True,
    initial_contacts_and_new_onsets=True,bounded_identification_and_rollback=True,
    gif_frames=101,failure_inventory=expected)
(out/'artifact_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
