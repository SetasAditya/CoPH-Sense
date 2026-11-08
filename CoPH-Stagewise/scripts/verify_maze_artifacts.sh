#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - "${1:-results/cooperative_maze_v2}" <<'PY'
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__import__('sys').argv[1]);manifest=json.loads((root/'manifest.json').read_text())
render=json.loads((root/'render_sources/manifest.json').read_text()) if (root/'render_sources/manifest.json').exists() else None
for name,h in manifest['source_hashes'].items():
    expected_hash=render['sha256'] if render and name==render['source'] else h
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected_hash,name
if render:assert hashlib.sha256((root/'render_sources'/render['file']).read_bytes()).hexdigest()==render['sha256']
expected={(s['seed'],a) for s in manifest['specs'] for a in manifest['arms']};results={};ledger=[];maxres=0.;gates=[]
def value(m,q,cfg):
    q=np.asarray(q);w=np.asarray(m['weights']);centers=np.asarray(m['centers']);d2=np.sum((q-m['target'])**2)
    risk=np.logaddexp(0,np.exp(-.5*np.sum(((q-centers)/cfg['rbf_width'])**2,axis=1))@w)/4
    B=0.;bound=np.exp(-.5)*np.abs(w).sum()/(4*cfg['rbf_width'])
    for ox,oy,r in m['geometry']:
        clearance=np.linalg.norm(q-[ox,oy])-r-.12;B+=np.logaddexp(0,6*(.35-clearance))**2/36
        bound+=np.logaddexp(0,6*(r+.47))/3
    L=cfg['policy_bounds'][1]*bound;alpha=1 if L==0 else min(1,2*cfg['shaping_eta']*cfg['shaping_k']*cfg['shaping_ell']/L)
    return .5*cfg['shaping_k']*d2+alpha*d2/(d2+cfg['shaping_ell']**2)*(m['policy'][0]*risk+m['policy'][1]*B)
for path in sorted((root/'episodes').glob('*.json.gz')):
    r=json.load(gzip.open(path,'rt'));key=(r['spec']['seed'],r['arm']);results[key]={k:r[k] for k in ('spec','config','receive_update_slots','agents')}
    for i,a in enumerate(r['agents']):
        assert a['start_time']==5*i and a['max_update_work']<=1024
        assert abs(a['cost']-sum(a['components'].values()))<1e-8
        assert sum(a['bytes'].values())<=r['communication']['byte_budget']
        maxres=max(maxres,a['max_energy_residual']);prev=None
        for t in r['trajectories'][i]:
            m=t['control_model'];assert m is not None and m['stage']==t['stage'] and m['version']==t['version']
            np.testing.assert_allclose(m['target'],t['navigation']['waypoint'],atol=1e-12)
            assert t['time']>=5*i;assert np.linalg.norm(t['u'])<=r['config']['force_limit']+1e-8
            assert np.isfinite(m['weights']).all();assert np.isfinite(t['u']).all()
            if prev:
                q=t['x'][:2];old=prev['control_model'];newV=value(m,q,r['config']);oldV=value(old,q,r['config'])
                fixed_old_target=dict(m,target=old['target']);middle=value(fixed_old_target,q,r['config'])
                route_jump=newV-middle;model_geometry_jump=middle-oldV
                assert abs(newV-oldV-route_jump-model_geometry_jump)<1e-10
                ledger.append(dict(seed=key[0],arm=key[1],agent=i,time=t['time'],route_jump=route_jump,model_geometry_jump=model_geometry_jump,total_shaping_jump=newV-oldV,physical_jump=0.))
            prev=t
        # Did the carrier enter the first blocked branch, rather than reject it at the fork?
        if i==1:
            for gx,gy,_ in r['spec']['gate_groups']:
                entered=[t['time'] for t in r['trajectories'][i] if gx-1.8<t['x'][0]<gx and abs(t['x'][1]-gy)<.5]
                gates.append(dict(seed=key[0],arm=key[1],gate=[gx,gy],carrier_entered_blocked_branch=bool(entered),first_entry=min(entered) if entered else None,criterion='centerline approach strip: gate_x-1.8 < x < gate_x and abs(y-gate_y) < .5'))
    observations=[{o['time']:o for o in os} for os in r['observations']]
    for m in r['packets']:
        sender=int(m['sender'])
        for o in m['geometry']:
            assert o[7].startswith(m['sender']+':geometry:')
            obs=observations[sender][o[6]]
            assert list(o[:3]) in obs['obstacles'] and list(o[3:5])==obs['sensor_center'] and o[5]==obs['sensor_radius']
            assert o[6]<=m['timestamp']
    assert all(a['operational_state_unchanged'] for a in r['audits'])
    del r
assert set(results)==expected;assert maxres<1e-10
for seed in {k[0] for k in expected}:
    a,b=results[(seed,'disabled')],results[(seed,'reciprocal_fixed')]
    assert a['spec']==b['spec'] and a['config']==b['config'] and a['receive_update_slots']==b['receive_update_slots']
pending=not (root/'figures/frames.json').exists()
if pending:
    assert __import__('os').environ.get('MAZE_ALLOW_PENDING_ANIMATION')=='1','animation not complete'
    frames={'times':[]}
else:
    frames=json.loads((root/'figures/frames.json').read_text());gif=Image.open(root/'figures/cooperative_maze.gif')
    durations=[]
    for k in range(gif.n_frames):gif.seek(k);durations.append(gif.info['duration'])
    assert all(d%180==0 for d in durations) and sum(durations)==180*len(frames['times'])
with gzip.open(root/'shaping_jump_ledger.json.gz','wt') as f:json.dump(ledger,f)
(root/'branch_entry_checks.json').write_text(json.dumps(gates,indent=2)+'\n')
summary=dict(episodes=len(results),goals=sum(a['success'] for r in results.values() for a in r['agents']),max_physical_energy_residual=maxres,source_fingerprints=len(manifest['source_hashes']),geometry_provenance_verified=True,model_snapshots_verified=True,shaping_jump_records=len(ledger),gif_frames=None if pending else gif.n_frames,sampled_frames=len(frames['times']),animation_verified=not pending)
(root/'artifact_checks.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print(gates)
PY
