#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - "${1:-results/attribution_v1}" <<'PY'
import gzip,hashlib,json,os,sys,zipfile
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);manifest=json.loads((root/'manifest.json').read_text())
execution=json.loads((root/'execution_context.json').read_text()) if (root/'execution_context.json').exists() else {}
source_root=Path(execution.get('frozen_source_root','.'))
with zipfile.ZipFile(root/'sources.zip') as archive:
    for name,sha in manifest['sources'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==sha,name
        assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==sha,name
with zipfile.ZipFile(root/'documents.zip') as archive:
    for name,sha in manifest['documents'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==sha,name
        assert hashlib.sha256((source_root.resolve().parent/name).read_bytes()).hexdigest()==sha,name
profile=os.environ.get('ATTRIBUTION_PROFILE','development')
assert profile in ('smoke','development')
seeds=manifest['smoke_seeds' if profile=='smoke' else 'development_seeds']
expected={(scene,seed,arm) for scene in ('blocked_maze','soft_risk_maze') for seed in seeds for arm in manifest['arms']}
seen=set();goals=collisions=timeouts=packets=audits=0;max_energy=0.;pairs={};disabled_checks=0;soft_controls={}
for status_path in sorted((root/'status').glob('*.json')):
    status=json.loads(status_path.read_text())
    if status['seed'] not in seeds:continue
    assert status['complete'],status
    path=root/'episodes'/status['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==status['sha256']
    r=json.load(gzip.open(path,'rt'));key=(r['scene'],r['spec']['seed'],r['attribution_arm'])
    assert key not in seen;seen.add(key)
    assert r['manifest_fingerprint']==manifest['fingerprint']
    assert r['spec']==manifest['specifications'][f'{key[0]}_{key[1]}']
    assert r['config']==manifest['config'] and r['communication']==manifest['communication']
    assert r['attribution']==manifest['arms'][key[2]]
    if key[0]=='soft_risk_maze' and key[2] in ('local_evidence','geometry_only'):
        soft_controls[key[1],key[2]]=hashlib.sha256(json.dumps([[dict(x=t['x'],u=t['u']) for t in ts] for ts in r['trajectories']],sort_keys=True).encode()).hexdigest()
    pair=(r['spec'],r['config'],r['receive_update_slots'])
    assert pairs.setdefault(key[:2],pair)==pair
    for i,a in enumerate(r['agents']):
        goals+=a['success'];collisions+=a['collision'];timeouts+=a['timeout']
        assert sum(a['bytes'].values())<=r['communication']['byte_budget']
        assert a['max_update_work']<=1024 and a['start_time']==i*5.
        assert abs(a['cost']-sum(a['components'].values()))<1e-8
        max_energy=max(max_energy,a['max_energy_residual'])
        for t in r['trajectories'][i]:
            model=t['control_model']
            assert model['version']==t['version'] and model['stage']==t['stage']
            np.testing.assert_allclose(model['target'],t['navigation']['waypoint'],atol=1e-12)
            assert np.linalg.norm(t['u'])<=r['config']['force_limit']+1e-8
            assert abs(t['cost']-sum(t['components'].values()))<1e-8
        for u in r['updates'][i]:
            assert u['diagnostic']['counterfactual_steps']<=1024
            p=u['evaluation']
            assert all((v is not None)==s for v,s in zip(p['values'],p['support']))
            if r['attribution']['replay']=='current':
                assert len(u['diagnostic'].get('replay_plan',[]))<=1
            if not r['attribution']['field_assimilation']:
                assert not u['diagnostic']['messages'];disabled_checks+=1
    observations=[{o['time']:o for o in obs} for obs in r['observations']]
    local_ids=[{str(i)+':'+reading['uid'] for obs in r['observations'][i] for reading in obs['readings']} for i in (0,1)]
    for packet in r['packets']:
        packets+=1;sender=int(packet['sender'])
        assert set(packet['sources'])<=local_ids[sender]
        for g in packet['geometry']:
            o=observations[sender][g[6]]
            assert list(g[:3]) in o['obstacles'] and list(g[3:5])==o['sensor_center'] and g[5]==o['sensor_radius']
            assert g[-1].startswith(str(sender)+':geometry:') and g[6]<=packet['timestamp']
    deliveries=[e for e in r['events'] if e['kind']=='delivered']
    assert len({e['packet'] for e in deliveries})==len(deliveries),'duplicate operational delivery'
    audit_keys=[]
    for a in r['audits']:
        audits+=1;audit_keys.append((a['receiver'],a['stage']))
        assert a['operational_state_unchanged'] and a['randomness']['shared_between_branches']
        assert a['decomposition_closure_residual']<1e-8
        for b in a['branches'].values():
            for h in ('8','16','32'):
                c=b['checkpoints'][h]
                if c is None:continue
                assert c['path'][0]==a['start'] and c['steps']<=int(h)
                assert abs(c['cost']-sum(c['components'].values()))<1e-8
                assert c['terminal_step'] is not None or c['steps']==int(h)
    assert len(set(audit_keys))==len(audit_keys),'more than one audit per receiver stage'
    del r
assert seen==expected,(len(seen),len(expected))
for seed in seeds:
    assert soft_controls[seed,'local_evidence']==soft_controls[seed,'geometry_only'],'geometry-only changed obstacle-free execution'
assert max_energy<1e-10
frames=json.loads((root/'figures/animation_frames.json').read_text())
from PIL import Image
for f in frames:
    assert f['seed']==manifest['animation_seed'] and f['arm']=='historical_replay'
    im=Image.open(root/'figures'/f"{f['scene']}.gif");duration=0
    for i in range(im.n_frames):im.seek(i);duration+=im.info['duration']
    assert duration==180*len(f['times'])
summary=dict(profile=profile,episodes=len(seen),agent_goals=goals,collisions=collisions,timeouts=timeouts,
    max_energy_residual=max_energy,packets_verified=packets,audits_verified=audits,
    disabled_field_updates_verified=disabled_checks,source_files_verified=len(manifest['sources']),
    document_files_verified=len(manifest['documents']),
    pairing_verified=True,animations_verified=True,soft_risk_geometry_only_isolation_verified=True,
    verifier_sha256=hashlib.sha256(Path('scripts/verify_attribution_artifacts.sh').read_bytes()).hexdigest())
(root/('artifact_checks_smoke.json' if profile=='smoke' else 'artifact_checks.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
