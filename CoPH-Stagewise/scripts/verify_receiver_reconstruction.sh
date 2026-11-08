#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - "${1:-results/receiver_reconstruction_v1}" <<'PY'
import json,sys,hashlib,gzip
from pathlib import Path
import numpy as np
from coph_stagewise.cli import source_hashes
from coph_stagewise.evaluation import load,dump
root=Path(sys.argv[1]);m=load(root/'manifest.json');current=source_hashes()
drift={k:dict(frozen=v,current=current.get(k)) for k,v in m['source_hashes'].items() if current.get(k)!=v}
revision=load(root/'reporting_revision.json') if (root/'reporting_revision.json').exists() else None
if revision:
 assert revision['path'] in drift
 assert drift[revision['path']]==dict(frozen=revision['frozen_sha256'],current=revision['corrected_sha256'])
 assert hashlib.sha256((root/revision['snapshot']).read_bytes()).hexdigest()==revision['corrected_sha256']
 assert len(drift)==1,drift
else:assert not drift,drift
checks=[]
for status_path in sorted((root/'status').glob('*.json')):
 status=load(status_path)
 assert status['complete'],status
 r=load(root/'episodes'/status['path']);scene=status['scene'];spec=m['settings'][scene]['spec']
 assert r['spec']==spec
 assert r['communication']==m['settings'][scene]['communication']
 expected=dict(m['settings'][scene]['config']);expected['route_field_mode']='supported_field' if status['method'].startswith('direct_') else 'anchors'
 assert r['config']==expected
 packets={p['identity']:p for p in r['packets']}
 for e in r['events']:
  if e['kind']=='sent':
   p=packets[e['packet']];n=len(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
   assert n==e['bytes'],(e['packet'],n,e['bytes'])
 for i in (0,1):
  ids={str(z['uid']):z for o in r['observations'][i] for z in o['readings']}
  assert all(str(z['uid']).startswith(str(i)+':') and dict(z,uid=str(z['uid']).split(':',1)[1])==ids.get(str(z['uid']).split(':',1)[1]) for u in r['updates'][i] for s in u['post']['stages'] for z in s['readings'])
  assert all(t['version']<=r['updates'][i][-1]['post']['version'] for t in r['trajectories'][i])
 samples=r['samples'][1];start=np.array(spec['initials'][1][:2]);entered=next((s['time'] for s in samples if np.linalg.norm(np.array(s['x'][:2])-start)>.25),float('inf'))
 seen=set();selected=[]
 for e in r['events']:
  if e['kind']!='delivered' or e['receiver']!=1:continue
  state=next(s for s in reversed(samples) if s['time']<=e['time']+1e-9);stage=state['stage']
  if e['time']<entered or stage not in seen:selected.append(e['packet']);seen.add(stage)
 assert selected==[a['packet'] for a in r['audits']],(status['id'],selected,[a['packet'] for a in r['audits']])
 for a in r['audits']:
  assert a['operational_state_unchanged'] and a['measurements_unchanged']
  assert a['branches']['no_fit']['message_fit_work']==0
  assert all(b['replay_work']<=1024 for b in a['branches'].values())
  assert a['queries']['supported_count']==sum(a['queries']['common_support'])
 checks.append(dict(id=status['id'],packet_count=len(packets),audit_count=len(selected),passed=True))
assert len(checks)==16,len(checks)
animations=load(root/'animation_frames.json')
from PIL import Image
for scene,frames in animations.items():
 with Image.open(root/'figures'/f'{scene}.gif') as im:assert im.n_frames==len(frames['times'])
assert len(animations)==4
dump(root/'artifact_verification.json',dict(source_drift=drift,episodes=checks,animations=list(animations),passed=True))
print('Verified frozen sources/configs, packet bytes, measurement provenance, audit schedule/isolation, flow ceiling and four GIFs.')
PY
