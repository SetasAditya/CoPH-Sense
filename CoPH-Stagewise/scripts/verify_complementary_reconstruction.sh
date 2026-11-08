#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - "${1:-results/complementary_reconstruction_v1}" <<'PY'
import sys,json,hashlib,zipfile
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load,dump
from coph_stagewise.cli import source_hashes
from coph_stagewise.receiver_reconstruction import fingerprint,file_hash
from coph_stagewise.cooperative import Packet
from coph_stagewise.complementary_reconstruction import METHODS,branch_entry
root=Path(sys.argv[1]);m=load(root/'manifest.json');body=dict(m);fp=body.pop('fingerprint');assert fingerprint(body)==fp
current=source_hashes();changed={k for k in set(current)|set(m['source_hashes']) if current.get(k)!=m['source_hashes'].get(k)}
if changed:
 revision=load(root/'reporting_revision.json');assert changed=={revision['path']}
 assert m['source_hashes'][revision['path']]==revision['archived_sha256']
 assert current[revision['path']]==revision['current_sha256']==file_hash(root/revision['snapshot'])
 assert revision['path']=='coph_stagewise/complementary_reporting.py'
with zipfile.ZipFile(root/'sources.zip') as z:
 assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==m['source_hashes']
g=load(root/'gate.json');gates={};checks=[];audit_count=0;query_count=0
for key,spec in m['specs'].items():
 gate=load(root/'gate'/f'{key}.json');gates[key]=gate
 assert file_hash(root/'snapshots'/gate['snapshot']['path'])==gate['snapshot']['sha256']
 for d in gate['directions']:
  assert d['passed']==all(d['checks'].values())
  if d['intended'] and g['passed']:assert d['passed']
 for a in load(root/'gate'/f'{key}_audits.json.gz'):
  assert a['operational_state_unchanged'] and a['measurements_unchanged']
  assert a['branches']['no_fit']['message_fit_work']==0
 if spec['scene']=='redundant':
  assert gate['noise']['initial_coordinates_identical'] and not gate['noise']['initial_values_identical']
  assert gate['noise']['independent_indexed_streams']
 for method,(mode,fit) in METHODS.items():
  status=root/'status'/f'{key}_{method}.json'
  if not g['passed']:assert not status.exists();continue
  s=load(status);assert s['complete'];p=root/'episodes'/s['path'];assert file_hash(p)==s['sha256'];r=load(p)
  assert r['spec']==spec and r['communication']==m['communication']
  assert r['config']==dict(m['config'],route_field_mode=mode)
  assert r['acquisition_check']['passed'];assert r['manifest_fingerprint']==fp
  deliveries=[e for e in r['events'] if e['kind']=='delivered']
  assert len({(e['receiver'],e['packet']) for e in deliveries})==len(deliveries)
  for e in deliveries:assert e['time']>=.96
  for i in (0,1):
   sent=[p for p in r['packets'] if p['sender']==str(i)]
   assert sum(len(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()) for p in sent)==r['agents'][i]['bytes']['payload']
   assert sum(r['agents'][i]['bytes'].values())<=4000000
   first=next((p for p in sent if abs(p['timestamp']-.96)<1e-10),None)
   assert first==gate['directions'][i]['packet'],'first actual export differs from acquisition export'
   for p in sent:
    sources={str(i)+':'+str(z['uid']) for o in r['observations'][i] if o['time']<=p['timestamp'] for z in o['readings']}
    assert set(p['sources'])<=sources,'unmeasured/foreign source in local-only export'
   assert all(u['diagnostic']['counterfactual_steps']<=1024 for u in r['updates'][i])
   if not fit:assert r['receiver_diagnostics'][i]['message_fit_work']==0
   assert abs(r['agents'][i]['components']['time']-len(r['trajectories'][i])*.12)<1e-9
   seen=set();required=set()
   for e in deliveries:
    if e['receiver']!=i:continue
    u=next(u for u in r['updates'][i] if e['packet'] in u['packets']);stage=u['pre']['stages'][-1]['id']
    if stage not in seen:required.add(e['packet']);seen.add(stage)
   entry=next((s['time'] for s in r['samples'][i] if branch_entry(s['x'][:2]) is not None),None)
   before=[e for e in deliveries if e['receiver']==i and entry is not None and e['time']<entry]
   if before:required.add(before[-1]['packet'])
   assert required=={a['packet'] for a in r['audits'] if a['receiver']==i},'audit selection mismatch'
  for a in r['audits']:
   audit_count+=1;query_count+=a['queries']['query_count']
   assert a['operational_state_unchanged'] and a['measurements_unchanged']
   assert file_hash(root/'snapshots'/a['snapshot']['path'])==a['snapshot']['sha256']
   assert a['branches']['no_fit']['message_fit_work']==0
   assert a['branches']['no_fit']['rankings']['anchors']==a['branches']['fit']['rankings']['anchors']
   for b in a['branches'].values():
    assert b['remote_anchors_identical'] and not b['hypothetical_measurements'] and b['replay_work']<=1024
    for ranking in b['rankings'].values():
     for q in ranking['quadrature']:
      assert np.isclose(sum(q['contributions']),q['cost'],atol=1e-12)
      assert np.allclose(np.array(q['lengths'])*(1+ranking['risk_weight']*np.array(q['values'])),q['contributions'])
  checks.append(dict(id=s['id'],goals=sum(a['success'] for a in r['agents']),audits=len(r['audits']),acquisition=r['acquisition_check']))
assert g['passed']==all(gates[k]['passed'] for k in m['gate_required'])
frames=load(root/'animation_frames.json')
from PIL import Image
for case,v in frames.items():
 with Image.open(root/'figures'/f'{case}.gif') as im:assert im.n_frames==len(v['times'])
 for method,h in v['sources'].items():assert file_hash(root/'episodes'/f'{case}_79100_{method}.json.gz')==h
result=dict(passed=True,manifest_fingerprint=fp,source_archive_sha256=file_hash(root/'sources.zip'),gate=g,
 episodes=len(checks),missions=len(checks)*2,audits=audit_count,queries=query_count,animations=len(frames),checks=checks,
 semantics='Source/config integrity, actual first packets, measured provenance, bytes, local flow bounds, exactly-once delivery, complete audit selection, immutable audit flags, weighted score sums and GIF frame manifests verified. Queries are correlated, not independent trials.')
dump(root/'artifact_verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
PY
