#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - "${1:-results/field_route_v1}" <<'PY'
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);count=bitwise=commits=rejections=points=0;peak=0;op=diag=audit=0;versions=[]
for f in sorted((root/'status').glob('*.json')):
 status=json.loads(f.read_text());assert status['complete'];r=json.load(gzip.open(root/'episodes'/status['path'],'rt'));count+=1
 if 'baseline_comparison' in r:bitwise+=r['baseline_comparison']['bitwise_passed']
 controls=[t['navigation'] for ts in r['trajectories'] for t in ts]+[n for us in r['updates'] for u in us for n in u['diagnostic'].get('diagnostic_controls',[])]
 operational=sum(n.get('route_field_work',{}).get('field_evaluations',0) for n in controls)
 diagnostic=sum(n.get('route_field_diagnostic_work',{}).get('field_evaluations',0) for n in controls)
 assert r['route_field_total_evaluations']==operational+diagnostic+r['route_field_audit_evaluations']
 op+=operational;diag+=diagnostic;audit+=r['route_field_audit_evaluations']
 if r['config']['route_field_mode']=='anchors':continue
 for us in r['updates']:
  for u in us:
   pre=u['pre']['navigation']['route_field_snapshot'];post=u['post']['navigation']['route_field_snapshot'];cfg=r['config'];ss={s['id']:s for s in u['post']['stages']}
   peak=max(peak,len(post['stages']));assert len(post['stages'])<=cfg['previous']+1
   if u['diagnostic']['accepted']:commits+=1
   else:rejections+=1
   old={s['id']:s for s in pre['stages']}
   for s in post['stages']:
    assert s['id'] in ss and s['version']<=u['post']['version'];assert len(s['support'])<=cfg['sample_cap']
    support={tuple(v['q']) for v in ss[s['id']]['readings'] if int(str(v['uid']).split(':')[-1])%5}
    assert set(map(tuple,s['support']))<=support
    if u['diagnostic']['accepted']:assert s['weights']==ss[s['id']]['weights'] and s['version']==u['post']['version']
    else:assert s['weights']==old[s['id']]['weights'] and s['version']==old[s['id']]['version']
 for ts,us in zip(r['trajectories'],r['updates']):
  index=-1
  for t in ts:
   while index+1<len(us) and us[index+1]['time']<=t['time']+1e-9:index+=1
   if index<0:continue
   snap=us[index]['post']['navigation']['route_field_snapshot'];models={s['id']:s for s in snap['stages']}
   for e in t['navigation']['route_field_trace']['quadrature']:
    meta=e['route_field']
    for q,source,stage,version,raw,clipped in zip(e['points'],meta['source'],meta['stage'],meta['version'],meta['raw'],meta['clipped']):
     if source!='field':continue
     points+=1;s=models[stage];assert version==s['version'];q=np.asarray(q)
     assert abs(q[0]-s['center'][0])<=snap['stage_width']/2+1e-10 and abs(q[1]-s['center'][1])<=snap['stage_height']/2+1e-10
     assert min(np.linalg.norm(np.asarray(s['support'])-q,axis=1))<=.65+1e-10
     z=np.exp(-.5*np.sum(((q-np.asarray(s['centers']))/snap['width'])**2,axis=1))@s['weights'];value=(z if z>20 else np.logaddexp(0,z))/4
     assert abs(value-raw)<1e-10 and abs(np.clip(value,0,20)-clipped)<1e-10
assert count==24
out=dict(episodes=count,bitwise_anchor_regressions=bitwise,direct_model_commits=commits,direct_model_rejections=rejections,verified_field_quadrature_points=points,peak_routing_stages=peak,operational_field_evaluations=op,tracing_field_evaluations=diag,audit_field_evaluations=audit,accounting_closure=True,script_sha256=hashlib.sha256(Path('scripts/verify_field_route_records.sh').read_bytes()).hexdigest())
(root/'record_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
PY
