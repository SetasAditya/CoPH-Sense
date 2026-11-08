#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - "${1:-results/decision_bottleneck_v1}" <<'PY'
import gzip,hashlib,json,sys,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_text())
with zipfile.ZipFile(root/'sources.zip') as z:
 for name,h in m['source_hashes'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
expected={f'{c}_{s}_{a}' for c in m['cases'] for s in m['seeds'] for a in m['arms']}
expected|={f"{r['case']}_{r['seed']}_{r['arm']}" for r in m['regressions']}
bottlenecks=[]
seen=set();goals=collisions=timeouts=audits=packets=updates=0;max_work=0;regressions=0;regression_goals=0;max_energy=0.;bitwise_regressions=0
for f in sorted((root/'status').glob('*.json')):
 s=json.loads(f.read_text());assert s['complete'],s;assert s['id'] not in seen;seen.add(s['id'])
 p=root/'episodes'/s['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256']
 r=json.load(gzip.open(p,'rt'));assert r['manifest_fingerprint']==m['fingerprint'];assert r['communication']==m['communication']
 if r['regression']:
  assert r['regression_comparison']['passed'] and all(r['regression_comparison']['fields'].values());regressions+=1;regression_goals+=sum(a['success'] for a in r['agents'])
  bitwise_regressions+=r['regression_comparison']['bitwise_passed'];assert r['regression_comparison']['tolerance']==dict(atol=1e-12,rtol=1e-10)
 else:
  assert r['spec']==m['specs'][f"{r['case']}_{r['seed']}"] and r['config']==m['config']
  assert not r['spec']['obstacles'] and not r['spec']['dynamic'];assert r['spec']['steps']==500
 for i,a in enumerate(r['agents']):
  if not r['regression']:goals+=a['success'];collisions+=a['collision'];timeouts+=a['timeout']
  assert abs(a['cost']-sum(a['components'].values()))<1e-8
  assert sum(a['bytes'].values())<=4000000;assert a['start_time']==5*i
  max_work=max(max_work,a['max_update_work']);assert a['max_update_work']<=1024
  max_energy=max(max_energy,a['max_energy_residual']);assert a['max_energy_residual']<1e-10
  for t in r['trajectories'][i]:
   assert t['control_model']['version']==t['version'];assert t['navigation']['decision_route']['reference']=='estimated'
   assert np.linalg.norm(t['u'])<=r['config']['force_limit']+1e-8
   np.testing.assert_allclose(t['navigation']['applied'],t['u'],atol=1e-12)
  for u in r['updates'][i]:
   updates+=1;d=u['diagnostic'];trace=d['policy_trace'];assert d['counterfactual_steps']<=1024
   np.testing.assert_allclose(trace['initial'],u['pre']['policy'],atol=0,rtol=0)
   np.testing.assert_allclose(trace['committed'],u['post']['policy'],atol=0,rtol=0)
   delta=np.array(u['post']['policy'])-u['pre']['policy'];np.testing.assert_allclose(trace['committed_delta'],delta,atol=1e-14)
   assert trace['model_accepted']==d['accepted']
   if not d['accepted']:assert np.array_equal(u['pre']['policy'],u['post']['policy'])
   if r['method'] in ('local_evidence','remote_route','reconstruction'):assert not trace['backtracking_accepted'] and not trace['policy_changed']
   for call in d['decision_replay']:
    if not call['valid']:assert call['missing_paths_reason'] is not None
    for path in call['paths']:
     assert path['reach']['reaches_decision']==bool(path['reach']['outgoing_entries'])
 if not r['regression']:
  first=[]
  for ts in r['trajectories']:
   event=next((dict(time=t['time'],edge='direct' if t['x'][0]>1.75 else 'bypass') for t in ts if t['x'][0]>1.75 or (t['x'][1]>.25 and t['x'][0]>1.25)),None)
   first.append(event)
  all_updates=[u for us in r['updates'] for u in us];delivery=[e for e in r['events'] if e['kind']=='delivered' and e['reason']=='eligible']
  early=[e for e in delivery if first[e['receiver']] is not None and e['time']<first[e['receiver']]['time']]
  audit_early=[a for a in r['audits'] if first[a['receiver']] is not None and a['time']<first[a['receiver']]['time']]
  gradient=[u['diagnostic']['policy_gradient_norm'] for u in all_updates if u['diagnostic']['policy_gradient_norm']>1e-10]
  plans=[p for u in all_updates for p in u['diagnostic'].get('replay_plan',[])]
  replay_paths=[p for u in all_updates for call in u['diagnostic'].get('decision_replay',[]) for p in call['paths']]
  changes=[float(np.linalg.norm(u['diagnostic']['policy_trace']['committed_delta'])) for u in all_updates if u['diagnostic']['policy_trace']['policy_changed']]
  rejected=[u['diagnostic'] for u in all_updates if not u['diagnostic']['accepted']]
  def describe(v):return dict(n=len(v),minimum=min(v) if v else None,median=float(np.median(v)) if v else None,maximum=max(v) if v else None)
  audit_transfer=[]
  for audit in r['audits']:
   b,d=[audit['branches'][name]['decision_audit']['forces'] for name in ('withhold','deliver')]
   nominal=float(np.linalg.norm(np.asarray(d['nominal'])-b['nominal']));applied=float(np.linalg.norm(np.asarray(d['applied'])-b['applied']))
   audit_transfer.append(dict(packet=audit['packet'],time=audit['time'],receiver=audit['receiver'],before_branch=audit in audit_early,nominal_delta=nominal,applied_delta=applied,ratio=applied/nominal if nominal>1e-10 else None))
  bottlenecks.append(dict(id=s['id'],case=r['case'],seed=r['seed'],arm=r['method'],first_branch_entries=first,
   eligible_packets=len(delivery),prebranch_eligible_packets=len(early),prebranch_field_components=sum(e['components']['field_assimilation']['used'] for e in early),
   replay_horizon_counts=dict(Counter(str(p['horizon']) for p in plans)),replay_plan_reasons=dict(Counter(p['reason'] for p in plans)),
   replay_handoffs=sum(len(call['handoffs']) for u in all_updates for call in u['diagnostic'].get('decision_replay',[])),
   replay_valid_calls=sum(call['valid'] for u in all_updates for call in u['diagnostic'].get('decision_replay',[])),
   replay_calls=sum(len(u['diagnostic'].get('decision_replay',[])) for u in all_updates),
   replay_distances=describe([p['reach']['distance'] for p in replay_paths]),
   replay_min_junction_distances=describe([p['reach']['minimum_junction_distance'] for p in replay_paths if p['reach']['minimum_junction_distance'] is not None]),
   audits=len(r['audits']),prebranch_audits=len(audit_early),gradient_norms=describe(gradient),committed_delta_norms=describe(changes),
   model_rejections=len(rejected),field_validation_failures=sum(d['field_validation_after']>d['field_validation_before']+r['config']['validation_slack'] for d in rejected),
   plant_validation_failures=sum(d['plant_validation_after']>d['plant_validation_before']+r['config']['validation_slack'] for d in rejected),
   backtracking_acceptances=sum(u['diagnostic']['policy_trace']['backtracking_accepted'] for u in all_updates),
   accepted_policy_proposals_rolled_back=sum(d['policy_trace']['backtracking_accepted'] for d in rejected),
   shaping_alpha=describe([t['navigation']['alpha'] for ts in r['trajectories'] for t in ts]),
   soft_to_attraction_ratio=describe([float(np.linalg.norm(t['navigation']['shaping_force'])/np.linalg.norm(t['navigation']['attraction_force'])) for ts in r['trajectories'] for t in ts if np.linalg.norm(t['navigation']['attraction_force'])>1e-10]),
   operational_control_modifications=sum(bool(np.linalg.norm(np.asarray(t['u'])-t['nominal'])>1e-10) for ts in r['trajectories'] for t in ts),
   operational_clipping_changes=sum(bool(np.linalg.norm(np.asarray(t['navigation']['nominal_clipped'])-t['nominal'])>1e-10) for ts in r['trajectories'] for t in ts),
   operational_safety_changes=sum(bool(np.linalg.norm(np.asarray(t['u'])-t['navigation']['nominal_clipped'])>1e-10) for ts in r['trajectories'] for t in ts),
   audit_force_transfer=audit_transfer))
 local=[{f"{i}:{v['uid']}" for obs in r['observations'][i] for v in obs['readings']} for i in (0,1)]
 payload_sizes={}
 for packet in r['packets']:
  assert set(packet['sources'])<=local[int(packet['sender'])];packets+=1
  payload_sizes[packet['identity']]=len(json.dumps(packet,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
 for e in r['events']:
  if e['kind']=='sent':assert e['bytes']==payload_sizes[e['packet']]
 for i in (0,1):assert r['agents'][i]['bytes']['payload']==sum(payload_sizes[p['identity']] for p in r['packets'] if int(p['sender'])==i)
 deliveries=[e for e in r['events'] if e['kind']=='delivered'];assert len(deliveries)==len({e['packet'] for e in deliveries})
 keys=[]
 for a in r['audits']:
  audits+=1;keys.append((a['receiver'],a['stage']));assert a['operational_state_unchanged'];assert a['randomness']['shared_between_branches'];assert a['decomposition_closure_residual']<1e-8
  for b in a['branches'].values():
   d=b['decision_audit'];assert d['control_evaluations']==5 and d['guard_flows']>=0
   for ref in d['route_references'].values():
    for edge in ref['quadrature']:
     assert abs(edge['cost']-sum(l*(1+4*v) for l,v in zip(edge['lengths'],edge['values'])))<1e-8
     if ref['reference']=='supported_truth':assert all(v==1. for v,supported in zip(edge['values'],edge['support']) if not supported)
   for sensitivity in d['sensitivity']:
    assert sensitivity['epsilon']==1e-4;assert np.isfinite(sensitivity['nominal_derivative']).all();assert np.isfinite(sensitivity['applied_derivative']).all()
   forces=d['forces'];np.testing.assert_allclose(np.array(forces['attraction'])+forces['soft_shaping']+np.array(forces['damping']),forces['nominal'],atol=1e-10)
   np.testing.assert_allclose(np.array(forces['nominal'])+forces['clipping']+np.array(forces['safety']),forces['applied'],atol=1e-10)
   for h,c in b['checkpoints'].items():
    if c is None:continue
    assert c['path'][0]==a['start'];assert c['steps']<=int(h);assert c['terminal_step'] is not None or c['steps']==int(h)
 assert len(keys)==len(set(keys))
assert seen==expected,(len(seen),len(expected));assert regressions==2
frames=json.loads((root/'figures/animation_frames.json').read_text());assert len(frames)==3
for entry in frames:
 assert not entry.get('unavailable');assert entry['seed']==78000
 im=Image.open(root/'figures'/f"{entry['case']}.gif");duration=0
 for k in range(im.n_frames):im.seek(k);duration+=im.info['duration']
 assert duration==160*entry['frames']
rows=json.loads((root/'rows.json').read_text());assert len(rows)==30
assert not json.loads((root/'incomplete.json').read_text())
result=dict(episodes=32,fork_episodes=30,regressions_passed=regressions,bitwise_regressions_passed=bitwise_regressions,fork_goals=goals,regression_goals=regression_goals,collisions=collisions,timeouts=timeouts,packets=packets,audits=audits,updates=updates,max_update_flows=max_work,max_energy_residual=max_energy,source_files=len(m['source_hashes']),animations=3,verifier_sha256=hashlib.sha256(Path('scripts/verify_decision_artifacts.sh').read_bytes()).hexdigest())
(root/'bottleneck_metrics.json').write_text(json.dumps(bottlenecks,indent=2)+'\n')
(root/'artifact_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
PY
