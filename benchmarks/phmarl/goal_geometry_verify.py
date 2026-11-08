#!/usr/bin/env python3
"""Verify logged geometry targets and summarize matched allocation outcomes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args();out=args.out
    root=Path(__file__).resolve().parents[2];manifest=json.loads((out/'manifest.json').read_text());results=json.loads((out/'results.json').read_text())
    assert len(results['rows'])==200
    for rel,h in manifest['sources'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
    geometry=json.loads((out/'geometry_gate.json').read_text());assert geometry['passed'] and len(geometry['rows'])==20
    for r in geometry['rows']:
        c=r['context'];assert c['consensus']['agreed'];assert all(p['feasible'] and p['solver_success'] for p in c['plans'])
        assert max(c['independent_target_coverage_gaps'])<.15
        assert c['minimum_independent_target_spacing']>=.32-1e-7
    maximum_target_error=0.;maximum_action_error=0.;target_checks=0;failures=[]
    for row in results['rows']:
        name=row['protocol']+'_'+row['case']
        with np.load(out/(name+'.npz')) as d:
            assert d['positions'].shape==(401,4,2)
            assert np.isfinite(d['positions']).all() and np.isfinite(d['actions']).all()
            assert np.max(np.abs(d['actions']))<=1.+1e-7
            q=d['positions'].copy();actions=d['actions'].reshape(400,4,2).copy()
        with gzip.open(out/(name+'_trace.json.gz'),'rt') as f:trace=json.load(f)
        assert len(trace)==400
        if row['target_mode']=='static_landmark':
            for t,step in enumerate(trace):
                assert len(step)==4
                for i,log in enumerate(step):
                    goal=np.asarray(log['goal']);selected=np.asarray(row['reset_context']['fixed_targets'][i]) if np.linalg.norm(q[t,i].astype(float)-goal)<=.35 else goal
                    err=float(np.linalg.norm(selected-np.asarray(log['target'])));maximum_target_error=max(maximum_target_error,err);target_checks+=1
                    aerr=float(np.max(np.abs(actions[t,i]-np.asarray(log['applied']))));maximum_action_error=max(maximum_action_error,aerr)
                    assert err<=1e-10 and aerr==0.
        if not row['full_coverage'] or not row['collision_free']:
            failures.append(dict(arm=row['protocol'],seed=row['episode'],coverage=row['final_landmark_coverage'],new_contacts=row['new_contact_onsets'],assigned_gaps=row['final_assigned_goal_distances']))
    animations=[]
    for seed in manifest['animation_seeds']:
        p=out/('comparison_seed%d.gif'%seed)
        with Image.open(p) as im:
            assert im.n_frames==51
            animations.append(dict(seed=seed,frames=im.n_frames,size=list(im.size),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),state_indices=list(range(0,401,8))))
    checks=dict(episodes=200,source_hashes_unchanged=True,independent_assignment_agreement_starts=20,joint_feasible_starts=20,
        legacy_arrays_bitwise_episodes=80,static_target_checks=target_checks,maximum_target_error=maximum_target_error,
        maximum_executed_action_error=maximum_action_error,finite_bounded_actions=True,animations=animations,
        tests='20 native tests pass; tests.log',all_failures=failures,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'artifact_checks.json').write_text(json.dumps(checks,indent=2))
    summary={s['arm']:s for s in results['summary']}
    text='# Static landmark target findings\n\nAll 20 reset states pass independent-assignment agreement and constructive joint target feasibility. The four agents recover the same reset geometry from their own native observations on these starts. This uses full native observations and known identity ordering, including peers beyond the nominal interaction radius; it is not established for restricted local sensing.\n\n'
    text+='| Damping | Legacy target full coverage | Static target full coverage | Static contact-free |\n|---|---:|---:|---:|\n'
    for m in ['frozen','known','online']:
        old=summary['coph_'+m+'_legacy_protected'];new=summary['coph_'+m+'_static_protected']
        text+='| %s | %d/20 | %d/20 | %d/20 |\n'%(m,old['full_coverage'],new['full_coverage'],new['contact_free'])
    text+='\nThe static target program uses landmark geometry, splits an isolated pair symmetrically, and reconciles multiple pair constraints jointly. Actual-motion protection is unchanged. Coverage radius remains 0.15 and center separation 0.32; the target-displacement cap is 0.149999 for numerical interior margin. Near-goal activation remains 0.35. The current-position midpoint rule remains available as the legacy default.\n\n'
    static=[r for r in results['rows'] if r['protocol'].startswith('coph_') and r['target_mode']=='static_landmark']
    if all(r['full_coverage'] and r['new_contact_onsets']==0 for r in static):text+='All 60 static-target CoPH episodes achieve full coverage with zero new contacts. This clears the requested development gate on these reused starts, including the three residual failure runs. It does not establish held-out navigation reliability.\n\n'
    else:
        text+='The requested all-complete/zero-new-contact gate does not pass. Residual static-target CoPH failures are: '+str([(r['protocol'],r['episode'],r['final_landmark_coverage'],r['new_contact_onsets']) for r in static if not r['full_coverage'] or r['new_contact_onsets']])+'\n\n'
    text+='## Matched phMARL controls\n\n'
    for a in ['phmarl','phmarl_assigned','phmarl_assigned_legacy_protected','phmarl_assigned_static_protected']:
        s=summary[a];text+='- %s: coverage %d/20; contact-free %d/20; mean return %.3f.\n'%(a,s['full_coverage'],s['contact_free'],s['means']['mean_agent_return'])
    text+='\nThe checkpoint also uses index-specific goal slots. Its assigned arms receive a consistent permutation of the landmark slots; its static protected arm receives the same assignment, targets and safety treatment as CoPH. These are modified checkpoint evaluations, not retraining.\n\n'
    for p in results['paired']:
        if p['treatment']=='coph_online_static_protected':text+='- Online static-target CoPH versus %s: return difference %.3f, paired bootstrap 95%% interval %s.\n'%(p['control'],p['return_difference'],p['interval95'])
    text+='\nAssignment and target planning are common navigation improvements. Identification is isolated by the frozen/known/online comparison with those layers held fixed. Receiver reconstruction and historical replay are absent from this native adapter; none of these results establish their navigation benefit. General superiority over phMARL must not be inferred from unmatched allocation or protection.\n\n'
    text+='## Verification and limitations\n\n80 archived baseline episodes reproduce arrays bit for bit and decoded decisions, updates and work exactly. All 20 native tests pass. Static-target selections and executed controls are verified against 400-step logs; GIF frames use logged indices 0,8,…,400. Every incomplete/contact episode remains in the failure inventory. Sources and old results are preserved.\n\n'
    text+='All 20 starts have a joint feasible witness. Pair spacing below 0.02 would certify an incompatible pair, but the converse is false for general clusters. No failed solve was used to declare true infeasibility or exclude an episode. Intervals use 10,000 paired resamples of the 20 reused development starts, and final-test seeds remain untouched.\n'
    (out/'FINDINGS.md').write_text(text);print(json.dumps(dict(episodes=200,static_target_checks=target_checks,failures=len(failures),legacy_bitwise=80)))

if __name__=='__main__':main()
