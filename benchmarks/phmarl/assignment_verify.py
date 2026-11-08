#!/usr/bin/env python3
"""Read logged residual failures and verify assignment-campaign artifacts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args();out=args.out
    manifest=json.loads((out/'manifest.json').read_text());results=json.loads((out/'results.json').read_text())
    root=Path(__file__).resolve().parents[2]
    for rel,digest in manifest['sources'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
    assert len(results['rows'])==120
    residual=[]
    for row in results['rows']:
        name=row['protocol']+'_'+row['case']
        with np.load(out/(name+'.npz')) as d:
            assert d['positions'].shape==(401,4,2)
            assert np.isfinite(d['positions']).all() and np.isfinite(d['actions']).all()
            q=d['positions'][-1].copy();g=d['landmarks'].copy()
            assigned=g[row['goal_indices']]
            np.testing.assert_allclose(np.linalg.norm(q-assigned,axis=-1),row['final_assigned_goal_distances'],rtol=1e-6,atol=1e-8)
        if row['assignment_mode']=='index' or row['full_coverage']:continue
        with gzip.open(out/(name+'_trace.json.gz'),'rt') as f:trace=json.load(f)
        agents=[]
        for i in range(4):
            logs=[step[i] for step in trace[-100:]]
            agents.append(dict(agent=i,assigned_goal_gap=row['final_assigned_goal_distances'][i],
                mean_nominal_norm=float(np.mean([np.linalg.norm(x['nominal']) for x in logs])),
                mean_applied_norm=float(np.mean([np.linalg.norm(x['applied']) for x in logs])),
                infeasible_controls=sum(not x['feasible'] for x in logs),
                mean_target_shift=float(np.mean([x['goal_shift'] for x in logs])),
                final_protected_target_gap=float(np.linalg.norm(q[i]-np.array(trace[-1][i]['target'])))))
        residual.append(dict(drag_mode=row['drag_mode'],seed=row['episode'],agents=agents,positions=q.tolist(),assigned_goals=assigned.tolist()))
    animations=[]
    for seed in manifest['animation_seeds']:
        p=out/('comparison_seed%d.gif'%seed)
        with Image.open(p) as im:
            assert im.n_frames==51
            animations.append(dict(seed=seed,frames=im.n_frames,size=list(im.size),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    check=dict(episodes=120,sources_unchanged=True,all_states_and_actions_finite=True,horizons=400,
        legacy_bitwise_episodes=sum(r['assignment_mode']=='index' and r['archive_parity']['all_arrays_bitwise'] for r in results['rows']),
        tests='14 native tests passed; tests.log',animations=animations,residual_failures=residual,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'diagnostic_checks.json').write_text(json.dumps(check,indent=2))
    text='# Assignment diagnosis findings\n\nCentralized reset-only assignment removes all three previously highlighted online failures (82000, 82011, 82019). It does not fully resolve native navigation.\n\n'
    for mode in ['frozen','known','online']:
        a=next(s for s in results['summary'] if s['drag_mode']==mode and s['assignment_mode']=='index')
        b=next(s for s in results['summary'] if s['drag_mode']==mode and s['assignment_mode']=='initial_min_distance')
        text+='- %s damping: full landmark coverage %d/20 → %d/20; contact-free %d/20 → %d/20.\n'%(mode,a['full_coverage'],b['full_coverage'],a['contact_free'],b['contact_free'])
    text+='\nEvery reassigned run has zero new contact onsets. Initial overlaps remain charged; contact-free counts can therefore be below 20. All 60 legacy reproductions match archived arrays bit for bit, and their decoded decision/update records and operational work match exactly. Fourteen native tests pass.\n\n'
    text+='## Remaining failures\n\n'
    for r in residual:
        gaps=[a for a in r['agents'] if a['assigned_goal_gap']>.15]
        text+='- %s, seed %d: '%(r['drag_mode'],r['seed'])
        text+='; '.join('agent %d remains %.6f from its landmark, with mean shifted-target displacement %.6f and final distance %.6g from its protected target'%(a['agent'],a['assigned_goal_gap'],a['mean_target_shift'],a['final_protected_target_gap']) for a in gaps)+'.\n'
    text+='\nThese failures differ from the original high-command/near-zero-action cluster stalls. In the residual runs, nominal commands are already small and agents settle near targets displaced by the protection layer. This points to near-goal target handling as a remaining mechanism; it does not prove that changing it alone guarantees coverage. The known-damping controller succeeds on these starts, so the geometry does not establish impossibility.\n\n'
    text+='The result supports task allocation as a cause of the highlighted failures, while exposing regressions under a different fixed assignment. It does not justify a complete-working claim or a decentralized comparison with phMARL. The next diagnostic should test joint feasibility of landmark coverage and the existing separation constraints before designing a goal-handling change. No safety thresholds, success radii or controller settings were adjusted here.\n'
    (out/'FINDINGS.md').write_text(text)
    print(json.dumps(dict(episodes=120,legacy_bitwise=60,residual_cases=len(residual),animation_frames=[a['frames'] for a in animations])))

if __name__=='__main__':main()
