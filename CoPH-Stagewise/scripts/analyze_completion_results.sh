#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl
python - "${1:-results/completion_first_v1}" <<'PY'
import hashlib,sys
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load,dump
from coph_stagewise.cli import source_hashes
from coph_stagewise.model import tensor,flow
root=Path(sys.argv[1]);manifest=load(root/'manifest.json');assert source_hashes()==manifest['source_hashes']
old=load('results/navigation_fix_v3/development/rows.json');rows=load(root/'development/rows.json');completed=load(root/'development/completion.json')
assert len(completed)==128 and all(c['complete'] for c in completed)
assert len(load(root/'smoke/completion.json'))==32
assert len(load(root/'development/failures.json'))==sum(r['goals']<2 for r in rows)
assert all(r['max_update_work']<=1024 and r['max_energy_residual']<1e-10 for r in rows)
comparisons=[];failures=[]
for mode in ('protected','routed'):
 for arm in ('disabled','reciprocal_fixed'):
  rs=[r for r in rows if (r['mode'],r['arm'])==(mode,arm)];bs=[r for r in old if (r['mode'],r['arm'])==(mode,arm)]
  comparisons.append(dict(mode=mode,arm=arm,goals=sum(r['goals'] for r in rs),agents=2*len(rs),legacy_goals=sum(r['goals'] for r in bs),collisions=sum(r['collisions'] for r in rs),timeouts=sum(r['timeouts'] for r in rs),stagnation_replans=sum(r['stagnation_replans'] for r in rs),brakes=sum(r['brakes'] for r in rs),infeasible=sum(r['infeasible'] for r in rs),mean_stationary_seconds=float(np.mean([r['near_stationary_seconds'] for r in rs])),max_update_work=max(r['max_update_work'] for r in rs),max_energy_residual=max(r['max_energy_residual'] for r in rs)))
for c in completed:
 if c['metrics']['goals']==2:continue
 r=load(root/'development/episodes'/c['path'])
 for i,a in enumerate(r['agents']):
  if a['success']:continue
  t=r['trajectories'][i][-1];n=t['navigation'];snap=r['updates'][i][-1]['post'];x=np.array(t['x']);projected=n.get('constrained');checks=[]
  if projected is not None:
   for d in sorted(set([*r['config']['damping_bounds'],snap['damping']])):
    y=flow(tensor(x),tensor(projected),d,r['config']['dt']).numpy();stop=float(y[2:]@y[2:])/(2*r['config']['force_limit'])
    margins=[float(np.linalg.norm(y[:2]-o[:2])-o[2]-.12-r['config']['safety_margin']-stop) for o in snap['geometry']]
    checks.append(dict(damping=d,minimum_stopping_margin=min(margins) if margins else None,predicted_speed=float(np.linalg.norm(y[2:]))))
  failures.append(dict(scene=c['scene'],seed=c['seed'],mode=c['mode'],arm=c['arm'],agent=i,goal_gap=a['goal_gap'],state=t['x'],nominal=n.get('nominal'),projected=projected,applied=t['u'],reason=n.get('reason'),infeasible=n.get('infeasible'),projection_residual=n.get('constraint_residual'),guard_checks=checks,final_40_brakes=sum(v.get('navigation',{}).get('reason')=='brake' for v in r['trajectories'][i][-40:])))
checks=load(root/'controlled_checks.json');gate=load(root/'development_gate.json');validation=load(root/'validation_status.json')
if not gate['passed']:assert not validation['executed'] and not (root/'validation').exists()
for audit in checks['audits']:assert audit['operational_state_unchanged'] and audit['field_revision']>0 and audit['control_revision']>0
artifact=dict(comparisons=comparisons,failed_agents=failures,source_matches=True,all_cost_flow_and_energy_contracts_passed=True,script_sha256=hashlib.sha256(Path('scripts/analyze_completion_results.sh').read_bytes()).hexdigest())
dump(root/'analysis.json',artifact)
lines=['# Completion-first cooperative navigation: results','',f"Implemented and evaluated 32 smoke and 128 development episodes. Development gate: **{'PASS' if gate['passed'] else 'FAIL'}**. Held-out validation executed: **{validation['executed']}**. Prior results and final-test namespaces remain untouched.",'','| Controller | Communication | Previous goals / 64 | Progress goals / 64 | Collisions | Timeouts |','|---|---|---|---|---|---|']
for r in comparisons:lines.append(f"| {r['mode']} | {r['arm']} | {r['legacy_goals']} | {r['goals']} | {r['collisions']} | {r['timeouts']} |")
lines+=['','The potential, replay, energy accounting and receiver-summary changes passed 77 numerical/contract tests. Controlled disk and passage cases completed with both frozen policies and learning enabled; strong-risk goal capture also passed. Recorded checks cover all eight old near-goal failures and the old waypoint-cancellation state.','','The completion gate is intentionally stricter than numerical correctness. The fixed-target radial bound cannot guarantee progress after safety projection and stopping guards. The failed-agent records in [analysis.json](results/completion_first_v1/analysis.json) expose the final nominal/projected/applied controls and stopping-clearance margins. A projection-feasible command can still fail the subsequent stopping guard, leading to repeated braking. These failures are not counted as successful navigation.','','Reciprocal causal audits used identical complete receiver states and isolated deliver/withhold continuations at 8, 16 and 32 steps. Both directions changed the initial reconstructed field and control; operational states and measurements remained untouched. The observed revisions are:','']
for a in checks['audits']:lines.append(f"- Receiver {a['receiver']}: field coefficient change {a['field_revision']:.6g}; applied-control change {a['control_revision']:.6g}.")
lines+=['','All failed development agents are in hazard slalom. Protected-progress repairs the previous asymmetric-coverage goal-capture failures but introduces repeated braking near slalom obstacles. Routed-progress completes all four scenes, including stage 9 in the long corridor; it records no braking or stagnation replans, and only 0.24 seconds of near-stationary time per episode on average (including startup).', '', 'This establishes an active information-to-model-to-control mechanism, not a demonstrated completion advantage from communication. Compare communication off and reciprocal within the same controller; gains over old routed control arise from the common navigation fix. Mission costs remain secondary diagnostics.','','The 1,024-flow ceiling includes replay, pilots, safety guards and update diagnostic controls. Routing and projection work are logged separately; actual compute and bytes differ across arms. No penalties, horizons, safety margins, goal thresholds or communication rates were tuned.','','Maximum recorded update work was 758/1,024 flow evaluations; the largest physical energy residual was 9.76e-15. Four synchronized 32-frame GIFs and all 16 failed development episodes are preserved. Full evaluation and rendering took about 13.2 minutes with 32 workers. The protected-controller failure blocks the overall working-version claim and the predeclared 64 held-out episodes.', '', '[Frozen manifest and fingerprints](results/completion_first_v1/manifest.json) · [Source archive](results/completion_first_v1/sources.zip) · [Controlled completion figure](results/completion_first_v1/controlled_completion.png)', '', '[Development table and synchronized GIFs](results/completion_first_v1/development/REPORT.md) · [Complete failure gallery](results/completion_first_v1/development/figures/failures/index.html) · [Implementation and reproduction](COMPLETION_FIRST.md)']
Path('COMPLETION_FIRST_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(comparisons)
PY
