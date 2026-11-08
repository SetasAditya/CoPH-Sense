#!/usr/bin/env python3
"""Development-only, fixed-protocol native controller transfer comparison."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import time
import numpy as np
import torch
from vmas import make_env
import reproduce as baseline
from adapter import NativeCoPH, SETTINGS, MODEL

ARMS=['phmarl','coph_frozen','coph_online','coph_known_native']
BASE=Path(__file__).parent/'results/native_v1'


def run(arm):
    torch.manual_seed(1);np.random.seed(1)
    env=make_env(scenario_name='simple_spread',num_envs=1,device='cpu',continuous_actions=True,
        wrapper=None,seed=1,max_steps=400,n_agents=4,n_agents_good=4,n_agents_adversaries=4,
        n_packages=1,share_reward=True,ratio=2)
    actor=baseline.cpu(torch.load(baseline.CHECKPOINT,map_location='cpu')) if arm=='phmarl' else None
    results=[]
    for episode in range(10):
        obs=env.reset(seed=None)
        controller=None if actor is not None else NativeCoPH(arm.removeprefix('coph_') if hasattr(arm,'removeprefix') else arm[5:])
        rec=baseline.record(env); controls=[]
        initial_collisions=sum(int(env.world.is_overlapping(a,b)[0]) for i,a in enumerate(env.world.agents) for b in env.world.agents[i+1:])
        control_seconds=0.;learning_seconds=0.
        for step in range(400):
            before=[o.detach().numpy().copy() for o in obs]
            begin=time.perf_counter()
            if actor is not None:
                with torch.no_grad():actions=list(actor(torch.stack(obs).squeeze(1).unsqueeze(0)).mean.transpose(0,1))
            else:
                actions=controller.action(before)
                controls.append(controller.last)
            control_seconds+=time.perf_counter()-begin
            obs,rewards,dones,_=env.step(actions)
            baseline.append(rec,env,actions,rewards)
            if controller is not None:
                begin=time.perf_counter()
                controller.observe(before,actions,[o.detach().numpy().copy() for o in obs])
                learning_seconds+=time.perf_counter()-begin
            if bool(torch.as_tensor(dones).any()):break
        metrics=baseline.metrics(rec,arm,episode)
        metrics.update(initial_overlap_pairs=initial_collisions,
            collision_pair_steps_after_50=int(sum(rec['collisions'][50:])),
            full_landmark_coverage=metrics['final_landmark_coverage']==1.,
            final_assigned_goal_distance=float(np.linalg.norm(rec['positions'][-1]-rec['landmarks'],axis=1).mean()),
            control_seconds=control_seconds,learning_seconds=learning_seconds,
            identification_record_evaluations=controller.work if controller is not None else 0,
            final_drag=controller.drag.tolist() if controller is not None else None,
            updates_accepted=sum(x['accepted'] for x in controller.updates) if controller is not None else 0,
            updates_rejected=sum(not x['accepted'] for x in controller.updates) if controller is not None else 0,
            peak_records_per_agent=min(step+1,120) if controller is not None else 0,
            clipped_controls=sum(x['clipped'] for batch in controls for x in batch))
        results.append((rec,metrics,controls,controller.updates if controller is not None else []))
    return results


def render(out,runs):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.patches import Circle
    fig,axes=plt.subplots(1,4,figsize=(16,4.5))
    colors=['tab:blue','tab:orange','tab:green','tab:red'];panels=[]
    lim=max(1.3,max(np.abs(np.asarray(runs[n][0][0]['positions'])).max() for n in ARMS)+.2)
    for ax,name in zip(axes,ARMS):
        rec=runs[name][0][0];p=np.asarray(rec['positions'])
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal')
        ax.scatter(*rec['landmarks'].T,marker='x',c='black',s=50)
        circles=[Circle(p[0,i],.15,color=colors[i],alpha=.5) for i in range(4)]
        lines=[ax.plot([],[],color=colors[i],lw=1)[0] for i in range(4)]
        for c in circles:ax.add_patch(c)
        title=ax.set_title(name)
        panels.append((p,circles,lines,title,name))
    fig.suptitle('Native VMAS navigation — fixed paired episode 0 (development)')
    def update(t):
        for p,cs,ls,title,name in panels:
            for i in range(4):
                cs[i].center=p[t,i];ls[i].set_data(p[:t+1,i,0],p[:t+1,i,1])
            title.set_text('%s\nstep %d / 400'%(name,t))
    FuncAnimation(fig,update,frames=range(0,401,4),interval=80).save(out/'comparison_episode0.gif',writer=PillowWriter(fps=12))
    update(400);fig.savefig(out/'comparison_episode0.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ARMS[1:]:
        differences=[runs[name][i][1]['mean_agent_return']-runs['phmarl'][i][1]['mean_agent_return'] for i in range(10)]
        ax.plot(range(10),differences,'o-',label=name)
    ax.axhline(0,color='black',lw=1);ax.set(xlabel='Paired episode',ylabel='Return difference versus phMARL (higher better)');ax.legend();fig.tight_layout();fig.savefig(out/'paired_returns.png',dpi=140);plt.close(fig)
    # Every incomplete-coverage/collision episode remains listed and plotted.
    gallery=out/'failure_gallery';gallery.mkdir()
    inventory=[]
    for name in ARMS:
        for i,(rec,row,_,_) in enumerate(runs[name]):
            if row['full_landmark_coverage'] and row['collision_free']:continue
            reasons=[]
            if not row['full_landmark_coverage']:reasons.append('incomplete_supplementary_coverage')
            if not row['collision_free']:reasons.append('overlap_observed')
            fig,ax=plt.subplots(figsize=(4,4));p=np.asarray(rec['positions'])
            ax.scatter(*rec['landmarks'].T,c='black',marker='x')
            for j in range(4):ax.plot(p[:,j,0],p[:,j,1],color=colors[j]);ax.add_patch(Circle(p[-1,j],.15,color=colors[j],alpha=.4))
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',title='%s episode %d\ncoverage %.0f%%; overlaps %d'%(name,i,100*row['final_landmark_coverage'],row['collision_pair_steps']))
            file='%s_%02d.png'%(name,i);fig.savefig(gallery/file,dpi=100);plt.close(fig)
            inventory.append(dict(arm=name,episode=i,reasons=reasons,figure=file))
    (gallery/'inventory.json').write_text(json.dumps(inventory,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results/controller_transfer_v1');args=parser.parse_args();out=args.out
    if out.exists():raise SystemExit('Refusing to overwrite output')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(baseline.UPSTREAM));import functions
    sources=[Path(__file__),Path(__file__).with_name('adapter.py'),Path(__file__).with_name('test_adapter.py'),Path(__file__).with_name('reproduce.py'),MODEL]
    manifest=dict(purpose='controller transfer, not full cooperative stagewise algorithm',arms=ARMS,
        base_manifest_sha256=baseline.digest(BASE/'manifest.json'),base_metrics_sha256=baseline.digest(BASE/'metrics.json'),
        settings=SETTINGS,seed=1,episodes=10,horizon=400,animation_episode=0,
        evaluation_role='development: same reset sequence as baseline, no held-out claim',
        primary='paired mean-agent return; same native reward',
        inactive=['environmental reconstruction','receiver assimilation','historical replay','A* routing','original safety projection'],
        sources={str(p.resolve().relative_to(baseline.ROOT)):baseline.digest(p) for p in sources})
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=out/'source';archive.mkdir()
    for p in sources:(archive/(p.name+'.txt')).write_text(p.read_text())
    runs={};rows=[]
    for arm in ARMS:
        runs[arm]=run(arm)
        for i,(rec,row,control,updates) in enumerate(runs[arm]):
            # Pair all arms against frozen native initial states, no reset edits.
            previous=np.load(BASE/('corrected_400_%02d.npz'%i))
            assert np.array_equal(rec['positions'][0],previous['positions'][0])
            assert np.array_equal(rec['landmarks'],previous['landmarks'])
            if arm=='phmarl':
                for key in previous.files:assert np.array_equal(rec[key],previous[key]),(i,key)
            rows.append(row)
            np.savez_compressed(out/('%s_%02d.npz'%(arm,i)),**{k:np.asarray(v) for k,v in rec.items()})
            (out/('%s_%02d_controls.json'%(arm,i))).write_text(json.dumps(control))
            (out/('%s_%02d_updates.json'%(arm,i))).write_text(json.dumps(updates))
        print(arm+' complete',flush=True)
    summary={}
    for arm in ARMS:
        rr=[r for r in rows if r['protocol']==arm]
        summary[arm]={k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ['protocol','episode','final_drag']}
    paired=[]
    for arm in ARMS[1:]:
        d=np.array([runs[arm][i][1]['mean_agent_return']-runs['phmarl'][i][1]['mean_agent_return'] for i in range(10)])
        rng=np.random.RandomState(179300);means=d[rng.randint(0,10,size=(10000,10))].mean(axis=1)
        paired.append(dict(arm=arm,differences=d.tolist(),mean=float(d.mean()),wins=int((d>0).sum()),
            ties=int((d==0).sum()),losses=int((d<0).sum()),development_paired_bootstrap_95=np.percentile(means,[2.5,97.5]).tolist()))
    (out/'metrics.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
    (out/'paired.json').write_text(json.dumps(paired,indent=2))
    (out/'checks.json').write_text(json.dumps(dict(phmarl_archived_bitwise_parity=True,all_initial_states_paired=True,episodes=40),indent=2))
    render(out,runs)
    table='| Arm | Mean-agent return ↑ | Final assignment distance ↓ | Landmark coverage | Collision pair-steps ↓ |\n|---|---:|---:|---:|---:|\n'
    for arm in ARMS:
        s=summary[arm];table+='| %s | %.2f | %.3f | %.1f%% | %.1f |\n'%(arm,s['mean_agent_return'],s['final_assignment_distance'],s['final_landmark_coverage']*100,s['collision_pair_steps'])
    report='''# CoPH controller transfer to native phMARL navigation

This is a first development comparison of a transferred CoPH scalar-potential controller and native plant identification, not the complete cooperative stagewise method. No retraining or outcome-dependent parameter tuning was performed. Settings, source snapshots and input fingerprints were written before executing any comparison episode.

Four agents, 400 simulator steps, ten identical sequential resets seeded with 1. The published phMARL actor was rerun and matched the archived baseline bitwise on all operational arrays. Every arm starts from the same states and landmarks. Native physics, observations, reward, action bounds, and horizon remain unchanged.

%s
## Exact adaptation

The adapter imports the existing CoPH potential, full spatial gradient, 1.4 velocity feedback, force bound 3, and continuous reference flow from `coph_stagewise/model.py`, without editing that source. Progress shaping retains k=1, ell=0.5, eta=0.5 and barrier weight 2. There are no field samples, and the environmental-risk coefficient is zero. Peers within 0.45 enter instantaneous soft barriers using native physical radius 0.15 for both agents. An effective obstacle radius of 0.18 compensates the original potential's built-in ego radius 0.12. This is moving-peer repulsion, not certified collision protection. Targets use the same indexed landmark as the published actor; there is no centralized assignment or extra map.

Reference velocity is obtained from the original exact-flow controller at dt=0.1, reference damping 0.6. Native action inverts the contact-free VMAS update v_next=(1-drag)*v+0.1*u, then clips each component to [-1,1]. All execution still occurs in VMAS, including native collision forces. This conversion is an explicit adapter extension, not an unchanged CoPH actuator path.

Frozen and online start from drag=1-exp(-0.6*0.1). Known-native uses the public simulator's drag 0.25 and isolates model mismatch; it is not a learned model. Online identification uses only native self-observations and issued actions, bounded memory of 120 transitions per agent, updates every eight steps, reserved uid%%5==0 records, drag bounds [0,0.95], a 0.15 trust step, and held-out validation before committing. Records with any peer within 0.45 at either endpoint are excluded to reduce contact contamination. This filter is approximate, not proof that collision forces vanished. Versions, rejected updates, excluded counts, controls and clipping are logged. Native drag units/bounds differ from the original continuous damping learner, so the original estimator is not reused falsely.

## What this comparison does not establish

No environmental field reconstruction, receiver fitting, historical replay, route planning, original obstacle-safety projection or environmental packet exchange is active. Native observations already supply positions; the adapter uses each agent's own target and the same 0.45 peer-radius restriction as the checkpoint, without peer velocities or privileged dynamics queries. No claim of full cooperative Hamiltonian learning superiority follows from this comparison. Offline phMARL training and online identification are different resources; control and learning wall times and identification work are reported separately. Network byte comparisons are unavailable because this native benchmark exposes observations rather than a serialized packet transport.

The ten episodes reuse development initializations already inspected in baseline reproduction. Paired bootstrap intervals are descriptive for this small fixed-checkpoint set, not multi-training-seed or held-out evidence. All arms and episodes are reported. Coverage within 0.15 is supplementary; the native task runs its full horizon and does not declare goal-completion success. Pair-step overlaps include contacts inherited from initial placement; initial overlaps and overlaps after step 50 are separately recorded. The failure gallery retains every episode with incomplete supplementary coverage or overlap; no favorable animation selection was used.

## Artifacts and reproduction

`metrics.json`, `paired.json`, per-episode arrays, control/identification traces, `manifest.json`, source snapshots, synchronized `comparison_episode0.gif`, `paired_returns.png`, and `failure_gallery/inventory.json`. Run from workspace root:

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B benchmarks/phmarl/compare.py --out benchmarks/phmarl/results/controller_transfer_v2
```

Original CoPH source and earlier results remain unchanged. Further adaptation of reconstruction, replay or safety requires a separately declared experiment.
'''%table
    (out/'REPORT.md').write_text(report)
    print(table,flush=True)
    print(json.dumps(paired,indent=2),flush=True)

if __name__=='__main__':main()
