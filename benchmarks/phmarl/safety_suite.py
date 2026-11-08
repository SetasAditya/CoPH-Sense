#!/usr/bin/env python3
"""Frozen six-arm native comparison; preserve every episode and original sources."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import time
import numpy as np
import torch
from vmas import make_env
import reproduce as base
from adapter import NativeCoPH,MODEL
from protection import Protection,SETTINGS

ARMS=['phmarl','attraction','attraction_protected','coph_frozen_protected','coph_online_protected','phmarl_protected']
FRESH=list(range(81000,81020))
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
HERE=Path(__file__).resolve().parent


def contacts(env):
    return np.array([bool(env.world.is_overlapping(env.world.agents[i],env.world.agents[j])[0]) for i,j in PAIRS])


def setup(spec):
    torch.manual_seed(1);np.random.seed(1)
    env=make_env(scenario_name='simple_spread',num_envs=1,device='cpu',continuous_actions=True,
        wrapper=None,seed=1,max_steps=400,n_agents=4,n_agents_good=4,n_agents_adversaries=4,
        n_packages=1,share_reward=True,ratio=2)
    if spec['profile']=='development':
        for _ in range(spec['index']+1):obs=env.reset(seed=None)
    else:obs=env.reset(seed=spec['seed'])
    return env,obs


def attraction(obs):
    result=[]
    for i,o in enumerate(obs):
        a=np.asarray(o).reshape(18)
        result.append(torch.tensor(np.clip(a[4+2*i:6+2*i]-1.4*a[2:4],-1,1),dtype=torch.float32).reshape(1,2))
    return result


def worker(task):
    arm,specs,out=task;out=Path(out)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    sys.path.insert(0,str(base.UPSTREAM));import functions
    actor=base.cpu(torch.load(base.CHECKPOINT,map_location='cpu')) if arm.startswith('phmarl') else None
    rows=[]
    for spec in specs:
        name='%s_%s'%(arm,spec['id']);env,obs=setup(spec)
        coph=NativeCoPH('online' if 'online' in arm else 'frozen') if arm.startswith('coph') else None
        protection=Protection() if arm.endswith('_protected') else None
        rec=base.record(env);initial=contacts(env);contact_history=[];trace=[]
        control_time=0.;protection_time=0.;learning_time=0.;begin_episode=time.perf_counter()
        for step in range(400):
            raw=[o.detach().numpy().copy() for o in obs];processed=raw
            begin=time.perf_counter()
            if protection is not None:processed=protection.prepare(raw)
            protection_time+=time.perf_counter()-begin
            begin=time.perf_counter()
            if actor is not None:
                with torch.no_grad():nominal=list(actor(torch.tensor(np.stack(processed)).squeeze(1).unsqueeze(0)).mean.transpose(0,1))
            elif coph is not None:nominal=coph.action(processed)
            else:nominal=attraction(processed)
            control_time+=time.perf_counter()-begin
            begin=time.perf_counter()
            if protection is not None:actions,logs=protection.apply(nominal)
            else:actions,logs=nominal,[]
            protection_time+=time.perf_counter()-begin
            obs,rewards,dones,_=env.step(actions)
            base.append(rec,env,actions,rewards);contact_history.append(contacts(env))
            begin=time.perf_counter()
            if coph is not None:coph.observe(raw,actions,[o.detach().numpy().copy() for o in obs])
            learning_time+=time.perf_counter()-begin
            trace.append(logs)
            if bool(torch.as_tensor(dones).any()):break
        row=base.metrics(rec,arm,spec.get('index',spec.get('seed')))
        cp=np.asarray(contact_history);prev=np.vstack([initial,cp[:-1]])
        recoveries=[int(np.where(~cp[:,j])[0][0]+1) if (~cp[:,j]).any() else None for j in range(6) if initial[j]]
        all_logs=[l for batch in trace for l in batch]
        pos=np.asarray(rec['positions']);g=rec['landmarks']
        away=np.linalg.norm(pos[16:]-g[None],axis=-1)>.15
        stalls=(np.linalg.norm(pos[16:]-pos[:-16],axis=-1)<.01)&away
        row.update(case=spec['id'],profile=spec['profile'],initial_overlap_pairs=int(initial.sum()),
            new_contact_onsets=int((cp&~prev).sum()),initial_overlap_recovery_steps=recoveries,
            collision_pair_steps_after_50=int(cp[50:].sum()),full_coverage=row['final_landmark_coverage']==1.,
            stall_agent_steps=int(stalls.sum()),control_seconds=control_time,protection_seconds=protection_time,
            learning_seconds=learning_time,total_seconds=time.perf_counter()-begin_episode,
            interventions=sum(x['intervened'] for x in all_logs),infeasible_controls=sum(not x['feasible'] for x in all_logs),
            goal_shifts=sum(x['goal_shift']>1e-8 for x in all_logs),goal_infeasible=sum(x['goal_infeasible'] for x in all_logs),
            unknown_peer_velocity_occurrences=sum(x['unknown_peer_velocities'] for x in all_logs),
            projection_candidates=sum(x['projection_candidates'] for x in all_logs),
            identification_work=coph.work if coph else 0,updates_accepted=sum(x['accepted'] for x in coph.updates) if coph else 0,
            final_drag=coph.drag.tolist() if coph else None)
        if spec['profile']=='development' and arm=='phmarl':
            original=np.load(HERE/'results/native_v1'/('corrected_400_%02d.npz'%spec['index']))
            for key in original.files:assert np.array_equal(original[key],rec[key]),(spec,key)
            row['archived_bitwise_parity']=True
        arrays={k:np.asarray(v) for k,v in rec.items()};arrays.update(contact_pairs=cp,initial_contacts=initial)
        np.savez_compressed(out/(name+'.npz'),**arrays)
        with gzip.open(out/(name+'_trace.json.gz'),'wt') as f:json.dump(trace,f)
        with gzip.open(out/(name+'_updates.json.gz'),'wt') as f:json.dump(coph.updates if coph else [],f)
        (out/(name+'_metrics.json')).write_text(json.dumps(row,indent=2));rows.append(row)
    return rows


def render(out,rows):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.patches import Circle
    case='seed81000';fig,axes=plt.subplots(2,3,figsize=(13,9));panels=[]
    colors=['tab:blue','tab:orange','tab:green','tab:red']
    records={arm:np.load(out/('%s_%s.npz'%(arm,case))) for arm in ARMS}
    lim=max(1.3,max(np.abs(r['positions']).max() for r in records.values())+.2)
    for ax,arm in zip(axes.flat,ARMS):
        r=records[arm];ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal')
        ax.scatter(*r['landmarks'].T,c='black',marker='x');lines=[];circles=[]
        for i in range(4):
            lines.append(ax.plot([],[],color=colors[i],lw=1)[0]);c=Circle(r['positions'][0,i],.15,color=colors[i],alpha=.5);ax.add_patch(c);circles.append(c)
        panels.append((arm,ax,r,lines,circles))
    def update(t):
        for arm,ax,r,lines,circles in panels:
            for i in range(4):lines[i].set_data(r['positions'][:t+1,i,0],r['positions'][:t+1,i,1]);circles[i].center=r['positions'][t,i]
            ax.set_title('%s\nstep %d; overlap pair-steps %d'%(arm,t,int(r['collisions'][:t].sum())))
    fig.suptitle('Native VMAS: fresh seed 81000, selected before execution');fig.tight_layout()
    FuncAnimation(fig,update,frames=range(0,401,4),interval=80).save(out/'comparison_seed81000.gif',writer=PillowWriter(fps=12))
    update(400);fig.savefig(out/'comparison_seed81000.png',dpi=140);plt.close(fig)
    fresh=[r for r in rows if r['profile']=='fresh'];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm in ARMS:
        rr=[r for r in fresh if r['protocol']==arm]
        axes[0].scatter([ARMS.index(arm)]*len(rr),[r['mean_agent_return'] for r in rr],s=15)
        axes[1].scatter([ARMS.index(arm)]*len(rr),[r['collision_pair_steps'] for r in rr],s=15)
    for ax in axes:ax.set_xticks(range(6));ax.set_xticklabels(ARMS,rotation=30,ha='right')
    axes[0].set_ylabel('Mean-agent return');axes[1].set_ylabel('Overlap pair-steps');fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=150);plt.close(fig)
    gallery=out/'failure_gallery';gallery.mkdir();inventory=[]
    for r in rows:
        if r['full_coverage'] and r['collision_free']:continue
        name=r['protocol']+'_'+r['case'];a=np.load(out/(name+'.npz'))
        fig,ax=plt.subplots(figsize=(4,4));ax.scatter(*a['landmarks'].T,c='black',marker='x')
        for i in range(4):ax.plot(a['positions'][:,i,0],a['positions'][:,i,1],color=colors[i]);ax.add_patch(Circle(a['positions'][-1,i],.15,color=colors[i],alpha=.5))
        ax.set_aspect('equal');ax.set_title('%s\ncoverage %.0f%%, overlaps %d'%(name,100*r['final_landmark_coverage'],r['collision_pair_steps']),fontsize=8)
        fig.savefig(gallery/(name+'.png'),dpi=100);plt.close(fig)
        inventory.append(dict(arm=r['protocol'],case=r['case'],initial_overlaps=r['initial_overlap_pairs'],
            overlap=not r['collision_free'],incomplete_coverage=not r['full_coverage'],figure=name+'.png'))
    (gallery/'inventory.json').write_text(json.dumps(inventory,indent=2))


def summarize(out,rows):
    fresh=[r for r in rows if r['profile']=='fresh'];summary={}
    keys=['mean_agent_return','final_assignment_distance','final_landmark_coverage','collision_pair_steps','new_contact_onsets',
        'collision_pair_steps_after_50','stall_agent_steps','interventions','infeasible_controls','goal_shifts',
        'control_seconds','protection_seconds','learning_seconds','projection_candidates','identification_work']
    for arm in ARMS:
        rr=[r for r in fresh if r['protocol']==arm]
        summary[arm]={k:float(np.mean([r[k] for r in rr])) for k in keys}
    comparisons=[]
    for treatment,control in [('attraction_protected','attraction'),('phmarl_protected','phmarl'),('coph_online_protected','coph_frozen_protected'),('coph_online_protected','attraction_protected'),('coph_online_protected','phmarl')]:
        tr={r['case']:r for r in fresh if r['protocol']==treatment};co={r['case']:r for r in fresh if r['protocol']==control}
        cases=sorted(tr);rng=np.random.RandomState(181000);idx=rng.randint(0,len(cases),size=(10000,len(cases)))
        row=dict(treatment=treatment,control=control,episodes=len(cases))
        for metric in ['mean_agent_return','collision_pair_steps','final_landmark_coverage']:
            d=np.array([tr[c][metric]-co[c][metric] for c in cases]);means=d[idx].mean(axis=1)
            row[metric]=dict(mean=float(d.mean()),interval95=np.percentile(means,[2.5,97.5]).tolist(),differences=d.tolist())
        row['return_wins']=sum(tr[c]['mean_agent_return']>co[c]['mean_agent_return'] for c in cases)
        comparisons.append(row)
    (out/'metrics.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2));(out/'paired.json').write_text(json.dumps(comparisons,indent=2))
    table='| Arm | Return ↑ | Coverage | Overlap pair-steps ↓ | New contact onsets ↓ | Infeasible controls |\n|---|---:|---:|---:|---:|---:|\n'
    for arm in ARMS:
        s=summary[arm];table+='| %s | %.2f | %.1f%% | %.1f | %.2f | %.1f |\n'%(arm,s['mean_agent_return'],100*s['final_landmark_coverage'],s['collision_pair_steps'],s['new_contact_onsets'],s['infeasible_controls'])
    text='''# Shared native protection: frozen six-arm comparison

All six arms use unchanged VMAS 1.2.6 simple_spread dynamics, native reward and bounds, four agents and 400 steps. Development checks replay old episodes 0 and 3 (12 runs); fresh evaluation uses seeds 81000–81019 (120 runs). The same sources/settings were frozen before both stages. No outcomes were used to change settings. The original phMARL development runs reproduce the archived arrays bitwise. The fresh initializations are new for this project; this is one checkpoint and one benchmark, not a multi-training-seed or paper-wide result.

%s
## Mechanisms and information

Attraction is native component-clipped u=(goal-q)-1.4v. CoPH frozen/online reuse the prior scalar-potential native adapter. Every protected arm uses exactly the same layer before and after its nominal policy, with public native dt=0.1, drag=0.25, radius=0.15. Known physics in this common layer is not inferred by the online learner. The online nominal controller still fits native drag from its own executed transitions, including the actual projected inputs. Environmental reconstruction, packets, historical replay and A* remain inactive.

Only peers within 0.45 in each agent's own observation enter the layer. Consecutive visible positions estimate peer velocity using the native semi-implicit identity; first sightings use zero and are counted as unknown. No peer private velocity, goal, proposed action or simulator truth is supplied. Near an own goal (distance <=0.35), its nominal target is projected onto local separating halfplanes through pair midpoints, with 0.32 desired center separation. The environment's actual landmark, native reward, coverage metric and terminal conditions never change. The same goal-offset rule also changes the phMARL actor's own-goal observation slots, so phMARL+protection is explicitly a modified policy.

Native action projection minimizes deviation from the nominal control over component bounds and reciprocal halfplanes. Halfplanes constrain pair separation along the currently observed radial direction under constant controls over one through four native steps. Each agent takes half the relative correction. Only the strongest parallel bound per peer is needed. A deterministic 2D active-set enumeration finds the exact projection when feasible. If infeasible, bounded braking is applied and logged; constraints are not silently relaxed. Goal-plane infeasibility retains the original target and is logged separately.

This is not certified safety: limited visibility, unknown first-sighting velocity, changing controls, and unmodelled contact forces invalidate any general guarantee. A successful unit test is not evidence that every rollout is safe. No extra global map, matching assignment, or enlarged sensing radius was introduced.

## Endpoints and interpretation

Primary diagnostics are native mean-agent return together with overlap pair-steps. Initial overlapping pairs, time to first separation, new contact onsets, overlaps after step 50, final coverage, and stalls are reported separately. New contact onsets exclude contacts already present at the immediately preceding step, including initialization; an initially overlapping pair can later produce a new onset after recovery. Pair-steps are not distinct collisions. A stall is <0.01 displacement over 16 steps while >0.15 from the original own goal; safe offsets can therefore be labeled as stalls, so this is diagnostic, not automatic task failure.

Paired differences and 10,000-resample episode-bootstrap intervals (seed 181000) are reported for protection versus no protection, online versus frozen, online versus protected attraction, and online versus original phMARL. No multiple-comparison or broad superiority claim is implied. The planned practical target is improved return without worse collisions; interpret both outcomes and coverage rather than selecting a favorable metric.

Computation includes candidate enumeration, identification work, interventions, and separately measured policy, projection and learning times. Worker contention makes wall times descriptive, not isolated deployment latency. The native task has no byte-budgeted network; no communication-efficiency claim is made.

## Reproduction and artifacts

Run `CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B benchmarks/phmarl/safety_suite.py --out benchmarks/phmarl/results/native_safety_v3 --workers 3` from the workspace root. Existing directories are never overwritten. The manifest and source archive precede execution; completed stages are sealed. JSON rows, paired outcomes, per-episode NPZ arrays, compressed protection/update traces, all failure-gallery entries, and a synchronized seed-81000 animation are retained. No episodes are dropped for collision, stall, infeasibility or incomplete supplementary coverage.
'''%table
    (out/'REPORT.md').write_text(text);print(table,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'results/native_safety_v2');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Output exists; refuse overwrite')
    development=[dict(id='old%d'%i,profile='development',index=i) for i in [0,3]]
    fresh=[dict(id='seed%d'%s,profile='fresh',seed=s) for s in FRESH]
    sources=[Path(__file__).resolve(),HERE/'protection.py',HERE/'test_protection.py',HERE/'adapter.py',HERE/'test_adapter.py',HERE/'reproduce.py',MODEL]
    manifest=dict(arms=ARMS,development=development,fresh=fresh,settings=SETTINGS,horizon=400,agents=4,
        animation_seed=81000,bootstrap_seed=181000,workers=args.workers,
        baseline_manifest_sha256=base.digest(HERE/'results/native_v1/manifest.json'),
        checkpoint_sha256=base.digest(base.CHECKPOINT),sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in sources},
        endpoint='native return improvement without worse overlap pair-steps, coverage reported jointly',
        thread_settings=dict(OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,torch=1))
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'source').mkdir()
    for p in sources:(out/'source'/(p.name+'.txt')).write_text(p.read_text())
    rows=[]
    for profile,specs in [('development',development),('fresh',fresh)]:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for arm,result in zip(ARMS,pool.map(worker,[(arm,specs,str(out)) for arm in ARMS])):
                rows+=result;print(profile+' '+arm+' complete',flush=True)
        for spec in specs:
            arrays=[np.load(out/('%s_%s.npz'%(arm,spec['id']))) for arm in ARMS]
            for a in arrays[1:]:
                assert np.array_equal(a['positions'][0],arrays[0]['positions'][0])
                assert np.array_equal(a['landmarks'],arrays[0]['landmarks'])
        (out/(profile+'_seal.json')).write_text(json.dumps(dict(episodes=len(specs)*6,paired=True,
            manifest_sha256=base.digest(out/'manifest.json'),sources_unchanged=all(base.digest(p)==manifest['sources'][str(p.relative_to(base.ROOT))] for p in sources)),indent=2))
    summarize(out,rows);render(out,rows)
    (out/'incomplete.json').write_text('[]\n')

if __name__=='__main__':main()
