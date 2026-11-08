#!/usr/bin/env python3
"""Frozen static-target experiment, independent reset allocation, matched phMARL."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
import torch
import adapter
import protection
import safety_suite as suite
import reproduce as base
from goal_geometry import assignment_consensus,local_reset_geometry,static_targets

SEEDS=list(range(82000,82020))
BASELINES=['coph_frozen_legacy_protected','coph_known_legacy_protected','coph_online_legacy_protected','phmarl']
TREATMENTS=['coph_frozen_static_protected','coph_known_static_protected','coph_online_static_protected',
    'phmarl_assigned','phmarl_assigned_legacy_protected','phmarl_assigned_static_protected']
ARMS=BASELINES+TREATMENTS
ORIGINAL_SETUP=suite.setup
ORIGINAL_CPU=base.cpu
ROOT=base.ROOT


def reset_context(observations):
    consensus=assignment_consensus(observations)
    if not consensus['agreed']:raise ValueError('Independent assignments disagree')
    mapping=consensus['assignments'][0]
    targets=[];plans=[]
    for i,o in enumerate(observations):
        positions,goals=local_reset_geometry(o,i)
        t,info=static_targets(goals)
        if not info['feasible']:raise ValueError('Fixed-normal target construction failed: '+str(info))
        targets.append(t[mapping[i]]);plans.append(info)
    targets=np.asarray(targets)
    distances=[np.linalg.norm(targets[i]-targets[j]) for i in range(4) for j in range(i+1,4)]
    if min(distances)<.32-1e-7:raise ValueError('Independent targets do not jointly clear separation')
    goals=local_reset_geometry(observations[0],0)[1]
    return dict(goal_indices=mapping,fixed_targets=targets.tolist(),consensus=consensus,plans=plans,
        minimum_independent_target_spacing=float(min(distances)),
        independent_target_coverage_gaps=np.linalg.norm(targets-goals[mapping],axis=1).tolist())


def parity(out,name,reference,row):
    bitwise={};max_error={}
    with np.load(out/(name+'.npz')) as a,np.load(reference.with_suffix('.npz')) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            assert a[key].shape==b[key].shape
            bitwise[key]=bool(np.array_equal(a[key],b[key]));max_error[key]=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float))))
            if not np.allclose(a[key],b[key],rtol=1e-7,atol=1e-8):raise AssertionError(('legacy array discrepancy',name,key))
    serialized={}
    for suffix in ['_trace.json.gz','_updates.json.gz']:
        ref=Path(str(reference)+suffix)
        with gzip.open(out/(name+suffix),'rt') as f:a=json.load(f)
        with gzip.open(ref,'rt') as f:b=json.load(f)
        if a!=b:raise AssertionError(('legacy decoded record discrepancy',name,suffix))
        serialized[suffix]=base.digest(out/(name+suffix))==base.digest(ref)
    ref=json.loads(Path(str(reference)+'_metrics.json').read_text())
    for key in ['identification_work','updates_accepted','final_drag','interventions','infeasible_controls','goal_shifts','goal_infeasible','unknown_peer_velocity_occurrences','projection_candidates']:
        if row[key]!=ref[key]:raise AssertionError(('legacy work discrepancy',name,key))
    return dict(arrays_bitwise=bitwise,maximum_errors=max_error,decoded_records_and_work_exact=True,compressed_bytes_equal=serialized)


def run(task):
    arm,seed,out=task;out=Path(out);context={}
    unmodified=arm=='phmarl';static='_static_' in arm
    def setup(spec):
        env,observations=ORIGINAL_SETUP(spec)
        context.update(reset_context(observations))
        if unmodified:context['goal_indices']=list(range(4))
        return env,observations
    class AssignedProtection(protection.Protection):
        def __init__(self):super().__init__(context['goal_indices'],context['fixed_targets'] if static else None)
    class AssignedCoPH(adapter.NativeCoPH):
        def __init__(self,unused):
            mode='known_native' if '_known_' in arm else ('online' if '_online_' in arm else 'frozen')
            super().__init__(mode,context['goal_indices'])
    class AssignedActor:
        def __init__(self,actor):self.actor=actor
        def __call__(self,x):
            y=x.clone();goals=x[:,:,4:12].reshape(x.shape[0],4,4,2)
            y[:,:,4:12]=goals[:,:,context['goal_indices'],:].reshape(x.shape[0],4,8)
            return self.actor(y)
    suite.setup=setup;suite.Protection=AssignedProtection;suite.NativeCoPH=AssignedCoPH
    base.cpu=(lambda actor:ORIGINAL_CPU(actor)) if unmodified else (lambda actor:AssignedActor(ORIGINAL_CPU(actor)))
    spec=dict(id='seed%d'%seed,profile='fresh',seed=seed)
    row=suite.worker((arm,[spec],str(out)))[0]
    name=arm+'_seed%d'%seed
    with np.load(out/(name+'.npz')) as data:
        q=data['positions'];g=data['landmarks'];assigned=g[context['goal_indices']]
        stationary=np.linalg.norm(q[16:]-q[:-16],axis=-1)<.01
        gap=np.linalg.norm(q[16:]-assigned[None],axis=-1)
        incomplete=~((np.linalg.norm(q[16:,:,None]-g[None,None,:],axis=-1).min(axis=1)<=.15).all(axis=1))
        row.update(assigned_goal_stall_agent_steps=int((stationary&(gap>.15)).sum()),
            stationary_agent_steps_with_incomplete_team_coverage=int((stationary&incomplete[:,None]).sum()),
            final_assigned_goal_distances=np.linalg.norm(q[-1]-assigned,axis=-1).tolist())
    row.update(reset_context=context,target_mode='static_landmark' if static else ('none' if unmodified or arm=='phmarl_assigned' else 'legacy'),
        assignment_mode='index' if unmodified else 'independent_initial_min_distance',
        reset_planning_seconds=sum(p['seconds'] for p in context['plans']),
        reset_planning_iterations=sum(p['iterations'] for p in context['plans']),
        reset_planning_residual_evaluations=sum(p['nfev'] for p in context['plans']))
    if arm in BASELINES:
        if unmodified:reference=ROOT/'CoPH-Stagewise/results/current_campaign_v1/native'/('phmarl_seed%d'%seed)
        else:reference=ROOT/'CoPH-Stagewise/results/native_assignment_v1'/('coph_%s_initial_min_distance_protected_seed%d'%(arm.split('_')[1],seed))
        row['parity']=parity(out,name,reference,row)
    (out/(name+'_metrics.json')).write_text(json.dumps(row,indent=2));return row


def report(out,rows):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.patches import Circle
    summary=[]
    keys=['mean_agent_return','final_landmark_coverage','new_contact_onsets','collision_pair_steps','assigned_goal_stall_agent_steps',
        'stationary_agent_steps_with_incomplete_team_coverage','interventions','infeasible_controls','identification_work','reset_planning_seconds']
    for arm in ARMS:
        rr=sorted([r for r in rows if r['protocol']==arm],key=lambda r:r['episode'])
        summary.append(dict(arm=arm,n=len(rr),full_coverage=sum(r['full_coverage'] for r in rr),contact_free=sum(r['collision_free'] for r in rr),
            incomplete_seeds=[r['episode'] for r in rr if not r['full_coverage']],means={k:float(np.mean([r[k] for r in rr])) for k in keys}))
    comparisons=[('coph_'+m+'_static_protected','coph_'+m+'_legacy_protected') for m in ['frozen','known','online']]
    comparisons += [('phmarl_assigned','phmarl'),('phmarl_assigned_static_protected','phmarl_assigned_legacy_protected'),
        ('coph_online_static_protected','phmarl_assigned_static_protected'),('coph_online_static_protected','coph_frozen_static_protected'),
        ('coph_online_static_protected','coph_known_static_protected')]
    rng=np.random.RandomState(182200);paired=[]
    for a,b in comparisons:
        aa={r['episode']:r for r in rows if r['protocol']==a};bb={r['episode']:r for r in rows if r['protocol']==b}
        d=np.array([aa[s]['mean_agent_return']-bb[s]['mean_agent_return'] for s in SEEDS]);idx=rng.randint(20,size=(10000,20))
        paired.append(dict(treatment=a,control=b,return_difference=float(d.mean()),interval95=np.percentile(d[idx].mean(axis=1),[2.5,97.5]).tolist(),
            return_differences=d.tolist(),coverage_rescued=[s for s in SEEDS if aa[s]['full_coverage'] and not bb[s]['full_coverage']],
            coverage_lost=[s for s in SEEDS if bb[s]['full_coverage'] and not aa[s]['full_coverage']],
            new_contacts_increased=[s for s in SEEDS if aa[s]['new_contact_onsets']>bb[s]['new_contact_onsets']]))
    # Pairing checked in logged trajectories rather than inferred from seeds.
    for seed in SEEDS:
        reference=None
        for arm in ARMS:
            with np.load(out/('%s_seed%d.npz'%(arm,seed))) as d:
                now=(d['positions'][0].copy(),d['landmarks'].copy())
            if reference is not None:
                for a,b in zip(now,reference):assert np.array_equal(a,b)
            reference=now
    (out/'results.json').write_text(json.dumps(dict(rows=rows,summary=summary,paired=paired,pairing_verified=True),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for i,s in enumerate(summary):
        for ax,key in zip(axes,['mean_agent_return','final_landmark_coverage','assigned_goal_stall_agent_steps']):ax.bar(i,s['means'][key]);ax.set_ylabel(key)
    for ax in axes:ax.set_xticks(range(len(summary)));ax.set_xticklabels([s['arm'] for s in summary],rotation=55,ha='right',fontsize=6)
    fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=150);plt.close(fig)
    failures=[];gallery=out/'failure_gallery';gallery.mkdir(exist_ok=True)
    for row in rows:
        if row['full_coverage'] and row['collision_free']:continue
        name=row['protocol']+'_'+row['case']
        with np.load(out/(name+'.npz')) as d:
            fig,ax=plt.subplots(figsize=(4,4));ax.scatter(*d['landmarks'].T,c='black',marker='x')
            for i in range(4):ax.plot(*d['positions'][:,i].T);ax.add_patch(Circle(d['positions'][-1,i],.15,alpha=.3))
            ax.set_aspect('equal');ax.set_title(name+'\ncoverage %.2f contacts %d'%(row['final_landmark_coverage'],row['new_contact_onsets']),fontsize=7)
            fig.tight_layout();fig.savefig(gallery/(name+'.png'),dpi=100);plt.close(fig)
        failures.append(dict(arm=row['protocol'],seed=row['episode'],coverage=row['final_landmark_coverage'],contact_free=row['collision_free'],figure='failure_gallery/'+name+'.png'))
    (out/'failure_inventory.json').write_text(json.dumps(failures,indent=2))
    animation_arms=['coph_online_legacy_protected','coph_online_static_protected','coph_known_static_protected','phmarl_assigned_static_protected']
    for seed in [82004,82017,82011]:
        fig,axes=plt.subplots(1,4,figsize=(15,4));panels=[]
        for ax,arm in zip(axes,animation_arms):
            row=next(r for r in rows if r['protocol']==arm and r['episode']==seed)
            with np.load(out/('%s_seed%d.npz'%(arm,seed))) as d:p=d['positions'].copy();g=d['landmarks'].copy()
            lines=[ax.plot([],[])[0] for _ in range(4)];circles=[]
            for i in range(4):
                color=lines[i].get_color();goal=g[row['reset_context']['goal_indices'][i]]
                ax.add_patch(Circle(goal,.15,fill=False,color=color,alpha=.5));ax.plot(*goal,'x',color=color)
                if '_static_' in arm:ax.plot(*row['reset_context']['fixed_targets'][i],'+',color=color)
                circle=Circle(p[0,i],.15,color=color,alpha=.5);ax.add_patch(circle);circles.append(circle)
            ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),aspect='equal');panels.append((ax,arm,p,lines,circles))
        def update(t):
            for ax,arm,p,lines,circles in panels:
                for i in range(4):lines[i].set_data(*p[:t+1,i].T);circles[i].center=p[t,i]
                ax.set_title(arm+'\nstep %d'%t,fontsize=8)
        fig.suptitle('Native goal geometry: seed %d; circles: coverage, crosses: landmarks, plus: static targets'%seed);fig.tight_layout()
        FuncAnimation(fig,update,frames=range(0,401,8),interval=100).save(out/('comparison_seed%d.gif'%seed),writer=PillowWriter(fps=10));plt.close(fig)
    text='# Native static landmark targets and matched allocation\n\n200 episodes, 20 reused development starts, 400 native steps. All 80 baseline episodes pass archive parity before the 120 new comparisons run. All agents independently reconstruct initial geometry and obtain matching reset assignments from their full native observations; the geometric input is not restricted to the 0.45 interaction radius. Peer and landmark identity ordering is assumed. No communication or consensus protocol is added.\n\n'
    text+='Targets solve a minimum-displacement convex program using fixed landmark-pair normals, 0.32 projected separation, and displacement at most 0.149999. This is the symmetric split for an isolated pair, extended jointly to clusters. Targets are frozen at reset and used inside the existing 0.35 near-goal region. Actual-motion projection and constraints are unchanged. Feasibility supplies a constructive witness; failed fixed-normal construction alone would not prove global geometric impossibility. Minimum spacing below 0.02 is a pairwise impossibility certificate, not a general characterization for clusters.\n\n'
    text+='| Arm | Full coverage | Contact-free | Mean return | New contact onsets |\n|---|---:|---:|---:|---:|\n'
    for s in summary:text+='| %s | %d/20 | %d/20 | %.3f | %.2f |\n'%(s['arm'],s['full_coverage'],s['contact_free'],s['means']['mean_agent_return'],s['means']['new_contact_onsets'])
    text+='\nThe checkpoint itself selects the index-specific goal slot. Assigned phMARL permutes all landmark slots consistently in every agent observation using the same reset assignment. Raw assigned, protected assigned and unmodified checkpoint arms are explicitly distinct. The static protected arm receives the same target and actual-motion protection treatment as CoPH. No retraining is performed.\n\n'
    for p in paired:text+='- %s versus %s: mean return difference %.3f, paired bootstrap 95%% interval %s; rescued %s, lost %s, increased new contacts %s.\n'%(p['treatment'],p['control'],p['return_difference'],p['interval95'],p['coverage_rescued'],p['coverage_lost'],p['new_contacts_increased'])
    text+='\nIntervals use 10,000 episode bootstrap draws. All prior runs, new contacts, incomplete coverage and initial overlaps remain reported. No new start is silently excluded as geometrically infeasible. Planning work is recorded separately from rollout work; timings include multiprocessing contention and are not deployment latency. These reused starts diagnose mechanisms and do not establish held-out superiority, training reproduction, or benefits from receiver reconstruction/historical replay.\n'
    (out/'REPORT.md').write_text(text);print(text,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output directory')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    sources=[Path(__file__).resolve(),Path(__file__).with_name('goal_geometry.py').resolve(),Path(__file__).with_name('test_goal_geometry.py').resolve(),
        Path(adapter.__file__),Path(protection.__file__),Path(suite.__file__),Path(base.__file__),adapter.MODEL,base.UPSTREAM/'functions.py']
    inputs={}
    for seed in SEEDS:
        names=[ROOT/'CoPH-Stagewise/results/native_assignment_v1'/('coph_%s_initial_min_distance_protected_seed%d'%(m,seed)) for m in ['frozen','known','online']]
        names.append(ROOT/'CoPH-Stagewise/results/current_campaign_v1/native'/('phmarl_seed%d'%seed))
        for name in names:
            for suffix in ['.npz','_trace.json.gz','_updates.json.gz','_metrics.json']:
                p=Path(str(name)+suffix);inputs[str(p.relative_to(ROOT))]=base.digest(p)
    manifest=dict(seeds=SEEDS,arms=ARMS,baseline_gate_arms=BASELINES,horizon=400,dt=.1,bootstrap_seed=182200,animation_seeds=[82004,82017,82011],
        role='reused development diagnostic; final-test namespaces untouched',threads=1,workers=args.workers,
        assignment='minimum total squared reset distance; independent full native observations, lexicographic exact ties; freeze for episode',
        targets='joint fixed-landmark-normal minimum squared displacement; cap .149999, separation .32; activate at existing near-goal radius .35',
        solver=dict(method='SLSQP',ftol=1e-12,maxiter=200,feasibility_tolerance=1e-9,independent_spacing_tolerance=1e-7),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
        adapter=adapter.SETTINGS,protection=protection.SETTINGS,checkpoint_sha256=base.digest(base.CHECKPOINT),
        sources={str(p.relative_to(ROOT)):base.digest(p) for p in sources},inputs=inputs)
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));archive=out/'sources';archive.mkdir()
    for p in sources:(archive/p.name).write_bytes(p.read_bytes())
    gates=[]
    for seed in SEEDS:
        env,obs=ORIGINAL_SETUP(dict(id='seed%d'%seed,profile='fresh',seed=seed))
        try:
            context=reset_context([o.detach().numpy().copy() for o in obs]);gates.append(dict(seed=seed,passed=True,context=context))
        except ValueError as error:gates.append(dict(seed=seed,passed=False,reason=str(error)))
    gate_passed=all(g['passed'] for g in gates)
    (out/'geometry_gate.json').write_text(json.dumps(dict(passed=gate_passed,starts=20,rows=gates),indent=2))
    if not gate_passed:raise SystemExit('Geometry/independent-assignment gate failed; no episodes run')
    rows=[];begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for arms in [BASELINES,TREATMENTS]:
            jobs=[pool.submit(run,(a,s,str(out))) for s in SEEDS for a in arms]
            for job in as_completed(jobs):
                rows.append(job.result());(out/'completion.json').write_text(json.dumps(rows,indent=2));print(len(rows),'/200',rows[-1]['protocol'],rows[-1]['case'],flush=True)
            if arms==BASELINES:(out/'legacy_gate.json').write_text(json.dumps(dict(passed=True,episodes=80,all_arrays_bitwise=all(all(r['parity']['arrays_bitwise'].values()) for r in rows)),indent=2))
    for p in sources:
        if base.digest(p)!=manifest['sources'][str(p.relative_to(ROOT))]:raise RuntimeError('Source changed during run: '+str(p))
    report(out,rows);(out/'execution.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,complete=len(rows),expected=200),indent=2))

if __name__=='__main__':main()
