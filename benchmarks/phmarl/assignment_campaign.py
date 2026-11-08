#!/usr/bin/env python3
"""Centralized, reset-only goal-assignment diagnosis; native task unchanged."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
import adapter
import protection
import safety_suite as suite
import reproduce as base

SEEDS=list(range(82000,82020))
MODES=['frozen','known','online']
ASSIGNMENTS=['index','initial_min_distance']
ARCHIVE=base.ROOT/'CoPH-Stagewise/results/current_campaign_v1/native'


def choose_assignment(observations, mode):
    if mode not in ASSIGNMENTS:raise ValueError(mode)
    obs=np.asarray(observations,dtype=np.float64).reshape(4,18)
    positions=obs[:,:2].copy()
    goals=positions[0]+obs[0,4:12].reshape(4,2)
    costs=((positions[:,None]-goals[None,:])**2).sum(axis=-1)
    permutations=list(itertools.permutations(range(4)))
    permutation=tuple(range(4)) if mode=='index' else min(permutations,key=lambda p:(sum(costs[i,p[i]] for i in range(4)),p))
    return permutation,dict(assignment_mode=mode,goal_indices=list(permutation),
        initial_positions=positions.tolist(),landmarks=goals.tolist(),
        assignment_squared_distance=float(sum(costs[i,permutation[i]] for i in range(4))),
        index_squared_distance=float(np.trace(costs)),centralized_reset_assignment=mode!='index')


def compare_archive(out, arm, old_arm, seed, row):
    stem='%s_seed%d'%(arm,seed);old='%s_seed%d'%(old_arm,seed)
    pairs=[];bitwise=True
    with np.load(out/(stem+'.npz')) as new,np.load(ARCHIVE/(old+'.npz')) as ref:
        assert set(new.files)==set(ref.files)
        for key in ref.files:
            a,b=new[key],ref[key]
            exact=bool(np.array_equal(a,b));close=bool(a.shape==b.shape and np.allclose(a,b,rtol=1e-7,atol=1e-8))
            pairs.append(dict(key=key,bitwise=exact,numerical=close));bitwise &= exact
            if not close:raise AssertionError(('archive array mismatch',stem,key))
    serialized={}
    for suffix in ['_trace.json.gz','_updates.json.gz']:
        with gzip.open(out/(stem+suffix),'rt') as f:a=json.load(f)
        with gzip.open(ARCHIVE/(old+suffix),'rt') as f:b=json.load(f)
        if a!=b:raise AssertionError(('archive decisions/updates mismatch',stem,suffix))
        serialized[suffix]=base.digest(out/(stem+suffix))==base.digest(ARCHIVE/(old+suffix))
    ref=json.loads((ARCHIVE/(old+'_metrics.json')).read_text())
    keys=['identification_work','updates_accepted','final_drag','interventions','infeasible_controls',
        'goal_shifts','goal_infeasible','unknown_peer_velocity_occurrences','projection_candidates']
    for key in keys:
        if row[key]!=ref[key]:raise AssertionError(('archive work mismatch',stem,key))
    return dict(arrays=pairs,all_arrays_bitwise=bitwise,decisions_updates_work_exact=True,
        compressed_bytes_identical=serialized,compressed_byte_note='gzip metadata can differ independently of decoded records')


def run(task):
    mode,assignment,seed,out=task;out=Path(out)
    arm='coph_%s_%s_protected'%(mode,assignment)
    metadata={};mapping={}
    class AssignedProtection(protection.Protection):
        def prepare(self,observations):
            if not mapping:
                indices,info=choose_assignment(observations,assignment)
                mapping['indices']=indices;metadata.update(info)
                self.goal_indices=indices
            return super().prepare(observations)
    class AssignedCoPH(adapter.NativeCoPH):
        def __init__(self,unused):super().__init__('known_native' if mode=='known' else mode)
        def action(self,observations):
            self.goal_indices=mapping['indices']
            return super().action(observations)
    suite.Protection=AssignedProtection;suite.NativeCoPH=AssignedCoPH
    spec=dict(id='seed%d'%seed,profile='fresh',seed=seed)
    row=suite.worker((arm,[spec],str(out)))[0]
    with np.load(out/('%s_seed%d.npz'%(arm,seed))) as data:
        positions=data['positions'];goals=data['landmarks'];targets=goals[list(mapping['indices'])]
        stalled=np.linalg.norm(positions[16:]-positions[:-16],axis=-1)<.01
        own_gap=np.linalg.norm(positions[16:]-targets[None],axis=-1)
        coverage=(np.linalg.norm(positions[16:,:,None]-goals[None,None,:],axis=-1).min(axis=1)<=.15).all(axis=1)
        row.update(assigned_goal_stall_agent_steps=int((stalled&(own_gap>.15)).sum()),
            stationary_agent_steps_with_incomplete_team_coverage=int((stalled&(~coverage[:,None])).sum()),
            final_assigned_goal_distances=np.linalg.norm(positions[-1]-targets,axis=-1).tolist(),
            legacy_stall_note='Original worker stall_agent_steps uses index goals; use assigned_goal_stall_agent_steps for reassigned runs')
    row.update(metadata,drag_mode=mode)
    if assignment=='index':row['archive_parity']=compare_archive(out,arm,'coph_%s_protected'%mode,seed,row)
    (out/('%s_seed%d_metrics.json'%(arm,seed))).write_text(json.dumps(row,indent=2))
    return row


def report(out,rows):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.patches import Circle
    rng=np.random.RandomState(182100);summary=[];paired=[]
    keys=['mean_agent_return','final_landmark_coverage','new_contact_onsets','collision_pair_steps',
        'assigned_goal_stall_agent_steps','stationary_agent_steps_with_incomplete_team_coverage',
        'interventions','infeasible_controls','identification_work','total_seconds']
    for mode in MODES:
        for assignment in ASSIGNMENTS:
            rr=[r for r in rows if r['drag_mode']==mode and r['assignment_mode']==assignment]
            summary.append(dict(drag_mode=mode,assignment_mode=assignment,n=len(rr),
                full_coverage=sum(r['full_coverage'] for r in rr),contact_free=sum(r['collision_free'] for r in rr),
                means={k:float(np.mean([r[k] for r in rr])) for k in keys}))
        old={r['episode']:r for r in rows if r['drag_mode']==mode and r['assignment_mode']=='index'}
        new={r['episode']:r for r in rows if r['drag_mode']==mode and r['assignment_mode']!='index'}
        assert set(old)==set(new)
        ids=sorted(old)
        for seed in ids:
            a,b=old[seed],new[seed]
            assert a['initial_positions']==b['initial_positions'] and a['landmarks']==b['landmarks']
        differences={k:[new[s][k]-old[s][k] for s in ids] for k in keys if k!='total_seconds'}
        d=np.array(differences['mean_agent_return']);indices=rng.randint(len(ids),size=(10000,len(ids)))
        paired.append(dict(drag_mode=mode,seeds=ids,differences=differences,
            mean_return_difference=float(d.mean()),interval95=np.percentile(d[indices].mean(axis=1),[2.5,97.5]).tolist(),
            coverage_rescued=[s for s in ids if not old[s]['full_coverage'] and new[s]['full_coverage']],
            coverage_lost=[s for s in ids if old[s]['full_coverage'] and not new[s]['full_coverage']],
            new_contact_increase=[s for s in ids if new[s]['new_contact_onsets']>old[s]['new_contact_onsets']]))
    (out/'results.json').write_text(json.dumps(dict(rows=rows,summary=summary,paired=paired,pairing_verified=True),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,s in enumerate(summary):
        for ax,key in zip(axes,['mean_agent_return','final_landmark_coverage','assigned_goal_stall_agent_steps']):
            ax.bar(i,s['means'][key]);ax.set_ylabel(key)
    for ax in axes:ax.set_xticks(range(6));ax.set_xticklabels([s['drag_mode']+'\n'+s['assignment_mode'] for s in summary],rotation=35,ha='right',fontsize=7)
    fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=150);plt.close(fig)
    gallery=out/'failure_gallery';gallery.mkdir(exist_ok=True);failures=[]
    for r in rows:
        if r['full_coverage'] and r['collision_free']:continue
        name=r['protocol']+'_'+r['case']
        with np.load(out/(name+'.npz')) as data:
            fig,ax=plt.subplots(figsize=(4,4));ax.scatter(*data['landmarks'].T,c='black',marker='x')
            for i in range(4):ax.plot(*data['positions'][:,i].T);ax.add_patch(Circle(data['positions'][-1,i],.15,alpha=.3))
            ax.set_aspect('equal');ax.set_title(name+'\ncoverage %.2f contacts %d'%(r['final_landmark_coverage'],r['new_contact_onsets']),fontsize=7)
            fig.tight_layout();fig.savefig(gallery/(name+'.png'),dpi=100);plt.close(fig)
        failures.append(dict(arm=r['protocol'],seed=r['episode'],coverage=r['final_landmark_coverage'],contact_free=r['collision_free'],figure='failure_gallery/'+name+'.png'))
    (out/'failure_inventory.json').write_text(json.dumps(failures,indent=2))
    for seed in [82000,82011,82019]:
        fig,axes=plt.subplots(2,3,figsize=(12,8));panels=[]
        for ax,(assignment,mode) in zip(axes.flat,itertools.product(ASSIGNMENTS,MODES)):
            arm='coph_%s_%s_protected'%(mode,assignment)
            with np.load(out/('%s_seed%d.npz'%(arm,seed))) as data:p=data['positions'].copy();g=data['landmarks'].copy()
            row=next(r for r in rows if r['protocol']==arm and r['episode']==seed)
            ax.scatter(*g.T,c='black',marker='x');lines=[ax.plot([],[])[0] for _ in range(4)]
            circles=[]
            for i in range(4):
                color=lines[i].get_color();c=Circle(p[0,i],.15,color=color,alpha=.5);ax.add_patch(c);circles.append(c)
                ax.scatter(*g[row['goal_indices'][i]],marker='x',color=color,s=60)
            ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),aspect='equal');panels.append((ax,mode,assignment,p,lines,circles))
        def update(t):
            for ax,mode,assignment,p,lines,circles in panels:
                for i in range(4):lines[i].set_data(*p[:t+1,i].T);circles[i].center=p[t,i]
                ax.set_title('%s / %s / step %d'%(mode,assignment,t),fontsize=9)
        fig.suptitle('Previously observed failure seed %d; centralized reset assignment diagnostic'%seed);fig.tight_layout()
        FuncAnimation(fig,update,frames=range(0,401,8),interval=100).save(out/('comparison_seed%d.gif'%seed),writer=PillowWriter(fps=10));plt.close(fig)
    text='# Native assignment diagnostic\n\n120 episodes, seeds 82000–82019, 400 steps. Centralized reset-only minimum squared-distance assignment; no dynamic reassignment. This uses all initial agent positions and is not a decentralized superiority comparison. Existing safety, native rewards, sensing, controller and identification parameters are unchanged.\n\n'
    text+='| Damping | Assignment | Full coverage | Contact-free | Mean return | Assigned-goal stall steps |\n|---|---|---:|---:|---:|---:|\n'
    for s in summary:text+='| %s | %s | %d/20 | %d/20 | %.3f | %.1f |\n'%(s['drag_mode'],s['assignment_mode'],s['full_coverage'],s['contact_free'],s['means']['mean_agent_return'],s['means']['assigned_goal_stall_agent_steps'])
    text+='\nAll 60 index-assignment runs reproduce archived decisions, model updates and operational work exactly. Array parity and gzip byte differences are recorded separately per episode. Pairing uses logged reset positions and goals. Intervals use 10,000 paired episode bootstrap resamples; these reused development starts do not establish generalization. Wall-clock timings include worker contention. Initial overlaps and all incomplete/contact episodes remain in the inventory.\n\n'
    for p in paired:text+='- %s: mean return difference %.3f, 95%% interval %s; coverage rescued %s, lost %s; seeds with increased new contacts %s.\n'%(p['drag_mode'],p['mean_return_difference'],p['interval95'],p['coverage_rescued'],p['coverage_lost'],p['new_contact_increase'])
    text+='\nAssignment changes test task allocation only. Residual feasible stalls indicate unresolved control/planning limitations. No recovery waypoints were added; receiver reconstruction and replay are absent from this native adapter.\n'
    (out/'REPORT.md').write_text(text);print(text,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    sources=[p.resolve() for p in [Path(__file__),Path(adapter.__file__),Path(protection.__file__),Path(suite.__file__),Path(base.__file__),adapter.MODEL,Path(__file__).with_name('test_assignment.py')]]
    scenario=base.ROOT/'CoPh-sense_wrong/external/envs/phmarl-py38/lib/python3.8/site-packages/vmas/scenarios/mpe/simple_spread.py';sources.append(scenario)
    manifest=dict(seeds=SEEDS,drag_modes=MODES,assignment_modes=ASSIGNMENTS,horizon=400,dt=.1,
        objective='initial squared-distance one-to-one assignment',tie_break='lexicographic exact tie',
        role='centralized development diagnostic',animation_seeds=[82000,82011,82019],bootstrap_seed=182100,
        numpy=np.__version__,torch=torch.__version__,python=platform.python_version(),threads=1,workers=args.workers,
        settings=adapter.SETTINGS,protection=protection.SETTINGS,checkpoint_sha256=base.digest(base.CHECKPOINT),
        sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in sources},
        archived_inputs={p.name:base.digest(p) for p in ARCHIVE.iterdir() if p.is_file() and any(p.name.startswith('coph_'+m+'_protected_') for m in MODES)})
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));archive=out/'sources';archive.mkdir()
    for p in sources:(archive/p.name).write_bytes(p.read_bytes())
    rows=[];begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for assignment in ASSIGNMENTS:
            jobs=[pool.submit(run,(m,assignment,s,str(out))) for s in SEEDS for m in MODES]
            for job in as_completed(jobs):
                rows.append(job.result());(out/'completion.json').write_text(json.dumps(rows,indent=2));print(len(rows),'/120',rows[-1]['protocol'],rows[-1]['case'],flush=True)
            if assignment=='index':(out/'legacy_gate.json').write_text(json.dumps(dict(passed=True,episodes=60,arrays_bitwise=all(r['archive_parity']['all_arrays_bitwise'] for r in rows)),indent=2))
    for p in sources:
        if base.digest(p)!=manifest['sources'][str(p.relative_to(base.ROOT))]:raise RuntimeError('Source changed during run: '+str(p))
    report(out,rows);(out/'execution.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,complete=len(rows),expected=120),indent=2))

if __name__=='__main__':main()
