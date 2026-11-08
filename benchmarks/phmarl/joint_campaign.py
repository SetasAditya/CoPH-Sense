#!/usr/bin/env python3
"""Frozen opt-in joint protection: development gate then conditional fresh starts."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import numpy as np
import scipy
import torch
import goal_geometry_campaign as gg
from protection import Protection as LegacyProtection
from joint_protection import JointProtection,SETTINGS
import adapter
import safety_suite as suite
import reproduce as base

ARMS=['coph_frozen_static_joint_protected','coph_known_static_joint_protected','coph_online_static_joint_protected','phmarl_assigned_static_joint_protected']
DEVELOPMENT=list(range(82000,82020))+[85048]
VALIDATION=list(range(85200,85250))
GOLDEN_SEEDS=[82000,82011]
ROOT=base.ROOT


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def unused_validation():
    used=[]
    pattern=re.compile(r'(?:seed|_)(852[0-4][0-9])(?:\D|$)')
    for directory in (ROOT/'CoPH-Stagewise/results',ROOT/'benchmarks/phmarl/results'):
        for p in directory.rglob('*'):
            if p.is_file() and pattern.search(p.name):used.append(str(p.relative_to(ROOT)))
    return dict(seeds=VALIDATION,unused=not used,matching_episode_named_artifacts=used,
        check='Exact seed-number episode filenames across both result trees; manifests reserve the namespace without executing it.')


def worker(task):
    arm,seed,out,joint=task;out=Path(out);gg.BASELINES=[]
    gg.protection.Protection=JointProtection if joint else LegacyProtection
    try:
        row=gg.run((arm,seed,str(out)))
        row['protection_mode']='joint' if joint else 'legacy'
        if joint:
            path=out/(arm+'_seed%d_trace.json.gz'%seed)
            with gzip.open(path,'rt') as f:trace=json.load(f)
            diagnostics=[batch[0]['joint'] for batch in trace]
            row.update(joint=dict(statuses={s:sum(d['status']==s for d in diagnostics) for s in sorted({d['status'] for d in diagnostics})},
                initial_unsafe_pairs=diagnostics[0]['initial_unsafe_pairs'],
                affine_infeasible_steps=sum(d['affine_infeasibility_certified'] for d in diagnostics),
                numerical_failure_steps=sum(d['status'] in ('numerical_failure','feasibility_probe_failed','float32_feasibility_failure') for d in diagnostics),
                minimum_constraint_residual=min(d['minimum_applied_constraint_residual'] for d in diagnostics),
                solver_iterations=sum(a['iterations'] for d in diagnostics for a in d['attempts']),
                solver_evaluations=sum(a['function_evaluations'] for d in diagnostics for a in d['attempts']),
                total_solver_seconds=sum(d['seconds'] for d in diagnostics),
                geometric_infeasibility_certified=False))
        write(out/(arm+'_seed%d_metrics.json'%seed),row)
        return dict(complete=True,**row)
    except Exception as error:
        failure=dict(complete=False,protocol=arm,episode=seed,case='seed%d'%seed,error=str(error),traceback=traceback.format_exc(),
            full_coverage=False,new_contact_onsets=None,geometric_infeasibility_certified=False,
            interpretation='Execution/planning unavailable is retained as failure; no geometric impossibility inferred.')
        write(out/(arm+'_seed%d_ERROR.json'%seed),failure);return failure


def golden_worker(task):
    arm,seed,out=task;row=worker((arm,seed,out,False))
    if not row['complete']:return dict(arm=arm,seed=seed,passed=False,error=row)
    reference=ROOT/'CoPH-Stagewise/results/native_goal_geometry_v1'/(arm+'_seed%d'%seed)
    result=gg.parity(Path(out),arm+'_seed%d'%seed,reference,row)
    return dict(arm=arm,seed=seed,passed=True,**result)


def run_phase(out,seeds,workers):
    out.mkdir(parents=True,exist_ok=False);rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,(a,s,str(out),True)) for s in seeds for a in ARMS]
        for job in as_completed(jobs):
            rows.append(job.result());write(out/'rows.json',rows)
            print(out.name,len(rows),'/',len(jobs),rows[-1]['protocol'],rows[-1]['episode'],rows[-1]['complete'],flush=True)
    by={a:{r['episode']:r for r in rows if r['protocol']==a} for a in ARMS}
    summary={}
    for arm in ARMS:
        rr=[by[arm][s] for s in seeds];complete=[r for r in rr if r['complete']]
        summary[arm]=dict(expected=len(rr),completed=len(complete),full_coverage=sum(r['full_coverage'] for r in rr),
            new_contact_onsets=sum(r['new_contact_onsets'] or 0 for r in rr),initial_overlap_pairs=sum(r.get('initial_overlap_pairs',0) for r in rr),
            mean_return=float(np.mean([r['mean_agent_return'] for r in complete])) if complete else None,
            mean_coverage=float(np.mean([r['final_landmark_coverage'] for r in complete])) if complete else None,
            incomplete_seeds=[r['episode'] for r in rr if not r['complete'] or not r['full_coverage']],
            numerical_failure_steps=sum(r.get('joint',{}).get('numerical_failure_steps',0) for r in rr),
            affine_infeasible_steps=sum(r.get('joint',{}).get('affine_infeasible_steps',0) for r in rr),
            geometric_infeasibility_certificates=0)
    rng=np.random.RandomState(192000);paired=[]
    online=ARMS[2]
    for control in [ARMS[0],ARMS[1],ARMS[3]]:
        if all(by[online][s]['complete'] and by[control][s]['complete'] for s in seeds):
            d=np.array([by[online][s]['mean_agent_return']-by[control][s]['mean_agent_return'] for s in seeds]);idx=rng.randint(0,len(seeds),size=(10000,len(seeds)))
            paired.append(dict(treatment=online,control=control,n=len(seeds),difference=float(d.mean()),interval95=np.percentile(d[idx].mean(axis=1),[2.5,97.5]).tolist(),differences=d.tolist()))
    match=next((r for r in paired if r['control']==ARMS[3]),None)
    gate=dict(all_complete=all(r['complete'] for r in rows),zero_new_contacts=all(r['complete'] and r['new_contact_onsets']==0 for r in rows),
        coverage=all(summary[a]['full_coverage']>=.95*len(seeds) for a in ARMS[:3]),
        online_superiority=bool(match and match['difference']>0 and match['interval95'][0]>0))
    gate['passed']=all(gate.values())
    result=dict(seeds=seeds,arms=ARMS,summary=summary,paired=paired,gate=gate,seconds=time.perf_counter()-start)
    write(out/'summary.json',result)
    # Pair actual initial states and landmarks, including the known failed start.
    pairing=[]
    for s in seeds:
        valid=[r for r in rows if r['episode']==s and r['complete']]
        reference=None
        for r in valid:
            with np.load(out/(r['protocol']+'_seed%d.npz'%s)) as d:now=(d['positions'][0].copy(),d['landmarks'].copy())
            if reference is not None:
                for x,y in zip(reference,now):assert np.array_equal(x,y)
            reference=now
        pairing.append(dict(seed=s,complete_arms=len(valid),initials_equal=True))
    write(out/'pairing.json',pairing);render(out,rows,seeds[0]);return result


def render(out,rows,animation_seed):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.patches import Circle
    gallery=out/'failure_gallery';gallery.mkdir();inventory=[]
    for row in rows:
        if row['complete'] and row['full_coverage'] and row['collision_free']:continue
        entry=dict(arm=row['protocol'],seed=row['episode'],complete=row['complete'],full_coverage=row['full_coverage'],new_contacts=row['new_contact_onsets'])
        if row['complete']:
            name=row['protocol']+'_'+row['case'];d=np.load(out/(name+'.npz'));fig,ax=plt.subplots(figsize=(4,4));ax.scatter(*d['landmarks'].T,c='black',marker='x')
            for i in range(4):ax.plot(*d['positions'][:,i].T);ax.add_patch(Circle(d['positions'][-1,i],.15,alpha=.4))
            ax.set_aspect('equal');ax.set_title(name+'\ncoverage%.2f newcontacts%d'%(row['final_landmark_coverage'],row['new_contact_onsets']),fontsize=7)
            fig.tight_layout();fig.savefig(gallery/(name+'.png'),dpi=100);plt.close(fig);entry['figure']='failure_gallery/'+name+'.png'
        inventory.append(entry)
    write(out/'failure_inventory.json',inventory)
    for seed in dict.fromkeys([animation_seed]+([85048] if any(r['episode']==85048 for r in rows) else [])):
        if not all(any(r['episode']==seed and r['protocol']==a and r['complete'] for r in rows) for a in ARMS):continue
        fig,axes=plt.subplots(1,4,figsize=(15,4));panels=[]
        for ax,a in zip(axes,ARMS):
            d=np.load(out/(a+'_seed%d.npz'%seed));p=d['positions'];ax.scatter(*d['landmarks'].T,c='black',marker='x');lines=[ax.plot([],[])[0] for _ in range(4)];circles=[]
            for i in range(4):
                c=Circle(p[0,i],.15,color=lines[i].get_color(),alpha=.45);ax.add_patch(c);circles.append(c)
            ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),aspect='equal');panels.append((ax,a,p,lines,circles))
        def update(t):
            for ax,a,p,lines,circles in panels:
                for i in range(4):lines[i].set_data(*p[:t+1,i].T);circles[i].center=p[t,i]
                ax.set_title(a+'\nstep%d'%t,fontsize=8)
        fig.suptitle('Centralized joint safety projection; seed%d (initial overlaps retained)'%seed);fig.tight_layout()
        FuncAnimation(fig,update,frames=range(0,401,8),interval=100).save(out/('comparison_seed%d.gif'%seed),writer=PillowWriter(fps=10));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,a in enumerate(ARMS):
        rr=[r for r in rows if r['protocol']==a and r['complete']]
        for ax,key in zip(axes,['mean_agent_return','final_landmark_coverage','new_contact_onsets']):ax.scatter([i]*len(rr),[r[key] for r in rr],s=15)
    for ax,label in zip(axes,['Native return','Final coverage','New contact onsets']):ax.set_xticks(range(4));ax.set_xticklabels(ARMS,rotation=35,ha='right',fontsize=7);ax.set_ylabel(label)
    fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=16);args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    unused=unused_validation();assert unused['unused'],'Proposed validation namespace already used'
    sources=[Path(__file__).resolve(),*[Path(__file__).resolve().with_name(n) for n in ['joint_protection.py','test_joint_protection.py','goal_geometry_campaign.py','goal_geometry.py','protection.py','adapter.py','safety_suite.py','reproduce.py']],adapter.MODEL,base.UPSTREAM/'functions.py']
    source_hashes={str(p.relative_to(ROOT)):base.digest(p) for p in sources}
    inputs={}
    for seed in GOLDEN_SEEDS:
        for a in ARMS:
            arm=a.replace('_joint_','_');stem=ROOT/'CoPH-Stagewise/results/native_goal_geometry_v1'/(arm+'_seed%d'%seed)
            for suffix in ['.npz','_metrics.json','_trace.json.gz','_updates.json.gz']:
                p=Path(str(stem)+suffix);inputs[str(p.relative_to(ROOT))]=base.digest(p)
    manifest=dict(development=DEVELOPMENT,validation_reserved=VALIDATION,validation_unused_check=unused,arms=ARMS,
        development_role='20 reused development starts plus85048 formerly-held-out failure now explicitly development regression',horizon=400,agents=4,
        golden_seeds=GOLDEN_SEEDS,known_regression_animation_seed=85048,development_animation_seed=82000,validation_animation_seed=85200,
        gates='all four arms zero new contact onsets; each CoPH arm >=95%fullcoverage; online-minus-matchedph mean>0 and paired95%bootstrap lower>0; no exclusions',
        bootstrap_samples=10000,bootstrap_seed=192000,joint=SETTINGS,legacy_protection=gg.protection.SETTINGS,adapter=adapter.SETTINGS,
        assumptions='Centralized protection pools measured own q/v and all actions; same layer for all four nominal arms. Assignment/static targets unchanged. No contact impulse prediction; no general safety guarantee.',
        sources=source_hashes,inputs=inputs,checkpoint_sha256=base.digest(base.CHECKPOINT),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,device='cpu',threads=1,workers=args.workers)
    out.mkdir(parents=True);write(out/'manifest.json',manifest);archive=out/'sources';archive.mkdir()
    for p in sources:(archive/p.name).write_bytes(p.read_bytes())
    golden=out/'legacy';golden.mkdir();checks=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,8)) as pool:
        for future in as_completed([pool.submit(golden_worker,(a.replace('_joint_','_'),s,str(golden))) for s in GOLDEN_SEEDS for a in ARMS]):
            checks.append(future.result());write(out/'legacy_parity.json',checks)
    if not all(r['passed'] for r in checks):raise RuntimeError('Legacy parity failed; stop before treatments')
    dev=run_phase(out/'development',DEVELOPMENT,args.workers);validation=None
    if dev['gate']['passed']:
        current=unused_validation();assert current['unused']
        assert all(base.digest(ROOT/p)==h for p,h in source_hashes.items())
        write(out/'validation_manifest.json',dict(seeds=VALIDATION,source_hashes=source_hashes,development_gate=dev['gate'],
            unused_check=current,settings_frozen_after_development=True,controller_or_solver_changes_after_development=False))
        validation=run_phase(out/'validation',VALIDATION,args.workers)
    assert all(base.digest(ROOT/p)==h for p,h in source_hashes.items())
    verdict=dict(component='native centralized joint peer action projection',complete=True,mechanism_correct=True,
        performance_demonstrated=bool(validation and validation['gate']['passed']),development_gate=dev['gate'],validation_gate=validation['gate'] if validation else None,
        metrics=dict(development=dev['summary'],validation=validation['summary'] if validation else None),
        limitations=['Pooled own velocities and all nominal actions require centralized coordination or an explicit state/action exchange.',
            'Native collision impulses are not predicted; initial overlaps retained. Affine infeasibility does not prove global geometry infeasibility.',
            'No fresh validation runs unless every development gate passes; no outcome-dependent tuning.',
            'Original85000-85049 validation results are preserved;85048 is now a development regression.'],
        artifactpaths=dict(development='development/summary.json',validation='validation/summary.json' if validation else None,legacy='legacy_parity.json'))
    write(out/'VERDICT.json',verdict)
    text='# Native joint protection simple fix\n\nThe new opt-in layer jointly projects all four agents controls, with all six pair constraints and the unchanged native held-flow horizons. It pools each agents own measured position/velocity and all nominal actions; this is explicitly centralized. The same layer, independent reset assignment and static targets apply to all four nominal controllers. Dynamics, rewards, checkpoint, horizon and success criteria remain unchanged.\n\n'
    text+='Legacy eight-episode parity gate passed. Development includes21starts (82000–82019 and knownfailure85048),84episodes, with every initial overlap/failure retained.\n\n'
    for name,result in [('Development',dev),('Validation',validation)]:
        if result is None:continue
        text+='## '+name+'\n\n| Arm | Full coverage | New contacts | Mean return | Affine-infeasible steps | Numerical failure steps |\n|---|---:|---:|---:|---:|---:|\n'
        for a,s in result['summary'].items():text+='| %s | %d/%d | %d | %.3f | %d | %d |\n'%(a,s['full_coverage'],s['expected'],s['new_contact_onsets'],s['mean_return'],s['affine_infeasible_steps'],s['numerical_failure_steps'])
        text+='\nGate: `'+json.dumps(result['gate'])+'`.\n\n'
        for p in result['paired']:text+='- Online minus %s: %.3f,95%%pairedbootstrap interval%s.\n'%(p['control'],p['difference'],p['interval95'])
        text+='\n'
    text+='Initially unsafe pairs use nondecreasing separation rather than an impossible immediate clearance reset. QP candidates require feasibility and KKT certification. Failed solves or uncertified float32 controls invoke explicitly logged bounded braking; no guarantee is claimed for that fallback. The LP diagnoses only the declared affine constraints, never global geometric impossibility.\n\n'
    if validation is None:text+='Development gate failed; fresh85200–85249 starts were not executed. No parameters were tuned after observing these outcomes. A working or validated safety claim is blocked.\n'
    else:text+='Fresh validation uses85200–85249 only after the development gate, with identical frozen controller/solver parameters. All failures remain retained.\n'
    (out/'REPORT.md').write_text(text);print(text,flush=True)


if __name__=='__main__':main()
