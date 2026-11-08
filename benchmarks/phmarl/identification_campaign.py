#!/usr/bin/env python3
"""Fresh native same-protection wrong/known/identified-drag comparison."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
import safety_suite as suite
import adapter
import reproduce as base

ARMS=['coph_frozen_protected','coph_known_protected','coph_online_protected','phmarl_protected','phmarl']
SEEDS=list(range(82000,82020))


def run(task):
    arm,spec,out=task
    # The production worker chooses frozen for non-online CoPH arms. Supply its
    # already-existing known-native constructor only in this isolated worker.
    suite.NativeCoPH=lambda mode:adapter.NativeCoPH('known_native' if arm=='coph_known_protected' else mode)
    return suite.worker((arm,[spec],out))[0]


def report(out,manifest,rows):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    summary={a:{k:float(np.mean([r[k] for r in rows if r['protocol']==a])) for k in
        ['mean_agent_return','final_landmark_coverage','collision_pair_steps','new_contact_onsets','interventions','identification_work','total_seconds']} for a in ARMS}
    comparisons=[];rng=np.random.RandomState(182000)
    for control in ['coph_frozen_protected','coph_known_protected','phmarl_protected','phmarl']:
        t={r['case']:r for r in rows if r['protocol']=='coph_online_protected'};c={r['case']:r for r in rows if r['protocol']==control}
        ids=sorted(set(t)&set(c));d=np.array([t[k]['mean_agent_return']-c[k]['mean_agent_return'] for k in ids]);idx=rng.randint(0,len(d),size=(10000,len(d)))
        comparisons.append(dict(treatment='coph_online_protected',control=control,n=len(ids),return_difference=float(d.mean()),
            interval95=np.percentile(d[idx].mean(axis=1),[2.5,97.5]).tolist(),differences=d.tolist()))
    # Pairing checked on actual logged initial states and landmarks, not RNG alone.
    for seed in manifest['seeds']:
        files=[np.load(out/('%s_seed%d.npz'%(a,seed))) for a in ARMS]
        for r in files[1:]:
            assert np.array_equal(files[0]['positions'][0],r['positions'][0])
            assert np.array_equal(files[0]['landmarks'],r['landmarks'])
    failure=[];gallery=out/'failure_gallery';gallery.mkdir(exist_ok=True)
    for r in rows:
        if r['collision_free'] and r['full_coverage']:continue
        name=r['protocol']+'_'+r['case'];d=np.load(out/(name+'.npz'));fig,ax=plt.subplots(figsize=(4,4));ax.scatter(*d['landmarks'].T,marker='x',c='black')
        for i in range(4):ax.plot(d['positions'][:,i,0],d['positions'][:,i,1])
        ax.set_aspect('equal');ax.set_title(name+'\noverlap=%d coverage=%.2f'%(r['collision_pair_steps'],r['final_landmark_coverage']),fontsize=8)
        fig.tight_layout();fig.savefig(gallery/(name+'.png'),dpi=100);plt.close(fig)
        failure.append(dict(arm=r['protocol'],case=r['case'],overlap=not r['collision_free'],incomplete_coverage=not r['full_coverage']))
    (out/'failure_inventory.json').write_text(json.dumps(failure,indent=2))
    (out/'results.json').write_text(json.dumps(dict(rows=rows,summary=summary,paired=comparisons,pairing_verified=True),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for j,a in enumerate(ARMS):
        rr=[r for r in rows if r['protocol']==a]
        for ax,k in zip(axes,['mean_agent_return','collision_pair_steps','final_landmark_coverage']):ax.scatter([j]*len(rr),[r[k] for r in rr],s=14)
    for ax,label in zip(axes,['Native mean-agent return','Overlap pair-steps','Final landmark coverage']):
        ax.set_xticks(range(len(ARMS)));ax.set_xticklabels(ARMS,rotation=35,ha='right',fontsize=7);ax.set_ylabel(label)
    fig.tight_layout();fig.savefig(out/'native_outcomes.png',dpi=160);fig.savefig(out/'native_outcomes.pdf');plt.close(fig)
    # Predeclared seed; same timestamps in every panel.
    records=[np.load(out/('%s_seed%d.npz'%(a,manifest['animation_seed']))) for a in ARMS]
    fig,axes=plt.subplots(1,5,figsize=(18,4));panels=[];lim=max(1.3,max(abs(r['positions']).max() for r in records)+.2)
    for ax,a,r in zip(axes,ARMS,records):
        ax.scatter(*r['landmarks'].T,c='black',marker='x');lines=[ax.plot([],[])[0] for _ in range(4)]
        dots=[ax.plot([],[],'o')[0] for _ in range(4)];ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal');panels.append((ax,a,r,lines,dots))
    def update(t):
        for ax,a,r,lines,dots in panels:
            for i,(line,dot) in enumerate(zip(lines,dots)):
                line.set_data(r['positions'][:t+1,i,0],r['positions'][:t+1,i,1]);dot.set_data([r['positions'][t,i,0]],[r['positions'][t,i,1]])
            ax.set_title(a+'\nstep %d'%t,fontsize=8)
    FuncAnimation(fig,update,frames=range(0,401,8),interval=100).save(out/'comparison_seed82000.gif',writer=PillowWriter(fps=10));plt.close(fig)
    table='| Arm | Return | Coverage | Overlap pair-steps | New contact onsets |\n|---|---:|---:|---:|---:|\n'
    for a in ARMS:
        s=summary[a];table+='| %s | %.3f | %.3f | %.2f | %.2f |\n'%(a,s['mean_agent_return'],s['final_landmark_coverage'],s['collision_pair_steps'],s['new_contact_onsets'])
    text='''# Native identification: correct-model control

Fresh development seeds 82000–82019, five arms, 100 episodes, four agents, 400 native steps. Frozen manifests and source copies precede execution. Initial positions and landmarks match across all arms exactly. This is the existing checkpoint/task, not full phMARL retraining.

The wrong/known/online nominal controllers use the same protection, observations, scalar potential and controller parameters. Only nominal drag initialization/identification differs. Correct drag is the native discrete coefficient 0.25, not continuous damping. The common protection already uses known native physics in all protected arms. Known drag uses the pre-existing known_native adapter mode; online learns from actual executed controls. Environmental learning, communication and historical replay are inactive. Unmodified phMARL is an external reference; protected phMARL is a modified policy.

%s
Paired return intervals are episode-start bootstrap intervals with 10,000 resamples. Initial overlaps, new contacts and incomplete supplementary coverage remain in the inventory and plots. Do not equate overlap pair-steps with distinct collision events. Computation is actual measured CPU work with worker contention, not deployment latency.

This tests recovery from one static nominal-model mismatch and comparison with a correct fixed model. It does not test a damping-shift mechanism, changing hazards, or full cooperative learning. All incomplete/contact episodes remain available. No outcome was used to tune controller or safety settings.
'''%table
    (out/'REPORT.md').write_text(text);print(text,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS);args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    sources=[p.resolve() for p in [Path(__file__),Path(suite.__file__),Path(adapter.__file__),Path(base.__file__),Path(suite.__file__).with_name('protection.py'),adapter.MODEL]]
    manifest=dict(arms=ARMS,seeds=args.seeds,horizon=400,animation_seed=args.seeds[0],bootstrap_seed=182000,
        protocol='native static mismatch: known-model control v1',device='cpu',threads=1,workers=args.workers,
        drag_true=.25,drag_wrong=float(-np.expm1(-.6*.1)),protection=suite.SETTINGS,adapter=adapter.SETTINGS,
        checkpoint_sha256=base.digest(base.CHECKPOINT),sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in sources},
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,evaluation_role='fresh development; final tests untouched')
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));archive=out/'sources';archive.mkdir()
    for p in sources:(archive/p.name).write_bytes(p.read_bytes())
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run,(a,dict(id='seed%d'%s,profile='fresh',seed=s),str(out))) for s in args.seeds for a in ARMS]
        for job in as_completed(jobs):
            rows.append(job.result());(out/'completion.json').write_text(json.dumps(rows,indent=2));print(len(rows),'/',len(jobs),rows[-1]['protocol'],rows[-1]['case'],flush=True)
    report(out,manifest,rows);(out/'execution.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,complete=len(rows),expected=len(jobs)),indent=2))


if __name__=='__main__':main()
