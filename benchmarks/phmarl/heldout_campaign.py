#!/usr/bin/env python3
"""Held-out validation of the frozen static-target native controller on unused starts.

The controller, assignment, targets and protection are exactly those of goal_geometry_campaign (imported, not copied).
Seeds 85000-85049 have not been used by any earlier experiment. Gate (fixed before running):
every CoPH static arm reaches full coverage on >=95% of starts with zero new contact onsets, and online beats
phmarl_assigned_static_protected in mean return with a paired 95% interval excluding zero.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
import goal_geometry_campaign as gg

SEEDS=list(range(85000,85050))
ARMS=['phmarl','phmarl_assigned','phmarl_assigned_static_protected',
      'coph_frozen_static_protected','coph_known_static_protected','coph_online_static_protected']
gg.BASELINES=[]  # no archived references exist for unused seeds, so parity is not checked


def job(task):
    gg.BASELINES=[];return gg.run(task)


def paired(by,a,b,rng):
    d=np.array([by[a][s]['mean_agent_return']-by[b][s]['mean_agent_return'] for s in SEEDS])
    m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1)
    return dict(treatment=a,control=b,difference=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],
                wins=int((d>0).sum()),median=float(np.median(d)))


def main():
    global SEEDS
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output directory')
    out.mkdir(parents=True);torch.set_num_threads(1)
    gates=[]
    for seed in SEEDS:
        env,obs=gg.ORIGINAL_SETUP(dict(id='seed%d'%seed,profile='fresh',seed=seed))
        try:gg.reset_context([o.detach().numpy().copy() for o in obs]);gates.append(dict(seed=seed,passed=True))
        except ValueError as e:gates.append(dict(seed=seed,passed=False,reason=str(e)))
    (out/'geometry_gate.json').write_text(json.dumps(gates,indent=2))
    valid=[g['seed'] for g in gates if g['passed']]
    (out/'manifest.json').write_text(json.dumps(dict(seeds=SEEDS,valid=valid,arms=ARMS,role='held-out validation; seeds unused before',
        gate='each CoPH static arm full coverage >=95% of starts and zero new contacts; online > phmarl_assigned_static_protected, CI excludes 0'),indent=2))
    rows=[];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job,(a,s,str(out))) for s in valid for a in ARMS]):
            rows.append(f.result());print(len(rows),rows[-1]['protocol'],rows[-1]['case'],flush=True)
    by={a:{int(r['case'][4:]):r for r in rows if r['protocol']==a} for a in ARMS}
    SEEDS=valid
    summary={a:dict(full_coverage=sum(by[a][s]['full_coverage'] for s in SEEDS),contact_free=sum(by[a][s]['collision_free'] for s in SEEDS),
             new_contacts=float(np.sum([by[a][s]['new_contact_onsets'] for s in SEEDS])),mean_return=float(np.mean([by[a][s]['mean_agent_return'] for s in SEEDS])),
             median_return=float(np.median([by[a][s]['mean_agent_return'] for s in SEEDS])),
             incomplete=[s for s in SEEDS if not by[a][s]['full_coverage']]) for a in ARMS}
    rng=np.random.RandomState(185000)
    pairs=[paired(by,'coph_online_static_protected',b,rng) for b in ['phmarl','phmarl_assigned','phmarl_assigned_static_protected','coph_frozen_static_protected','coph_known_static_protected']]
    n=len(SEEDS);cop=[a for a in ARMS if a.startswith('coph')]
    gate=dict(coverage=all(summary[a]['full_coverage']>=.95*n for a in cop),contacts=all(summary[a]['new_contacts']==0 for a in cop),
              superiority=pairs[2]['interval95'][0]>0)
    gate['passed']=all(gate.values())
    res=dict(starts=n,excluded=[g for g in gates if not g['passed']],summary=summary,paired=pairs,gate=gate,seconds=time.perf_counter()-t)
    (out/'summary.json').write_text(json.dumps(res,indent=2));print(json.dumps(dict(summary=summary,paired=pairs,gate=gate),indent=1))


if __name__=='__main__':main()
