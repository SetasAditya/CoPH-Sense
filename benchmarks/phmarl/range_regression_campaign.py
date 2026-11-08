#!/usr/bin/env python3
"""Regression check of the braking-derived interaction radius on the main native benchmark.

Same controller, allocation, static targets and matched phMARL as heldout_campaign; only the shared
protection radius differs (0.45 m versus protection.braking_radius() = 0.641 m). Fresh seeds 88000-88049.
Pre-declared criteria for the 0.641 m layer: online CoPH full coverage >= 48/50; zero new contacts in starts
without an initial overlap (starts that begin in contact are reported separately); online CoPH beats matched
phMARL under the same layer with a paired 95% interval excluding zero. The return change versus 0.45 m is reported.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import protection
import goal_geometry_campaign as gg

SEEDS=list(range(88000,88050));ARMS=('coph_online_static_protected','phmarl_assigned_static_protected')


def job(task):
    arm,seed,radius,out=task;gg.BASELINES=[];protection.Protection.peer_radius=radius
    d=Path(out)/('r%03d'%round(radius*1000));d.mkdir(exist_ok=True);return dict(gg.run((arm,seed,str(d))),radius=radius)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);R=(.45,protection.braking_radius())
    (out/'manifest.json').write_text(json.dumps(dict(seeds=SEEDS,arms=ARMS,radii=R,criteria=__doc__),indent=1))
    with ProcessPoolExecutor(44) as pool:rows=list(pool.map(job,[(a,s,r,str(out)) for r in R for s in SEEDS for a in ARMS]))
    (out/'rows.json').write_text(json.dumps(rows,default=str))
    by={(r['protocol'],r['radius'],int(r['case'][4:])):r for r in rows};rng=np.random.RandomState(188000);res={}
    def ci(d):m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1);return dict(mean=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],wins=int((d>0).sum()))
    for r in R:
        for a in ARMS:
            rr=[by[a,r,s] for s in SEEDS]
            res[f'{a}@{r:.3f}']=dict(full_coverage=sum(x['full_coverage'] for x in rr),mean_return=float(np.mean([x['mean_agent_return'] for x in rr])),
                new_contacts_clean_starts=sum(x['new_contact_onsets'] for x in rr if x['initial_overlap_pairs']==0),
                new_contacts_overlap_starts=sum(x['new_contact_onsets'] for x in rr if x['initial_overlap_pairs']>0),overlap_starts=sum(x['initial_overlap_pairs']>0 for x in rr))
    big=R[1];on,ph=ARMS
    vs_ph=ci(np.array([by[on,big,s]['mean_agent_return']-by[ph,big,s]['mean_agent_return'] for s in SEEDS]))
    vs_small=ci(np.array([by[on,big,s]['mean_agent_return']-by[on,.45,s]['mean_agent_return'] for s in SEEDS]))
    o=res[f'{on}@{big:.3f}']
    gate=dict(coverage=o['full_coverage']>=48,contacts=o['new_contacts_clean_starts']==0,superiority=vs_ph['interval95'][0]>0)
    out_=dict(arms=res,online_vs_phmarl_same_layer=vs_ph,online_range_minus_online_045=vs_small,gate=gate,passed=all(gate.values()))
    (out/'summary.json').write_text(json.dumps(out_,indent=1));print(json.dumps(out_,indent=1))


if __name__=='__main__':main()
