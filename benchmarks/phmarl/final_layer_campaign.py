#!/usr/bin/env python3
"""Final shared protection layer on fresh seeds: native benchmark and liveness families.

Layer 'final' = braking-derived peer radius (0.641 m) + contact-impulse prediction from public VMAS physics
+ least-violation fallback when constraints are infeasible, + deterministic priority (liveness rule).
Layer 'base' = the original 0.45 m layer without priority, for comparison. Development used seeds 85000-85049
and 88000-88049 (native) and 0-9 (liveness); evaluation seeds: native 90000-90099, liveness 300-329.

Pre-declared gate (all must hold):
 native: online CoPH under 'final' full coverage >= 95/100; zero avoidable new contacts for online CoPH and
   matched phMARL under 'final' (contact_certificate: a contact is unavoidable only if a best-escape replay
   with native physics still touches); online CoPH beats matched phMARL under 'final' (paired 95% CI > 0).
 liveness: under 'final', attraction and CoPH-online reach all goals in >= 27/30 of every family with zero
   avoidable new contacts, and more successes than 'base' where 'base' < 27, no fewer elsewhere.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import protection,adapter
import goal_geometry_campaign as gg
import liveness_campaign as lv
from contact_certificate import certify

NATIVE=list(range(90000,90100));LIVE=range(300,330);ARMS=('coph_online_static_protected','phmarl_assigned_static_protected')


def configure(layer):
    P=protection.Protection;final=layer=='final'
    P.peer_radius=protection.braking_radius() if final else .45;P.contact_prediction=final;P.least_violation_fallback=final


def native(task):
    layer,arm,seed,out=task;configure(layer);gg.BASELINES=[];d=Path(out)/'native'/layer;d.mkdir(parents=True,exist_ok=True)
    r=gg.run((arm,seed,str(d)))
    with np.load(d/f'{arm}_seed{seed}.npz') as z:c=certify(z)
    return dict(layer=layer,arm=arm,seed=seed,coverage=r['full_coverage'],ret=r['mean_agent_return'],new=r['new_contact_onsets'],
                initial=r['initial_overlap_pairs'],avoidable=sum(not x['unavoidable'] for x in c),certificates=c)


def live(task):
    layer,family,seed,arm=task;configure(layer)
    r=lv.run((family,seed,arm+('_priority' if layer=='final' else '')))
    return dict(layer=layer,family=family,seed=seed,arm=arm,success=r['success'],new=r['new_contacts'],time=r['time_to_goals'],path=r['path'],
                starts=r['starts'],goals=r['goals'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(dict(native=NATIVE,liveness=list(LIVE),radius=protection.braking_radius(),gate=__doc__),indent=1))
    with ProcessPoolExecutor(44) as pool:
        nrows=list(pool.map(native,[(L,a,s,str(out)) for L in ('base','final') for a in ARMS for s in NATIVE]))
        lrows=list(pool.map(live,[(L,f,s,a) for L in ('base','final') for f in lv.FAMILIES for s in LIVE for a in ('attraction','coph_online')]))
    (out/'native_rows.json').write_text(json.dumps(nrows));(out/'liveness_rows.json').write_text(json.dumps(lrows))
    # Liveness contacts are certified from the recorded paths only for counts; the scenarios start separated.
    res=dict(native={},liveness={});rng=np.random.RandomState(190000)
    for L in ('base','final'):
        for a in ARMS:
            rr=[r for r in nrows if r['layer']==L and r['arm']==a]
            res['native'][f'{L}/{a}']=dict(coverage=sum(r['coverage'] for r in rr),mean_return=float(np.mean([r['ret'] for r in rr])),
                new_contacts=sum(r['new'] for r in rr),avoidable=sum(r['avoidable'] for r in rr),overlap_starts=sum(r['initial']>0 for r in rr))
    by={(r['layer'],r['arm'],r['seed']):r['ret'] for r in nrows}
    def ci(d):m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1);return dict(mean=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],wins=int((d>0).sum()))
    res['native']['final_online_vs_phmarl']=ci(np.array([by['final',ARMS[0],s]-by['final',ARMS[1],s] for s in NATIVE]))
    res['native']['final_minus_base_online']=ci(np.array([by['final',ARMS[0],s]-by['base',ARMS[0],s] for s in NATIVE]))
    for L in ('base','final'):
        for f in lv.FAMILIES:
            for a in ('attraction','coph_online'):
                rr=[r for r in lrows if r['layer']==L and r['family']==f and r['arm']==a];t=sorted(r['time'] for r in rr if r['time'])
                res['liveness'][f'{L}/{f}/{a}']=dict(success=sum(r['success'] for r in rr),new_contacts=sum(r['new'] for r in rr),median_time=t[len(t)//2] if t else None)
    n=res['native'];g=dict(native_coverage=n[f'final/{ARMS[0]}']['coverage']>=95,
        native_avoidable=n[f'final/{ARMS[0]}']['avoidable']==0 and n[f'final/{ARMS[1]}']['avoidable']==0,
        native_superiority=n['final_online_vs_phmarl']['interval95'][0]>0)
    ok=True
    for f in lv.FAMILIES:
        for a in ('attraction','coph_online'):
            t=res['liveness'][f'final/{f}/{a}'];b=res['liveness'][f'base/{f}/{a}']
            ok&=t['success']>=27 and t['new_contacts']==0 and t['success']>=b['success'] and (b['success']>=27 or t['success']>b['success'])
    g['liveness']=bool(ok);res['gate']=g;res['passed']=all(g.values())
    (out/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))


if __name__=='__main__':main()
