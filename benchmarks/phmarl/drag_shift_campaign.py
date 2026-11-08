#!/usr/bin/env python3
"""Two-phase native task: the drag changes and the landmarks move at step 200.

Phase 1 (steps 0-199): native drag 0.25. At step 200 the world drag switches to D2 in {0.10, 0.45}
(a lighter/slippery and a heavier/viscous payload), and the four landmarks are resampled with the same
uniform [-1,1]^2 rule as the native reset, so the team must reallocate and travel again.
Allocation and static targets are recomputed at the switch with exactly the reset procedure of
goal_geometry_campaign, from each agent's own native observation. Landmark draws failing that
geometry gate are redrawn (logged; outcome-blind). Protection, controller and checkpoint are unchanged.

Arms: CoPH frozen (prior 0.058), known (fixed 0.25, i.e. correct only before the switch), online;
phMARL with the same allocation, targets and protection. Fresh seeds 87000-87029.
Pre-declared gate: in phase 2, online beats known in mean return with a paired 95% interval excluding
zero for both D2 values, with >=95% full coverage at the end of phase 2 and no more new contacts than known.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
import adapter,protection
import safety_suite as suite
import reproduce as base
import goal_geometry_campaign as gg

SEEDS=list(range(87000,87030));SWITCH=200;D2=(.10,.45)
ARMS=['coph_frozen_static_protected','coph_known_static_protected','coph_online_static_protected','phmarl_assigned_static_protected']


def run(task):
    arm,seed,d2,out=task;out=Path(out);torch.set_num_threads(1);context={};live={};log={}
    def setup(spec):
        env,obs=gg.ORIGINAL_SETUP(spec);context.update(gg.reset_context(obs));original=env.step;count=[0]
        rng=np.random.RandomState(seed*10+int(round(d2*100)))
        def step(actions):
            if count[0]==SWITCH:
                env.world._drag=d2;draws=0
                while True:
                    draws+=1;new=rng.uniform(-1,1,(4,2))
                    for lm,p in zip(env.world.landmarks,new):lm.set_pos(torch.tensor(p,dtype=torch.float32).reshape(1,2),batch_index=0)
                    obs=[env.scenario.observation(a)[0].detach().numpy().copy() for a in env.world.agents]
                    try:new_context=gg.reset_context(obs);break
                    except ValueError:continue
                context.update(new_context);log.update(landmark_draws=draws,phase2_landmarks=new.tolist(),phase2_context=new_context)
                live['protection'].goal_indices=tuple(new_context['goal_indices']);live['protection'].fixed_targets=np.asarray(new_context['fixed_targets'],float)
                if 'coph' in live:live['coph'].goal_indices=tuple(new_context['goal_indices'])
            count[0]+=1;return original(actions)
        env.step=step;return env,obs
    class P(protection.Protection):
        def __init__(self):super().__init__(context['goal_indices'],context['fixed_targets']);live['protection']=self
    class C(adapter.NativeCoPH):
        def __init__(self,unused):
            super().__init__('known_native' if '_known_' in arm else ('online' if '_online_' in arm else 'frozen'),context['goal_indices']);live['coph']=self
    class A:
        def __init__(self,actor):self.actor=actor
        def __call__(self,x):
            y=x.clone();g=x[:,:,4:12].reshape(x.shape[0],4,4,2);y[:,:,4:12]=g[:,:,context['goal_indices'],:].reshape(x.shape[0],4,8);return self.actor(y)
    suite.setup=setup;suite.Protection=P;suite.NativeCoPH=C;base.cpu=lambda actor:A(gg.ORIGINAL_CPU(actor))
    tag='%s_seed%d_d%03d'%(arm,seed,round(d2*100))
    row=suite.worker((arm,[dict(id=tag,profile='fresh',seed=seed)],str(out)))[0]
    with np.load(out/('%s_%s.npz'%(arm,tag))) as d:q=d['positions'];r=d['rewards'].reshape(-1,4)
    g2=np.asarray(log['phase2_landmarks']);cov=lambda t,g:float((np.linalg.norm(q[t][:,None]-g[None],axis=-1).min(0)<=.15).mean())
    g1=np.load(out/('%s_%s.npz'%(arm,tag)))['landmarks']
    drag=[float(np.mean(u['committed'])) for u in []]
    row.update(seed=seed,d2=d2,phase1_return=float(r[:SWITCH].mean(1).sum()),phase2_return=float(r[SWITCH:].mean(1).sum()),
        phase1_coverage=cov(SWITCH,g1),phase2_coverage=cov(-1,g2),landmark_draws=log['landmark_draws'],
        drag_trace=[[float(x) for x in live['coph'].drag]] if 'coph' in live else None,
        drag_updates=[dict(step=u['step'],agent=u['agent'],committed=u['committed']) for u in live['coph'].updates] if 'coph' in live else None)
    (out/(tag+'_shift.json')).write_text(json.dumps(row,indent=1));return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(dict(seeds=SEEDS,switch=SWITCH,d2=D2,arms=ARMS,gate=__doc__),indent=1))
    rows=[];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,(a,s,d,str(out))) for d in D2 for s in SEEDS for a in ARMS]):
            rows.append(f.result());print(len(rows),rows[-1]['protocol'],rows[-1]['seed'],rows[-1]['d2'],flush=True)
    (out/'rows.json').write_text(json.dumps(rows));rng=np.random.RandomState(187100);res=dict(conditions={})
    for d2 in D2:
        by={a:{r['seed']:r for r in rows if r['protocol']==a and r['d2']==d2} for a in ARMS};c={}
        for a in ARMS:c[a]=dict(**{k:float(np.mean([by[a][s][k] for s in SEEDS])) for k in ('phase1_return','phase2_return','phase1_coverage','phase2_coverage')},
            full_phase2=sum(by[a][s]['phase2_coverage']==1 for s in SEEDS),new_contacts=sum(by[a][s]['new_contact_onsets'] for s in SEEDS))
        pairs={}
        for b in ARMS[:2]+ARMS[3:]:
            for ph in ('phase1_return','phase2_return'):
                d=np.array([by['coph_online_static_protected'][s][ph]-by[b][s][ph] for s in SEEDS]);m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1)
                pairs[f'online_minus_{b}_{ph}']=dict(mean=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],wins=int((d>0).sum()))
        o=c['coph_online_static_protected']
        gate=pairs['online_minus_coph_known_static_protected_phase2_return']['interval95'][0]>0 and o['full_phase2']>=.95*len(SEEDS) and o['new_contacts']<=c['coph_known_static_protected']['new_contacts']
        res['conditions'][str(d2)]=dict(arms=c,paired=pairs,gate=bool(gate))
    res['gate']=all(v['gate'] for v in res['conditions'].values());res['seconds']=time.perf_counter()-t
    (out/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))


if __name__=='__main__':main()
