#!/usr/bin/env python3
"""Liveness under symmetric encounters: shared protection only versus protection plus deterministic priority.

Priority rule (opt-in, shared layer, identical for every nominal controller): an agent more than .35 from
its goal whose path is blocked by a visible lower-index peer (peer within 60 deg of the goal direction)
replaces its target that step by q + .5*right - .2*forward (sidestep right and back). Agent 0 never yields.
This is a target change, recorded like every other target change; projection and actual-motion
constraints are unchanged.
Development history (seeds 0-9, see LIVENESS.md): a velocity-only gyroscopic term u += 3 S v (no work)
and a .3 keep-right waypoint shift both left 10/10 circle swaps gridlocked as a 4-agent square.
Settings were then frozen; evaluation uses seeds 100-129 only.

Families (30 evaluation seeds each, 400 steps; each agent assigned its designed goal):
  pair_exact  two agents swap head-on along a random direction, zero lateral offset; two parked far away
  pair_swap   as above with lateral offset U(0,.03)
  circle_swap four agents on a circle of radius U(.6,.9) move to the antipodal point (angle jitter .05 rad)
  crossing    two agents on perpendicular paths timed to meet; two parked far away
Arms: attraction or CoPH-online nominal, each with protection only or protection + priority.
Pre-declared gate: each priority arm reaches all goals (<= .15) in >= 90% of episodes of every family,
with zero new contacts; it has more successes than its protection-only counterpart in every family where
that counterpart is below 90%, and no fewer successes in any family.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from vmas import make_env
from safety_suite import attraction,contacts
from protection import Protection,braking_radius
import adapter

FAMILIES=('pair_exact','pair_swap','circle_swap','crossing');SEEDS=range(100,130);DEV=range(10);W_SHIFT=.3;W_CONE=np.deg2rad(60);MODE='priority';YIELD=.5;BACK=.2
ARMS=('attraction','attraction_priority','coph_online','coph_online_priority');PARK=[[2.5,2.5],[-2.5,2.5]]


class Priority(Protection):
    def prepare(self,observations):
        out=super().prepare(observations);self.shifts=0
        for i,(o,c) in enumerate(zip(out,self.context)):
            to_goal=c['goal']-c['q'];dist=np.linalg.norm(to_goal)
            if dist<=.35:continue
            d=to_goal/dist
            ahead=[p for p in c['peers'] if np.dot(np.asarray(p['position'])-c['q'],d)>np.cos(W_CONE)*p['distance']]
            if not ahead:continue
            right=np.array([d[1],-d[0]])
            if MODE=='priority' and any(p['id']<i for p in ahead):target=c['q']+YIELD*right-BACK*d
            elif MODE=='priority':continue
            else:target=c['target']+W_SHIFT*right
            gi=self.goal_indices[i]
            o[0,4+2*gi:6+2*gi]=(target-c['q']).astype(np.float32);c['target']=target;self.shifts+=1
        return out


def scenario(family,seed):
    rng=np.random.RandomState(9000+100000*FAMILIES.index(family)+seed)
    if family in ('pair_swap','pair_exact'):
        th=rng.uniform(0,2*np.pi);d=np.array([np.cos(th),np.sin(th)]);n=np.array([-d[1],d[0]]);r=rng.uniform(.6,.9);off=0. if family=='pair_exact' else rng.uniform(0,.03)
        a,b=-r*d+off/2*n,r*d-off/2*n;return [a,b,*PARK],[b,a,*PARK]
    if family=='circle_swap':
        r=rng.uniform(.6,.9);th=rng.uniform(0,2*np.pi)+np.arange(4)*np.pi/2+rng.uniform(-.05,.05,4)
        p=np.stack([r*np.cos(th),r*np.sin(th)],1);return list(p),list(-p)
    th=rng.uniform(0,2*np.pi);d=np.array([np.cos(th),np.sin(th)]);n=np.array([-d[1],d[0]]);r=rng.uniform(.6,.9)
    return [-r*d,-r*n,*PARK],[r*d,r*n,*PARK]


def run(task):
    family,seed,arm=task;torch.set_num_threads(1);starts,goals=scenario(family,seed)
    env=make_env(scenario_name='simple_spread',num_envs=1,device='cpu',continuous_actions=True,n_agents=4,max_steps=400,seed=82);env.reset(seed=82)
    for a,p in zip(env.world.agents,starts):a.set_pos(torch.tensor(p,dtype=torch.float32),batch_index=None);a.set_vel(torch.zeros(2),batch_index=None)
    for l,p in zip(env.world.landmarks,goals):l.set_pos(torch.tensor(p,dtype=torch.float32),batch_index=None)
    obs=[env.scenario.observation(a)[0].numpy().copy() for a in env.agents]
    shield=Priority() if '_priority' in arm else Protection()
    if arm.endswith('_range'):shield.peer_radius=braking_radius()
    coph=adapter.NativeCoPH('online') if arm.startswith('coph') else None
    initial=contacts(env);previous=initial;onsets=0;work=[];q=[];g=np.asarray(goals);done_at=None;minsep=np.inf
    for k in range(400):
        processed=shield.prepare(obs)
        nominal=coph.action(processed) if coph else attraction(processed)
        u,logs=shield.apply(nominal);new,_,_,_=env.step(u);new=[o[0].numpy().copy() for o in new]
        if coph:coph.observe(obs,u,new)
        obs=new;cp=contacts(env);onsets+=int((cp&~previous).sum());previous=cp;work.append(getattr(shield,'shifts',0))
        pos=np.stack([a.state.pos[0].numpy() for a in env.world.agents]);q.append(pos)
        minsep=min(minsep,min(np.linalg.norm(pos[i]-pos[j]) for i in range(4) for j in range(i+1,4)))
        if done_at is None and (np.linalg.norm(pos-g,axis=1)<=.15).all():done_at=k+1
    q=np.asarray(q);final=bool((np.linalg.norm(q[-1]-g,axis=1)<=.15).all())
    return dict(family=family,seed=seed,arm=arm,success=final,time_to_goals=done_at,new_contacts=onsets,initial_contacts=int(initial.sum()),
        min_separation=float(minsep),shift_agent_steps=int(sum(work)),path=q[::4].tolist(),starts=np.asarray(starts).tolist(),goals=g.tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40);args=ap.parse_args()
    if args.out.exists():raise SystemExit('Refuse existing output')
    args.out.mkdir(parents=True);(args.out/'manifest.json').write_text(json.dumps(dict(families=FAMILIES,seeds=list(SEEDS),arms=ARMS,mode=MODE,shift=W_SHIFT,cone_deg=60,yield_=YIELD,back=BACK,gate=__doc__),indent=1))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,(fa,s,a)) for fa in FAMILIES for s in SEEDS for a in ARMS]):rows.append(f.result())
    (args.out/'rows.json').write_text(json.dumps(rows));summary={}
    for fa in FAMILIES:
        for a in ARMS:
            rr=[r for r in rows if r['family']==fa and r['arm']==a];t=[r['time_to_goals'] for r in rr if r['time_to_goals']]
            summary[f'{fa}/{a}']=dict(success=sum(r['success'] for r in rr),new_contacts=sum(r['new_contacts'] for r in rr),
                median_time=float(np.median(t)) if t else None,min_separation=float(min(r['min_separation'] for r in rr)),
                shift_agent_steps=int(np.mean([r['shift_agent_steps'] for r in rr])))
    gate=all(summary[f'{fa}/{a}']['success']>=27 and summary[f'{fa}/{a}']['new_contacts']==0 for fa in FAMILIES for a in ARMS if a.endswith('_priority'))
    for fa in FAMILIES:
        for a in ('attraction','coph_online'):
            t,b=summary[f'{fa}/{a}_priority']['success'],summary[f'{fa}/{a}']['success']
            gate&=t>=b and (b>=27 or t>b)
    res=dict(summary=summary,gate=bool(gate));(args.out/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))


if __name__=='__main__':main()
