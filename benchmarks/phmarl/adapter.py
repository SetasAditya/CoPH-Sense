"""Observation-only CoPH scalar-potential transfer to native VMAS navigation.

This is a controller/plant-identification adapter, NOT the complete cooperative
stagewise learner. No risk samples, maps, receiver fitting or replay are invented.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'CoPH-Stagewise/coph_stagewise/model.py'
spec = importlib.util.spec_from_file_location('coph_native_model', MODEL)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

SETTINGS = dict(dt=.1,agents=4,communication_radius=.45,agent_radius=.15,
    original_controller_radius=.12,barrier_weight=2.,risk_weight=0.,
    shaping_k=1.,shaping_ell=.5,shaping_eta=.5,policy_upper_bound=10.,
    reference_damping=.6,reference_force_limit=3.,update_every=8,
    transition_cap=120,drag_bounds=[0.,.95],drag_trust=.15,
    validation_relative_slack=.05,validation_absolute_slack=1e-10,
    validation_uid_modulus=5,minimum_training_records=3,
    contact_exclusion_distance=.45)


def native_flow(x, action, drag, dt=.1):
    """Native contact-free semi-implicit Euler, not CoPH's exact flow."""
    velocity=(1-drag)*x[...,2:]+dt*action
    return torch.cat((x[...,:2]+dt*velocity,velocity),-1)


def unpack(obs, agent, goal_index=None):
    # Only this agent's native observation enters its controller.
    a=np.asarray(obs,dtype=np.float64)
    if a.shape!=(18,) or not np.isfinite(a).all():
        raise ValueError('Expected finite native four-agent observation, length 18')
    x=a[:4].copy()
    goal_index=agent if goal_index is None else goal_index
    target=x[:2]+a[4+2*goal_index:6+2*goal_index]
    relative=a[12:].reshape(3,2)
    peers=relative[np.linalg.norm(relative,axis=1)<=SETTINGS['communication_radius']]
    return x,target,x[:2]+peers


class NativeCoPH:
    def __init__(self,mode,goal_indices=None):
        if mode not in ('frozen','online','known_native'):
            raise ValueError(mode)
        self.mode=mode
        self.goal_indices=tuple(range(4)) if goal_indices is None else tuple(goal_indices)
        if sorted(self.goal_indices)!=list(range(4)):raise ValueError('Expected a goal permutation')
        initial=.25 if mode=='known_native' else -math.expm1(-.6*.1)
        self.drag=np.full(4,initial)
        self.versions=np.zeros(4,dtype=int)
        self.records=[[] for _ in range(4)]
        self.updates=[]
        self.last=[]
        self.step=0
        self.work=0
        self.cfg=SimpleNamespace(shaping_mode='progress',shaping_k=1.,shaping_ell=.5,
            shaping_eta=.5,policy_bounds=(0.,10.))

    def action(self,observations):
        actions=[]; self.last=[]
        for i,obs in enumerate(observations):
            x,target,peers=unpack(np.asarray(obs).reshape(-1),i,self.goal_indices[i])
            # The original potential subtracts ego radius .12. Using effective
            # peer radius .18 yields native physical center clearance .15+.15.
            obstacles=tuple((float(q[0]),float(q[1]),.18) for q in peers)
            state=model.tensor(x)
            command,nominal=model.command(state,model.tensor(target),model.tensor([]),
                torch.empty((0,2),dtype=torch.float64),.8,model.tensor([0.,2.]),
                obstacles,3.,self.cfg)
            desired=model.flow(state,command,.6,.1)[2:]
            raw=(desired-(1-self.drag[i])*state[2:])/.1
            applied=raw.clamp(-1.,1.).detach().to(torch.float32)
            actions.append(applied.unsqueeze(0))
            self.last.append(dict(drag=float(self.drag[i]),version=int(self.versions[i]),
                nominal=nominal.detach().tolist(),reference_command=command.detach().tolist(),
                native_unclipped=raw.detach().tolist(),applied=applied.tolist(),
                clipped=bool((raw.abs()>1).any()),peer_count=len(peers),target=target.tolist(),
                shaping_energy=float(model.potential(state[:2],model.tensor(target),model.tensor([]),
                    torch.empty((0,2),dtype=torch.float64),.8,model.tensor([0.,2.]),obstacles,self.cfg))))
        return actions

    def observe(self,before,actions,after):
        # Inputs are measured observations and actually issued native controls.
        for i in range(4):
            x,_,peers=unpack(np.asarray(before[i]).reshape(-1),i)
            y,_,next_peers=unpack(np.asarray(after[i]).reshape(-1),i)
            eligible=len(peers)==0 and len(next_peers)==0
            rec=dict(uid=self.step,x=tuple(x),u=tuple(np.asarray(actions[i]).reshape(2)),
                after=tuple(y),eligible=eligible,version=int(self.versions[i]))
            self.records[i].append(rec)
            self.records[i]=self.records[i][-120:]
        self.step+=1
        if self.mode=='online' and self.step%8==0:
            for i in range(4): self.update(i)

    def loss(self,records,drag):
        return float(np.mean([np.mean(((1-drag)*np.asarray(r['x'][2:])+.1*np.asarray(r['u'])-
            np.asarray(r['after'][2:]))**2) for r in records]))

    def update(self,i):
        train=[r for r in self.records[i] if r['eligible'] and r['uid']%5!=0]
        valid=[r for r in self.records[i] if r['eligible'] and r['uid']%5==0]
        old=float(self.drag[i])
        row=dict(agent=i,step=self.step,old=old,version_before=int(self.versions[i]),
            train=len(train),validation=len(valid),excluded=sum(not r['eligible'] for r in self.records[i]),
            accepted=False,reason='insufficient_measured_data')
        if len(train)>=3 and valid:
            v=np.array([r['x'][2:] for r in train]); u=np.array([r['u'] for r in train])
            y=np.array([r['after'][2:] for r in train]); denominator=float((v*v).sum())
            self.work+=len(train)+2*len(valid)
            if denominator>1e-12:
                estimate=float((v*(v+.1*u-y)).sum()/denominator)
                proposal=float(np.clip(estimate,max(0.,old-.15),min(.95,old+.15)))
                pre=self.loss(valid,old); post=self.loss(valid,proposal)
                accepted=bool(np.isfinite(proposal) and post<=pre+.05*max(pre,1e-12)+1e-10)
                if accepted:
                    self.drag[i]=proposal;self.versions[i]+=1
                row.update(proposal=proposal,unbounded_estimate=estimate,validation_before=pre,
                    validation_after=post,accepted=accepted,reason='accepted' if accepted else 'validation_rejected')
            else: row['reason']='unexcited_velocity'
        row.update(committed=float(self.drag[i]),version_after=int(self.versions[i]))
        self.updates.append(row)
        return row
