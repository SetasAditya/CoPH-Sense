"""Isolated paired continuations over a supplied environment interface."""
import copy
import time
import numpy as np
from .records import Transition
from .stage_prediction import DomainExit

def continuation(agent,state,world,packet,tick,horizons):
    """Same update, costs, termination, and randomness for every factorial cell."""
    results={};started=time.perf_counter()
    for deliver in (False,True):
        a=copy.deepcopy(agent);w=copy.deepcopy(world);x=state.copy()
        replay0=a.counterfactual_steps;fit0=a.identification_steps
        event=a.assimilate(packet) if deliver else None
        update=a.update(x);updates=[dict(step=0,diagnostic=update)]
        # Diagnostic control must not advance the branch's routing clock/context.
        navigation=copy.deepcopy(a.navigation);navigation_step=a.navigation_step;last_control=copy.deepcopy(a.last_control)
        initial_control=a.action(x)[0].tolist()
        initial_guard_steps=a.last_control.get('guard_steps',0)
        a.navigation=navigation;a.navigation_step=navigation_step;a.last_control=last_control
        initial_model=dict(version=a.version,policy=a.policy.tolist(),damping=a.damping,
                           fields=[s.weights.tolist() for s in a.stages])
        replay_handoffs=sum(len(c.get('stage_events',[])) for c in a.replay_calls)
        parts=dict(time=0.,risk=0.,effort=0.,failure=0.)
        path=[x.tolist()];checkpoints={};ended=False;invalid=None;work=0
        for n in range(1,max(horizons)+1):
            if not ended and invalid is None:
                w.time=(tick+n)*a.cfg.dt
                try:
                    if hasattr(w,"prepare"):w.prepare(x)
                    u,_=a.action(x);y,collision,c,_=w.step(x,u)
                    parts['time']+=a.cfg.dt
                    parts['risk']+=a.cfg.dt*4*float(w.risk_truth(y[:2]))
                    parts['effort']+=a.cfg.dt*.03*float(u@u)
                    work+=1
                    a.observe(w.sense(y,tick+n),Transition(tick+n-1,w.time,tuple(x),tuple(u),tuple(y),a.version))
                    x=y;path.append(x.tolist())
                    success=np.linalg.norm(x[:2]-a.goal)<.25
                    ended=bool(collision or success or tick+n>=a.cfg.steps)
                    if collision or (tick+n>=a.cfg.steps and not success):parts['failure']+=40.
                    if not ended:
                        if not getattr(w,"frozen_prediction",False):a.advance(x)
                        if n%a.cfg.update_every==0:
                            report=a.update(x);updates.append(dict(step=n,diagnostic=report))
                            replay_handoffs+=sum(len(c.get('stage_events',[])) for c in a.replay_calls)
                except DomainExit as exc:invalid=str(exc) or 'domain_exit'
            if n in horizons:
                checkpoints[str(n)]=dict(valid=invalid is None,reason=invalid or 'valid',
                    cost=sum(parts.values()) if invalid is None else None,components=parts.copy(),
                    executed_steps=work,terminated=ended,state=x.tolist(),version=a.version)
        results['deliver' if deliver else 'withhold']=dict(checkpoints=checkpoints,path=path,
            packet_event=event,initial_update=update,initial_control=initial_control,initial_model=initial_model,initial_control_guard_steps=initial_guard_steps,
            updates=updates,replay_handoffs=replay_handoffs,replay_steps=a.counterfactual_steps-replay0,
            identification_work=a.identification_steps-fit0,steps=work,
            stage_events=list(getattr(w,'context_jumps',[])),environment_seed=w.seed,
            observation_provenance='hypothetical' if getattr(w,'frozen_prediction',False) else 'evaluator_only_truth')
    rows=[]
    for h in horizons:
        d=results['deliver']['checkpoints'][str(h)];w=results['withhold']['checkpoints'][str(h)]
        eligible=results['deliver']['packet_event']['reason']=='eligible'
        valid=d['valid'] and w['valid'] and eligible
        rows.append(dict(horizon=h,valid=valid,reason='valid' if valid else
                         ('packet_'+results['deliver']['packet_event']['reason'] if not eligible else 'domain_exit'),
                         benefit=w['cost']-d['cost'] if valid else None))
    return dict(rows=rows,branches=results,seconds=time.perf_counter()-started,start=state.tolist(),tick=tick)
