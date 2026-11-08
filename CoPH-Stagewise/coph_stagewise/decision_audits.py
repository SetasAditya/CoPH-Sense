"""Truth references and finite differences confined to isolated evaluator copies."""
import copy
import math
import time
import numpy as np
from .model import tensor
from .shapes import is_slab,slab_offset
from .attribution_audits import packet_audit, fixed_state, digest
from .decision_trace import route_ranking, force_components


def active_signature(agent,x,state):
    nav=state['navigation'];u=nav.get('constrained');active=[]
    if u is None:return [nav['reason'],'no_feasible_projection']
    u=np.asarray(u);v=np.asarray(x[2:]);q=tensor(x[:2]);vt=tensor(v)
    residuals={'actuator':np.linalg.norm(u)-agent.cfg.force_limit}
    for d in sorted(set((*agent.cfg.damping_bounds,float(agent.damping)))):
        a=math.exp(-d*agent.cfg.dt);b=-math.expm1(-d*agent.cfg.dt)/d
        residuals[f'speed_{d}']=np.linalg.norm(u+a/b*v)-1/b
    for i,o in enumerate(agent.geometry_store):
        if is_slab(o):
            z,interior=slab_offset(q,o);curvature=(z@vt).square()/z.square().sum().clamp_min(1e-20) if bool(interior) else vt.square().sum()
        else:z=q-tensor(o[:2]);curvature=vt.square().sum()
        h=z.square().sum()-(o[2]+.12+agent.cfg.safety_margin)**2;rv=z@vt
        bound=max(2*agent.cfg.damping_bounds[0]*float(rv),2*agent.cfg.damping_bounds[1]*float(rv))-2*float(curvature)-8*float(rv)-4*float(h)
        residuals[f'barrier_{i}']=bound-float(2*z@tensor(u))
    active=[k for k,v in residuals.items() if abs(v)<=1e-7]
    return [nav['reason'],*active]


def branch_observer(agent,x,world):
    start=time.perf_counter();before=digest(agent);world_before=digest(world)
    base=fixed_state(agent,x);base_signature=active_signature(agent,x,base);sens=[];guards=base['navigation'].get('guard_steps',0)
    for j in range(2):
        lower=max(agent.cfg.policy_bounds[0],float(agent.policy[j])-1e-4)
        upper=min(agent.cfg.policy_bounds[1],float(agent.policy[j])+1e-4)
        samples=[]
        for value in (lower,upper):
            a=copy.deepcopy(agent);a.policy=a.policy.clone();a.policy[j]=value
            state=fixed_state(a,x);guards+=state['navigation'].get('guard_steps',0)
            samples.append(dict(value=value,state=state,signature=active_signature(a,x,state)))
        denominator=upper-lower
        sens.append(dict(coefficient=j,epsilon=1e-4,scheme='central' if lower<float(agent.policy[j])<upper else 'one_sided',
            interval=[lower,upper],nominal_derivative=((np.asarray(samples[1]['state']['nominal'])-samples[0]['state']['nominal'])/denominator).tolist() if denominator else None,
            applied_derivative=((np.asarray(samples[1]['state']['control'])-samples[0]['state']['control'])/denominator).tolist() if denominator else None,
            signatures=[s['signature'] for s in samples],active_set_changed=any(s['signature']!=base_signature for s in samples),
            route_changed=any(s['state']['navigation'].get('route_nodes')!=base['navigation'].get('route_nodes') for s in samples)))
    refs={name:route_ranking(x[:2],agent.goal,agent.navigation,truth=world.risk_truth,reference=name)
          for name in ('estimated','supported_truth','full_truth')}
    assert digest(agent)==before and digest(world)==world_before
    return dict(forces=force_components(base,x),sensitivity=sens,route_references=refs,
                guard_flows=guards,control_evaluations=5,seconds=time.perf_counter()-start,
                interpretation='oracle route rankings only; no oracle navigation-success claim')


def decision_packet_audit(agent,x,packet,world,tick):
    result=packet_audit(agent,x,packet,world,tick,branch_observer=branch_observer)
    result['decision_intermediates']=[dict(label=s['label'],forces=force_components(s,x),
        route=s['navigation'].get('decision_route')) for s in result['intermediate_states']]
    result['decision_guard_flows']=sum(b['decision_audit']['guard_flows'] for b in result['branches'].values())
    result['decision_seconds']=sum(b['decision_audit']['seconds'] for b in result['branches'].values())
    return result
