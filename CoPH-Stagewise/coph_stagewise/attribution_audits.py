"""Evaluator-only packet interventions; private snapshots never enter runtime."""
import copy
import hashlib
import pickle
import time
import numpy as np
from .records import Transition
from .staggered_demo import sense
from .model import tensor, potential
from .diagnostics import probe_metrics, probes


def digest(value):
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def fixed_state(agent, x):
    a = copy.deepcopy(agent)
    u, raw = a.action(x.copy())
    s = a.stages[-1]
    kinetic = .5*float(x[2:]@x[2:])
    shaping = float(potential(tensor(x[:2]), tensor(a.control_target(s)), s.weights,
        s.centers, a.cfg.rbf_width, a.policy, a.control_obstacles(s), a.cfg))
    return dict(model=a.control_model(), control=u.tolist(), nominal=raw.tolist(),
        navigation=copy.deepcopy(a.last_control), H=kinetic+shaping, kinetic=kinetic,
        shaping=shaping, version=a.version, policy=a.policy.tolist(),
        fields=[s.weights.tolist() for s in a.stages])


def evaluate_probes(agent, world):
    return probe_metrics(agent, world, probes(world.spec))


def packet_audit(agent, x, packet, world, tick, branch_observer=None):
    started = time.perf_counter()
    original = digest(agent)
    world_before = digest(world)
    measurement_hash = digest([(s.readings,s.transitions) for s in agent.stages])
    branches = {}
    learners = {}
    for delivery in (False, True):
        a = copy.deepcopy(agent)
        event = a.assimilate(packet) if delivery else None
        assert digest([(s.readings,s.transitions) for s in a.stages]) == measurement_hash
        report = a.update(x.copy())
        state = fixed_state(a, x)
        if branch_observer is not None:state['decision_audit']=branch_observer(a,x,world)
        key = 'deliver' if delivery else 'withhold'
        branches[key] = dict(state, accepted=report['accepted'], reason=report['reason'],
            event=event, replay_work=report['counterfactual_steps'],
            identification_work=a.identification_steps-agent.identification_steps,
            diagnostic=report, checkpoints={})
        learners[key] = a

    # Ordered telescoping decomposition from the updated withhold model.
    # Route/geometry receipt precedes reconstructed field and then policy changes.
    base = copy.deepcopy(learners['withhold'])
    stages = [dict(label='withhold_updated', **fixed_state(base,x))]
    base.assimilate(packet)
    stages.append(dict(label='received_geometry_and_route_context', **fixed_state(base,x)))
    delivered = learners['deliver']
    for b, d in zip(base.stages, delivered.stages):
        b.weights = d.weights.clone(); b.version = d.version
    base.damping = delivered.damping
    base.local_model_anchors = copy.deepcopy(delivered.local_model_anchors)
    base.local_cells = copy.deepcopy(delivered.local_cells)
    base.refresh_terrain()
    if agent.cfg.route_field_mode=='supported_field':
        from .route_field import sync
        base.routing_fields=copy.deepcopy(delivered.routing_fields)
        sync(base)
    stages.append(dict(label='reconstructed_field_and_damping', **fixed_state(base,x)))
    base.policy = delivered.policy.clone()
    stages.append(dict(label='optimized_policy', **fixed_state(base,x)))
    decomposition = []
    for left,right in zip(stages, stages[1:]):
        decomposition.append(dict(label=right['label'],
            shaping_jump=right['shaping']-left['shaping'],
            target_before=left['model']['target'],target_after=right['model']['target'],
            nominal_delta=(np.array(right['nominal'])-left['nominal']).tolist(),
            applied_delta=(np.array(right['control'])-left['control']).tolist()))

    horizon = min(32, agent.cfg.steps-tick)
    starts=world.spec.get('start_seconds',[0.,5.]);receiver=int(agent.agent_id)
    receive_slots={int(np.ceil((starts[1-receiver]+k*agent.cfg.dt-starts[receiver])/agent.cfg.dt-1e-9))
        for k in range(0,agent.cfg.steps,agent.cfg.update_every)
        if starts[1-receiver]+k*agent.cfg.dt>=starts[receiver]-1e-9}
    work = 0
    for name, a in learners.items():
        w = copy.deepcopy(world); state=x.copy(); path=[state.tolist()]
        components=dict(time=0.,risk=0.,effort=0.,failure=0.)
        collision=False; success=False; terminal_step=None; completed=0
        for step in range(1,horizon+1):
            w.time=agent.now+step*a.cfg.dt
            u,_=a.action(state)
            after,collision,_,_=w.step(state,u);work+=1
            components['time']+=a.cfg.dt
            components['risk']+=a.cfg.dt*4*float(w.risk_truth(after[:2]))
            components['effort']+=a.cfg.dt*.03*float(u@u)
            observation=sense(w,after,tick+step,w.time)
            a.observe(observation,Transition(tick+step-1,w.time,tuple(state),tuple(u),tuple(after),a.version))
            state=after;path.append(state.tolist());completed=step
            success=bool(np.linalg.norm(state[:2]-a.goal)<.25)
            terminal=collision or success or tick+step>=a.cfg.steps
            if terminal:
                terminal_step=step
                if not success:components['failure']=40.
            else:
                a.advance(state)
                if (tick+step)%a.cfg.update_every==0 or tick+step in receive_slots:a.update(state)
            if step in (8,16,32) or terminal:
                branches[name]['checkpoints'][str(step)]=dict(cost=sum(components.values()),
                    components=copy.deepcopy(components),success=success,collision=bool(collision),
                    terminal_step=terminal_step,path=copy.deepcopy(path),steps=step)
            if terminal:break
        for h in (8,16,32):
            if str(h) not in branches[name]['checkpoints']:
                branches[name]['checkpoints'][str(h)]=(dict(cost=sum(components.values()),
                    components=copy.deepcopy(components),success=success,collision=bool(collision),
                    terminal_step=terminal_step,path=copy.deepcopy(path),steps=completed)
                    if terminal_step is not None else None)
        branches[name]['total_replay_work']=a.counterfactual_steps-agent.counterfactual_steps
        branches[name]['total_identification_work']=a.identification_steps-agent.identification_steps
        if hasattr(a,'route_field_evaluations'):
            branches[name]['route_field_audit_work']=dict(field_evaluations=a.route_field_evaluations-agent.route_field_evaluations,
                diagnostic_field_evaluations=a.route_field_diagnostic_evaluations-agent.route_field_diagnostic_evaluations,
                seconds=a.route_field_seconds-agent.route_field_seconds)
    assert digest(agent)==original and digest(world)==world_before, 'audit mutated runtime'
    d,b=branches['deliver'],branches['withhold']
    return dict(packet=packet.identity,receiver=int(agent.agent_id),time=agent.now,stage=agent.stages[-1].id,
        start=x.tolist(),tick=tick,horizons=[8,16,32],branches=branches,
        field_delta=float(np.linalg.norm(np.asarray(d['fields'])-b['fields'])),
        control_delta=float(np.linalg.norm(np.asarray(d['control'])-b['control'])),H_delta=d['H']-b['H'],
        seconds=time.perf_counter()-started,truth_steps=work,operational_state_unchanged=True,
        retained_measurements_hash=measurement_hash,source_ids=packet.sources,
        geometry_source_ids=[g[-1] for g in packet.geometry],
        intermediate_states=stages,decomposition=decomposition,
        safety_modification={k:(np.array(v['control'])-v['navigation']['nominal_clipped']).tolist() for k,v in branches.items()},
        clipping_modification={k:(np.array(v['navigation']['nominal_clipped'])-v['nominal']).tolist() for k,v in branches.items()},
        decomposition_closure_residual=float(np.linalg.norm(np.array(stages[-1]['control'])-d['control'])),
        interpretation='ordered decomposition, not unique attribution; true-world audit only; no future packets',
        randomness=dict(seed=world.seed,local_ticks=[tick+1,tick+horizon],shared_between_branches=True),
        realized_improvement={str(h):(b['checkpoints'][str(h)]['cost']-d['checkpoints'][str(h)]['cost']
            if b['checkpoints'][str(h)] is not None and d['checkpoints'][str(h)] is not None else None) for h in (8,16,32)})
