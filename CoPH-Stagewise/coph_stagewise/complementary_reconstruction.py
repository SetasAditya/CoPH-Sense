"""Gated complementary-coverage diagnosis; all truth use is evaluator-only."""
import copy,gzip,json,pickle,time,traceback
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from .receiver_reconstruction import METHODS,fingerprint,file_hash
from .decision_suite import NODES,EDGES,specification as fork_spec
from .crossing_demo import configuration
from .hazard_demo import communication
from .records import Config
from .cooperative import CommunicationConfig
from .maze_family import NetworkWorld
from .staggered_demo import execute
from .reconstruction_diagnostics import ReconstructionLearner,receiver_audit,eligibility,ranking
from .route_field import hypothetical_context
from .evaluation import dump,load
from .cli import source_hashes,archive_sources

CASES=('complementary','reverse','redundant')
SEEDS=(79100,79101)
ACQUISITION=.96


def specification(case,seed):
    if case not in CASES:raise ValueError(case)
    starts=[[.4,.45,0.,0.],[.4,-.45,0.,0.]]
    if case=='reverse':starts=starts[::-1]
    if case=='redundant':starts=[starts[1].copy(),starts[1].copy()]
    s=fork_spec('clear_fork',seed)
    s.update(name=case,scene=case,initials=starts,initial=starts[0],start_seconds=[0.,0.],
        communication_start_seconds=ACQUISITION,risk_weights=[.25,6.],unknown_risk=.02,
        patches=[[1.25,.45,.6,1.,.5]],roles=['agent 0: risk weight .25','agent 1: risk weight 6'])
    return s


def manifest():
    m=dict(schema=1,profile='complementary',development_only=True,seeds=list(SEEDS),animation_seed=79100,
        specs={f'{c}_{s}':specification(c,s) for c in CASES for s in SEEDS},methods=METHODS,
        config=asdict(configuration(500)),communication=asdict(communication('field')),source_hashes=source_hashes(),
        acquisition=dict(seconds=ACQUISITION,boundary='after both sensing steps, before first export and boundary update',
            ordinary_movement_and_fitting=True,counted_in_mission=True),
        gate_required=['complementary_79100','complementary_79101','reverse_79100','reverse_79101'],
        intended_sender=dict(complementary=0,reverse=1,redundant=None),
        branch_marker='projection > .25 m along edge (1,2) or (1,4), within .25 m perpendicular distance; geometric traversal, not planning',
        expected_episodes=18,expected_missions=36,policy_replay=False,
        non_overlap_reference='results/receiver_reconstruction_v1: 25-second relay remains useful for routing, not an irrelevant-information control',
        seed_semantics='independent indexed sensor noise, identical geometry; query counts are not independent trials',
        held_out_executed=False,gate_failure_action='stop; no navigation comparisons, fixture search or adjustment')
    return dict(m,fingerprint=fingerprint(m))


def factory_for(spec,fit):
    def factory(cfg,x,goal,agent_id,communication):
        return ReconstructionLearner(cfg,x,goal,agent_id=agent_id,communication=communication,mode='evidence',fit_received=fit,
            network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))),
            risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'])
    return factory


def branch_entry(q):
    nodes=np.asarray(NODES);q=np.asarray(q)
    for edge in ((1,2),(1,4)):
        a,b=nodes[list(edge)];d=b-a;length=np.linalg.norm(d);along=float((q-a)@(d/length))
        if .25<along<=length+.25 and np.linalg.norm(q-a-np.clip(along,0,length)*d/length)<.25:return list(edge)
    return None


def prefix(ctx):
    # Compare measured/control/committed-state evidence, not timing or scorer metadata.
    return dict(observations=copy.deepcopy(ctx['observations']),samples=copy.deepcopy(ctx['samples']),
        trajectories=[[{k:t[k] for k in ('time','x','u','after','version','stage','energy')} for t in ts] for ts in ctx['trajectories']],
        updates=[[dict(time=u['time'],accepted=u['diagnostic']['accepted'],reason=u['diagnostic']['reason'],
            pre=u['pre'],post=u['post'],work=u['diagnostic']['counterfactual_steps']) for u in us] for us in ctx['updates']])


def equivalent_prefix(a,b):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k not in ('navigation','entry_navigation','route_field_mode','route_field_snapshot','wall_seconds') and not k.startswith('rr_')}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        return v
    a,b=clean(a),clean(b)
    def eq(x,y):
        if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(eq(x[k],y[k]) for k in x)
        if isinstance(x,list):return isinstance(y,list) and len(x)==len(y) and all(eq(v,w) for v,w in zip(x,y))
        if isinstance(x,float):return isinstance(y,(float,int)) and bool(np.isclose(x,y,atol=1e-12,rtol=1e-10))
        return x==y
    return dict(passed=eq(a,b),bitwise=fingerprint(a)==fingerprint(b),reference=fingerprint(a),actual=fingerprint(b))


def save_snapshot(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wb') as f:pickle.dump(value,f,protocol=5)
    return dict(path=path.name,sha256=file_hash(path),format='trusted local evaluator snapshot; never runtime input')


class AcquisitionComplete(Exception):pass


def acquire(spec,cfg,comm):
    captured={}
    def boundary(ctx):
        if ctx['time']+1e-9>=ACQUISITION:
            captured.update(copy.deepcopy(ctx));raise AcquisitionComplete()
    try:
        execute(spec,'disabled',comm,audit=False,learner_factory=factory_for(spec,True),world_factory=NetworkWorld,config=cfg,boundary_callback=boundary)
    except AcquisitionComplete:pass
    if not captured:raise RuntimeError('Acquisition boundary not reached')
    return captured


def gate_snapshot(ctx,spec):
    directions=[];audits=[];agents=ctx['agents'];states=ctx['states']
    for sender in (0,1):
        receiver=1-sender;a=agents[receiver];x=states[receiver]
        exported=copy.deepcopy(agents[sender]);packet=exported.export(states[sender].copy(),str(receiver))
        quad=ranking(a,x,hypothetical_context(a),True)['quadrature']
        outgoing=[e for e in quad if tuple(e['edge']) in ((1,2),(1,4))]
        support=[dict(edge=e['edge'],points=e['points'],supported=e['route_field']['field_support'],
            count=sum(e['route_field']['field_support']),total=len(e['points'])) for e in outgoing]
        pairs=eligibility(a,packet) if packet else []
        inside=sorted({r['anchor'] for r in pairs if r['rectangle']})
        outside=[k for k in inside if np.linalg.norm(np.asarray(packet.anchors[k][:2])-x[:2])>a.cfg.sensor_radius+1e-10]
        entered=any(branch_entry(s['x'][:2]) is not None for s in ctx['samples'][receiver])
        audit=receiver_audit(a,x,packet,ctx['worlds'][receiver],8,capture_proposals=True) if packet else None
        if audit:audits.append(audit)
        checks=dict(exported=packet is not None,rectangle_anchor=bool(inside),complementary_anchor=bool(outside),
            committed_both_outgoing=len(support)==2 and all(e['count']>0 for e in support),before_branch=not entered,
            valid_packet=bool(audit and audit['event']['reason']=='eligible'),
            nonzero_fitting=bool(audit and audit['branches']['fit']['message_fit_work']>0))
        directions.append(dict(sender=sender,receiver=receiver,intended=sender==dict(complementary=0,reverse=1,redundant=None)[spec['scene']],
            passed=all(checks.values()),checks=checks,packet=asdict(packet) if packet else None,
            eligibility=pairs,rectangle_anchor_ids=inside,outside_footprint_ids=outside,exported_anchor_count=len(packet.anchors) if packet else 0,
            outgoing=support,receiver_state=x.tolist(),sender_state=states[sender].tolist(),sensor_radius=a.cfg.sensor_radius,
            stages=[dict(id=s.id,version=s.version,center=list(s.center),width=a.cfg.stage_width,height=a.cfg.stage_height) for s in a.stages]))
    first=[o[0] for o in ctx['observations']]
    coords=[[r['q'] for r in o['readings']] for o in first];vals=[[r['value'] for r in o['readings']] for o in first]
    noise=dict(initial_coordinates_identical=coords[0]==coords[1],initial_values_identical=vals[0]==vals[1],
        seeds=[w.seed for w in ctx['worlds']],independent_indexed_streams=ctx['worlds'][0].seed!=ctx['worlds'][1].seed,
        acquisition_centers_identical=[o['sensor_center'] for o in ctx['observations'][0]]==[o['sensor_center'] for o in ctx['observations'][1]],
        note='Redundant starts share initial sensor coordinates. Different risk weights and independent noise can move agents differently afterward; subsequent coverage is measured, not assumed identical.')
    return dict(scene=spec['scene'],seed=spec['seed'],time=ctx['time'],directions=directions,noise=noise,
        required=spec['scene']!='redundant',passed=all(d['passed'] for d in directions if d['intended']),
        acquisition_work=[dict(flow=a.counterfactual_steps,identification=a.identification_steps) for a in agents]),audits


class DeliverySnapshots:
    """Bounded pending copies; retain first-stage and last pre-branch delivery snapshots."""
    def __init__(self,expected):
        self.expected=expected;self.acquisition=None;self.seen=set();self.selected={};self.pending={};self.entered={};self.seconds=0.
    def boundary(self,ctx):
        start=time.perf_counter()
        if abs(ctx['time']-ACQUISITION)<1e-9:
            self.acquisition=equivalent_prefix(self.expected,prefix(ctx))
            if not self.acquisition['passed']:raise RuntimeError('Acquisition histories differ; navigation comparison stopped')
        for i,x in enumerate(ctx['states']):
            edge=branch_entry(x[:2])
            if edge is not None and i not in self.entered:
                self.entered[i]=dict(time=ctx['time'],edge=edge)
                if i in self.pending:
                    snap=self.pending.pop(i);key=snap[2].identity
                    self.selected.setdefault(key,dict(snapshot=snap,purposes=[]))['purposes'].append('last_delivery_before_first_branch_entry')
        self.seconds+=time.perf_counter()-start
    def delivery(self,a,x,m,w,tick):
        start=time.perf_counter();i=int(a.agent_id);key=(i,a.stages[-1].id)
        if key not in self.seen or i not in self.entered:
            snap=copy.deepcopy((a,x,m,w,tick))
            if key not in self.seen:
                self.selected[m.identity]=dict(snapshot=snap,purposes=['first_delivery_in_receiver_stage']);self.seen.add(key)
            if i not in self.entered:self.pending[i]=snap
        self.seconds+=time.perf_counter()-start


def run_episode(root,key,method):
    root=Path(root);m=load(root/'manifest.json');spec=m['specs'][key];mode,fit=METHODS[method]
    cfg=replace(Config(**m['config']),route_field_mode=mode);comm=CommunicationConfig(**m['communication'])
    capture=DeliverySnapshots(load(root/'acquisition'/f'{key}.json.gz'));started=time.perf_counter()
    r=execute(spec,'reciprocal_fixed',comm,audit=False,learner_factory=factory_for(spec,fit),world_factory=NetworkWorld,config=cfg,
        boundary_callback=capture.boundary,delivery_callback=capture.delivery)
    r.update(method=method,scene=spec['scene'],manifest_fingerprint=m['fingerprint'],acquisition_check=capture.acquisition,
        first_branch_entries=capture.entered,snapshot_seconds=capture.seconds)
    r['audits']=[]
    for n,(identity,selected) in enumerate(capture.selected.items()):
        snap=selected['snapshot'];a=receiver_audit(*snap,capture_proposals=True)
        a['purposes']=selected['purposes'];a['snapshot']=save_snapshot(root/'snapshots'/f'{key}_{method}_{n}.pkl.gz',snap);r['audits'].append(a)
    r['audit_seconds']=sum(a['seconds'] for a in r['audits']);r['audit_replay_work']=sum(b['replay_work'] for a in r['audits'] for b in a['branches'].values())
    path=root/'episodes'/f'{key}_{method}.json.gz';dump(path,r)
    row=dict(id=f'{key}_{method}',complete=True,path=path.name,sha256=file_hash(path),seconds=time.perf_counter()-started)
    dump(root/'status'/f'{key}_{method}.json',row);return row


def suite(root):
    root=Path(root);m=manifest();torch.set_num_threads(1)
    if (root/'manifest.json').exists():
        if load(root/'manifest.json')['fingerprint']!=m['fingerprint']:raise ValueError('Frozen source/config mismatch; use archive or fresh output')
    else:
        root.mkdir(parents=True,exist_ok=False)
        for sub in ('acquisition','gate','snapshots','episodes','status','figures'):(root/sub).mkdir()
        dump(root/'manifest.json',m);archive_sources(root)
    rows=[]
    for key,spec in m['specs'].items():
        path=root/'gate'/f'{key}.json'
        if path.exists():row=load(path)
        else:
            ctx=acquire(spec,Config(**m['config']),CommunicationConfig(**m['communication']))
            dump(root/'acquisition'/f'{key}.json.gz',prefix(ctx))
            row,audits=gate_snapshot(ctx,spec)
            row['snapshot']=save_snapshot(root/'snapshots'/f'{key}_acquisition.pkl.gz',ctx)
            dump(root/'gate'/f'{key}_audits.json.gz',audits);dump(path,row)
        rows.append(row);print(json.dumps(dict(case=key,passed=row['passed'],required=row['required'])),flush=True)
    passed=all(r['passed'] for r in rows if r['required']);dump(root/'gate.json',dict(passed=passed,required_cases=4,evaluated_cases=len(rows)))
    if passed:
        for key in m['specs']:
            for method in METHODS:
                status=root/'status'/f'{key}_{method}.json'
                if status.exists():
                    s=load(status)
                    if s['complete']:assert file_hash(root/'episodes'/s['path'])==s['sha256'];continue
                    raise RuntimeError('Preserved incomplete run; investigate before resuming')
                try:print(json.dumps(run_episode(root,key,method)),flush=True)
                except Exception as e:
                    dump(status,dict(id=f'{key}_{method}',complete=False,error=str(e),traceback=traceback.format_exc()))
                    from .complementary_reporting import report
                    report(root);raise
    from .complementary_reporting import report,render
    report(root);render(root)


def dispatch(args):
    if args.command=='suite':return suite(args.out)
    from . import complementary_reporting
    return getattr(complementary_reporting,'report' if args.command=='analyze' else args.command)(args.out)
