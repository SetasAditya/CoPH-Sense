"""Opt-in receiver instrumentation. Truth is confined to isolated evaluator functions."""
import copy,time
import numpy as np
from .sharing import LedgerExport,FieldLearner
from .route_field import hypothetical_context,sync
from .decision_trace import route_ranking
from .cooperative import field_vector
from .attribution_audits import digest,fixed_state


def ranking(agent,x,context=None,details=False):
    r=route_ranking(x[:2],agent.goal,context or agent.navigation,trace_fields=True,respect_context=True)
    if not details:
        r['coverage']={k:sum(e['route_field']['source'].count(k) for e in r['quadrature']) for k in ('field','anchor','unknown')}
        r['query_count']=sum(len(e['points']) for e in r.pop('quadrature'))
    return r


def eligibility(agent,packet):
    rows=[]
    for s in agent.stages:
        local=np.asarray([r.q for r in s.readings if int(r.uid.split(':')[-1])%5]).reshape(-1,2)
        for k,a in enumerate(packet.anchors):
            inside=bool(s.contains(a[:2],agent.cfg));near=bool(len(local) and np.min(np.linalg.norm(local-np.asarray(a[:2]),axis=1))<=.65)
            rows.append(dict(stage=s.id,version=s.version,anchor=k,q=list(a[:2]),rectangle=inside,local_support=inside and near))
    return rows


def residuals(agent):
    rows=[]
    for m in agent.received.values():
        for s in agent.stages:
            ids=[k for k,a in enumerate(m.anchors) if s.contains(a[:2],agent.cfg)]
            if not ids:continue
            v=field_vector(s.weights,s.centers,[m.anchors[k][:2] for k in ids],agent.cfg.rbf_width,getattr(s,'local_residual',None)).reshape(-1,3)
            z=np.array([m.anchors[k][2:] for k in ids])
            rows.append(dict(packet=m.identity,stage=s.id,anchors=ids,value_residual=(v[:,0]-z[:,0]).tolist(),gradient_residual=(v[:,1:]-z[:,1:]).tolist()))
    return rows


class ReconstructionLearner(LedgerExport,FieldLearner):
    def __init__(self,*args,fit_received=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.fit_received=fit_received;self.track_route_field=True
        self.rr_events=[];self.rr_seconds=0.;self.rr_queries=0;self.rr_routing_evals=0;self.rr_trace_evals=0
        self.rr_origin=np.asarray(self.current_state[:2]);self.rr_entered=False

    def observe(self,observation,transition=None):
        super().observe(observation,transition)
        if hasattr(self,"rr_origin") and np.linalg.norm(np.asarray(observation.sensor_center)-self.rr_origin)>.25:self.rr_entered=True

    def _field_fit(self):
        if self.fit_received:return super()._field_fit()
        received=self.received;self.received={}
        try:return super()._field_fit()
        finally:self.received=received

    def _expire(self):
        before=dict(self.received);super()._expire()
        if hasattr(self,'rr_events'):
            for k,m in before.items():
                if k not in self.received:self.rr_events.append(dict(kind='expiry_or_stage_eviction',time=self.now,packet=m.identity,expired=m.expires<self.now))

    def assimilate(self,packet):
        started=time.perf_counter();before={k:m.identity for k,m in self.received.items()};state=self.current_state
        e=super().assimilate(packet)
        after={k:m.identity for k,m in self.received.items()}
        self.rr_events.append(dict(kind='receipt',time=self.now,packet=packet.identity,sources=list(packet.sources),sender_stage=packet.stage,
            event=copy.deepcopy(e),before=before,after=after,replaced=before.get(packet.sender) if before.get(packet.sender)!=after.get(packet.sender) else None,
            fit_enabled=self.fit_received,eligibility=eligibility(self,packet)))
        assert self.current_state==state
        self.rr_seconds+=time.perf_counter()-started
        return e

    def action(self,x):
        from . import route_field
        before=route_field.FIELD_EVALUATIONS
        u,raw=super().action(x);began=time.perf_counter()
        self.rr_routing_evals+=route_field.FIELD_EVALUATIONS-before;before=route_field.FIELD_EVALUATIONS
        r=ranking(self,x);self.rr_trace_evals+=route_field.FIELD_EVALUATIONS-before;self.last_control['rr_route']=r
        self.rr_queries+=r['query_count'];self.rr_seconds+=time.perf_counter()-began
        return u,raw

    def diagnostic_summary(self):
        return dict(seconds=self.rr_seconds,queries=self.rr_queries,routing_field_evaluations=self.rr_routing_evals,trace_field_evaluations=self.rr_trace_evals,
                    export_work=self.export_work,message_fit_work=self.message_fit_work)

    def update(self,x):
        began=time.perf_counter();before=residuals(self);messages={k:m.identity for k,m in self.received.items()}
        self.rr_seconds+=time.perf_counter()-began
        report=super().update(x)
        began=time.perf_counter()
        report['rr_trace']=dict(fit_enabled=self.fit_received,messages=messages,residuals_before=before,residuals_after=residuals(self),
            events=copy.deepcopy(self.rr_events),routing_versions=[(f['id'],f['version']) for f in getattr(self,'routing_fields',[])])
        self.rr_events=[];self.rr_seconds+=time.perf_counter()-began
        report['rr_trace'].update(diagnostic_seconds_total=self.rr_seconds,diagnostic_queries_total=self.rr_queries)
        return report


class CarrierAuditSelector:
    """Audit every delivery before leaving .25 m of the shared start; then first per stage.
    Branch entry is a geometric crossing, not a claim about decision timing.
    """
    def __init__(self,start):self.start=np.array(start);self.entered=False;self.seen=set()
    def __call__(self,agent,x,packet,tick,eligible):
        if agent.agent_id!='1':return False
        if getattr(agent,"rr_entered",False) or np.linalg.norm(x[:2]-self.start)>.25:self.entered=True
        key=agent.stages[-1].id
        selected=not self.entered or key not in self.seen
        if selected:self.seen.add(key)
        return selected  # rejected/unsupported deliveries remain diagnostically useful


def field_queries(agent,points):
    ctx=hypothetical_context(agent)
    from .route_field import values
    _,meta=values(points,ctx,details=True)
    output=np.full((len(points),3),np.nan)
    for s in agent.stages:
        ids=[i for i,k in enumerate(meta['stage']) if k==s.id]
        if ids:output[ids]=field_vector(s.weights,s.centers,np.asarray(points)[ids],agent.cfg.rbf_width,getattr(s,'local_residual',None)).reshape(-1,3)
    return output,meta


def receiver_audit(agent,x,packet,world,tick,capture_proposals=False):
    started=time.perf_counter();original=digest(agent);physical=x.copy();truth_before=digest(world)
    measured=digest([(s.readings,s.transitions) for s in agent.stages])
    from . import route_field
    initial_evals=route_field.FIELD_EVALUATIONS
    base=copy.deepcopy(agent);event=base.assimilate(packet)
    anchors=copy.deepcopy(base.navigation.get('terrain_anchors',[]))
    quad=ranking(base,x,details=True)['quadrature'];points=np.array([p for e in quad for p in e['points']])
    truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
    initial,imeta=field_queries(base,points)
    branches={};predictions={}
    for enabled in (False,True):
        a=copy.deepcopy(base);a.fit_received=enabled
        work=a.counterfactual_steps;ident=a.identification_steps;fit=a.message_fit_work
        d=a.update(x.copy());fixed=fixed_state(a,x)
        assert digest([(s.readings,s.transitions) for s in a.stages])==measured
        assert digest(a.remote_terrain)==digest(base.remote_terrain)
        ranks={};natural={}
        for mode in ('anchors','supported_field'):
            ctx=copy.deepcopy(a.navigation) if mode=='anchors' else hypothetical_context(a)
            ctx['route_field_mode']=mode
            natural[mode]=ranking(a,x,ctx)
            ctx['terrain_anchors']=copy.deepcopy(anchors)
            ranks[mode]=ranking(a,x,ctx,True)
        pred,meta=field_queries(a,points);predictions[str(enabled)]=pred
        branches['fit' if enabled else 'no_fit']=dict(accepted=d['accepted'],reason=d['reason'],version=a.version,
            fields=[s.weights.tolist() for s in a.stages],policy=a.policy.tolist(),control=fixed['control'],target=fixed['model']['target'],
            kinetic=fixed['kinetic'],shaping=fixed['shaping'],replay_work=a.counterfactual_steps-work,identification_work=a.identification_steps-ident,
            message_fit_work=a.message_fit_work-fit,residuals=residuals(a),rankings=ranks,natural_rankings=natural,
            support=meta,validation={k:v for k,v in d.items() if 'validation' in k},
            hypothetical_measurements=False,remote_anchors_identical=True)
        if capture_proposals:
            branches['fit' if enabled else 'no_fit']['proposal']=dict(
                initial_fields=[s.weights.tolist() for s in base.stages],
                proposed_fields=[w.tolist() for w in a.reference] if getattr(a,'reference',None) is not None else None,
                residuals_before=residuals(base),diagnostic=d)
    common=np.isfinite(initial).all(1)&np.isfinite(predictions['True']).all(1)&np.isfinite(predictions['False']).all(1)
    def errors(v):
        delta=v[common]-truth[common]
        return dict(value_mae=float(np.abs(delta[:,0]).mean()) if len(delta) else None,
                    gradient_mae=float(np.linalg.norm(delta[:,1:],axis=1).mean()) if len(delta) else None)
    data=dict(points=points.tolist(),truth=truth.tolist(),common_support=common.tolist(),initial_support=imeta,
        predictions={k:[[float(z) for z in row] if np.isfinite(row).all() else None for row in v] for k,v in {'initial':initial,**predictions}.items()},
        errors={k:errors(v) for k,v in {'initial':initial,'fit':predictions['True'],'no_fit':predictions['False']}.items()},
        supported_count=int(common.sum()),query_count=len(points))
    assert digest(agent)==original and digest(world)==truth_before and np.array_equal(x,physical)
    return dict(packet=packet.identity,receiver=int(agent.agent_id),time=agent.now,stage=agent.stages[-1].id,tick=tick,event=event,
        **({'source_ids':list(packet.sources)} if capture_proposals else {}),
        eligibility=eligibility(base,packet),branches=branches,queries=data,seconds=time.perf_counter()-started,
        field_evaluations=route_field.FIELD_EVALUATIONS-initial_evals,operational_state_unchanged=True,measurements_unchanged=True,
        semantics='identical packet received in both clones; fit on/off; fixed-anchor ranks isolate reconstruction; natural ranks include regenerated local anchors; no mission benefit inferred')
