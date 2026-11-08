"""Compact operational summaries; not environmental evidence or measured history."""
from dataclasses import dataclass,asdict,replace
import hashlib
import json
import numpy as np
from .records import Config,Reading,Stage
from .model import tensor,risk
from .stage_prediction import StageModel
from .cooperative import CooperativeLearner,CommunicationConfig,CONVENTION


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


@dataclass(frozen=True)
class ReceiverSummary:
    schema: int
    sender: str
    recipient: str
    version: int
    timestamp: float
    expires: float
    state: tuple
    model_digest: str
    model_json: str
    convention: str = CONVENTION

    def bytes(self):return len(canonical(asdict(self)).encode())

    @classmethod
    def from_dict(cls,d):
        d=dict(d);d['state']=tuple(d['state']);return cls(**d)

    def validate(self,recipient,now):
        if self.schema not in (1,2,3) or self.convention!=CONVENTION:raise ValueError('incompatible_summary')
        if self.recipient!=str(recipient) or self.sender==self.recipient:raise ValueError('summary_identity')
        if not np.isfinite([self.timestamp,self.expires,now,*self.state]).all() or len(self.state)!=4:
            raise ValueError('nonfinite_summary')
        if self.timestamp>now+1e-9 or self.expires<now or self.expires<self.timestamp:raise ValueError('stale_summary')
        if hashlib.sha256(self.model_json.encode()).hexdigest()!=self.model_digest:raise ValueError('summary_digest')
        b=json.loads(self.model_json);cfg=Config(**b['config'])
        if 'route_decision_options' in b:
            from .route_decision import RouteDecisionOptions
            RouteDecisionOptions(**b['route_decision_options'])
        elif b.get('navigation',{}).get('route_candidate_schema') is not None:
            raise ValueError('missing_route_decision_options')
        if cfg.controller_mode!='legacy' and (self.schema not in (2,3) or not {'navigation','geometry','geometry_overflow','navigation_step'}<=b.keys() or any(not {'footprints','entry_navigation'}<=s.keys() for s in b['stages'])):raise ValueError('missing_navigation_context')
        if cfg.controller_mode!='legacy' and (len(b['geometry'])>cfg.obstacle_cap or any(len(s.get('footprints',[]))>cfg.footprint_cap for s in b['stages'])):raise ValueError('navigation_memory_bounds')
        if cfg.controller_mode!='legacy':
            contexts=[b['navigation']]+[s.get('entry_navigation') for s in b['stages']]
            for context in contexts:
                if not isinstance(context,dict):raise ValueError('missing_navigation_context')
                from .route_field import validate_context
                validate_context(context,cfg,[s['id'] for s in b['stages']])
                history=np.asarray(context.get('progress_history',[]))
                if history.size and (history.shape[0]>17 or history.ndim!=2 or history.shape[1]!=3 or not np.isfinite(history).all()):raise ValueError('invalid_progress_history')
                path=np.asarray(context.get('path',[]))
                if len(path)>cfg.route_expansion_cap or (path.size and (path.ndim!=2 or path.shape[1]!=2 or not np.isfinite(path).all())):raise ValueError('invalid_route')
                if 'target' in context and (np.asarray(context['target']).shape!=(2,) or not np.isfinite(context['target']).all()):raise ValueError('invalid_waypoint')
            geometry=np.asarray(b['geometry'])
            if geometry.size and (geometry.ndim!=2 or geometry.shape[1]!=3 or not np.isfinite(geometry).all() or (geometry[:,2]<=0).any()):raise ValueError('invalid_geometry')
            for stage in b['stages']:
                footprints=np.asarray(stage.get('footprints',[]))
                if footprints.size and (footprints.ndim!=2 or footprints.shape[1]!=4 or not np.isfinite(footprints).all() or (footprints[:,2]<=0).any()):raise ValueError('invalid_footprints')
        if cfg.shaping_mode=='progress' and (self.schema!=3 or b.get('shaping_convention')!='progress_taper_v1'):raise ValueError('incompatible_shaping_context')
        if cfg.shaping_mode=='legacy' and b.get('shaping_convention','legacy')!='legacy':raise ValueError('incompatible_shaping_context')
        if b['version']!=self.version or not 0<len(b['stages'])<=cfg.previous+1:raise ValueError('summary_bounds')
        if b['dynamic']:raise ValueError('static_summary_only')
        if not np.isfinite([b['damping'],*b['policy'],*b['goal']]).all() or b['damping']<=0:raise ValueError('invalid_model')
        if len(b['policy'])!=2 or len(b['goal'])!=2:raise ValueError('invalid_model_shape')
        if b['mode'] not in ('frozen','evidence','full'):raise ValueError('invalid_mode')
        ids=[]
        for d in b['stages']:
            ids.append(d['id'])
            if len(d['weights'])!=cfg.rbf_side**2 or np.asarray(d['centers']).shape!=(cfg.rbf_side**2,2):raise ValueError('invalid_basis')
            from .localized_receiver import validate
            validate(d.get('local_residual'),cfg.anchor_cap)
            if d.get('local_residual') is not None and b.get('receiver_localization_mode','global')!='compact':raise ValueError('incompatible_localized_context')
            if len(d['support'])>cfg.sample_cap or len(d['anchors'])>cfg.anchor_cap:raise ValueError('summary_memory')
            if not np.isfinite([*d['weights'],*np.asarray(d['centers']).reshape(-1),*np.asarray(d['support']).reshape(-1),
                                   *d['center'],*d['entry'],*d['target'],*np.asarray(d['obstacles']).reshape(-1)]).all():raise ValueError('invalid_stage')
        if ids!=sorted(set(ids)):raise ValueError('invalid_stage_order')
        if any(not source.startswith(self.sender+':') for source in b['local_source_ids']):raise ValueError('invalid_provenance')
        return b


def summarize_receiver(agent,state,recipient):
    if agent.dynamic:raise ValueError('static_summary_only')
    body=dict(version=agent.version,config=asdict(agent.cfg),communication=asdict(agent.communication),
        goal=agent.goal,policy=agent.policy.tolist(),damping=agent.damping,mode=agent.mode,reconcile=agent.reconcile,
        dynamic=False,stages=[asdict(StageModel.freeze(s)) for s in agent.stages],
        acknowledged_sources=sorted({source for p in agent.received.values() for source in p.sources}),
        last_sequences=agent.last_sequences,
        local_source_ids=sorted({r.uid for s in agent.stages for r in s.readings if int(r.uid.split(':')[-1])%5}),
        semantics='operational_model_context; never independent environmental evidence; no measured values or reserved validation')
    if getattr(agent,'receiver_localization_mode','global')=='compact':
        body.update(receiver_localization_mode='compact',receiver_fit_mode=agent.receiver_fit_mode,
            receiver_validation_mode=agent.receiver_validation_mode,receiver_solver_mode=agent.receiver_solver_mode,
            receiver_search_mode=agent.receiver_search_mode)
    else:
        for stage in body['stages']:stage.pop('local_residual',None)
    if body['config'].get('stage_blend_mode','legacy')=='legacy':body['config'].pop('stage_blend_mode',None)
    if hasattr(agent,'route_options'):body['route_decision_options']=asdict(agent.route_options)
    if agent.cfg.controller_mode!='legacy':body.update(navigation=agent.navigation,navigation_step=agent.navigation_step,geometry=agent.geometry_store,geometry_overflow=agent.geometry_overflow)
    if agent.cfg.shaping_mode=='progress':body['shaping_convention']='progress_taper_v1'
    payload=canonical(body)
    return ReceiverSummary(3 if agent.cfg.shaping_mode=='progress' else (1 if agent.cfg.controller_mode=='legacy' else 2),agent.agent_id,str(recipient),agent.version,agent.now,agent.now+agent.communication.ttl,
        tuple(map(float,state)),hashlib.sha256(payload.encode()).hexdigest(),payload)


class SummaryCache:
    """Explicit version+content references; state/time always travel in an envelope."""
    def __init__(self,capacity=8):
        if capacity<1:raise ValueError("positive cache capacity required")
        self.capacity=capacity;self.models={};self.latest={}

    def transfer(self,summary,recipient,now,budget):
        body=summary.validate(recipient,now);key=(summary.sender,summary.recipient,summary.version,summary.model_digest)
        previous=self.latest.get((summary.sender,summary.recipient))
        if previous and (summary.timestamp<previous[0] or summary.version<previous[1]):raise ValueError('reordered_summary')
        cached=key in self.models
        wire=asdict(summary)
        if cached:wire.pop('model_json')
        size=len(canonical(wire).encode())
        if size>budget:return None,0,'summary_budget'
        # Only an actually transmitted envelope updates the local knowledge cache.
        if not cached:self.models[key]=summary.model_json
        self.latest[(summary.sender,summary.recipient)]=(summary.timestamp,summary.version)
        while len(self.models)>self.capacity:del self.models[next(iter(self.models))]
        while len(self.latest)>self.capacity:del self.latest[next(iter(self.latest))]
        reconstructed=replace(summary,model_json=self.models[key])
        return reconstructed,size,'cached_reference' if cached else 'full_summary'


def reconstruct_receiver(summary,recipient,now,cross_stage=False):
    body=summary.validate(recipient,now)
    cfg=replace(Config(**body['config']),replay_domain='retained' if cross_stage else 'single')
    learner=CooperativeLearner;extra={}
    if 'route_decision_options' in body:
        from .route_decision import RouteDecisionLearner,RouteDecisionOptions
        learner=RouteDecisionLearner
        extra=dict(network=tuple(tuple(map(tuple,v)) for v in body['navigation']['network']),
            route_options=RouteDecisionOptions(**body['route_decision_options']),
            risk_weight=body['navigation'].get('risk_weight',1.),unknown_risk=body['navigation'].get('unknown_risk',1.))
    a=learner(cfg,np.array(summary.state),body['goal'],mode=body['mode'],reconcile=body['reconcile'],
                         agent_id=summary.sender,communication=CommunicationConfig(**body['communication']),
                         **{k:body[k] for k in ('receiver_localization_mode','receiver_fit_mode','receiver_validation_mode','receiver_solver_mode','receiver_search_mode') if k in body},**extra)
    a.stages=[]
    if cfg.controller_mode!='legacy':
        a.navigation=__import__('copy').deepcopy(body['navigation']);a.navigation_step=body['navigation_step']
        if cfg.route_field_mode=='supported_field':a.routing_fields=__import__('copy').deepcopy(a.navigation['route_field_snapshot']['stages'])
        a.geometry_store=tuple(map(tuple,body['geometry']));a.geometry_overflow=body['geometry_overflow']
    for d in body['stages']:
        stage=Stage(d['id'],tuple(d['center']),tuple(d['entry']),tuple(d['target']),tensor(d['centers']),tensor(d['weights']),
                    footprints=list(map(tuple,d.get('footprints',[]))),entry_navigation=__import__('copy').deepcopy(d.get('entry_navigation') or {}),
                    obstacles=tuple(map(tuple,d['obstacles'])),anchors=tuple(map(tuple,d['anchors'])),version=d['version'])
        if d.get('local_residual') is not None:stage.local_residual=__import__('copy').deepcopy(d['local_residual'])
        # Synthetic values reproduce the advertised fitted function, not past measurements.
        # Keep validation explicitly synthetic as well; exact acceptance is not recoverable.
        for k,q in enumerate(d['support']):
            value=float(risk(tensor(q),stage.weights,stage.centers,cfg.rbf_width,getattr(stage,'local_residual',None)))
            stage.readings.append(Reading(f'hypothetical_summary:{summary.sender}:{stage.id}:{k*5+1}',summary.timestamp,tuple(q),value))
        for k,q in enumerate(d['support'][:min(12,max(0,cfg.sample_cap-len(stage.readings)))]):
            value=float(risk(tensor(q),stage.weights,stage.centers,cfg.rbf_width,getattr(stage,'local_residual',None)))
            stage.readings.append(Reading(f'hypothetical_validation:{summary.sender}:{stage.id}:{k*5}',summary.timestamp,tuple(q),value))
        a.stages.append(stage)
    a.policy=tensor(body['policy']);a.damping=body['damping'];a.version=summary.version;a.now=summary.timestamp
    a.last_sequences={k:tuple(v) for k,v in body['last_sequences'].items()}
    a.hypothetical=True;a.summary_digest=summary.model_digest
    a.advertised_acknowledged_sources=tuple(body['acknowledged_sources'])
    a.validation_interpretation='synthetic_model_consistency; not measured held-out validation'
    return a
