"""Receiver-local packet assessment. No evaluator/simulator imports or truth input."""
from dataclasses import dataclass,asdict
import copy,hashlib,json,time
import numpy as np
from .cooperative import advertise
from .receiver_summary import reconstruct_receiver
from .model_prediction import ModelEnvironment
from .continuations import continuation

HORIZONS=(8,16,32)
DRAW_COUNT=4

@dataclass(frozen=True)
class PacketFeedback:
    schema: int
    packet: str
    receiver: str
    receiver_version: int
    timestamp: float
    state: tuple
    horizon_mode: str
    statistics_json: str

    def bytes(self):return len(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def aggregate(draws):
    rows=[]
    for horizon in HORIZONS:
        results=[next(r for r in d['rows'] if r['horizon']==horizon) for d in draws]
        valid=[r['benefit'] for r in results if r['valid']]
        rows.append(dict(horizon=horizon,valid=bool(valid),valid_draws=len(valid),total_draws=DRAW_COUNT,
            benefit=float(np.mean(valid)) if valid else None,
            dispersion=float(np.std(valid,ddof=1)) if len(valid)>1 else None,
            exclusions=[r['reason'] for r in results if not r['valid']],
            interpretation='valid-draw conditional mean and sample dispersion; not calibrated confidence'))
    return rows


def evaluate_model(agent,state,packet,status,seed):
    """Shared evaluator for exact local state or legally reconstructed compact state."""
    begin=time.perf_counter();draws=[]
    for index in range(DRAW_COUNT):
        draw_seed=int(np.random.SeedSequence([seed,index,2718]).generate_state(1)[0])
        world=ModelEnvironment(agent,draw_seed,cross_stage=True)
        if not world.sample_shared_field(packet,status,draw_seed):
            draws.append(dict(draw=index,seed=draw_seed,rows=[dict(horizon=h,valid=False,reason='missing_common_uncertainty',benefit=None) for h in HORIZONS],branches={},seconds=0.))
            continue
        result=continuation(agent,np.array(state),world,packet,int(round(agent.now/agent.cfg.dt)),HORIZONS)
        result.update(draw=index,seed=draw_seed,sampled_anchors=world.sampled_anchors,field_draw_work=world.field_draw_work)
        draws.append(result)
    rows=aggregate(draws);revisions=[]
    for d in draws:
        if not d['branches']:continue
        a=d['branches']['deliver'];b=d['branches']['withhold']
        revisions.append(dict(draw=d['draw'],deliver_accepted=a['initial_update']['accepted'],withhold_accepted=b['initial_update']['accepted'],
            deliver_reason=a['initial_update']['reason'],withhold_reason=b['initial_update']['reason'],
            shaping_revision=a['initial_update'].get('shaping_jump',0.)-b['initial_update'].get('shaping_jump',0.),
            force_revision=float(np.linalg.norm(np.array(a['initial_control'])-b['initial_control'])),
            policy_revision=(np.array(a['initial_model']['policy'])-b['initial_model']['policy']).tolist(),
            field_revision=float(sum(np.linalg.norm(np.array(x)-y) for x,y in zip(a['initial_model']['fields'],b['initial_model']['fields'])))))
    work=dict(transitions=sum(b['steps'] for d in draws for b in d['branches'].values()),
        replay_steps=sum(b['replay_steps'] for d in draws for b in d['branches'].values()),
        identification_work=sum(b['identification_work'] for d in draws for b in d['branches'].values()),
        field_draw_work=sum(d.get('field_draw_work',0) for d in draws),
        pilot_steps=sum(u['diagnostic'].get('replay_pilot_steps',0) for d in draws for b in d['branches'].values() for u in b['updates']),
        replay_handoffs=sum(b.get('replay_handoffs',0) for d in draws for b in d['branches'].values()),
        seconds=time.perf_counter()-begin)
    payload=dict(rows=rows,initial_revisions=revisions,work={k:v for k,v in work.items() if k!='seconds'},
        provenance='receiver_model_prediction; never independent environmental evidence',uncertainty='draw dispersion, not a calibrated confidence interval')
    feedback=PacketFeedback(1,packet.identity,agent.agent_id,agent.version,agent.now,tuple(map(float,state)),
                            agent.cfg.replay_horizon_mode,json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False))
    return dict(feedback=asdict(feedback),wire_bytes=feedback.bytes(),rows=rows,draws=draws,initial_revisions=revisions,work=work)


def evaluate_receiver(agent,packet,seed=None):
    if agent.dynamic:raise ValueError('receiver evaluation currently supports static fields')
    started=time.perf_counter();local=copy.deepcopy(agent);state=tuple(local.current_state)
    eligibility=copy.deepcopy(local).assimilate(packet)
    if eligibility['reason']!='eligible':
        reason='packet_'+eligibility['reason']
        rows=aggregate([dict(rows=[dict(horizon=h,valid=False,benefit=None,reason=reason) for h in HORIZONS]) for _ in range(DRAW_COUNT)])
        feedback=PacketFeedback(1,packet.identity,local.agent_id,local.version,local.now,state,local.cfg.replay_horizon_mode,
                                json.dumps(dict(rows=rows,acceptance=eligibility),sort_keys=True,separators=(',',':')))
        return dict(mode='receiver',feedback=asdict(feedback),wire_bytes=feedback.bytes(),rows=rows,draws=[],initial_revisions=[],
                    work=dict(transitions=0,replay_steps=0,identification_work=0,field_draw_work=0,pilot_steps=0,replay_handoffs=0),
                    seconds=time.perf_counter()-started,initial_validation='packet rejected before branch construction')
    if seed is None:seed=int(hashlib.sha256(packet.identity.encode()).hexdigest()[:8],16)
    status=advertise(copy.deepcopy(local),state,packet.sender,[z[:2] for z in packet.anchors])
    result=evaluate_model(local,state,packet,status,seed)
    result.update(mode='receiver',seconds=time.perf_counter()-started,status=asdict(status) if status else None,
                  initial_validation='actual retained measured evidence; future observations hypothetical')
    return result


def evaluate_compact(summary,packet,status,seed):
    started=time.perf_counter()
    local=reconstruct_receiver(summary,packet.sender,summary.timestamp,cross_stage=True)
    result=evaluate_model(local,summary.state,packet,status,seed)
    result.update(mode='compact',seconds=time.perf_counter()-started,
                  initial_validation='synthetic summary-derived fitting and validation')
    return result
