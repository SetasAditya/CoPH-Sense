"""Opt-in wire coding, advertised payload selection and local consistency checks.

No evaluator truth, episode loaders or simulator interfaces are imported here.
Existing packet and learner defaults are unchanged.
"""
import copy
import hashlib
import json
import struct
import time
import zlib
from dataclasses import dataclass, asdict, replace
import numpy as np
from .cooperative import Packet, CooperativeLearner, CONVENTION
from .maze_learning import MazePacket
from .model import risk, tensor

MAGIC=b'CPH1'
MAX_WIRE_BYTES=2_000_000
MAX_DECODED_BYTES=8_000_000


@dataclass(frozen=True)
class CommunicationRepairOptions:
    wire_mode:str='legacy'
    payload_selection:str='rotation'
    consistency_mode:str='off'

    def __post_init__(self):
        if self.wire_mode not in ('legacy','compact_v1'):raise ValueError('wire_mode')
        if self.payload_selection not in ('rotation','receiver_relevant'):raise ValueError('payload_selection')
        if self.consistency_mode not in ('off','local_residual'):raise ValueError('consistency_mode')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def encode_packet(packet,mode='compact_v1'):
    """Lossless canonical JSON compressed into a versioned binary transport frame.

    This is a codec, not evidence removal or lower-precision floating arithmetic.
    The full transmitted support/covariance/provenance is retained and charged.
    """
    raw=canonical(asdict(packet))
    if mode=='legacy':return raw
    if mode!='compact_v1':raise ValueError('wire_mode')
    if len(raw)>MAX_DECODED_BYTES:raise ValueError('decoded_size_limit')
    compressed=zlib.compress(raw,level=9)
    frame=MAGIC+struct.pack('>BI',int(isinstance(packet,MazePacket)),len(raw))+hashlib.sha256(raw).digest()+compressed
    if len(frame)>MAX_WIRE_BYTES:raise ValueError('wire_size_limit')
    return frame


def decode_packet(frame,mode='compact_v1'):
    if mode=='legacy':raw=frame;kind=None
    elif mode=='compact_v1':
        if len(frame)>MAX_WIRE_BYTES or len(frame)<41 or frame[:4]!=MAGIC:raise ValueError('invalid_frame')
        kind,length=struct.unpack('>BI',frame[4:9])
        if kind not in (0,1) or length>MAX_DECODED_BYTES:raise ValueError('decoded_size_limit')
        obj=zlib.decompressobj();raw=obj.decompress(frame[41:],MAX_DECODED_BYTES+1)
        if len(raw)!=length or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('invalid_compressed_frame')
        if hashlib.sha256(raw).digest()!=frame[9:41]:raise ValueError('frame_digest')
    else:raise ValueError('wire_mode')
    if len(raw)>MAX_DECODED_BYTES:raise ValueError('decoded_size_limit')
    body=json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite_wire')))
    def tuples(v):return tuple(tuples(z) for z in v) if isinstance(v,list) else v
    body={k:tuples(v) for k,v in body.items()}
    maze='geometry_schema' in body
    if kind is not None and bool(kind)!=maze:raise ValueError('packet_kind')
    return (MazePacket if maze else Packet)(**body)


@dataclass(frozen=True)
class PayloadAdvertisement:
    schema:int
    receiver:str
    recipient:str
    timestamp:float
    expires:float
    version:int
    route:tuple
    support:tuple
    acknowledged_sources:tuple
    convention:str=CONVENTION

    def bytes(self):return len(canonical(asdict(self)))

    def validate(self,sender,recipient,now):
        if self.schema!=1 or self.convention!=CONVENTION:raise ValueError('advertisement_convention')
        if self.receiver!=str(recipient) or self.recipient!=str(sender) or self.receiver==self.recipient:raise ValueError('advertisement_identity')
        if self.timestamp>now+1e-9 or self.expires<now or self.expires<self.timestamp:raise ValueError('advertisement_age')
        if len(self.route)>64 or len(self.support)>128 or len(self.acknowledged_sources)>2048:raise ValueError('advertisement_bounds')
        if not np.isfinite([self.timestamp,self.expires,*np.asarray(self.route).reshape(-1),*np.asarray(self.support).reshape(-1)]).all():raise ValueError('advertisement_nonfinite')
        if any(len(q)!=2 for q in self.route+self.support):raise ValueError('advertisement_shape')


def advertise_payload(receiver,state,sender):
    path=receiver.navigation.get('path',[])
    route=[tuple(map(float,state[:2]))]+[tuple(map(float,q)) for q in path]
    if len(route)>64:route=[route[k] for k in np.linspace(0,len(route)-1,64).round().astype(int)]
    support=list(dict.fromkeys(r.q for s in receiver.stages for r in s.readings if int(r.uid.split(':')[-1])%5))
    if len(support)>128:support=[support[k] for k in np.linspace(0,len(support)-1,128).round().astype(int)]
    acknowledged=sorted({v for m in receiver.received.values() for v in m.sources})[-2048:]
    return PayloadAdvertisement(1,receiver.agent_id,str(sender),receiver.now,receiver.now+receiver.cfg.dt*receiver.cfg.update_every,
                                receiver.version,tuple(route),tuple(support),tuple(acknowledged))


def select_payload(sender,state,recipient,advertisement=None,mode='rotation'):
    """Called at existing fixed opportunities; only stage/payload selection changes.

    Returns payload and charged-advertisement metadata. The caller must charge
    returned status bytes against the receiver's transmission budget.
    """
    if mode=='rotation':return sender.export(state,recipient),dict(mode=mode,status_bytes=0,reason='baseline_rotation')
    if mode!='receiver_relevant':raise ValueError('payload_selection')
    if advertisement is None:return sender.export(state,recipient),dict(mode=mode,status_bytes=0,reason='missing_advertisement_rotation_fallback')
    advertisement.validate(sender.agent_id,recipient,sender.now)
    route=np.asarray(advertisement.route,float).reshape(-1,2);support=np.asarray(advertisement.support,float).reshape(-1,2)
    if not len(route):return sender.export(state,recipient),dict(mode=mode,status_bytes=advertisement.bytes(),reason='empty_route_rotation_fallback')
    acknowledged=set(advertisement.acknowledged_sources);scores=[]
    for s in sender.stages:
        local=[r for r in s.readings if int(r.uid.split(':')[-1])%5]
        supported=np.array([s.supported(q,sender.cfg) for q in route]);missing=np.ones(len(route),bool)
        if len(support):missing=np.linalg.norm(route[:,None]-support[None],axis=-1).min(1)>.65
        new_sources=sum(r.uid not in acknowledged for r in local)
        distance=min((np.linalg.norm(np.asarray(r.q)-q) for r in local for q in route),default=float('inf'))
        score=float(np.sum(supported & missing)+.25*np.sum(supported))/(1+distance) if new_sources else 0.
        scores.append(dict(stage=s.id,score=score,supported_route_queries=int(supported.sum()),novel_supported_queries=int(np.sum(supported&missing)),new_sources=new_sources))
    chosen=max(zip(sender.stages,scores),key=lambda z:(z[1]['score'],z[0].id))[0]
    # Preserve stage order: LedgerExport recognizes historical stages by their
    # position in the original ledger and uses their own measured locations.
    # Reordering an old stage to the end incorrectly invokes current-position
    # anchor selection and can produce an empty packet far from that stage.
    branch=copy.deepcopy(sender)
    branch._ledger_turn=len(branch.stages)-1-next(k for k,s in enumerate(branch.stages) if s.id==chosen.id)
    before=branch.export_work;packet=branch.export(state,recipient)
    sender.export_work+=branch.export_work-before
    return packet,dict(mode=mode,status_bytes=advertisement.bytes(),reason='advertised_route_support',selected_stage=chosen.id,
                       advertisement_version=advertisement.version,advertisement_timestamp=advertisement.timestamp,scores=scores)


def consistency_check(receiver,packet,threshold=4.):
    """Optional local residual quarantine; not a calibrated probabilistic test.

    It checks only anchors supported by local measured evidence against a local-
    only proposal, using reserved local RMSE and known sensor noise as scales.
    Unsupported anchors remain unchecked. No covariance-posterior claim is made.
    """
    branch=copy.deepcopy(receiver)
    weights,_=super(CooperativeLearner,branch)._field_fit()
    checks=[]
    for s,w in zip(branch.stages,weights):
        reserved=[r for r in s.readings if int(r.uid.split(':')[-1])%5==0]
        if not reserved:continue
        fit=risk(tensor([r.q for r in reserved]),w,s.centers,branch.cfg.rbf_width).detach().numpy()
        mse=float(np.mean((fit-np.array([r.value for r in reserved]))**2));scale=np.sqrt(mse+.003**2)
        for k,a in enumerate(packet.anchors):
            if s.supported(a[:2],branch.cfg):
                estimate=float(risk(tensor(a[:2]),w,s.centers,branch.cfg.rbf_width))
                checks.append(dict(stage=s.id,anchor=k,standardized_residual=float(abs(a[2]-estimate)/scale),local_estimate=estimate,scale=float(scale)))
    score=max((r['standardized_residual'] for r in checks),default=None)
    return dict(available=bool(checks),accepted=bool(score is None or score<=threshold),score=score,threshold=threshold,
                supported_anchor_checks=len(checks),checks=checks,interpretation='heuristic local consistency, not calibrated confidence')


class FixedRateRepairLink:
    """Opt-in bounded transport through existing assimilation and update methods.

    Call offer only at the unchanged sender update interval, after ordinary
    sensing; call deliver_due at receiver control boundaries. Payload selection
    never changes opportunity times. Status, wire payload and acknowledgment
    are charged to the endpoint that transmits them, independently.
    """
    def __init__(self,options=None,byte_budget=4000000,delay_seconds=0.):
        self.options=options or CommunicationRepairOptions()
        if byte_budget<1 or delay_seconds<0:raise ValueError('channel_limits')
        self.byte_budget=byte_budget;self.delay_seconds=delay_seconds
        self.accounting={};self.queue=[];self.opportunities={};self.events=[]

    def _ledger(self,agent):
        return self.accounting.setdefault(str(agent),dict(status=0,payload=0,acknowledgment=0,processing_seconds=0.))

    def _charge(self,agent,component,size):
        ledger=self._ledger(agent)
        if sum(ledger[k] for k in ('status','payload','acknowledgment'))+size>self.byte_budget:return False
        ledger[component]+=size;return True

    def offer(self,sender,state,receiver,receiver_state,tick):
        if tick%sender.cfg.update_every:raise ValueError('not_fixed_rate_opportunity')
        if tick<=self.opportunities.get(sender.agent_id,-1):return dict(reason='duplicate_opportunity',tick=tick)
        self.opportunities[sender.agent_id]=tick;started=time.perf_counter()
        advertisement=None;status_bytes=0
        if self.options.payload_selection=='receiver_relevant':
            advertisement=advertise_payload(receiver,receiver_state,sender.agent_id);status_bytes=advertisement.bytes()
            if not self._charge(receiver.agent_id,'status',status_bytes):return dict(reason='status_budget',tick=tick)
        packet,selection=select_payload(sender,state,receiver.agent_id,advertisement,self.options.payload_selection)
        if packet is None:return dict(reason='no_supported_packet',tick=tick,status_bytes=status_bytes)
        wire=encode_packet(packet,self.options.wire_mode)
        if not self._charge(sender.agent_id,'payload',len(wire)):
            event=dict(reason='payload_budget',tick=tick,status_bytes=status_bytes);self.events.append(event);return event
        self.queue.append((sender.now+self.delay_seconds,receiver.agent_id,wire))
        self._ledger(sender.agent_id)['processing_seconds']+=time.perf_counter()-started
        event=dict(reason='sent',sender=sender.agent_id,receiver=receiver.agent_id,packet=packet.identity,tick=tick,
                   time=sender.now,release=sender.now+self.delay_seconds,payload_bytes=len(wire),status_bytes=status_bytes,selection=selection)
        self.events.append(event);return event

    def deliver_due(self,receiver,state,commit=True):
        started=time.perf_counter();now=receiver.now
        due=[z for z in self.queue if z[0]<=now+1e-9 and z[1]==receiver.agent_id]
        self.queue=[z for z in self.queue if not(z[0]<=now+1e-9 and z[1]==receiver.agent_id)]
        events=[];physical=tuple(receiver.current_state);position=np.asarray(state).copy()
        for release,_,wire in due:
            packet=decode_packet(wire,self.options.wire_mode)
            check=None
            if self.options.consistency_mode=='local_residual':
                trial=copy.deepcopy(receiver);eligibility=trial.assimilate(packet)
                if eligibility['reason']=='eligible':check=consistency_check(receiver,packet)
            if check is not None and not check['accepted']:
                event=dict(reason='local_consistency_quarantine',packet=packet.identity,sender=packet.sender,
                           recipient=receiver.agent_id,time=now,consistency=check)
            else:event=receiver.assimilate(packet)
            ack=canonical(event);ack_sent=self._charge(receiver.agent_id,'acknowledgment',len(ack))
            events.append(dict(event,wire_bytes=len(wire),acknowledgment_bytes=len(ack) if ack_sent else 0,
                               acknowledgment_unsent=not ack_sent,release=release))
        update=receiver.update(np.asarray(state).copy()) if due and commit else None
        assert tuple(receiver.current_state)==physical and np.array_equal(position,state),'receipt changed physical state'
        self._ledger(receiver.agent_id)['processing_seconds']+=time.perf_counter()-started
        return dict(events=events,update=update,exactly_once=True,physical_state_unchanged=True)
