"""Opt-in acknowledgment, advertisement coding and payload-reserving admission.

Uses only explicit receiver advertisements and locally measured sender models.
The environmental packet schema and all existing defaults remain unchanged.
"""
import copy,hashlib,json,time,zlib
from dataclasses import dataclass,asdict,replace
import numpy as np
from .communication_repairs import (PayloadAdvertisement,FixedRateRepairLink,CommunicationRepairOptions,
    advertise_payload,select_payload,encode_packet,canonical,MAX_DECODED_BYTES)


@dataclass(frozen=True)
class SimpleOptions:
    ack_route_used:bool=False
    compact_status:bool=False


@dataclass(frozen=True)
class RouteAdvertisement(PayloadAdvertisement):
    known_route_anchors:tuple=()
    interface_schema:int=2

    def validate(self,sender,recipient,now):
        super().validate(sender,recipient,now)
        if self.interface_schema!=2 or len(self.known_route_anchors)>128:raise ValueError('route_advertisement_bounds')
        if any(len(q)!=2 for q in self.known_route_anchors) or not np.isfinite(np.asarray(self.known_route_anchors)).all():raise ValueError('route_advertisement_shape')
        if any(not x.startswith(str(sender)+':') for x in self.acknowledged_sources):raise ValueError('acknowledgment_provenance')


def route_advertisement(receiver,state,sender):
    base=advertise_payload(receiver,state,sender);now=receiver.now;sources=set();points=[]
    for packet in receiver.received.values():
        if packet.expires>=now and packet.sender==str(sender):sources.update(packet.sources)
    live_ids=set()
    for cell in getattr(receiver,'remote_cells',{}).values():
        if len(cell)>=4 and cell[0]>=now and all(str(s).startswith(str(sender)+':') for s in cell[3]):
            points.append(tuple(cell[1][:2]));sources.update(cell[3]);live_ids.add(cell[2])
    for packet in getattr(receiver,'remote_terrain',{}).values():
        if packet.expires>=now and packet.sender==str(sender) and packet.identity in live_ids:sources.update(packet.sources)
    # Bounded exact subsets; omitted IDs are not invented acknowledgments.
    points=tuple(dict.fromkeys(points))[-128:];sources=tuple(sorted(sources))[-2048:]
    return RouteAdvertisement(**dict(asdict(base),acknowledged_sources=sources),known_route_anchors=points)


def novel_payload(sender,state,recipient,advertisement):
    advertisement.validate(sender.agent_id,recipient,sender.now)
    route=np.asarray(advertisement.route,float).reshape(-1,2)
    known=np.asarray(tuple(advertisement.support)+tuple(advertisement.known_route_anchors),float).reshape(-1,2)
    unseen=np.ones(len(route),bool)
    if len(known) and len(route):unseen=np.linalg.norm(route[:,None]-known[None],axis=-1).min(1)>.65
    acknowledged=set(advertisement.acknowledged_sources);candidates=[];work=0
    for k,stage in enumerate(sender.stages):
        branch=copy.deepcopy(sender);branch._ledger_turn=len(branch.stages)-1-k;before=branch.export_work
        packet=branch.export(state,recipient);work+=branch.export_work-before
        if packet is None:continue
        anchors=np.asarray([a[:2] for a in packet.anchors]);fresh=bool(set(packet.sources)-acknowledged)
        covered=np.zeros(len(route),bool)
        if len(route):covered=(np.linalg.norm(route[:,None]-anchors[None],axis=-1).min(1)<=.65)&np.array([stage.supported(q,sender.cfg) for q in route])
        novel=covered&unseen if fresh else np.zeros(len(route),bool)
        n=int(novel.sum());first=int(np.flatnonzero(novel)[0]) if n else len(route)
        novel_anchors=int((np.linalg.norm(anchors[:,None]-known[None],axis=-1).min(1)>.65).sum()) if len(known) else len(anchors)
        candidates.append((packet,(n,novel_anchors,-first,stage.id),dict(stage=stage.id,novel_covered_queries=n,covered_queries=int(covered.sum()),new_source_ids=len(set(packet.sources)-acknowledged),first_novel_query=first)))
    sender.export_work+=work
    usable=[c for c in candidates if c[1][0]>0]
    if not usable:return None,dict(reason='no_novel_covered_query',scores=[c[2] for c in candidates])
    packet,score,_=max(usable,key=lambda c:c[1])
    return packet,dict(reason='novel_actual_anchor_coverage',selected_stage=packet.stage,scores=[c[2] for c in candidates])


def _body(ad):return json.loads(canonical(asdict(ad)))


def _digest(body):return hashlib.sha256(canonical(body)).hexdigest()


class AdvertisementCodec:
    """Separate sending/receiving caches; only transmitted decoded state commits."""
    def __init__(self):self.sending={};self.receiving={}

    def prepare(self,ad):
        body=_body(ad);key=(ad.receiver,ad.recipient);old=self.sending.get(key)
        full=dict(schema=1,mode='full',digest=_digest(body),body=body)
        wire=b'CAD1'+zlib.compress(canonical(full),9);mode='full'
        if old is not None and old['expires']>=ad.timestamp and body['timestamp']>=old['timestamp'] and body['version']>=old['version']:
            delta=dict(schema=1,mode='delta',base=_digest(old),digest=_digest(body),changes={k:v for k,v in body.items() if old.get(k)!=v},removed=[k for k in old if k not in body])
            candidate=b'CAD1'+zlib.compress(canonical(delta),9)
            if len(candidate)<len(wire):wire=candidate;mode='delta'
        return wire,dict(mode=mode,raw_bytes=len(canonical(body)),wire_bytes=len(wire))

    def decode(self,wire,receiver,sender,now):
        if wire[:4]!=b'CAD1' or len(wire)>MAX_DECODED_BYTES:raise ValueError('advertisement_frame')
        obj=zlib.decompressobj();raw=obj.decompress(wire[4:],MAX_DECODED_BYTES+1)
        if len(raw)>MAX_DECODED_BYTES or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('advertisement_size')
        envelope=json.loads(raw);key=(str(receiver),str(sender))
        if envelope.get('schema')!=1:raise ValueError('advertisement_schema')
        if envelope['mode']=='full':body=envelope['body']
        elif envelope['mode']=='delta':
            old=self.receiving.get(key)
            if old is None or _digest(old)!=envelope['base']:raise ValueError('advertisement_cache_miss')
            body=dict(old);body.update(envelope['changes'])
            for k in envelope.get('removed',[]):body.pop(k,None)
        else:raise ValueError('advertisement_mode')
        if _digest(body)!=envelope['digest']:raise ValueError('advertisement_digest')
        old=self.receiving.get(key)
        if old and (body['timestamp']<old['timestamp'] or body['version']<old['version']):raise ValueError('advertisement_reordered')
        for field in ('route','support','known_route_anchors'):
            if field in body:body[field]=tuple(tuple(q) for q in body[field])
        body['acknowledged_sources']=tuple(body['acknowledged_sources'])
        ad=(RouteAdvertisement if 'interface_schema' in body else PayloadAdvertisement)(**body)
        ad.validate(sender,receiver,now)
        return ad

    def commit(self,ad,wire,now):
        decoded=self.decode(wire,ad.receiver,ad.recipient,now)
        key=(ad.receiver,ad.recipient);body=_body(decoded)
        self.sending[key]=body;self.receiving[key]=body
        return decoded


class SimpleRepairLink(FixedRateRepairLink):
    def __init__(self,*args,simple_options=None,**kwargs):
        super().__init__(*args,**kwargs);self.simple_options=simple_options or SimpleOptions();self.advertisement_codec=AdvertisementCodec();self.total_offer_seconds=0.

    def _remaining(self,agent):return self.byte_budget-sum(self._ledger(agent)[k] for k in ('status','payload','acknowledgment'))

    def offer(self,sender,state,receiver,receiver_state,tick):
        if self.options.payload_selection=='rotation' or not(self.simple_options.ack_route_used or self.simple_options.compact_status):return super().offer(sender,state,receiver,receiver_state,tick)
        began=time.perf_counter();receiver_processing=self._ledger(receiver.agent_id)['processing_seconds']
        try:return self._offer_fixed(sender,state,receiver,receiver_state,tick)
        finally:
            elapsed=time.perf_counter()-began;receiver_elapsed=self._ledger(receiver.agent_id)['processing_seconds']-receiver_processing
            self._ledger(sender.agent_id)['processing_seconds']+=max(0.,elapsed-receiver_elapsed);self.total_offer_seconds+=elapsed

    def _offer_fixed(self,sender,state,receiver,receiver_state,tick):
        options=self.simple_options
        if self.options.payload_selection=='rotation' or not(options.ack_route_used or options.compact_status):return super().offer(sender,state,receiver,receiver_state,tick)
        if tick%sender.cfg.update_every:raise ValueError('not_fixed_rate_opportunity')
        if tick<=self.opportunities.get(sender.agent_id,-1):return dict(reason='duplicate_opportunity',tick=tick)
        self.opportunities[sender.agent_id]=tick;start=time.perf_counter()
        # Local previews reserve each endpoint's own normal payload. The sender
        # receives no advertisement contents until admission and wire charging.
        own_reserve=reserve=0
        if options.compact_status:
            preview=copy.deepcopy(sender);rotation=preview.export(state,receiver.agent_id)
            sender.export_work=preview.export_work
            reserve=len(encode_packet(rotation,self.options.wire_mode)) if rotation else 0
        receiver_began=time.perf_counter()
        if options.compact_status:
            peer_preview=copy.deepcopy(receiver);peer_packet=peer_preview.export(receiver_state,sender.agent_id)
            receiver.export_work=peer_preview.export_work
            own_reserve=len(encode_packet(peer_packet,self.options.wire_mode)) if peer_packet else 0
        ad=route_advertisement(receiver,receiver_state,sender.agent_id) if options.ack_route_used else advertise_payload(receiver,receiver_state,sender.agent_id)
        wire,codec=self.advertisement_codec.prepare(ad) if options.compact_status else (canonical(asdict(ad)),dict(mode='json',raw_bytes=ad.bytes(),wire_bytes=ad.bytes()))
        self._ledger(receiver.agent_id)['processing_seconds']+=time.perf_counter()-receiver_began
        admitted=self._remaining(receiver.agent_id)>=len(wire)+own_reserve and self._remaining(sender.agent_id)>=reserve
        if not options.compact_status and not admitted:return dict(reason='status_budget',tick=tick)
        status_bytes=0;fallback=None
        if admitted:
            assert self._charge(receiver.agent_id,'status',len(wire));status_bytes=len(wire)
            if options.compact_status:decoded=self.advertisement_codec.commit(ad,wire,sender.now)
            else:
                body=json.loads(wire)
                for field in ('route','support','known_route_anchors'):
                    if field in body:body[field]=tuple(tuple(q) for q in body[field])
                body['acknowledged_sources']=tuple(body['acknowledged_sources']);decoded=(RouteAdvertisement if options.ack_route_used else PayloadAdvertisement)(**body);decoded.validate(sender.agent_id,receiver.agent_id,sender.now)
            if options.ack_route_used:packet,selection=novel_payload(sender,state,receiver.agent_id,decoded)
            else:packet,selection=select_payload(sender,state,receiver.agent_id,decoded,'receiver_relevant')
            if packet is None:fallback='no_novel_payload'
            elif options.compact_status and len(encode_packet(packet,self.options.wire_mode))>self._remaining(sender.agent_id):fallback='selected_payload_budget'
        else:packet=None;selection={};fallback='joint_admission'
        if fallback:
            # Re-run the actual normal export so its rotation cursor advances.
            packet=sender.export(state,receiver.agent_id)
            selection=dict(reason='normal_rotation_fallback',fallback=fallback)
        if packet is None:return dict(reason='no_supported_packet',tick=tick,status_bytes=status_bytes,selection=selection)
        payload=encode_packet(packet,self.options.wire_mode)
        if not self._charge(sender.agent_id,'payload',len(payload)):
            return dict(reason='payload_budget',tick=tick,status_bytes=status_bytes,selection=selection)
        self.queue.append((sender.now+self.delay_seconds,receiver.agent_id,payload))
        event=dict(reason='sent',sender=sender.agent_id,receiver=receiver.agent_id,packet=packet.identity,tick=tick,time=sender.now,release=sender.now+self.delay_seconds,
            payload_bytes=len(payload),status_bytes=status_bytes,selection=dict(selection,advertisement_codec=codec,admission=admitted,peer_payload_reserve=own_reserve,sender_payload_reserve=reserve))
        self.events.append(event);return event
