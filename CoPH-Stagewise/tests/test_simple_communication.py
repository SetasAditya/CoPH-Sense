import copy,json
from dataclasses import asdict,replace
import pytest
import torch
from coph_stagewise.simple_communication import *
from coph_stagewise import complementary_reconstruction as cr
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.attribution_audits import digest


@pytest.fixture(scope='module')
def ctx():
    torch.set_num_threads(1)
    return cr.acquire(cr.specification('complementary',86200),configuration(500),CommunicationConfig(byte_budget=4000000,ttl=300.))


def pair(ctx):return copy.deepcopy(ctx['agents'][0]),copy.deepcopy(ctx['agents'][1])


def test_route_only_acknowledgment_exact_deduplicated_and_expired(ctx):
    s,a=pair(ctx);p=s.export(ctx['states'][0],'1');a.assimilate(p);a.received={}
    before=digest(a);ad=route_advertisement(a,ctx['states'][1],'0')
    assert set(ad.acknowledged_sources)==set(p.sources) and len(ad.acknowledged_sources)==len(set(p.sources))
    assert ad.known_route_anchors and digest(a)==before
    a.now=p.expires+1;expired=route_advertisement(a,ctx['states'][1],'0')
    assert not expired.acknowledged_sources and not expired.known_route_anchors
    with pytest.raises(ValueError):replace(ad,acknowledged_sources=('1:forged',)).validate('0','1',s.now)


def test_compact_full_delta_exact_caches_and_staleness(ctx):
    s,a=pair(ctx);ad=route_advertisement(a,ctx['states'][1],'0');codec=AdvertisementCodec()
    wire,info=codec.prepare(ad);assert info['mode']=='full';decoded=codec.commit(ad,wire,s.now);assert asdict(decoded)==asdict(ad)
    later=replace(ad,timestamp=ad.timestamp+.12,expires=ad.expires+.12)
    delta,info=codec.prepare(later);assert info['mode']=='delta';decoded=codec.commit(later,delta,later.timestamp);assert asdict(decoded)==asdict(later)
    broken=AdvertisementCodec()
    with pytest.raises(ValueError):broken.decode(delta,'1','0',later.timestamp)
    stale,unused=codec.prepare(ad)
    with pytest.raises(ValueError):codec.commit(ad,stale,later.timestamp)
    expired=replace(later,timestamp=later.expires+.01,expires=later.expires+1.)
    full,info=codec.prepare(expired);assert info['mode']=='full'


def test_payload_reservation_falls_back_without_charging_unsent_ad(ctx):
    s,a=pair(ctx);ps=copy.deepcopy(s).export(ctx['states'][0],'1');pa=copy.deepcopy(a).export(ctx['states'][1],'0')
    budget=max(len(encode_packet(ps)),len(encode_packet(pa)))+1
    link=SimpleRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'),byte_budget=budget,simple_options=SimpleOptions(True,True))
    e=link.offer(s,ctx['states'][0],a,ctx['states'][1],8)
    assert e['reason']=='sent' and e['selection']['fallback']=='joint_admission' and e['status_bytes']==0
    assert link.accounting['1']['status']==0 and not link.advertisement_codec.sending


def test_novelty_is_actual_supported_anchor_coverage(ctx):
    s,a=pair(ctx);p=copy.deepcopy(s).export(ctx['states'][0],'1');route=tuple(v[:2] for v in p.anchors)
    ad=replace(route_advertisement(a,ctx['states'][1],'0'),route=route,support=(),known_route_anchors=(),acknowledged_sources=())
    packet,meta=novel_payload(s,ctx['states'][0],'1',ad);assert packet is not None and meta['scores'][0]['novel_covered_queries']>0
    none,meta=novel_payload(s,ctx['states'][0],'1',replace(ad,acknowledged_sources=p.sources));assert none is None
    none,meta=novel_payload(s,ctx['states'][0],'1',replace(ad,known_route_anchors=route));assert none is None


def test_normal_validation_and_no_hidden_measurements_in_status(ctx):
    s,a=pair(ctx);original=digest([(st.readings,st.transitions) for st in a.stages]);physical=a.current_state
    link=SimpleRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'),simple_options=SimpleOptions(True,True))
    e=link.offer(s,ctx['states'][0],a,ctx['states'][1],8);r=link.deliver_due(a,ctx['states'][1])
    assert e['reason']=='sent' and r['events'][0]['reason']=='eligible' and r['update'] is not None
    assert a.current_state==physical and digest([(st.readings,st.transitions) for st in a.stages])==original
    assert not link.deliver_due(a,ctx['states'][1])['events']
    ad=asdict(route_advertisement(a,ctx['states'][1],'0'))
    assert not {'readings','transitions','values','weights','validation','world','truth'}&set(ad)


def test_default_link_parity(ctx):
    s,a=pair(ctx);s2,a2=pair(ctx)
    old=FixedRateRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'))
    new=SimpleRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'))
    x=old.offer(s,ctx['states'][0],a,ctx['states'][1],8);y=new.offer(s2,ctx['states'][0],a2,ctx['states'][1],8)
    assert x==y and old.queue==new.queue


def test_ack_only_does_not_enable_reserving_admission(ctx):
    s,a=pair(ctx);ps=copy.deepcopy(s).export(ctx['states'][0],'1');pa=copy.deepcopy(a).export(ctx['states'][1],'0')
    budget=max(len(encode_packet(ps)),len(encode_packet(pa)))+1
    link=SimpleRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'),byte_budget=budget,simple_options=SimpleOptions(True,False))
    e=link.offer(s,ctx['states'][0],a,ctx['states'][1],8)
    assert e['reason']=='status_budget' and not link.queue and link.accounting['1']['status']==0
