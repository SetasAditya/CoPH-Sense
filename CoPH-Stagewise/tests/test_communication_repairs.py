import copy
from dataclasses import asdict,replace
import numpy as np
import pytest
from coph_stagewise.communication_repairs import *
from coph_stagewise import complementary_reconstruction as cr
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.attribution_audits import digest


@pytest.fixture(scope='module')
def context():
    import torch
    torch.set_num_threads(1)
    return cr.acquire(cr.specification('complementary',79100),configuration(500),CommunicationConfig(byte_budget=4000000,ttl=300.))


def test_lossless_packet_codec(context):
    a=copy.deepcopy(context['agents'][0]);p=a.export(context['states'][0],'1');wire=encode_packet(p)
    assert asdict(decode_packet(wire))==asdict(p)
    assert encode_packet(decode_packet(wire))==wire
    assert len(wire)<p.bytes()
    assert asdict(decode_packet(encode_packet(p,'legacy'),'legacy'))==asdict(p)
    for corrupted in (wire[:30],b'bad!'+wire[4:],wire[:-1],wire+b'extra'):
        with pytest.raises((ValueError,zlib.error)):decode_packet(corrupted)


def test_default_selection_parity_and_advertisement(context):
    a,b=copy.deepcopy(context['agents'][0]),copy.deepcopy(context['agents'][0]);x=context['states'][0]
    p,meta=select_payload(a,x,'1');expected=b.export(x,'1')
    assert asdict(p)==asdict(expected) and meta['status_bytes']==0
    advertisement=advertise_payload(context['agents'][1],context['states'][1],'0')
    receiver_before=digest(context['agents'][1]);q,m=select_payload(b,x,'1',advertisement,'receiver_relevant')
    assert q is not None and m['status_bytes']==advertisement.bytes() and m['selected_stage'] in [s.id for s in b.stages]
    assert digest(context['agents'][1])==receiver_before
    with pytest.raises(ValueError):select_payload(b,x,'1',replace(advertisement,expires=-1),'receiver_relevant')


def test_consistency_check_isolation_and_bias(context):
    s=copy.deepcopy(context['agents'][0]);a=context['agents'][1];p=s.export(context['states'][0],'1');before=digest(a)
    clean=consistency_check(a,p);bad=consistency_check(a,replace(p,anchors=tuple((*v[:2],v[2]+1.,*v[3:]) for v in p.anchors)))
    import json
    json.dumps(clean,allow_nan=False);json.dumps(bad,allow_nan=False)
    assert clean['available'] and bad['available'] and bad['score']>clean['score'] and not bad['accepted']
    assert digest(a)==before


def test_option_validation():
    assert CommunicationRepairOptions()==CommunicationRepairOptions('legacy','rotation','off')
    with pytest.raises(ValueError):CommunicationRepairOptions(wire_mode='unknown')


def test_fixed_rate_compact_transport_normal_receiver_update(context):
    s=copy.deepcopy(context['agents'][0]);a=copy.deepcopy(context['agents'][1]);x=context['states'][1].copy()
    before=digest([(st.readings,st.transitions) for st in a.stages]);physical=a.current_state
    link=FixedRateRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'),byte_budget=4000000)
    sent=link.offer(s,context['states'][0],a,x,8);assert sent['reason']=='sent' and sent['status_bytes']>0
    result=link.deliver_due(a,x)
    assert result['events'][0]['reason']=='eligible' and result['update'] is not None
    assert result['update']['counterfactual_steps']<=1024 and a.current_state==physical
    assert digest([(st.readings,st.transitions) for st in a.stages])==before
    assert link.deliver_due(a,x)['update'] is None
    assert link.offer(s,context['states'][0],a,x,8)['reason']=='duplicate_opportunity'
    assert link.accounting['0']['payload']==sent['payload_bytes']
    assert link.accounting['1']['status']==sent['status_bytes']
    with pytest.raises(ValueError):link.offer(s,context['states'][0],a,x,9)


def test_fixed_rate_byte_ceiling(context):
    s=copy.deepcopy(context['agents'][0]);a=copy.deepcopy(context['agents'][1]);link=FixedRateRepairLink(CommunicationRepairOptions('compact_v1','receiver_relevant'),byte_budget=1)
    result=link.offer(s,context['states'][0],a,context['states'][1],8)
    assert result['reason']=='status_budget' and not link.queue
    assert all(sum(v[k] for k in ('status','payload','acknowledgment'))<=1 for v in link.accounting.values())


def test_saved_state_fault_runner_isolation(context):
    from coph_stagewise.communication_repair_suite import fault_snapshot
    s=copy.deepcopy(context['agents'][0]);a=context['agents'][1];x=context['states'][1]
    packet=s.export(context['states'][0],'1');before=digest(a)
    rows=fault_snapshot((a,x,packet,context['worlds'][1],8),dict(case='unit_fixture',seed=79100,group='unit'))
    assert len(rows)==16 and all(r['isolated'] for r in rows)
    assert {r['mode'] for r in rows}=={'legacy','corrected'} and digest(a)==before


def test_transport_quarantine_does_not_insert_biased_evidence(context):
    s=copy.deepcopy(context['agents'][0]);a=copy.deepcopy(context['agents'][1]);x=context['states'][1]
    p=s.export(context['states'][0],'1');bad=replace(p,anchors=tuple((*v[:2],v[2]+1.,*v[3:]) for v in p.anchors))
    s.export=lambda state,recipient:bad
    stores=digest((a.received,a.remote_terrain,a.remote_cells));link=FixedRateRepairLink(CommunicationRepairOptions('compact_v1','rotation','local_residual'))
    link.offer(s,context['states'][0],a,x,8);result=link.deliver_due(a,x)
    assert result['events'][0]['reason']=='local_consistency_quarantine' and result['update'] is not None
    assert digest((a.received,a.remote_terrain,a.remote_cells))==stores


def test_historical_relevant_selection_exports_nonempty_supported_anchors(context):
    from coph_stagewise.records import Reading
    from coph_stagewise.model import tensor
    s=copy.deepcopy(context['agents'][0]);old=s.stages[0];current=copy.deepcopy(old);current.id=1
    current.center=(old.center[0]+2.1,old.center[1]);current.centers=current.centers+tensor([2.1,0.])
    current.readings=[replace(r,q=(r.q[0]+2.1,r.q[1])) for r in current.readings]
    s.stages=[old,current];x=context['states'][0].copy();x[0]+=2.1
    route=tuple(r.q for r in old.readings if int(r.uid.split(':')[-1])%5)[:16]
    ad=PayloadAdvertisement(1,'1','0',s.now,s.now+.96,0,route,(),())
    packet,meta=select_payload(s,x,'1',ad,'receiver_relevant')
    assert meta['selected_stage']==old.id and packet is not None and packet.stage==old.id
    assert packet.anchors and all(old.supported(a[:2],s.cfg) for a in packet.anchors)
    assert [st.id for st in s.stages]==[old.id,current.id]
