import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coph_stagewise.decision_packet import (
    DecisionAdvertisement, advertise, candidates, costs, packet_prediction, rank, shortest)
from coph_stagewise.decision_packet_suite import actual_route, true_route_cost
from coph_stagewise.attribution_audits import digest
from coph_stagewise import complementary_reconstruction as cr
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.crossing_demo import configuration


def synthetic(weight=6.):
    def row(i,j,p,length,value=0.,support=False):
        return dict(i=i,j=j,points=[p],lengths=[length],values=[value],support=[support])
    return DecisionAdvertisement(1,'1','0',1.,2.,1,2,weight,(
        row(0,1,[1.,0.],1),row(1,2,[2.,0.],1),
        row(0,3,[1.,2.],2),row(3,2,[2.,2.],2),row(-1,0,[0.,0.],0)))


def test_competing_routes_respect_recipient_preferences():
    low=synthetic(.25); high=synthetic(6.)
    packet=SimpleNamespace(anchors=((1.,0.,.15,0.,0.),(2.,0.,.15,0.,0.)))
    assert shortest(low,costs(low,packet_prediction(low,packet)))['nodes']==[0,1,2]
    assert shortest(high,costs(high,packet_prediction(high,packet)))['nodes']==[0,3,2]


def test_prior_replacement_envelope_and_no_remote_support_certification():
    ad=synthetic(); packet=SimpleNamespace(anchors=((1.,0.,.1,0.,0.),))
    changed=packet_prediction(ad,packet)
    assert changed[0][0]==.1 and changed[2][0]==0
    records=copy.deepcopy(ad.records); records[0]['support']=[True];records[0]['values']=[.3]
    assert packet_prediction(replace(ad,records=records),packet)[0][0]==.3
    assert ad.records[0]['support']==[False]


def test_deterministic_ties_and_fixed_path_pricing():
    ad=synthetic(); scores=costs(ad)
    assert shortest(ad,scores)['nodes']==[0,1,2]
    assert true_route_cost(ad,[0,3,2],scores)==4.
    assert true_route_cost(ad,[],scores) is None
    assert true_route_cost(ad,[0,2],scores) is None


def test_advertisement_validation_and_wire():
    ad=synthetic();ad.validate('0','1',1.)
    assert len(ad.wire())>0
    with pytest.raises(ValueError):replace(ad,expires=.5).validate('0','1',1.)
    with pytest.raises(ValueError):replace(ad,risk_weight=float('nan')).validate('0','1',1.)
    with pytest.raises(ValueError):ad.validate('1','0',1.)


@pytest.fixture(scope='module')
def acquisition():
    torch.set_num_threads(1)
    return cr.acquire(cr.specification('complementary',86200),configuration(500),CommunicationConfig(byte_budget=4000000,ttl=300.))


def test_production_route_and_advertised_scores_agree(acquisition):
    receiver=copy.deepcopy(acquisition['agents'][1]); x=acquisition['states'][1]
    ad=advertise(receiver,x,'0')
    assert shortest(ad,costs(ad))['nodes']==actual_route(receiver,x)
    assert not {'readings','transitions','world','validation'}&set(ad.__dict__)


def test_ranking_and_exports_are_isolated_and_use_local_evidence(acquisition):
    sender=copy.deepcopy(acquisition['agents'][0]);receiver=copy.deepcopy(acquisition['agents'][1])
    x=acquisition['states'][0];rx=acquisition['states'][1];before=digest((sender,receiver))
    ad=advertise(receiver,rx,sender.agent_id); packets,work=candidates(sender,x,receiver.agent_id)
    chosen,meta=rank(sender,x,receiver.agent_id,ad,packets)
    assert digest((sender,receiver))==before and packets and work>0
    assert meta['status_bytes']==len(ad.wire())
    assert meta['sender_supported_queries']<=meta['queries']
    if chosen is not None: assert any(p.identity==chosen.identity for p in packets)


def test_fallback_when_no_candidate_and_unsupported_mode(acquisition):
    s=copy.deepcopy(acquisition['agents'][0]);r=copy.deepcopy(acquisition['agents'][1])
    ad=advertise(r,acquisition['states'][1],'0')
    packet,meta=rank(s,acquisition['states'][0],'1',ad,[])
    assert packet is None and meta['reason']=='rotation_fallback'
    r.navigation['route_field_mode']='supported_field'
    with pytest.raises(ValueError):advertise(r,acquisition['states'][1],'0')
