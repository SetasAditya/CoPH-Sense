import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coph_stagewise.additive_routing import MODE,graph_and_connections,portion,price,profiles,records,route_path
from coph_stagewise.attribution_audits import digest
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.crossing_demo import configuration
from coph_stagewise import complementary_reconstruction as cr
from coph_stagewise.decision_packet import costs,shortest
from coph_stagewise.decision_packet_suite import actual_route
from coph_stagewise.decision_sequence import SequenceAdvertisement,advertise,first_branch,merge,predict_sequence,rank_sequences
from coph_stagewise.decision_sequence_suite import enable
from coph_stagewise.receiver_summary import summarize_receiver
from coph_stagewise.route_decision import connectors,graph_data


def context():
    return dict(network=(((0.,0.),(4.,0.),(0.,2.),(4.,2.)),((0,1),(0,2),(2,3),(3,1))),
        terrain_anchors=[(1.7,0.,2.,0.,0.),(2.4,0.,1.,2.,0.)],unknown_risk=.02,
        risk_weight=6.,route_integral_mode=MODE)


def test_partial_integrals_are_additive_reversible_nonnegative():
    tables,_=profiles(context());rng=np.random.default_rng(19001)
    for p in tables:
        full=price(portion(p,0,p['bounds'][-1]),6.)
        for split in rng.uniform(0,p['bounds'][-1],20):
            left=price(portion(p,0,split),6.);right=price(portion(p,split,p['bounds'][-1]),6.)
            assert left>=0 and right>=0
            assert abs(left+right-full)<1e-12
            assert price(portion(p,split,0),6.)==left


def test_backtracking_never_lowers_same_edge_integral():
    ctx=context()
    for x in np.linspace(.05,3.95,40):
        graph,blocked,conn,work=graph_and_connections(np.array([x,0.]),ctx)
        whole=next(c for j,c in graph[0] if j==1)
        assert conn[0]+whole>conn[1]


def test_projected_spur_is_charged_and_clear_path_matches():
    ctx=context();q=np.array([1.,.15]);rows,work=records(q,ctx)
    graph,blocked,conn,_=graph_and_connections(q,ctx)
    assert all(c>=0 for c in conn.values())
    path=route_path([1],q,ctx,work)
    assert np.array_equal(path[0],q) and np.array_equal(path[-1],[4.,0.])
    assert np.min(np.linalg.norm(path-np.array([1.,0.]),axis=1))<1e-12
    assert all('connector' in r for r in rows if r['i']==-1)


def test_graph_continuation_uses_same_integrals_and_cache_invalidates():
    ctx=context();q=np.array([1.,0.])
    graph,blocked,con,work=graph_and_connections(q,ctx)
    assert graph_data(ctx)==(graph,blocked)
    assert connectors(q,ctx)[0]==con
    changed=copy.deepcopy(ctx);changed['terrain_anchors']=[(1.7,0.,5.,0.,0.)]
    assert profiles(changed)[0]!=profiles(ctx)[0]


@pytest.fixture(scope='module')
def acquired():
    torch.set_num_threads(1)
    return cr.acquire(cr.specification('complementary',86200),configuration(500),CommunicationConfig(byte_budget=4000000,ttl=300.))


def test_execution_advertisement_summary_and_branch_isolation(acquired):
    a=enable(copy.deepcopy(acquired['agents'][1]));x=acquired['states'][1]
    before=digest(a);ad=advertise(a,x,'0');decoded=SequenceAdvertisement.decode(ad.wire())
    decoded.validate('0','1',a.now)
    assert asdict(ad)==asdict(decoded)
    assert shortest(ad.ad,costs(ad.ad))['nodes']==actual_route(a,x)
    summary=summarize_receiver(a,x,'0');body=summary.validate('0',a.now)
    assert body['navigation']['route_integral_mode']==MODE
    assert all(s['entry_navigation']['route_integral_mode']==MODE for s in body['stages'])
    assert digest(a)==before
    a.navigation['route_integral_mode']='bogus'
    with pytest.raises(ValueError):summarize_receiver(a,x,'0').validate('0',a.now)


def test_empty_sequence_and_roundtrip_nominal_no_observations(acquired):
    s=copy.deepcopy(acquired['agents'][0]);a=enable(copy.deepcopy(acquired['agents'][1]));x=acquired['states'][1]
    ad=advertise(a,x,'0');before=digest((s,a))
    reference=[np.asarray(r['values']) for r in ad.ad.records]
    result=predict_sequence(ad,[],reference)
    assert result['valid'] and len(result['path'])==33
    assert all(k in result['checkpoints'] for k in ('8','16','32'))
    selected=rank_sequences(s,acquired['states'][0],'1',ad,[],[],None)
    assert selected['choices']['sequence']==[]
    assert digest((s,a))==before
    with pytest.raises(ValueError):SequenceAdvertisement.decode(b'bad')


def test_branch_detection_requires_observed_junction_and_exit():
    nodes=((0.,0.),(2.,0.),(0.,2.),(-2.,0.));edges=((0,1),(0,2),(0,3))
    assert first_branch([[0.,0.],[.1,0.],[.4,0.]],nodes,edges)['edge']==0
    assert not first_branch([[.5,0.],[.8,0.]],nodes,edges)['valid']
    assert not first_branch([[0.,0.],[.1,0.]],nodes,edges)['valid']


def test_two_packets_can_change_ranking_when_each_alone_cannot():
    ctx=context();ctx['terrain_anchors']=[]
    receiver=SimpleNamespace(navigation=ctx,agent_id='1',now=1.,version=1,goal=[4.,0.],cfg=SimpleNamespace(dt=.12,update_every=8))
    ad=advertise(receiver,np.array([0.,0.,0.,0.]),'0').ad
    first=SimpleNamespace(anchors=((1.,0.,.15,0.,0.),))
    second=SimpleNamespace(anchors=((3.,0.,.15,0.,0.),))
    assert shortest(ad,costs(ad))['nodes']==[1]
    for packet in (first,second):
        one=merge(ad,packet);assert shortest(one,costs(one))['nodes']==[1]
    both=merge(merge(ad,first),second)
    assert shortest(both,costs(both))['nodes']==[2,3,1]
