import copy
import numpy as np
import pytest
from coph_stagewise import campaign_communication as c
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise import complementary_reconstruction as cr
from coph_stagewise.attribution_audits import digest


def test_unselected_layout_protocol_contains_every_draw():
    p=c.protocol('reciprocity')
    assert len(p['specs'])==8
    assert all(r['selected'] for r in p['catalogue'])
    assert sorted({r['doors'] for r in p['catalogue']})==[0,1,2]
    assert [s['seed'] for s in p['specs']]==list(c.LAYOUT_SEEDS)
    assert p['specs']==c.protocol('reciprocity')['specs']


def test_manifest_immutable_resume(tmp_path):
    frozen=c.freeze(tmp_path,'representation')
    assert c.freeze(tmp_path,'representation')==frozen
    with pytest.raises(ValueError):c.freeze(tmp_path,'faults')


def test_costmap_supported_grid_and_fault_isolation():
    import torch
    torch.set_num_threads(1)
    s=cr.specification('complementary',89600)
    ctx=cr.acquire(s,c.configuration(500),CommunicationConfig(byte_budget=4000000,ttl=300.,anchors=12))
    sender=copy.deepcopy(ctx['agents'][0]);receiver=ctx['agents'][1];before=digest(receiver)
    packet=sender.export(ctx['states'][0],'1');assert packet is not None
    # Directly exercise scalar-grid conversion on the real retained local history.
    sender.__class__=c.LedgerCostmap
    m=sender.export(ctx['states'][0],'1');assert m is not None
    assert len(m.anchors)<=12 and all(a[3:]==(0.,0.) for a in m.anchors)
    assert all(any(st.supported(a[:2],sender.cfg) for st in sender.stages) for a in m.anchors)
    for name,packets in c.fault_variants(packet,receiver).items():
        branch=copy.deepcopy(receiver);events=[branch.assimilate(p) for p in packets]
        if name=='duplicate':assert events[-1]['reason']=='duplicate_or_reordered'
        if name=='echo':assert events[-1]['reason']=='returned_evidence'
        if name=='stale':assert events[-1]['reason']=='invalid_age'
        if name=='correlated':assert len(branch.received)<=1
    assert digest(receiver)==before
