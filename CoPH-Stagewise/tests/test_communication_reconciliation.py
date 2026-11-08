"""Contract tests for the evaluator-only reconciliation harness."""
import copy
import json
import pytest
from coph_stagewise import communication_reconciliation as reconciliation


def test_parity_keeps_decisions_and_work_exact_and_allows_only_float_tolerance():
    original=dict(version=3, accepted=True, counterfactual_steps=12, position=[.1, .2])
    near=copy.deepcopy(original);near['position'][0]+=1e-12
    result=reconciliation.compare_values(original,near)
    assert result['passed'] and not result['bitwise_serialized_equal']
    changed=copy.deepcopy(original);changed['counterfactual_steps']+=1
    assert not reconciliation.compare_values(original,changed)['passed']
    changed=copy.deepcopy(original);changed['accepted']=False
    assert not reconciliation.compare_values(original,changed)['passed']


def test_resume_refuses_changed_manifest_without_overwriting(tmp_path,monkeypatch):
    manifest=dict(schema=1,sources={},runtime={'threads':1},inputs={'archive':'frozen'})
    monkeypatch.setattr(reconciliation,'frozen_manifest',lambda:copy.deepcopy(manifest))
    reconciliation.freeze(tmp_path)
    initial=(tmp_path/'manifest.json').read_bytes()
    assert reconciliation.freeze(tmp_path)==manifest
    manifest['inputs']['archive']='changed'
    with pytest.raises(ValueError,match='Refuse resume'):reconciliation.freeze(tmp_path)
    assert (tmp_path/'manifest.json').read_bytes()==initial


def test_no_communication_reference_reused_and_per_agent_byte_rows():
    agents=[]
    for i in (0,1):
        agents.append(dict(success=True,collision=False,timeout=False,cost=12.+i,time_risk_cost=11.+i,
                           bytes=dict(status=100*i,payload=400,acknowledgment=20)))
    episode=dict(spec=dict(scene='mild',seed=86100),arm='disabled',selector='rotation',agents=agents,events=[],
                 wall_seconds=1.,update_seconds=.1)
    row=reconciliation.episode_row(episode,'key',8000)
    assert row['bytes_team']==dict(status=100,payload=800,acknowledgment=40)
    assert row['bytes_per_agent'][0]['status']==0 and row['bytes_per_agent'][1]['status']==100
    assert row['carrier_time_risk']==12. and row['carrier_cost']==13.
    assert row['team_cost']==25. and row['team_time_risk']==23.
