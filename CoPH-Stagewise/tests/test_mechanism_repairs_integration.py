"""Independent integration checks; no runtime changes or new mission fixtures."""
import copy
from dataclasses import replace
import json
import numpy as np
import torch
from coph_stagewise.records import Reading,new_stage
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.campaign_reconstruction import NODES,EDGES
from coph_stagewise.route_decision import RouteDecisionLearner,RouteDecisionOptions,evaluate_routes,conditioned_path,connectors
from coph_stagewise.receiver_summary import summarize_receiver,reconstruct_receiver
from coph_stagewise.model import tensor
from coph_stagewise.stage_prediction import FrozenStages
from coph_stagewise.maze_visuals import landscape


def combined():
    cfg=replace(configuration(60),stage_blend_mode='smooth',stage_width=12.,stage_height=6.,route_field_mode='supported_field')
    options=RouteDecisionOptions(mode='prefix',starts='historical',candidates=2)
    a=RouteDecisionLearner(cfg,np.array([.3,0.,0.,0.]),[5.,0.],agent_id='0',mode='evidence',network=(NODES,EDGES),
        route_options=options,unknown_risk=.02,receiver_fit_mode='precision_bounded',receiver_validation_mode='relative_local',
        receiver_solver_mode='active_set',receiver_search_mode='diagnostic',receiver_localization_mode='compact')
    a.stages.append(new_stage(1,(2.1,0.),(1.06,0.,.2,0.),a.goal,cfg))
    for k,stage in enumerate(a.stages):
        stage.version=1;stage.weights=tensor(np.linspace(-1.,1.,16)+k*.2)
        stage.readings=[]
        for edge,(i,j) in enumerate(EDGES):
            for m,t in enumerate(np.linspace(0,1,9)):
                q=np.asarray(NODES[i])*(1-t)+np.asarray(NODES[j])*t
                stage.readings.append(Reading('0:integration:%d'%(edge*50+m*5+1),0.,tuple(q),.02))
        stage.local_residual=dict(schema=1,radius=.65,centers=[[1.,0.]],coefficients=[.3],packets=['1:peer'],source_ids=['1:measured:1'])
        stage.entry_navigation=copy.deepcopy(a.navigation)
    a.version=1
    from coph_stagewise.route_field import sync
    sync(a,publish=True)
    return a


def test_smooth_compact_prefix_snapshot_isolation_and_flow():
    torch.set_num_threads(1);a=combined();x=np.array([.3,0.,0.,0.]);measurements=copy.deepcopy([s.readings for s in a.stages]);navigation=copy.deepcopy(a.navigation)
    result=evaluate_routes(a,x,a.route_options)
    assert result['reason']=='supported_route_choice'
    assert len(result['starts'])==2 and all(s['horizon']==8 for s in result['starts'])
    assert result['flow_work']<=result['reserved_flow']<=1024
    assert [s.readings for s in a.stages]==measurements and a.navigation==navigation
    frozen_states=[s['state'] for s in result['starts']]
    for candidate in result['candidates']:
        assert candidate['valid'] and len(candidate['starts'])==2
        assert [s['states'][0] for s in candidate['starts']]==frozen_states
        assert all(s['horizon']==8 for s in candidate['starts'])
    a.navigation.update(route_candidate_schema=1,route_candidate_nodes=result['selected'])
    first=next(c for c in result['candidates'] if c['nodes']==result['selected'])['starts'][0]
    actual,nominal=a.action(x)
    np.testing.assert_allclose(actual,first['controls'][0]['applied'],atol=1e-12,rtol=0)
    model=a.control_model();H,F,supported=landscape(model,np.array([x[:2]]),x[2:],a.cfg)
    assert supported[0]
    np.testing.assert_allclose(F[0]-1.4*x[2:],nominal,atol=1e-12,rtol=0)
    assert abs(float(H[0])-float(a.shaping_energy(x)))<1e-12


def test_all_modes_receiver_summary_roundtrip_and_controls():
    torch.set_num_threads(1);a=combined();x=np.array([.3,0.,0.,0.]);decision=evaluate_routes(a,x,a.route_options)
    a.navigation.update(route_candidate_schema=1,route_candidate_nodes=decision['selected'])
    summary=summarize_receiver(a,x,'1');body=summary.validate('1',a.now)
    assert body['config']['stage_blend_mode']=='smooth'
    assert body['receiver_localization_mode']=='compact'
    assert body['route_decision_options']['mode']=='prefix'
    restored=reconstruct_receiver(summary,'1',a.now,cross_stage=True)
    assert restored.route_options==a.route_options and restored.cfg.stage_blend_mode=='smooth'
    assert restored.receiver_localization_mode=='compact'
    assert restored.navigation['route_candidate_nodes']==a.navigation['route_candidate_nodes']
    assert all(getattr(s,'local_residual')==getattr(t,'local_residual') for s,t in zip(a.stages,restored.stages))
    u,nominal=a.action(x);other,other_nominal=restored.action(x)
    np.testing.assert_allclose(u,other,atol=1e-12,rtol=0);np.testing.assert_allclose(nominal,other_nominal,atol=1e-12,rtol=0)
    assert restored.hypothetical and 'not measured' in restored.validation_interpretation


def test_forced_route_is_best_eligible_suffix_not_strict_first_connector():
    # Exposes the actual API semantics for review: a candidate can join at a
    # later production-eligible node. It does not force its first listed node.
    context=dict(network=(((0.,0.),(0.,1.),(4.,1.),(4.,0.)),((0,1),(1,2),(2,3),(0,3))),
        terrain_anchors=[],reported_geometry=[],unknown_risk=.02)
    q=np.array([0.,0.]);eligible,_=connectors(q,context)
    assert 3 in eligible
    suffix,cost,_=conditioned_path([0,1,2,3],q,context)
    assert suffix==[3] and abs(cost-4*(1+4*.02))<1e-12
