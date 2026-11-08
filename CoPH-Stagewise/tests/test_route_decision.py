import copy
import unittest
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.route_decision import RouteDecisionOptions,RouteDecisionLearner,evaluate_routes,conditioned_path,connectors,prefix
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.records import Reading
from coph_stagewise.campaign_reconstruction import NODES,EDGES
from coph_stagewise.stage_prediction import FrozenStages
from coph_stagewise.model import tensor


def agent():
    cfg=replace(configuration(60),stage_width=12.,stage_height=6.,route_field_mode='supported_field')
    a=RouteDecisionLearner(cfg,np.array([.3,0,0,0]),[5,0],agent_id='0',mode='evidence',network=(NODES,EDGES),
        route_options=RouteDecisionOptions(mode='prefix'),unknown_risk=.02)
    s=a.stages[0]
    for k,(i,j) in enumerate(EDGES):
        for m,t in enumerate(np.linspace(0,1,9)):
            q=np.asarray(NODES[i])*(1-t)+np.asarray(NODES[j])*t
            s.readings.append(Reading(f'local:{k*50+m*5+1}',0.,tuple(q),.02))
    return a


class RouteDecisionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_candidates_and_prefix_terminal_are_explicit(self):
        a=agent();x=np.array([.3,0,0,0]);before=copy.deepcopy(a.navigation)
        result=evaluate_routes(a,x,RouteDecisionOptions(mode='prefix'))
        self.assertEqual(result['reason'],'supported_route_choice');self.assertGreaterEqual(len(result['candidates']),2)
        self.assertEqual(a.navigation,before);self.assertLessEqual(result['flow_work'],1024)
        self.assertTrue(all(c['valid'] for c in result['candidates']))
        for c in result['candidates']:
            self.assertEqual(len(c['starts']),1);s=c['starts'][0]
            self.assertGreater(s['remaining_cost'],0);self.assertAlmostEqual(c['cost'],s['prefix_cost']+s['remaining_cost'])
            self.assertEqual(s['horizon'],8);self.assertEqual(s['executed_steps'],8)
    def test_budget_refusal_never_uses_shorter_candidate(self):
        a=agent();before=a.counterfactual_steps
        r=evaluate_routes(a,[.3,0,0,0],RouteDecisionOptions(mode='prefix'),flow_allowance=1)
        self.assertEqual(r['reason'],'route_batch_budget');self.assertIsNone(r['selected']);self.assertEqual(a.counterfactual_steps,before)
    def test_unsupported_batch_keeps_incumbent(self):
        a=agent();a.stages[0].readings=[Reading('only:1',0.,(.3,0.),.02)]
        r=evaluate_routes(a,[.3,0,0,0],RouteDecisionOptions(mode='prefix',horizon=32,candidates=2))
        self.assertEqual(r['reason'],'candidate_batch_unsupported');self.assertIsNone(r['selected'])
    def test_conditional_path_rejects_closed_or_nonedge_route(self):
        a=agent()
        self.assertIsNone(conditioned_path([0,5],np.array([.3,0]),a.navigation)[0])
        self.assertIsNone(conditioned_path([0,1,0,5],np.array([.3,0]),a.navigation)[0])
    def test_execution_uses_selected_prefix_controller(self):
        a=agent();x=np.array([.3,0,0,0]);r=evaluate_routes(a,x,RouteDecisionOptions(mode='prefix'))
        a.navigation.update(route_candidate_schema=1,route_candidate_nodes=r['selected'])
        u,raw=a.action(x)
        chosen=next(c for c in r['candidates'] if c['nodes']==r['selected'])
        np.testing.assert_allclose(u,chosen['starts'][0]['controls'][0]['applied'],atol=1e-12)
    def test_historical_start_and_measurement_isolation(self):
        a=agent();x=np.array([.4,0,0,0]);saved=copy.deepcopy(a.stages[0].readings)
        r=evaluate_routes(a,x,RouteDecisionOptions(mode='prefix',starts='historical',candidates=2))
        self.assertEqual(len(r['starts']),2);self.assertEqual(a.stages[0].readings,saved)
        b=copy.deepcopy(a);r2=evaluate_routes(b,x,RouteDecisionOptions(mode='prefix',starts='historical',candidates=2))
        self.assertEqual([c['cost'] for c in r['candidates']],[c['cost'] for c in r2['candidates']])
    def test_optin_and_invalid_options(self):
        self.assertEqual(RouteDecisionOptions().mode,'graph')
        for kw in [dict(horizon=33),dict(candidates=5),dict(starts='future'),dict(nominal_speed=2)]:
            with self.assertRaises(ValueError):RouteDecisionOptions(**kw)

if __name__=='__main__':unittest.main()
