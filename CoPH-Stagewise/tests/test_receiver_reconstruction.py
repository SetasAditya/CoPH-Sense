import copy,pickle,unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.hazard_demo import specification,communication,learner
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.maze_family import NetworkWorld
from coph_stagewise.staggered_demo import sense,execute
from coph_stagewise.reconstruction_diagnostics import ReconstructionLearner,receiver_audit,eligibility,ranking,CarrierAuditSelector
from coph_stagewise.maze_learning import graph_costs
from coph_stagewise.route_field import field_key
from coph_stagewise.receiver_reconstruction import compare

class ReceiverReconstructionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.spec=specification('moderate');self.cfg=configuration(16)
    def agent(self,i=1,q=(6.4,4.),fit=True,mode='supported_field'):
        x=np.array([*q,0.,0.]);cfg=replace(self.cfg,route_field_mode=mode)
        a=ReconstructionLearner(cfg,x,self.spec['goals'][i],mode='evidence',agent_id=str(i),communication=communication('field'),fit_received=fit,
            network=(tuple(map(tuple,self.spec['network_nodes'])),tuple(map(tuple,self.spec['network_edges']))),risk_weight=self.spec['risk_weights'][i],unknown_risk=.02)
        w=NetworkWorld(dict(self.spec,initial=x.tolist(),goal=list(a.goal)),cfg)
        for k in range(3):a.observe(sense(w,x,k,1+k*.12))
        a.update(x);return a,x,w
    def test_audit_isolation_and_component_control(self):
        s,xx,_=self.agent(0,(6.,4.));m=s.export(xx,'1');a,x,w=self.agent();original=pickle.dumps(a);r=receiver_audit(a,x,m,w,2)
        self.assertEqual(original,pickle.dumps(a));self.assertGreater(r['queries']['supported_count'],0)
        self.assertEqual(r['branches']['no_fit']['message_fit_work'],0);self.assertGreater(r['branches']['fit']['message_fit_work'],0)
        self.assertEqual(r['branches']['fit']['rankings']['anchors'],r['branches']['no_fit']['rankings']['anchors'])
    def test_replacement_and_state(self):
        s,x,_=self.agent(0,(6.,4.));a,q,_=self.agent();m=s.export(x,'1');state=a.current_state
        a.assimilate(m);a.assimilate(m);self.assertEqual(a.current_state,state)
        m2=replace(m,identity=m.identity+'new',timestamp=m.timestamp+.01);a.now+=.02;a.assimilate(m2)
        self.assertEqual(len(a.received),1);self.assertEqual(a.received['0'].identity,m2.identity)
        self.assertTrue(any(e.get('replaced')==m.identity for e in a.rr_events))
    def test_rectangle_not_local_support(self):
        s,x,_=self.agent(0,(6.,4.));a,q,_=self.agent();m=s.export(x,'1');st=a.stages[-1]
        point=(st.center[0]+a.cfg.stage_width/2-.01,st.center[1]+a.cfg.stage_height/2-.01)
        m=replace(m,anchors=((point[0],point[1],1.,0.,0.),));r=eligibility(a,m)[0]
        self.assertTrue(r['rectangle']);self.assertFalse(r['local_support'])
    def test_rejected_update_snapshot(self):
        a,x,_=self.agent();before=copy.deepcopy(a.routing_fields);measured=pickle.dumps([(s.readings,s.transitions) for s in a.stages])
        with patch.object(a,'_field_validation',side_effect=[0.,100.]):r=a.update(x)
        self.assertFalse(r['accepted']);self.assertEqual(a.routing_fields,before)
        self.assertEqual(measured,pickle.dumps([(s.readings,s.transitions) for s in a.stages]));self.assertLessEqual(r['counterfactual_steps'],1024)
    def test_weighted_graph_scores(self):
        a,x,_=self.agent();r=ranking(a,x,details=True);c=a.navigation
        graph,_,_=graph_costs(tuple(map(tuple,c['terrain_anchors'])),tuple(map(tuple,c.get('reported_geometry',[]))),c['network'],field_key(c),6.,.02)
        self.assertEqual(r['risk_weight'],24.);self.assertEqual(r['unknown_risk'],.02)
        for e in r['quadrature']:
            i,j=e['edge']
            if i>=0:self.assertAlmostEqual(e['cost'],dict(graph[i])[j],places=11)
    def test_selector_includes_unsupported(self):
        a,x,_=self.agent(q=(0.,4.));sel=CarrierAuditSelector((0.,4.))
        self.assertTrue(sel(a,x,None,0,False));self.assertTrue(sel(a,x,None,1,True))
        x[0]=.3;self.assertFalse(sel(a,x,None,2,True));a.stages[-1].id+=1;self.assertTrue(sel(a,x,None,3,False))
    def test_short_regression(self):
        spec=dict(self.spec,steps=12,start_seconds=[0.,.48]);cfg=configuration(12);fs=learner(spec,'field')
        def old(cfg,x,goal,agent_id,communication):return fs[int(agent_id)](cfg,x,goal,agent_id=agent_id,communication=communication)
        def new(cfg,x,goal,agent_id,communication):return ReconstructionLearner(cfg,x,goal,agent_id=agent_id,communication=communication,network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))),risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=.02)
        kw=dict(spec=spec,arm='reciprocal_fixed',comm=communication('field'),audit=False,world_factory=NetworkWorld,config=cfg)
        a=execute(**kw,learner_factory=old);b=execute(**kw,learner_factory=new);self.assertTrue(compare(b,a)['passed'],compare(b,a))
