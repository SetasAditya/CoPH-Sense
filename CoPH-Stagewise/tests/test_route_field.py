import copy,unittest,functools
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.route_field import values,sync,field_key
from coph_stagewise.field_route_diagnostics import FieldRouteLearner,representation_audit,compatible_baseline
from coph_stagewise.decision_suite import specification,ForkWorld,NODES,EDGES
from coph_stagewise.maze_demo import configuration
from coph_stagewise.attribution import ARMS
from coph_stagewise.staggered_demo import sense,execute
from coph_stagewise.records import new_stage
from coph_stagewise.model import risk,tensor
from coph_stagewise.decision_trace import route_ranking,DecisionLearner
from coph_stagewise.attribution_audits import digest
from coph_stagewise.receiver_summary import summarize_receiver,reconstruct_receiver
from coph_stagewise.maze_learning import graph_costs

class RouteFieldTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_suite_initializes_output_directories(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  from coph_stagewise.field_route_suite import initialize
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'run'
   with patch('coph_stagewise.field_route_suite.archive_sources'):initialize(p,{'fixture':True})
   self.assertTrue(all((p/s).is_dir() for s in ('episodes','status','progress','figures')))
 def agent(self,mode='supported_field'):
  cfg=replace(configuration(56),route_field_mode=mode);x=np.zeros(4);w=ForkWorld(specification('direct_risk',78000),cfg)
  a=FieldRouteLearner(cfg,x,[8,0],attribution=ARMS['reconstruction'],layout=(NODES,EDGES),agent_id='0')
  a.observe(sense(w,x,1,0));return a,w
 def test_commit_reject_and_no_new_support(self):
  a,w=self.agent();self.assertEqual(values([[0,0]],a.navigation).tolist(),[1.]);r=a.update(np.zeros(4));self.assertTrue(r['accepted'])
  snapshot=copy.deepcopy(a.navigation['route_field_snapshot']);a.action(np.zeros(4));self.assertEqual(snapshot,a.navigation['route_field_snapshot'])
  calls=[]
  def bad(_):calls.append(1);return 0 if len(calls)==1 else float('inf')
  a._field_validation=bad;r=a.update(np.zeros(4));self.assertFalse(r['accepted']);self.assertEqual(snapshot,a.navigation['route_field_snapshot'])
  a.observe(sense(w,np.array([1.,0,0,0]),5,.6));a.action(np.zeros(4));self.assertTrue(set(map(tuple,a.navigation['route_field_snapshot']['stages'][0]['support']))<=set(map(tuple,snapshot['stages'][0]['support'])))
 def test_precedence_clipping_support_and_values(self):
  a,w=self.agent();a.update(np.zeros(4));ctx=copy.deepcopy(a.navigation);ctx['terrain_anchors']=[[0,0,19,0,0],[8,0,25,0,0]]
  v,m=values([[0,0],[8,0],[20,0]],ctx,True);self.assertEqual(m['source'],['field','anchor','unknown']);self.assertEqual(v[1],20);self.assertEqual(m['raw'][1],25);self.assertEqual(v[2],1)
  s=a.stages[0];self.assertAlmostEqual(v[0],float(risk(tensor([0,0]),s.weights,s.centers,a.cfg.rbf_width)))
  self.assertEqual(values([[0,0]],dict(ctx,route_field_mode='anchors'))[0],19)
 def test_uncommitted_weights_and_evicted_measurements(self):
  a,w=self.agent();a.update(np.zeros(4));before=values([[0,0]],a.navigation).copy()
  a.stages[0].weights+=100
  sync(a);np.testing.assert_array_equal(values([[0,0]],a.navigation),before)
  a.stages[0].readings=[];sync(a)
  self.assertFalse(values([[0,0]],a.navigation,True)[1]['field_support'][0])
  self.assertFalse(a.navigation['route_field_snapshot']['stages'][0]['support'])
 def test_overlap_and_eviction(self):
  a,w=self.agent();a.update(np.zeros(4));first=copy.deepcopy(a.routing_fields[0]);other=copy.deepcopy(first);other.update(id=1,version=99,weights=[1.]*16)
  ctx=copy.deepcopy(a.navigation);ctx['route_field_snapshot']['stages']=[first,other]
  self.assertEqual(values([[0,0]],ctx,True)[1]['stage'],[1])
  a.stages=[new_stage(1,(2.1,0),[2.1,0,0,0],[8,0],a.cfg)];sync(a);self.assertFalse(a.routing_fields)
  self.assertEqual(values([[0,0]],a.navigation,True)[1]['source'],['anchor'])
 def test_cache_fingerprint_and_graph_accounting(self):
  a,w=self.agent();a.update(np.zeros(4));ctx=a.navigation;key=field_key(ctx);layout=(NODES,EDGES)
  args=(tuple(map(tuple,ctx['terrain_anchors'])),(),layout,key);graph_costs.cache_clear();g=graph_costs(*args);graph_costs(*args);self.assertEqual(graph_costs.cache_info().hits,1)
  changed=copy.deepcopy(ctx);changed['route_field_snapshot']['stages'][0]['weights'][0]+=1;self.assertNotEqual(key,field_key(changed))
  rank=route_ranking([0,0],[8,0],ctx,trace_fields=True)
  for e in rank['quadrature']:self.assertAlmostEqual(e['cost'],sum(e['contributions']))
  for i,j in EDGES:
   edge=next(e for e in rank['quadrature'] if e['edge']==[i,j]);self.assertAlmostEqual(dict(g[0][i])[j],edge['cost'])
  a.action(np.zeros(4));self.assertEqual(a.last_control['route_nodes'][-1],3)
 def test_summary_and_branch_isolation(self):
  a,w=self.agent();a.update(np.zeros(4));s=summarize_receiver(a,np.zeros(4),'1');s.validate('1',0.)
  b=reconstruct_receiver(s,'1',0.,cross_stage=True);self.assertEqual(a.navigation['route_field_snapshot'],b.navigation['route_field_snapshot'])
  b.routing_fields[0]['weights'][0]+=1;self.assertNotEqual(a.routing_fields,b.routing_fields)
  from coph_stagewise.route_field import validate_context
  with self.assertRaises(ValueError):validate_context({},a.cfg,[0])
  old,_=self.agent('anchors');old.update(np.zeros(4));summarize_receiver(old,np.zeros(4),'1').validate('1',0.)
  before=digest(a);truth=digest(w);audit=representation_audit(a,np.zeros(4),w)
  self.assertEqual(before,digest(a));self.assertEqual(truth,digest(w));self.assertIn('route_field_comparison',audit)
  from coph_stagewise.field_route_diagnostics import field_packet_audit
  sender,_=self.agent();sender.agent_id='1';sender.update(np.zeros(4))
  packet=sender.export(np.zeros(4),'0')
  result=field_packet_audit(a,np.zeros(4),packet,w,0)
  self.assertTrue(result['operational_state_unchanged']);self.assertLess(result['decomposition_closure_residual'],1e-8)
 def test_anchor_operational_parity(self):
  s=specification('clear_fork',78000);s['steps']=56;cfg=configuration(56)
  kw=dict(audit=False,world_factory=ForkWorld,config=cfg)
  old=execute(s,'reciprocal_fixed',learner_factory=functools.partial(DecisionLearner,attribution=ARMS['reconstruction'],layout=(NODES,EDGES)),**kw)
  new=execute(s,'reciprocal_fixed',learner_factory=functools.partial(FieldRouteLearner,attribution=ARMS['reconstruction'],layout=(NODES,EDGES)),**kw)
  self.assertTrue(old['packets']);self.assertTrue(compatible_baseline(new,old)['passed'],compatible_baseline(new,old))

if __name__=='__main__':unittest.main()
