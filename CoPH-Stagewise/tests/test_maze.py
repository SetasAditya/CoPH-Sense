import copy,pickle,unittest
from dataclasses import replace
import numpy as np
import torch
from coph_stagewise.maze_demo import specification,configuration,MazeWorld
from coph_stagewise.maze_geometry import corridor_clear,distance_to_network
from coph_stagewise.maze_learning import MazeLearner,MazePacket,maze_route
from coph_stagewise.maze_visuals import landscape
from coph_stagewise.model import tensor,potential,flow
from coph_stagewise.stage_prediction import FrozenStages
from coph_stagewise.staggered_demo import sense

class MazeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.cfg=configuration(50)
    def agent(self,x=(0.,0.,0.,0.),i='0',now=10):
        s=specification(75000);w=MazeWorld(s,self.cfg);a=MazeLearner(self.cfg,np.array(x),s['goal'],agent_id=i);a.observe(sense(w,np.array(x),0,now));return a,w
    def test_packet_geometry_changes_route_before_local_sensing(self):
        scout,w=self.agent((4.,0.,0.,0.));carrier,_=self.agent((2.,0.,0.,0.),'1');self.assertFalse(carrier.local_geometry_reports)
        m=scout.export(np.array(scout.current_state),'1');self.assertTrue(m.geometry)
        before=pickle.dumps([s.readings for s in carrier.stages]);state=carrier.current_state
        event=carrier.assimilate(m);self.assertEqual(event['reason'],'eligible');self.assertEqual(state,carrier.current_state)
        self.assertEqual(before,pickle.dumps([s.readings for s in carrier.stages]));atlas=FrozenStages.freeze(carrier)
        _,meta=maze_route(atlas,tensor(state),carrier.navigation,0);self.assertIn(1,meta['blocked_edges']);self.assertNotIn(2,meta['route_nodes'][:2])
        outbound=carrier.export(np.array(state),'0');self.assertFalse(outbound.geometry,'remote geometry was echoed as local evidence')
        self.assertEqual(carrier.assimilate(m)['reason'],'duplicate_or_reordered')
    def test_fast_projection_matches_complete_enumeration(self):
        from coph_stagewise.navigation import project
        for q,v,raw in [([0.,0.],[0.,0.],[.5,.1]),([0.,0.],[.7,0.],[3.,0.]),([.5,0.],[.5,0.],[2.,0.])]:
            x=tensor(q+v);obstacles=((1.5,0.,.34),)
            a,_,_=project(tensor(raw),x,obstacles,.85,self.cfg)
            b,_,_=project(tensor(raw),x,obstacles,.85,self.cfg,fast=True)
            if a is None:self.assertIsNone(b)
            else:np.testing.assert_allclose(a.detach().numpy(),b.detach().numpy(),atol=1e-12)

    def test_bad_geometry_is_rejected(self):
        scout,_=self.agent((4.,0.,0.,0.));carrier,_=self.agent(i='1');m=scout.export(np.array(scout.current_state),'1')
        for bad in (replace(m,geometry_schema=9),replace(m,geometry=((5.,0.,.3,0.,0.,.1,10.,'0:geometry:0'),))):
            self.assertNotEqual(carrier.assimilate(bad)['reason'],'eligible');self.assertFalse(carrier.remote_geometry_reports)
    def test_real_corridor_and_gate_collisions(self):
        a,w=self.agent();self.assertTrue(corridor_clear([0,0],[3,0]));self.assertFalse(corridor_clear([1,0],[1,3]))
        _,hit,_,_=w.step(np.array([1.,.74,0.,1.]),np.zeros(2));self.assertTrue(hit)
        _,hit,_,_=w.step(np.array([4.8,0.,1.,0.]),np.zeros(2));self.assertTrue(hit)
    def test_spatial_stages_allow_vertical_and_return_moves(self):
        a,w=self.agent();self.assertTrue(a.advance(np.array([0.,1.41,0.,0.])));self.assertAlmostEqual(a.stages[-1].center[1],2.8)
        self.assertTrue(a.advance(np.array([0.,1.39,0.,0.])));self.assertAlmostEqual(a.stages[-1].center[1],0.)
        self.assertEqual(FrozenStages.freeze(a).stage_mode,'spatial');self.assertLessEqual(len(a.stages),3)
    def test_exact_landscape_and_controller_gradients(self):
        a,_=self.agent();a.update(np.array(a.current_state));atlas=FrozenStages.freeze(a);q=np.array([.23,.02,.1,0.]);before=copy.deepcopy(a.navigation)
        u,_=a.action(q);v,_=atlas.control(tensor(q),len(atlas.stages)-1,a.policy,damping=a.damping,context=copy.deepcopy(before),step=0)
        np.testing.assert_allclose(u,v.detach().numpy(),atol=1e-12);m=a.control_model();points=np.array([[.2,.1],[.3,.2]])
        H,F,mask=landscape(m,points,q[2:],a.cfg)
        for i,p in enumerate(points):
            z=tensor(p).requires_grad_(True);V=potential(z,tensor(m['target']),tensor(m['weights']),tensor(m['centers']),a.cfg.rbf_width,tensor(m['policy']),m['geometry'],a.cfg)
            self.assertAlmostEqual(H[i],float(V.detach())+.5*np.dot(q[2:],q[2:]),places=11)
            np.testing.assert_allclose(F[i],-torch.autograd.grad(V,z)[0].numpy(),atol=1e-11)
        def objective(z):
            context=copy.deepcopy(before)
            for k in range(3):
                control,_=atlas.control(z,0,a.policy,damping=a.damping,context=context,step=k);z=flow(z,control,a.damping,a.cfg.dt)
            return z.square().sum()
        z=tensor(q).requires_grad_(True);gradient=torch.autograd.grad(objective(z),z)[0].numpy()
        fd=[float((objective(tensor(q+np.eye(4)[i]*1e-6))-objective(tensor(q-np.eye(4)[i]*1e-6)))/(2e-6)) for i in range(4)]
        np.testing.assert_allclose(gradient,fd,atol=1e-5,rtol=1e-4)

if __name__=='__main__':unittest.main()
