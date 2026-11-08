import unittest
import numpy as np
import torch
from coph_stagewise.maze_family import grid_maze,screen,specification,learner_for,NetworkWorld,SCREEN
from coph_stagewise.maze_geometry import distance_to_network
from coph_stagewise.shapes import ends
from coph_stagewise.staggered_demo import execute
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise import crossing_demo


def accepted(layout,count=3,seed=5):
    rng=np.random.default_rng(seed);out=[]
    while len(out)<count:
        m=grid_maze(rng,layout=layout)
        if not screen(m):out.append(m)
    return out


class MazeFamilyTests(unittest.TestCase):
    def test_doors_hidden_from_junctions_and_other_corridors(self):
        for layout in ('crossing','relay'):
            for m in accepted(layout):
                nodes=np.asarray(m['nodes'])
                for d in m['doors']:
                    p,s=ends(d);core=np.linspace(p,s,20)
                    self.assertGreaterEqual(min(np.linalg.norm(core-n,axis=1).min() for n in nodes)-d[2],SCREEN['junction_clearance']-1e-9)
                    others=[e for e in m['edges'] if distance_to_network(np.asarray(d[:2])[None],m['nodes'],[e])[0]>1e-6]
                    # The nearest other corridor centreline is beyond sensor range of the door surface.
                    self.assertGreater(distance_to_network(core,m['nodes'],others).min()-d[2],1.25+.2)

    def test_layouts(self):
        c=accepted('crossing',1)[0];r=accepted('relay',1)[0]
        self.assertEqual(c['goals'][0],list(c['initials'][1][:2]));self.assertEqual(r['initials'][0],r['initials'][1]);self.assertGreater(r['start_seconds'][1],0)

    def test_generated_maze_executes(self):
        torch.set_num_threads(1);spec=specification(accepted('crossing',1)[0],'t',0,12)
        r=execute(spec,'reciprocal_fixed',CommunicationConfig(byte_budget=4000000,ttl=300.),audit=False,learner_factory=learner_for(spec),world_factory=NetworkWorld,config=crossing_demo.configuration(12))
        self.assertFalse(any(a['collision'] for a in r['agents']));self.assertTrue(all(len(s)>5 for s in r['samples']))

    def test_learner_pickles_for_packet_audits(self):
        import pickle
        spec=specification(accepted('relay',1)[0],'t',0,12);a=learner_for(spec)(crossing_demo.configuration(12),np.array(spec['initials'][0]),spec['goals'][0],agent_id='0')
        self.assertEqual(pickle.loads(pickle.dumps(a)).navigation['network'],a.navigation['network'])


if __name__=='__main__':unittest.main()
