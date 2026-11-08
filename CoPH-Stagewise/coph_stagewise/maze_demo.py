"""Static maze with unknown physical closures; evaluator truth lives here only."""
import argparse,copy
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .cooperative_evaluation import CooperativeWorld
from .staggered_demo import execute
from .maze_learning import MazeLearner
from .maze_geometry import NODES,EDGES,corridor_clear,CORRIDOR_RADIUS
from .cooperative import CommunicationConfig
from .records import Config
from .evaluation import dump,load
from .cli import source_hashes,archive_sources


def specification(seed):
    # Unknown gates span the full traversable corridor. Three overlapping disks
    # are the simulator's conservative rounded-barrier representation, not props.
    gates=[(5.3,0.,'vertical'),(8.3,3.,'vertical'),(11.3,0.,'vertical')]
    obstacles=[]
    for x,y,d in gates:
        obstacles.extend([[x,y+dy,.34] for dy in (-.55,0.,.55)])
    return dict(name='surveyed_maze_hidden_gates',seed=seed,initials=[[0.,0.,0.,0.]]*2,goals=[[12.,0.]]*2,
        initial=[0.,0.,0.,0.],goal=[12.,0.],start_seconds=[0.,5.],steps=1500,damping=.85,
        obstacles=obstacles,patches=[],bounds=[-1.,13.,-1.,7.],dynamic=False,
        gate_groups=gates,network_nodes=NODES,network_edges=EDGES,corridor_radius=CORRIDOR_RADIUS,
        prior='Surveyed corridor layout only. Closures unknown. Independent plants; no inter-agent contact dynamics.',
        experiment='development maze mechanism demo, not held-out evaluation')


class MazeWorld(CooperativeWorld):
    def step(self,x,u):
        y,hit,cost,energy=super().step(x,u)
        # Same public corridor footprint as the controller; actual agent radius .12.
        hit=bool(hit or not corridor_clear(x[:2],y[:2],margin=.12))
        return y,hit,cost,energy


def configuration(steps):
    return Config(controller_mode='routed',shaping_mode='progress',steps=steps,replay_objective='mission',replay_domain='retained',replay_horizon_mode='supported_boundary',replay_update_cap=1024)


def run(arg):
    seed,arm,root,steps=arg;torch.set_num_threads(1);s=specification(seed);s['steps']=steps
    def progress(event):
        import resource
        event.update(seed=seed,arm=arm,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(Path(root)/f'progress_{seed}_{arm}.json',event);print(event,flush=True)
    r=execute(s,arm,CommunicationConfig(byte_budget=4000000,ttl=300.),learner_factory=MazeLearner,world_factory=MazeWorld,config=configuration(steps),progress=progress)
    path=f'{seed}_{arm}.json.gz';dump(Path(root)/'episodes'/path,r)
    return dict(seed=seed,arm=arm,path=path,goals=sum(a['success'] for a in r['agents']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=1500);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
    if not args.render_only:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'episodes').mkdir()
        dump(args.out/'manifest.json',dict(specs=[dict(specification(s),steps=args.steps) for s in (75000,75001)],arms=['disabled','reciprocal_fixed'],animation_seed=75000,
            config=asdict(configuration(args.steps)),communication=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.)),source_hashes=source_hashes(),
            main_visualization='Single reciprocal run, both agents, exact logged Hamiltonian slices and forces; disabled arm is diagnostic only.'))
        archive_sources(args.out);rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(run,(s,a,str(args.out),args.steps)) for s in (75000,75001) for a in ('disabled','reciprocal_fixed')]
            for job in as_completed(jobs):rows.append(job.result());dump(args.out/'completion.json',rows);print(rows[-1],flush=True)
    from .maze_visuals import report
    report(args.out)

if __name__=='__main__':main()
