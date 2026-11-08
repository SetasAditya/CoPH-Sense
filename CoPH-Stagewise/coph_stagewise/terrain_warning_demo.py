"""Development-only surveyed-route hazard warning demonstration."""
import argparse,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
import torch
from .terrain_warning import TerrainLearner,ROUTES
from .staggered_demo import execute
from .cooperative import CommunicationConfig
from .evaluation import dump,load
from .cli import source_hashes,archive_sources


def specification(seed):
    return dict(name='unsafe_shortcut',seed=seed,initials=[[0.,0.,0.,0.]]*2,goals=[[8.,0.]]*2,
        initial=[0.,0.,0.,0.],goal=[8.,0.],start_seconds=[0.,5.],steps=500,damping=.85,
        obstacles=[],patches=[[3.6,0.,.5,.38,4.]],bounds=[-1.,9.,-1.5,3.],dynamic=False,
        surveyed_routes=ROUTES,prior='surveyed route topology only; terrain unknown, fixed risk prior 1',
        experiment='development mechanism demonstration; not held-out validation')


def run(args):
    seed,arm,root=args;torch.set_num_threads(1)
    r=execute(specification(seed),arm,CommunicationConfig(byte_budget=2000000,ttl=60.),learner_factory=TerrainLearner)
    dump(Path(root)/'episodes'/f'{seed}_{arm}.json.gz',r)
    return dict(seed=seed,arm=arm,goals=sum(a['success'] for a in r['agents']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
    if not args.render_only:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'episodes').mkdir()
        dump(args.out/'manifest.json',dict(specs=[specification(s) for s in (74000,74001)],arms=['disabled','reciprocal_fixed'],animation_seed=74000,
            communication=asdict(CommunicationConfig(byte_budget=2000000,ttl=60.)),source_hashes=source_hashes(),
            assumptions='Known route topology shared equally. No risk truth given. Static remote route evidence expires after 60s; no remote replay support. Development only.'))
        archive_sources(args.out)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed=list(pool.map(run,[(s,a,str(args.out)) for s in (74000,74001) for a in ('disabled','reciprocal_fixed')]))
        dump(args.out/'completion.json',completed);print(completed,flush=True)
    from .terrain_warning_visuals import report
    report(args.out)

if __name__=='__main__':main()
