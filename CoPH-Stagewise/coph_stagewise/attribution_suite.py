"""Frozen development-only attribution experiment; existing suites are untouched."""
import argparse
import functools
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import torch
from .attribution import ARMS, AttributionLearner
from .attribution_audits import packet_audit, evaluate_probes
from .maze_demo import specification as maze_specification, configuration, MazeWorld
from .cooperative import CommunicationConfig
from .staggered_demo import execute
from .evaluation import dump, load
from .cli import source_hashes, archive_sources

SCENES=('blocked_maze','soft_risk_maze')


def specification(scene, seed):
    if scene not in SCENES:raise ValueError(scene)
    spec=maze_specification(seed)
    if scene=='soft_risk_maze':
        spec.update(name='surveyed_maze_soft_risk',obstacles=[],
            patches=[[x,y,.5,.38,4.] for x,y,_ in spec['gate_groups']],gate_groups=[])
    return spec


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def manifest():
    cfg=asdict(configuration(1500))
    comm=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.))
    specs={f'{scene}_{seed}':specification(scene,seed) for scene in SCENES for seed in range(75000,75008)}
    base=Path(__file__).resolve().parents[1]
    documents=[base/'IMPLEMENTATION_CONTRACT.md',base.parent/'CoPH-Sense-yi-0110/CoPH-Sense/docs/COOPERATIVE_STAGEWISE_PH_FORMULATION.md']
    settings=dict(schema=1,config=cfg,communication=comm,specifications=specs,
        arms={k:asdict(v) for k,v in ARMS.items()},animation_seed=75000,
        smoke_seeds=[75000,75001],development_seeds=list(range(75000,75008)),
        development_only=True,final_test_executed=False,
        seed_semantics='Seeds change indexed Gaussian sensor noise only. Two fixed layouts; identical geometry and damping across seeds.',
        audits='First valid packet per receiver stage; horizons 8/16/32; truth-only paired continuations; no future packets',
        documents={str(p.relative_to(base.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in documents},
        sources=source_hashes())
    return dict(settings,fingerprint=fingerprint(settings))


def episode_id(scene,seed,arm):return f'{scene}_{seed}_{arm}'


def job(task):
    scene,seed,arm,root,run_fingerprint=task
    torch.set_num_threads(1)
    root=Path(root);key=episode_id(scene,seed,arm)
    spec=specification(scene,seed);started=time.perf_counter()
    def progress(event):
        dump(root/'progress'/f'{key}.json',dict(event,scene=scene,seed=seed,arm=arm,seconds=time.perf_counter()-started))
    try:
        result=execute(spec,'disabled' if arm.startswith('local_') else 'reciprocal_fixed',
            comm=CommunicationConfig(byte_budget=4000000,ttl=300.),
            learner_factory=functools.partial(AttributionLearner,attribution=ARMS[arm]),
            world_factory=MazeWorld,config=configuration(1500),progress=progress,
            audit_callback=packet_audit,update_callback=evaluate_probes)
        result.update(attribution_arm=arm,attribution=asdict(ARMS[arm]),scene=scene,
            manifest_fingerprint=run_fingerprint)
        path=root/'episodes'/f'{key}.json.gz'
        dump(path,result)
        row=dict(id=key,scene=scene,seed=seed,arm=arm,path=path.name,complete=True,
            goals=sum(a['success'] for a in result['agents']),seconds=time.perf_counter()-started,
            fingerprint=run_fingerprint,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception as error:
        row=dict(id=key,scene=scene,seed=seed,arm=arm,complete=False,error=str(error),
            traceback=traceback.format_exc(),seconds=time.perf_counter()-started,fingerprint=run_fingerprint)
    dump(root/'status'/f'{key}.json',row)
    return row


def initialize(root):
    root=Path(root);expected=manifest()
    if (root/'manifest.json').exists():
        saved=load(root/'manifest.json')
        if saved!=json.loads(json.dumps(expected)):
            raise ValueError('Source/configuration fingerprints changed; choose a new output directory')
        return saved
    root.mkdir(parents=True,exist_ok=False)
    for name in ('episodes','status','progress','figures'): (root/name).mkdir()
    dump(root/'manifest.json',expected);archive_sources(root)
    return expected


def run_profile(root, profile, workers):
    root=Path(root);m=initialize(root)
    seeds=m['smoke_seeds' if profile=='smoke' else 'development_seeds']
    expected=[(scene,seed,arm) for seed in seeds for scene in SCENES for arm in ARMS]
    tasks=[]
    for scene,seed,arm in expected:
        key=episode_id(scene,seed,arm);status=root/'status'/f'{key}.json'
        if status.exists():
            row=load(status)
            if row.get('complete'):
                path=root/'episodes'/row['path']
                if row['fingerprint']!=m['fingerprint'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('Completed episode fingerprint mismatch: '+key)
            # Preserve failed attempts too; do not silently rerun favorable outcomes.
            continue
        tasks.append((scene,seed,arm,str(root),m['fingerprint']))
    dump(root/f'{profile}_execution.json',dict(expected=[episode_id(*t) for t in expected],
        reused=len(expected)-len(tasks),scheduled=len(tasks),workers=workers,driver_pid=os.getpid()))
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending=[pool.submit(job,task) for task in tasks]
            for future in as_completed(pending):
                row=future.result();print(json.dumps(row),flush=True)
    from .attribution_reporting import report
    report(root)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['suite','report','render'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--profile',choices=['smoke','development','all'],default='all')
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args(argv)
    if args.workers<1:ap.error('--workers must be positive')
    if args.command=='suite':
        for profile in (('smoke','development') if args.profile=='all' else (args.profile,)):
            run_profile(args.out,profile,args.workers)
    else:
        from .attribution_reporting import report,render
        (report if args.command=='report' else render)(args.out)


if __name__=='__main__':main()
