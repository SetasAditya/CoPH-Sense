"""Focused decision bottleneck suite. No algorithm or runtime-truth intervention."""
import argparse
import functools
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import torch
from .attribution import ARMS
from .attribution_suite import specification as old_spec
from .maze_demo import configuration,MazeWorld
from .maze_geometry import corridor_clear
from .cooperative_evaluation import CooperativeWorld
from .cooperative import CommunicationConfig
from .decision_trace import DecisionLearner
from .decision_audits import decision_packet_audit
from .attribution_audits import evaluate_probes
from .staggered_demo import execute
from .cli import source_hashes,archive_sources
from .evaluation import dump,load

NODES=((0.,0.),(1.5,0.),(5.5,0.),(8.,0.),(1.5,1.7),(5.5,1.7))
EDGES=((0,1),(1,2),(2,3),(1,4),(4,5),(5,2))
CASES=('clear_fork','direct_risk','bypass_risk')
METHODS=('local_evidence','remote_route','reconstruction','current_replay','historical_replay')


def specification(case,seed):
    if case not in CASES:raise ValueError(case)
    patch=[] if case=='clear_fork' else [[3.6,0. if case=='direct_risk' else 1.7,.5,.38,4.]]
    return dict(name=case,seed=seed,initials=[[0.,0.,0.,0.]]*2,goals=[[8.,0.]]*2,
        initial=[0.,0.,0.,0.],goal=[8.,0.],start_seconds=[0.,5.],steps=500,damping=.85,
        obstacles=[],patches=patch,bounds=[-1.,9.,-1.,2.7],dynamic=False,gate_groups=[],
        network_nodes=NODES,network_edges=EDGES,corridor_radius=.85,
        prior='surveyed graph only; terrain unknown; independent plants',development_only=True)


class ForkWorld(CooperativeWorld):
    def step(self,x,u):
        y,hit,cost,energy=super().step(x,u)
        hit=bool(hit or not corridor_clear(x[:2],y[:2],margin=.12,nodes=self.spec['network_nodes'],edges=self.spec['network_edges']))
        return y,hit,cost,energy


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def manifest(reference):
    ref=load(Path(reference)/'manifest.json')
    m=dict(schema=1,reference=str(Path(reference).resolve()),reference_fingerprint=ref['fingerprint'],
        source_hashes=source_hashes(),seeds=[78000,78001],arms=list(METHODS),cases=list(CASES),
        specs={f'{c}_{s}':specification(c,s) for c in CASES for s in (78000,78001)},
        regressions=[dict(case=c,seed=75000,arm='historical_replay') for c in ('blocked_maze','soft_risk_maze')],
        config=asdict(configuration(500)),communication=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.)),
        animation_seed=78000,development_only=True,held_out_executed=False,
        seed_semantics='Two indexed sensor-noise seeds, not independent graph layouts.',
        regression_comparison=dict(atol=1e-12,rtol=1e-10,discrete='exact',serialized_bytes='report separately; never claim bitwise parity'),
        reference_semantics='Exact risk in route scoring only; evaluator-only; no oracle controller or support expansion')
    return dict(m,fingerprint=fingerprint(m))


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not k.startswith('decision_') and k not in ('policy_trace','wall_seconds')}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def compare_operational(a,b):
    keys=('spec','config','communication','arm','agents','trajectories','samples','observations','updates','packets','events','geometry_events','learner_events','receive_update_slots','pending_packets')
    atol=1e-12;rtol=1e-10;maximum=0.
    def equivalent(x,y):
        nonlocal maximum
        if isinstance(x,dict) and isinstance(y,dict):
            # JSON float digit counts may differ despite numerical equivalence.
            # Preserve and disclose bytes separately; identities/schedules/budgets
            # and all executed learning-work counters remain strict comparisons.
            xx={k:v for k,v in x.items() if k!='bytes'};yy={k:v for k,v in y.items() if k!='bytes'}
            return xx.keys()==yy.keys() and all(equivalent(xx[k],yy[k]) for k in xx)
        if isinstance(x,list) and isinstance(y,list):return len(x)==len(y) and all(equivalent(v,w) for v,w in zip(x,y))
        if isinstance(x,float) and isinstance(y,float):
            delta=abs(x-y);maximum=max(maximum,delta)
            return delta<=atol+rtol*max(abs(x),abs(y))
        return type(x)==type(y) and x==y
    result={};strict={}
    for k in keys:
        x,y=clean(a[k]),clean(b[k]);strict[k]=fingerprint(x)==fingerprint(y);result[k]=equivalent(x,y)
    byte_difference=[{k:a['agents'][i]['bytes'][k]-b['agents'][i]['bytes'][k] for k in a['agents'][i]['bytes']} for i in (0,1)]
    return dict(passed=all(result.values()),fields=result,bitwise_passed=all(strict.values()),bitwise_fields=strict,
                tolerance=dict(atol=atol,rtol=rtol),maximum_float_delta=maximum,byte_accounting_difference=byte_difference,
                semantics='Numerical equivalence only; raw serialized byte counts are reported, not asserted identical. All packet identities, schedules, acceptance and work counts remain exact.')


def job(task):
    case,seed,arm,root,regression,run_fp,reference=task;torch.set_num_threads(1)
    root=Path(root);key=f'{case}_{seed}_{arm}';start=time.perf_counter()
    try:
        spec=old_spec(case,seed) if regression else specification(case,seed)
        layout=None if regression else (NODES,EDGES)
        def progress(e):dump(root/'progress'/f'{key}.json',dict(e,seconds=time.perf_counter()-start))
        r=execute(spec,'disabled' if arm=='local_evidence' else 'reciprocal_fixed',
            comm=CommunicationConfig(byte_budget=4000000,ttl=300.),
            learner_factory=functools.partial(DecisionLearner,attribution=ARMS[arm],layout=layout),
            world_factory=MazeWorld if regression else ForkWorld,config=configuration(spec['steps']),progress=progress,
            audit_callback=decision_packet_audit,update_callback=evaluate_probes)
        r.update(case=case,seed=seed,method=arm,regression=regression,manifest_fingerprint=run_fp)
        if regression:r['regression_comparison']=compare_operational(r,load(Path(reference)/'episodes'/f'{key}.json.gz'))
        path=root/'episodes'/f'{key}.json.gz';dump(path,r)
        row=dict(id=key,case=case,seed=seed,arm=arm,regression=regression,complete=True,
                 goals=sum(a['success'] for a in r['agents']),path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 regression_pass=r.get('regression_comparison',{}).get('passed'),seconds=time.perf_counter()-start)
    except Exception as e:row=dict(id=key,complete=False,case=case,seed=seed,arm=arm,regression=regression,error=str(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
    dump(root/'status'/f'{key}.json',row);return row


def suite(root,reference,workers):
    root=Path(root);expected=manifest(reference)
    if (root/'manifest.json').exists():
        if load(root/'manifest.json')!=json.loads(json.dumps(expected)):raise ValueError('Frozen sources/settings differ; use the archived checkout or a new result directory')
    else:
        root.mkdir(parents=True,exist_ok=False)
        for sub in ('status','episodes','progress','figures'): (root/sub).mkdir()
        dump(root/'manifest.json',expected);archive_sources(root)
    for regression in (True,False):
        tasks=[]
        entries=[(c,75000,'historical_replay') for c in ('blocked_maze','soft_risk_maze')] if regression else [(c,s,a) for c in CASES for s in (78000,78001) for a in METHODS]
        for c,s,a in entries:
            status=root/'status'/f'{c}_{s}_{a}.json'
            if status.exists():
                r=load(status)
                if not r['complete']:raise ValueError(f'Preserved failed attempt: {status}')
                assert hashlib.sha256((root/'episodes'/r['path']).read_bytes()).hexdigest()==r['sha256']
                if regression and not r['regression_pass']:raise ValueError('Operational regression failed; fork execution blocked')
                continue
            tasks.append((c,s,a,str(root),regression,expected['fingerprint'],str(reference)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows=[f.result() for f in as_completed([pool.submit(job,t) for t in tasks])]
        for row in rows:print(json.dumps(row),flush=True)
        if any(not r['complete'] or (regression and not r.get('regression_pass')) for r in rows):raise ValueError('Failed run preserved; inspect status before proceeding')
    from .decision_reporting import report
    report(root)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['analyze','suite','render','report']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('results/attribution_v1'));p.add_argument('--workers',type=int,default=4);a=p.parse_args(argv)
    from .decision_reporting import analyze,report,render
    if a.command=='analyze':analyze(a.reference,a.out)
    elif a.command=='suite':suite(a.out,a.reference,a.workers)
    elif a.command=='report':report(a.out)
    else:render(a.out)

if __name__=='__main__':main()
