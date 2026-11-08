"""Four-arm, frozen development comparison of anchor versus direct field routing."""
import argparse,functools,hashlib,json,time,traceback
from pathlib import Path
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
from .decision_suite import specification,ForkWorld,NODES,EDGES,CASES,fingerprint
from .maze_demo import configuration
from .attribution import ARMS
from .cooperative import CommunicationConfig
from .staggered_demo import execute
from .attribution_audits import evaluate_probes
from .field_route_diagnostics import FieldRouteLearner,field_packet_audit,compatible_baseline
from .cli import source_hashes,archive_sources
from .evaluation import dump,load

METHODS={'anchor_baseline':('anchors','remote_route'),'anchor_reconstruction':('anchors','reconstruction'),
         'direct_local':('supported_field','remote_route'),'direct_cooperative':('supported_field','reconstruction')}

def manifest(reference):
    ref=load(Path(reference)/'manifest.json')
    m=dict(schema=1,reference=str(Path(reference).resolve()),reference_fingerprint=ref['fingerprint'],source_hashes=source_hashes(),
        seeds=[78000,78001],cases=list(CASES),arms=METHODS,animation_seed=78000,development_only=True,
        specs={f'{c}_{s}':specification(c,s) for c in CASES for s in (78000,78001)},
        configs={a:asdict(replace(configuration(500),route_field_mode=v[0])) for a,v in METHODS.items()},
        communication=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.)),
        acceptance='mechanism/isolation and reproducibility, not navigation superiority',
        seed_semantics='indexed sensor noise varies; maps and dynamics do not',regression_tolerance=dict(atol=1e-12,rtol=1e-10))
    return dict(m,fingerprint=fingerprint(m))

def job(args):
    case,seed,arm,root,reference,fp=args;torch.set_num_threads(1);start=time.perf_counter();root=Path(root);key=f'{case}_{seed}_{arm}'
    try:
        mode,base=METHODS[arm];spec=specification(case,seed)
        from . import route_field
        before=route_field.FIELD_EVALUATIONS
        def progress(e):dump(root/'progress'/f'{key}.json',dict(e,seconds=time.perf_counter()-start))
        r=execute(spec,'reciprocal_fixed',comm=CommunicationConfig(byte_budget=4000000,ttl=300.),
            learner_factory=functools.partial(FieldRouteLearner,attribution=ARMS[base],layout=(NODES,EDGES)),world_factory=ForkWorld,
            config=replace(configuration(500),route_field_mode=mode),progress=progress,audit_callback=field_packet_audit,update_callback=evaluate_probes)
        r.update(case=case,seed=seed,method=arm,regression=False,manifest_fingerprint=fp)
        r['route_field_total_evaluations']=route_field.FIELD_EVALUATIONS-before
        r['route_field_audit_evaluations']=sum(a['route_field_total_audit_evaluations'] for a in r['audits'])
        if mode=='anchors':r['baseline_comparison']=compatible_baseline(r,load(Path(reference)/'episodes'/f'{case}_{seed}_{base}.json.gz'))
        path=root/'episodes'/f'{key}.json.gz';dump(path,r)
        status=dict(id=key,complete=True,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),goals=sum(a['success'] for a in r['agents']),
                    baseline_pass=r.get('baseline_comparison',{}).get('passed'),seconds=time.perf_counter()-start)
    except Exception as e:status=dict(id=key,complete=False,error=str(e),traceback=traceback.format_exc())
    dump(root/'status'/f'{key}.json',status);return status

def initialize(root,m):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for sub in ('status','episodes','progress','figures'):(root/sub).mkdir()
    dump(root/'manifest.json',m);archive_sources(root)

def suite(root,reference,workers):
    root=Path(root);m=manifest(reference)
    if (root/'manifest.json').exists():
        if load(root/'manifest.json')['fingerprint']!=m['fingerprint']:raise ValueError('frozen manifest/source mismatch; use original archive')
    else:
        initialize(root,m)
    for modes in (list(METHODS)[:2],list(METHODS)[2:]):
        tasks=[]
        for c in CASES:
            for seed in m['seeds']:
                for arm in modes:
                    key=f'{c}_{seed}_{arm}';status=root/'status'/f'{key}.json'
                    if status.exists():
                        s=load(status)
                        if not s['complete'] or s.get('baseline_pass') is False:raise ValueError(f'preserved failed run: {key}')
                        assert hashlib.sha256((root/'episodes'/s['path']).read_bytes()).hexdigest()==s['sha256'];continue
                    tasks.append((c,seed,arm,str(root),str(reference),m['fingerprint']))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=[]
            for f in as_completed([pool.submit(job,t) for t in tasks]):
                s=f.result();results.append(s);print(json.dumps(s),flush=True)
        if any(not s['complete'] or s.get('baseline_pass') is False for s in results):raise ValueError('failed baseline/execution; later phase blocked')
    from .field_route_reporting import report
    report(root)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['suite','report','render','verify']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('results/decision_bottleneck_v1'));p.add_argument('--workers',type=int,default=6);a=p.parse_args(argv)
    if a.command=='suite':return suite(a.out,a.reference,a.workers)
    from . import field_route_reporting as reporting
    return getattr(reporting,a.command)(a.out)
