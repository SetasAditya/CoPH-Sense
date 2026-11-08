"""Frozen development diagnosis of receiver reconstruction in the existing hazard pilot."""
import argparse,copy,hashlib,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import torch
from .evaluation import load,dump
from .cli import source_hashes,archive_sources
from .hazard_demo import SCENES
from .records import Config
from .cooperative import CommunicationConfig
from .maze_family import NetworkWorld
from .staggered_demo import execute
from .reconstruction_diagnostics import ReconstructionLearner,CarrierAuditSelector,receiver_audit

METHODS={'shared_anchors':('anchors',False),'direct_local':('supported_field',False),'direct_cooperative':('supported_field',True)}


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(reference,out):
    """Read all saved arms; unlogged store lifetimes are not reconstructed as facts."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];lineage=[]
    for path in sorted(Path(reference).glob('*/episodes/*.json.gz')):
        r=load(path);scene=path.parents[1].name
        for i in (0,1):
            start=np.array(r['spec']['initials'][i][:2]);cross=next((s['time'] for s in r['samples'][i] if np.linalg.norm(np.array(s['x'][:2])-start)>.25),None)
            deliveries=[e for e in r['events'] if e['kind']=='delivered' and e['receiver']==i]
            for e in deliveries:
                m=next(p for p in r['packets'] if p['identity']==e['packet'])
                u=next((u for u in r['updates'][i] if m['identity'] in u['packets']),None)
                pairs=[]
                if u:
                    for s in u['pre']['stages']:
                        train=np.array([z['q'] for z in s['readings'] if int(str(z['uid']).split(':')[-1])%5]).reshape(-1,2)
                        for k,a in enumerate(m['anchors']):
                            b=s['bounds'];inside=b[0]-1e-10<=a[0]<=b[1]+1e-10 and b[2]-1e-10<=a[1]<=b[3]+1e-10
                            near=bool(len(train) and np.min(np.linalg.norm(train-a[:2],axis=1))<=.65)
                            pairs.append(dict(stage=s['id'],anchor=k,rectangle=bool(inside),local_support=bool(inside and near)))
                lineage.append(dict(scene=scene,arm=r['arm'],agent=i,packet=m['identity'],sources=m['sources'],sender_stage=m['stage'],time=e['time'],
                    before_first_entry=cross is None or e['time']<cross,reason=e['reason'],accepted=e.get('accepted'),
                    version_before=e['version_before'],version_after=e.get('version_after'),eligibility=pairs,
                    available_snapshot=u is not None,received_store_lifetime='unavailable in legacy records',
                    route_score_margin='unavailable in legacy records',actual_fit_residuals='unavailable in legacy records'))
            rows.append(dict(scene=scene,arm=r['arm'],agent=i,outcome=r['agents'][i],first_branch_entry=cross,
                deliveries=len(deliveries),pre_entry_deliveries=sum(e['time']<(cross if cross is not None else float('inf')) for e in deliveries),
                accepted_updates=sum(u['diagnostic']['accepted'] for u in r['updates'][i]),updates=len(r['updates'][i]),
                input_path=str(path.resolve()),sha256=file_hash(path)))
    dump(out/'legacy_rows.json',rows);dump(out/'legacy_lineage.json.gz',lineage)
    dump(out/'legacy_analysis.json',dict(episodes=len(rows)//2,agent_rows=len(rows),delivery_rows=len(lineage),
        availability='Eligibility reconstructed from recorded pre-update stages; no invented private stores or margins.',
        reference=str(Path(reference).resolve()),source_hashes=source_hashes()))
    return rows


def compare(new,old):
    from .decision_suite import compare_operational
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if not k.startswith('rr_') and k not in ('audit_ids',)}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    a,b=clean(new),clean(old)
    # The saved pilot relabels the runner's reciprocal arm as field.
    a['arm']=b['arm']
    result=compare_operational(a,b)
    result['excluded_metadata']='rr_* additive diagnostics and audit_ids only; audit scheduling is evaluator-only'
    result['serialized_packet_bytes_identical']=json.dumps(new['packets'],sort_keys=True)==json.dumps(old['packets'],sort_keys=True)
    return result


def manifest(reference):
    reference=Path(reference).resolve();settings={};refs={}
    for scene in SCENES:
        m=load(reference/scene/'manifest.json');settings[scene]=dict(spec=m['specs'][0],config=m['config'],communication=m['communication']['field'])
        refs[scene]=dict(manifest=file_hash(reference/scene/'manifest.json'),episode=file_hash(reference/scene/'episodes/79000_field.json.gz'))
    m=dict(schema=1,reference=str(reference),reference_hashes=refs,settings=settings,methods=METHODS,source_hashes=source_hashes(),
        development_only=True,seed=79000,animation_seed=79000,regressions=4,comparison_episodes=12,
        branch_entry='first logged position > .25 m from shared start; not decision timing',
        audit_schedule='carrier every delivery before first branch entry, then first delivery per retained stage; include rejections',
        comparison_replay='disabled; local fitting and validation retained',tolerance=dict(atol=1e-12,rtol=1e-10))
    return dict(m,fingerprint=fingerprint(m))


def run_job(task):
    root,scene,method=task;root=Path(root);m=load(root/'manifest.json');settings=m['settings'][scene];key=f'{scene}_{method}'
    torch.set_num_threads(1);start=time.perf_counter()
    try:
        regression=method=='regression';route,fit=('anchors',True) if regression else METHODS[method]
        cfg=replace(Config(**settings['config']),route_field_mode=route);spec=settings['spec'];comm=CommunicationConfig(**settings['communication'])
        def factory(cfg,x,goal,agent_id,communication):
            return ReconstructionLearner(cfg,x,goal,agent_id=agent_id,communication=communication,mode='full' if regression else 'evidence',fit_received=fit,
                network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))),risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'])
        def progress(e):dump(root/'progress'/f'{key}.json',dict(e,seconds=time.perf_counter()-start))
        r=execute(spec,'reciprocal_fixed',comm,learner_factory=factory,world_factory=NetworkWorld,config=cfg,progress=progress,
            audit_callback=receiver_audit,audit_selector=CarrierAuditSelector(spec['initials'][1][:2]))
        r.update(method=method,scene=scene,manifest_fingerprint=m['fingerprint'])
        if regression:r['regression_comparison']=compare(r,load(Path(m['reference'])/scene/'episodes/79000_field.json.gz'))
        path=root/'episodes'/f'{key}.json.gz';dump(path,r)
        row=dict(id=key,path=path.name,sha256=file_hash(path),complete=True,scene=scene,method=method,goals=sum(a['success'] for a in r['agents']),
            regression_pass=r.get('regression_comparison',{}).get('passed'),seconds=time.perf_counter()-start)
    except Exception as e:row=dict(id=key,scene=scene,method=method,complete=False,error=str(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
    dump(root/'status'/f'{key}.json',row);return row


def suite(root,reference,workers):
    root=Path(root);m=manifest(reference)
    if (root/'manifest.json').exists():
        if load(root/'manifest.json')['fingerprint']!=m['fingerprint']:raise ValueError('Frozen source/config mismatch; use archive or fresh output')
    else:
        root.mkdir(parents=True,exist_ok=False)
        for sub in ('episodes','status','progress','figures'):(root/sub).mkdir()
        dump(root/'manifest.json',m);archive_sources(root);analyze(reference,root/'legacy')
    for methods in (['regression'],list(METHODS)):
        jobs=[];results=[]
        for s in SCENES:
            for a in methods:
                p=root/'status'/f'{s}_{a}.json'
                if p.exists():
                    r=load(p);results.append(r)
                    if r['complete']:assert file_hash(root/'episodes'/r['path'])==r['sha256']
                else:jobs.append((str(root),s,a))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(run_job,j) for j in jobs]):
                r=f.result();results.append(r);print(json.dumps(r),flush=True)
        if any(not r['complete'] or r.get('regression_pass') is False for r in results):
            from .reconstruction_reporting import report
            report(root);raise RuntimeError('Preserved incomplete run or failed tracing regression; comparison blocked')
    from .reconstruction_reporting import report
    report(root)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['analyze','suite','render','report'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path,default=Path('results/hazard_pilot_v2'));p.add_argument('--workers',type=int,default=4)
    p.add_argument("--profile",choices=["hazard","complementary"],default="hazard")
    a=p.parse_args(argv)
    if a.profile=="complementary":
        from .complementary_reconstruction import dispatch
        return dispatch(a)
    if a.command=='analyze':return analyze(a.reference,a.out)
    if a.command=='suite':return suite(a.out,a.reference,a.workers)
    from . import reconstruction_reporting
    return getattr(reconstruction_reporting,a.command)(a.out)

if __name__=='__main__':main()
