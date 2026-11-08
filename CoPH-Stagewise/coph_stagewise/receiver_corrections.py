"""Frozen staged diagnosis and navigation comparison for bounded receiver fitting."""
import argparse,copy,gzip,json,pickle,time,traceback
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .evaluation import dump,load
from .cli import source_hashes,archive_sources
from .receiver_reconstruction import fingerprint,file_hash
from .receiver_fitting import SETTINGS
from .reconstruction_quality import _snapshot_jobs,_local
from .reconstruction_diagnostics import ReconstructionLearner,field_queries,ranking
from .attribution_audits import digest,fixed_state
from .route_field import hypothetical_context
from .complementary_reconstruction import specification,CASES,acquire,gate_snapshot,prefix,DeliverySnapshots,save_snapshot
from .records import Config
from .crossing_demo import configuration
from .hazard_demo import communication
from .cooperative import CommunicationConfig
from .maze_family import NetworkWorld
from .staggered_demo import execute

SEEDS=(79200,79201)
ARMS=dict(local_only=(False,'legacy','legacy'),legacy=(True,'legacy','legacy'),corrected_fit=(True,'precision_bounded','legacy'),validation_only=(True,'legacy','relative_local'),combined=(True,'precision_bounded','relative_local'))
VARIANTS=dict(no_fit=None,original=(False,False),precision_single=(True,False),legacy_iterative=(False,True),precision_iterative=(True,True))


def configure(a,variant,validation):
    a.receiver_solver_mode='projected';a.receiver_search_mode='legacy'
    a.receiver_fit_mode='precision_bounded' if variant=='precision_iterative' else 'legacy'
    a.receiver_validation_mode=validation;a.fit_received=variant!='no_fit'
    a.receiver_diagnostic_variant=VARIANTS[variant] if variant not in ('no_fit','original','precision_iterative') else None
    a.receiver_fit_diagnostic=None
    return a


def errors(pred,truth,mask):
    delta=pred[mask]-truth[mask]
    return dict(value_mae=float(np.abs(delta[:,0]).mean()) if len(delta) else None,gradient_mae=float(np.linalg.norm(delta[:,1:],axis=1).mean()) if len(delta) else None,
        supported=int(mask.sum()),queries=len(mask))


def audit_snapshot(snapshot,meta,variants=None,solver_modes=None):
    agent,x,packet,world,tick=snapshot;original=digest((agent,x,world));started=time.perf_counter()
    base=copy.deepcopy(agent);event=base.assimilate(packet);measurements=digest([(s.readings,s.transitions) for s in base.stages]);remote=digest(base.remote_terrain)
    quad=ranking(base,x,details=True)['quadrature'];points=np.array([q for e in quad for q in e['points']])
    truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
    initial,_=field_queries(base,points);support=np.isfinite(initial).all(1);anchors=copy.deepcopy(base.navigation['terrain_anchors'])
    results={};predictions={};local,_=_local(copy.deepcopy(base))
    for variant in (variants or VARIANTS):
        previous=None
        for validator in (('legacy',) if variant=='no_fit' else ('legacy','relative_local')):
            a=configure(copy.deepcopy(base),variant,validator)
            if variant=='precision_iterative' and solver_modes is not None:a.receiver_solver_mode,a.receiver_search_mode=solver_modes
            d=a.update(x.copy());state=fixed_state(a,x)
            proposal=getattr(a,'reference',None)
            if proposal is None:proposal=local
            proposed=[w.detach().clone() for w in proposal]
            if previous is not None:assert all(torch.equal(u,v) for u,v in zip(previous,proposed)),'validator altered proposal'
            previous=proposed
            probe=copy.deepcopy(base)
            for stage,w in zip(probe.stages,proposed):stage.weights=w
            # Score the proposal with the stage selection it would publish.
            # Old overlap selection can disagree even when coefficients match.
            from .route_field import sync
            sync(probe,publish=True)
            pred,pm=field_queries(probe,points);committed,cm=field_queries(a,points)
            mask=support&np.isfinite(pred).all(1)&np.isfinite(committed).all(1)
            ctx=hypothetical_context(a);ctx['terrain_anchors']=copy.deepcopy(anchors)
            ranks=ranking(a,x,ctx)
            natural=ranking(a,x,hypothetical_context(a))
            assert digest([(s.readings,s.transitions) for s in a.stages])==measurements and digest(a.remote_terrain)==remote
            key=f'{variant}/{validator}'
            results[key]=dict(variant=variant,validator=validator,accepted=d['accepted'],reason=d['reason'],version=a.version,diagnostic=d,
                proposed_fields=[w.tolist() for w in proposed],committed_fields=[s.weights.tolist() for s in a.stages],
                proposed_errors=errors(pred,truth,mask),committed_errors=errors(committed,truth,mask),common_support=mask.tolist(),
                proposed_support=pm,committed_support=cm,control=state['control'],target=state['model']['target'],kinetic=state['kinetic'],shaping=state['shaping'],
                ranking=ranks,natural_ranking=natural,proposal_max_correction=max(float((w-z).abs().max()) for w,z in zip(proposed,local)))
            predictions[key]=(pred,committed)
    common=support.copy()
    for p,c in predictions.values():common &= np.isfinite(p).all(1)&np.isfinite(c).all(1)
    # Fixed common mask across methods; preserve unsupported predictions explicitly.
    for key,(p,c) in predictions.items():
        results[key]['common_errors']=dict(proposed=errors(p,truth,common),committed=errors(c,truth,common))
    assert digest((agent,x,world))==original
    return dict(meta,packet=packet.identity,source_ids=list(packet.sources),receiver=int(agent.agent_id),time=agent.now,stage=agent.stages[-1].id,
        event=event,points=points.tolist(),truth=truth.tolist(),common_support=common.tolist(),initial_support=support.tolist(),results=results,
        state_unchanged=True,measurements_unchanged=True,seconds=time.perf_counter()-started)


def load_saved_snapshot(path):
    """Load trusted local artifacts, including experiments launched with -m."""
    class Loader(pickle.Unpickler):
        def find_class(self,module,name):
            if module=='__main__' and name=='CorrectionLearner':return CorrectionLearner
            return super().find_class(module,name)
    with gzip.open(path,'rb') as f:return Loader(f).load()


def snapshot_job(task):
    root,path,meta=task;torch.set_num_threads(1);root=Path(root)
    ctx=load_saved_snapshot(path)
    rows=[]
    if meta['group']=='navigation':snapshots=[(ctx,meta)]
    else:
        snapshots=[]
        for sender in (0,1):
            a=copy.deepcopy(ctx['agents'][sender]);packet=a.export(ctx['states'][sender],str(1-sender))
            snapshots.append(((ctx['agents'][1-sender],ctx['states'][1-sender],packet,ctx['worlds'][1-sender],8),dict(meta,intended=sender==dict(complementary=0,reverse=1,redundant=None)[meta['case']])))
    for snap,labels in snapshots:rows.append(audit_snapshot(snap,labels))
    out=root/'offline'/f'{Path(path).name[:-7]}.json.gz';dump(out,rows)
    return dict(path=out.name,sha256=file_hash(out),audits=len(rows))


def manifest(source):
    jobs,_=_snapshot_jobs(source)
    m=dict(schema=1,development_only=True,source=str(Path(source).resolve()),source_hashes=source_hashes(),settings=SETTINGS,
        snapshot_inputs=[dict(path=str(Path(p).resolve()),sha256=file_hash(p),meta=meta) for p,meta in jobs],
        variants=VARIANTS,arms=ARMS,seeds=list(SEEDS),animation_seed=79200,
        specs={f'{c}_{s}':specification(c,s) for c in CASES for s in SEEDS},
        config=asdict(replace(configuration(500),route_field_mode='supported_field')),communication=asdict(communication('field')),
        required_gate=[f'{c}_{s}' for c in ('complementary','reverse') for s in SEEDS],expected_episodes=30,expected_missions=60,
        policy_replay=False,held_out_executed=False,validator_reference='same-update local-only proposal; whole-model checks retained; whole-model rollback',
        solver_failure='whole update rejected, including iteration exhaustion',query_semantics='correlated quadrature occurrences, not independent trials')
    return dict(m,fingerprint=fingerprint(m))


def default_checks(root):
    root=Path(root);m=load(root/'manifest.json');checks=[]
    source=Path(m['source']);archived={}
    for p in (source/'gate').glob('*_audits.json.gz'):
        parts=p.name.split('_');case=parts[0];seed=int(parts[1])
        for a in load(p):archived[(case,seed,'gate',a['packet'],a['receiver'])]=a
    for p in (source/'episodes').glob('*.json.gz'):
        r=load(p)
        for a in r['audits']:archived[(r['scene'],r['spec']['seed'],r['method'],a['packet'],a['receiver'])]=a
    for p in (root/'offline').glob('*.json.gz'):
        for a in load(p):
            old=archived[(a['case'],a['seed'],a.get('method','gate'),a['packet'],a['receiver'])];ok=True;fields=True;decisions=True
            for key,oldkey in [('no_fit/legacy','no_fit'),('original/legacy','fit')]:
                b=a['results'][key];z=old['branches'][oldkey]
                fields &= np.array_equal(b['committed_fields'],z['fields'])
                decisions &= b['accepted']==z['accepted'] and b['version']==z['version'] and np.allclose(b['control'],z['control'],rtol=1e-10,atol=1e-12) and b['target']==z['target']
                for metric in ('value_mae','gradient_mae'):
                    new=b['committed_errors'][metric];prev=old['queries']['errors'][oldkey][metric]
                    ok &= (new==prev) if new is None or prev is None else abs(new-prev)<1e-12
            checks.append(dict(case=a['case'],seed=a['seed'],packet=a['packet'],receiver=a['receiver'],errors=bool(ok),fields_bitwise=bool(fields),decisions=bool(decisions)))
    result=dict(passed=len(checks)==160 and all(c['errors'] and c['fields_bitwise'] and c['decisions'] for c in checks),audits=len(checks),checks=checks)
    dump(root/'default_checks.json',result);return result


def clean_prefix(p):
    if isinstance(p,dict):return {k:clean_prefix(v) for k,v in p.items() if k!='receiver_options'}
    if isinstance(p,(list,tuple)):return [clean_prefix(v) for v in p]
    return p


class Captures(DeliverySnapshots):
    def boundary(self,ctx):
        c=dict(ctx)
        if abs(ctx['time']-.96)<1e-9:c['updates']=clean_prefix(ctx['updates'])
        super().boundary(c)


class CorrectionLearner(ReconstructionLearner):
    """Accounting-only experiment subclass; fitting delegates to operational code."""
    def _field_fit(self):
        before=self.message_fit_work;started=time.perf_counter()
        result=super()._field_fit();diag=getattr(self,'receiver_fit_diagnostic',None)
        if not hasattr(self,'receiver_work'):self.receiver_work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0,information_inversions=0,seconds=0.)
        if diag is None:
            delta=self.message_fit_work-before
            self.receiver_work['residual_rows']+=delta;self.receiver_work['jacobian_rows']+=delta
            self.receiver_work['linear_solves']+=sum(any(s.contains(a[:2],self.cfg) for m in self.received.values() for a in m.anchors) for s in self.stages) if delta else 0
        else:
            for stage in diag.get('stages',[]):
                for k,v in stage['work'].items():self.receiver_work[k]=self.receiver_work.get(k,0)+v
            if diag.get('local_information') is not None:self.receiver_work['information_inversions']+=1+len(self.stages)
        self.receiver_work['seconds']+=time.perf_counter()-started
        return result
    def diagnostic_summary(self):
        return dict(super().diagnostic_summary(),receiver_work=getattr(self,'receiver_work',{}))


def episode_job(task):
    root,key,arm=task;root=Path(root);m=load(root/'manifest.json');spec=m['specs'][key];fit,fitmode,validator=ARMS[arm]
    torch.set_num_threads(1);started=time.perf_counter();identity=f'{key}_{arm}'
    try:
        cfg=Config(**m['config']);comm=CommunicationConfig(**m['communication'])
        def factory(cfg,x,goal,agent_id,communication):
            return CorrectionLearner(cfg,x,goal,agent_id=agent_id,communication=communication,mode='evidence',fit_received=fit,
                receiver_fit_mode=fitmode,receiver_validation_mode=validator,**(dict(receiver_solver_mode=m['selected_solver'][0],receiver_search_mode=m['selected_solver'][1]) if arm in ('corrected_fit','combined') and 'selected_solver' in m else {}),network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))),
                risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'])
        capture=Captures(clean_prefix(load(root/'acquisition'/f'{key}.json.gz')))
        r=execute(spec,'reciprocal_fixed',comm,audit=False,learner_factory=factory,world_factory=NetworkWorld,config=cfg,boundary_callback=capture.boundary,delivery_callback=capture.delivery)
        audits=[]
        for n,selected in enumerate(capture.selected.values()):
            a=audit_snapshot(selected['snapshot'],dict(case=spec['scene'],seed=spec['seed'],method=arm,group='navigation'),variants=('no_fit','original','precision_iterative'),solver_modes=m.get('selected_solver'))
            a['purposes']=selected['purposes'];a['snapshot']=save_snapshot(root/'snapshots'/f'{identity}_{n}.pkl.gz',selected['snapshot']);audits.append(a)
        r.update(method=arm,scene=spec['scene'],receiver_options=dict(fit=fitmode,validation=validator,received_fit=fit),manifest_fingerprint=m['fingerprint'],
            acquisition_check=capture.acquisition,first_branch_entries=capture.entered,snapshot_seconds=capture.seconds,receiver_audits=audits,
            audit_seconds=sum(a['seconds'] for a in audits),audit_flow=sum(b['diagnostic']['counterfactual_steps'] for a in audits for b in a['results'].values()))
        path=root/'episodes'/f'{identity}.json.gz';dump(path,r)
        status=dict(id=identity,complete=True,path=path.name,sha256=file_hash(path),seconds=time.perf_counter()-started,goals=sum(a['success'] for a in r['agents']))
    except Exception as e:status=dict(id=identity,complete=False,error=str(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
    dump(root/'status'/f'{identity}.json',status);return status


def suite(root,source,workers):
    root=Path(root);m=manifest(source);torch.set_num_threads(1)
    if (root/'manifest.json').exists():
        if load(root/'manifest.json')['fingerprint']!=m['fingerprint']:raise ValueError('Frozen source/config mismatch; use archive or fresh output')
    else:
        root.mkdir(parents=True,exist_ok=False)
        for sub in ('offline','acquisition','gate','snapshots','episodes','status','figures','verification'):(root/sub).mkdir()
        dump(root/'manifest.json',m);archive_sources(root)
    tasks=[]
    for entry in m['snapshot_inputs']:
        if file_hash(entry['path'])!=entry['sha256']:raise ValueError('Input snapshot changed')
        if not (root/'offline'/f"{Path(entry['path']).name[:-7]}.json.gz").exists():tasks.append((str(root),entry['path'],entry['meta']))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,f in enumerate(as_completed([pool.submit(snapshot_job,t) for t in tasks])):
            f.result()
            if n%20==0:print(json.dumps(dict(offline_files_complete=n+1,total=len(tasks))),flush=True)
    if not default_checks(root)['passed']:raise RuntimeError('Default reproduction failed; navigation blocked')
    gates=[]
    for key,spec in m['specs'].items():
        path=root/'gate'/f'{key}.json'
        if not path.exists():
            ctx=acquire(spec,Config(**m['config']),CommunicationConfig(**m['communication']))
            dump(root/'acquisition'/f'{key}.json.gz',prefix(ctx));g,audits=gate_snapshot(ctx,spec)
            g['snapshot']=save_snapshot(root/'snapshots'/f'{key}_acquisition.pkl.gz',ctx);dump(path,g);dump(root/'gate'/f'{key}_audits.json.gz',audits)
        gates.append(load(path))
    gate=dict(passed=all(g['passed'] for g in gates if g['required']),cases=gates);dump(root/'gate.json',gate);print(json.dumps(dict(gate_passed=gate['passed'])),flush=True)
    if gate['passed']:
        tasks=[]
        for key in m['specs']:
            for arm in ARMS:
                p=root/'status'/f'{key}_{arm}.json'
                if p.exists():
                    s=load(p)
                    if not s['complete']:raise RuntimeError('Preserved incomplete episode; diagnose before resuming')
                    assert file_hash(root/'episodes'/s['path'])==s['sha256']
                else:tasks.append((str(root),key,arm))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(episode_job,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    from .receiver_corrections_report import report,render
    report(root);render(root)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['suite','analyze','report','render'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('results/complementary_reconstruction_v1'));p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(argv)
    if a.command=='suite':return suite(a.out,a.source,a.workers)
    from . import receiver_corrections_report
    return getattr(receiver_corrections_report,'report' if a.command=='analyze' else a.command)(a.out)

if __name__=='__main__':main()
