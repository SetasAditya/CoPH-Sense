"""Frozen four-way receiver solver isolation. Runtime code has no evaluator imports."""
import argparse,copy,gzip,json,os,pickle,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from . import receiver_fitting as rf
from . import solver_diagnosis as sd
from .receiver_corrections import configure,load_saved_snapshot,manifest as navigation_manifest,episode_job,ARMS
from .receiver_reconstruction import fingerprint,file_hash
from .cli import source_hashes,archive_sources
from .evaluation import load,dump
from .attribution_audits import digest
from .complementary_reconstruction import acquire,gate_snapshot,prefix,save_snapshot
from .records import Config
from .cooperative import CommunicationConfig

CONFIGS={'projected_legacy':('projected','legacy'),'active_set_legacy':('active_set','legacy'),'projected_diagnostic':('projected','diagnostic'),'active_set_diagnostic':('active_set','diagnostic')}
ORDERS=('numpy_first','torch_first')
SOURCE=Path('results/solver_diagnosis_v1')


def key(a):return (a['file'],a.get('sender'),a['packet'])


def worker(root,order,name):
    root=Path(root);m=load(root/'manifest.json');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    archived={key(a):a for a in load(SOURCE/'audits.json.gz')};results=[];began=time.perf_counter()
    with gzip.open(root/'captured_inputs.pkl.gz','rb') as f:canonical=pickle.load(f)
    for entry in m['inputs']:
        value=load_saved_snapshot(entry['path']);meta=entry['meta'];snapshots=[]
        if isinstance(value,tuple):snapshots=[(value,meta)]
        else:
            for sender in (0,1):
                a=copy.deepcopy(value['agents'][sender]);packet=a.export(value['states'][sender].copy(),str(1-sender))
                if packet:snapshots.append(((value['agents'][1-sender],value['states'][1-sender],packet,value['worlds'][1-sender],8),dict(meta,sender=sender)))
        for snapshot,labels in snapshots:
            agent,x,packet,world,tick=snapshot;before=digest((agent,x,world));base=copy.deepcopy(agent);base.assimilate(packet)
            quad=sd.ranking(base,x,details=True)['quadrature'];points=np.array([q for edge in quad for q in edge['points']]);truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
            initial,_=sd.field_queries(base,points);support=np.isfinite(initial).all(1);local,_=sd._local(copy.deepcopy(base))
            def query(weights,publish):
                probe=copy.deepcopy(base)
                for st,ww in zip(probe.stages,weights):st.weights=ww.clone()
                if publish:sd.sync(probe,publish=True)
                pred,_=sd.field_queries(probe,points);return pred
            local_pred=query(local,True)
            a=configure(copy.deepcopy(base),'precision_iterative','legacy');captured=[];orig=rf.solve
            def capture(stage,w0,H,blocks,width,iterative=True,max_iterations=None):
                frozen=canonical[(*key(dict(labels,packet=packet.identity)),stage.id)]
                stage=sd.Stage(stage.id,frozen['centers']);w0,H,blocks,width=(frozen[k] for k in ('w0','H','blocks','width'))
                w,d=orig(stage,w0,H,blocks,width,iterative,max_iterations)
                captured.append((stage,w0,H,blocks,width,w,d));return w,d
            rf.solve=capture
            try:a.update(x.copy())
            finally:rf.solve=orig
            archived_a=archived[key(dict(labels,packet=packet.identity))];problems=[];proposals={}
            for stage,w0,H,blocks,width,original,old in captured:
                w,d=orig(stage,w0,H,blocks,width,solver_mode=CONFIGS[name][0],search_mode=CONFIGS[name][1]);proposals[stage.id]=(w,d)
                ap=next(p for p in archived_a['problems'] if p['stage']==stage.id)
                baseline_exact=all(old[k]==ap[k] for k in ('status','iterations','objective_final','stationarity'))
                p=dict(stage=stage,w0=w0,H=H,blocks=blocks,width=width)
                p['scale']=old['normalization'];p['initial']=old['objective_initial']
                traced,_=sd.trace(p);baseline_bitwise=bool(torch.equal(original,traced))
                candidate_match=None;port_bitwise=None
                if name=='active_set_diagnostic':
                    expected=ap['candidates']['active_set_gauss_newton'];reference,rd=sd.active_set_gauss_newton(p)
                    port_bitwise=bool(torch.equal(w,reference)) and all(d[k]==rd[k] for k in ('status','iterations','objective_final','stationarity'))
                    candidate_match=d['status']==expected['status'] and d['iterations']==expected['iterations'] and d['objective_final']==expected['objective'] and d['stationarity']==expected['stationarity']
                problems.append(dict(stage=stage.id,result=d,weights=w.tolist(),original=old,baseline_archive_exact=baseline_exact,baseline_bitwise=baseline_bitwise,
                    candidate_archive_exact=candidate_match,port_bitwise=port_bitwise,input_sha256=canonical[(*key(dict(labels,packet=packet.identity)),stage.id)]['sha256'],original_active_exhaustion=ap['candidates']['active_set_gauss_newton']['status']!='converged',
                    bound_ok=bool((w-w0).abs().max()<=.5+1e-12),finite=bool(torch.isfinite(w).all()),near_tolerance=bool(.5e-6<=d['stationarity']<=2e-6)))
            # Reuse the already solved proposals through both unchanged validators.
            branches={}
            for validator in ('legacy','relative_local'):
                b=configure(copy.deepcopy(base),'precision_iterative',validator)
                b.receiver_solver_mode,b.receiver_search_mode=CONFIGS[name]
                def frozen(stage,w0,H,blocks,width,iterative=True,max_iterations=None,**kwargs):
                    w,d=proposals[stage.id];return w.clone(),copy.deepcopy(d)
                rf.solve=frozen
                try:diag=b.update(x.copy())
                finally:rf.solve=orig
                proposal=query([proposals[st.id][0] if st.id in proposals else ww for st,ww in zip(base.stages,local)],True)
                committed,_=sd.field_queries(b,points);common=support&np.isfinite(local_pred).all(1)&np.isfinite(proposal).all(1)&np.isfinite(committed).all(1)
                def error(pred):
                    delta=pred[common]-truth[common]
                    return dict(value_mae=float(np.abs(delta[:,0]).mean()) if len(delta) else None,gradient_mae=float(np.linalg.norm(delta[:,1:],axis=1).mean()) if len(delta) else None,supported=int(common.sum()),queries=len(common))
                branches[validator]=dict(local_error=error(local_pred),proposal_error=error(proposal),committed_error=error(committed),accepted=diag['accepted'],reason=diag['reason'],version=b.version,diagnostic=diag,fields=[s.weights.tolist() for s in b.stages])
            assert digest((agent,x,world))==before
            results.append(dict(labels,packet=packet.identity,receiver=agent.agent_id,problems=problems,branches=branches,isolated=True))
    env=dict(order=order,numpy=np.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),torch_config=torch.__config__.show(),python=sys.version)
    dump(root/'diagnostics'/f'{name}_{order}.json.gz',dict(config=name,canonical_inputs_sha256=file_hash(root/'captured_inputs.pkl.gz'),environment=env,audits=results,seconds=time.perf_counter()-began))


def capture_inputs(root):
    """Freeze numerical problems once; alternate kernels must not refit their inputs."""
    import hashlib
    root=Path(root);m=load(root/'manifest.json');torch.set_num_threads(1);problems={}
    for entry in m['inputs']:
        value=load_saved_snapshot(entry['path']);snapshots=[];meta=entry['meta']
        if isinstance(value,tuple):snapshots=[(value,meta)]
        else:
            for sender in (0,1):
                a=copy.deepcopy(value['agents'][sender]);packet=a.export(value['states'][sender],str(1-sender))
                if packet:snapshots.append(((value['agents'][1-sender],value['states'][1-sender],packet,None,8),dict(meta,sender=sender)))
        for snapshot,labels in snapshots:
            agent,x,packet,_,_=snapshot;a=configure(copy.deepcopy(agent),'precision_iterative','legacy');a.assimilate(packet);orig=rf.solve
            def capture(stage,w0,H,blocks,width,iterative=True,max_iterations=None):
                h=hashlib.sha256()
                for t in [stage.centers,w0,H]+[t for block in blocks for t in block]:h.update(str(tuple(t.shape)).encode());h.update(t.detach().numpy().tobytes())
                h.update(str(width).encode());identifier=(*key(dict(labels,packet=packet.identity)),stage.id)
                problems[identifier]=dict(centers=stage.centers.clone(),w0=w0.clone(),H=H.clone(),blocks=[tuple(t.clone() for t in block) for block in blocks],width=width,sha256=h.hexdigest())
                return orig(stage,w0,H,blocks,width,iterative,max_iterations)
            rf.solve=capture
            try:a._field_fit()
            finally:rf.solve=orig
    assert len(problems)==697
    with gzip.open(root/'captured_inputs.pkl.gz','wb') as f:pickle.dump(problems,f,protocol=5)
    dump(root/'capture.json',dict(problems=len(problems),import_order='numpy_first',sha256=file_hash(root/'captured_inputs.pkl.gz'),inputs=[dict(key=k,sha256=v['sha256']) for k,v in problems.items()]))


def select(root):
    root=Path(root);datasets={(o,n):load(root/'diagnostics'/f'{n}_{o}.json.gz') for o in ORDERS for n in CONFIGS}
    rows={};summary=[];disagreements=[];remaining=[];reproduction=True
    for (order,name),data in datasets.items():
        rr={}
        for a in data['audits']:
            for p in a['problems']:rr[(*key(a),p['stage'])]=p
        rows[order,name]=rr
        if data['canonical_inputs_sha256']!=file_hash(root/'captured_inputs.pkl.gz'):raise ValueError('Canonical input bytes changed')
        if order=='numpy_first':
            reproduction &= len(data['audits'])==409 and len(rr)==697 and all(p['baseline_archive_exact'] and p['baseline_bitwise'] for p in rr.values())
            if name=='active_set_diagnostic':reproduction &= all(p['candidate_archive_exact'] and p['port_bitwise'] for p in rr.values())
        summary.append(dict(order=order,config=name,audits=len(data['audits']),problems=len(rr),converged=sum(p['result']['success'] for p in rr.values()),
            near_tolerance=sum(p['near_tolerance'] for p in rr.values()),accepted={v:sum(a['branches'][v]['accepted'] for a in data['audits']) for v in ('legacy','relative_local')},
            residual_rows=sum(p['result']['work']['residual_rows'] for p in rr.values()),linear_solves=sum(p['result']['work']['linear_solves'] for p in rr.values()),seconds=data['seconds']))
    for name in CONFIGS:
        for k,p in rows['numpy_first',name].items():
            q=rows['torch_first',name][k]
            if p['input_sha256']!=q['input_sha256']:raise ValueError('Cross-order problem mismatch')
            if p['result']['status']!=q['result']['status']:disagreements.append(dict(key=k,config=name,numpy=p['result']['stationarity'],torch=q['result']['stationarity'],numpy_status=p['result']['status'],torch_status=q['result']['status']))
            if p['original_active_exhaustion']:remaining.append(dict(key=k,config=name,numpy=p['result'],torch=q['result']))
    candidates=[]
    for name in ('active_set_legacy','active_set_diagnostic'):
        orders=[]
        for order in ORDERS:
            base=rows[order,'projected_legacy'];rr=rows[order,name]
            regressions=[k for k,p in base.items() if p['result']['success'] and p['result']['stationarity']<.5e-6 and not rr[k]['result']['success']]
            failed=sum(not p['result']['success'] for p in rr.values());baseline_failed=sum(not p['result']['success'] for p in base.values())
            valid=all(p['bound_ok'] and p['finite'] and (not p['result']['success'] or p['result']['stationarity']<=1e-6) for p in rr.values())
            orders.append(dict(order=order,stable_regressions=regressions,failed=failed,baseline_failed=baseline_failed,valid=valid,passed=valid and not regressions and failed<baseline_failed))
        candidates.append(dict(config=name,orders=orders,eligible=reproduction and all(o['passed'] for o in orders)))
    selected=next((c['config'] for c in candidates if c['eligible']),None)
    result=dict(reproduction_passed=bool(reproduction),candidates=candidates,selected=selected,selected_solver=CONFIGS[selected] if selected else None,
        tests_required=True,selection_uses_estimation_or_navigation_outcomes=False)
    dump(root/'selection.json',result);dump(root/'solver_summary.json',summary);dump(root/'import_disagreements.json',disagreements);dump(root/'archived_exhaustions.json',remaining)
    return result


def freeze(root):
    root=Path(root);tasks=sd.jobs();m=dict(schema=1,source_hashes=source_hashes(),inputs=[dict(path=str(Path(p).resolve()),sha256=file_hash(p),meta=meta) for p,meta in tasks],
        diagnosis_sha256=file_hash(SOURCE/'audits.json.gz'),configs=CONFIGS,orders=ORDERS,settings=rf.SETTINGS,search=dict(legacy_trials=13,diagnostic_trials=40,diagnostic_rounding_floor='10 * float64 epsilon * abs(objective)'),
        navigation=navigation_manifest(Path('results/complementary_reconstruction_v1')),expected_audits=409,expected_problems=697,
        selection='Prefer active_set_legacy then active_set_diagnostic; bounds/certification, no stable convergence losses, fewer failures under both import orders; no truth-based selection')
    m['fingerprint']=fingerprint(m)
    if root.exists():
        if load(root/'manifest.json')['fingerprint']!=m['fingerprint']:raise ValueError('Frozen source/config mismatch')
    else:
        root.mkdir(parents=True);(root/'diagnostics').mkdir();(root/'verification').mkdir();(root/'figures').mkdir();dump(root/'manifest.json',m);archive_sources(root)
    return m


def run(root):
    root=Path(root);m=freeze(root)
    if not (root/'verification/tests.json').exists():
        with (root/'verification/full-tests.log').open('w') as f:
            test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],stdout=f,stderr=f,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/coph-mpl'))
        import re
        log=(root/'verification/full-tests.log').read_text();errors=re.findall(r'^(?:ERROR|FAIL): (.+)$',log,re.M)
        allowed=not errors if test.returncode==0 else len(errors)==1 and 'test_frozen_sources_specs_and_runtime_config' in errors[0]
        dump(root/'verification/tests.json',dict(passed_except_preserved_archive_check=allowed,returncode=test.returncode,errors=errors,total=int(re.search(r'Ran (\d+) tests',log).group(1))))
    if not load(root/'verification/tests.json')['passed_except_preserved_archive_check']:raise RuntimeError('Tests failed; evaluation blocked')
    # Capture once, before either import-order stress test. All eight solvers see these bytes.
    if not (root/'captured_inputs.pkl.gz').exists():
        code='import numpy; import torch; from coph_stagewise.active_set_suite import capture_inputs; capture_inputs('+repr(str(root))+')'
        subprocess.run([sys.executable,'-c',code],check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    # Separate interpreters per configuration AND import order; never change kernels mid-process.
    processes=[]
    for order in ORDERS:
        for name in CONFIGS:
            if (root/'diagnostics'/f'{name}_{order}.json.gz').exists():continue
            imports='import numpy; import torch' if order=='numpy_first' else 'import torch; import numpy'
            code=imports+'; from coph_stagewise.active_set_suite import worker; worker('+repr(str(root))+','+repr(order)+','+repr(name)+')'
            log=(root/'diagnostics'/f'{name}_{order}.log').open('w');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            processes.append((name,order,subprocess.Popen([sys.executable,'-c',code],stdout=log,stderr=log,env=env),log))
    for name,order,p,log in processes:
        rc=p.wait();log.close()
        if rc:raise RuntimeError(f'{name}/{order} failed; log preserved')
        print(name,order,'complete',flush=True)
    decision=select(root);print(json.dumps(decision),flush=True)
    if decision['selected'] is not None:navigation(root,decision)
    from .active_set_reporting import report
    report(root)


def baseline_compare(root):
    root=Path(root);records=[]
    def operational(r):
        return dict(samples=r['samples'],observations=r['observations'],packets=r['packets'],events=r['events'],
            transitions=[[{k:t[k] for k in ('time','x','u','after','version','stage','energy')} for t in ts] for ts in r['trajectories']],
            updates=[[dict(time=u['time'],tick=u['tick'],packets=u['packets'],
                models=[{k:z[k] for k in ('version','damping','policy','stages')} for z in (u['pre'],u['post'])],
                decision={k:u['diagnostic'][k] for k in ('accepted','reason','counterfactual_steps','field_validation_before','field_validation_after','plant_validation_before','plant_validation_after')}) for u in us] for us in r['updates']],
            work=[{k:a[k] for k in ('replay_work','identification_work','max_update_work','bytes')} for a in r['agents']])
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k not in ('entry_navigation',)}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    def equivalent(a,b):
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
        if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
        if isinstance(a,float):return isinstance(b,(float,int)) and bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
        return a==b
    for key in load(root/'manifest.json')['specs']:
        for arm in ('local_only','legacy','validation_only'):
            name=f'{key}_{arm}.json.gz';p=root/'episodes'/name;old=Path('results/receiver_corrections_v1/episodes')/name
            if not p.exists():records.append(dict(id=name,passed=False,reason='missing episode'));continue
            a=load(p);b=load(old);x=clean(operational(a));y=clean(operational(b))
            records.append(dict(id=name,passed=equivalent(x,y),operational_bitwise=fingerprint(x)==fingerprint(y),full_serialized_equal=fingerprint(a)==fingerprint(b),compressed_bytes_equal=file_hash(p)==file_hash(old)))
    result=dict(passed=len(records)==18 and all(r['passed'] for r in records),records=records);dump(root/'baseline_checks.json',result);return result


def navigation(root,selection):
    root=Path(root);out=root/'navigation';m=copy.deepcopy(load(root/'manifest.json')['navigation']);m.pop('fingerprint',None)
    m.update(selected_solver=selection['selected_solver'],parent_manifest=load(root/'manifest.json')['fingerprint']);m['fingerprint']=fingerprint(m)
    if not out.exists():
        out.mkdir()
        for sub in ('offline','acquisition','gate','snapshots','episodes','status','figures','verification'):(out/sub).mkdir()
        dump(out/'manifest.json',m)
    gates=[]
    for key,spec in m['specs'].items():
        p=out/'gate'/f'{key}.json'
        if not p.exists():
            ctx=acquire(spec,Config(**m['config']),CommunicationConfig(**m['communication']));dump(out/'acquisition'/f'{key}.json.gz',prefix(ctx));g,aa=gate_snapshot(ctx,spec)
            g['snapshot']=save_snapshot(out/'snapshots'/f'{key}_acquisition.pkl.gz',ctx);dump(p,g);dump(out/'gate'/f'{key}_audits.json.gz',aa)
        gates.append(load(p))
    gate=dict(passed=all(g['passed'] for g in gates if g['required']),cases=gates);dump(out/'gate.json',gate)
    if gate['passed']:
        for arms in (('local_only','legacy','validation_only'),('corrected_fit','combined')):
            if arms[0]=='corrected_fit' and not baseline_compare(out)['passed']:break
            tasks=[]
            for key in m['specs']:
                for arm in arms:
                    status=out/'status'/f'{key}_{arm}.json'
                    if not status.exists():tasks.append((str(out),key,arm))
                    elif not load(status)['complete']:raise RuntimeError('Preserved incomplete episode')
            with ProcessPoolExecutor(max_workers=4) as pool:
                for f in as_completed([pool.submit(episode_job,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    from .receiver_corrections_report import report,render
    report(out);render(out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('suite','report'));p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.command=='suite':run(a.out)
    else:
        from .active_set_reporting import report
        report(a.out)

if __name__=='__main__':main()
