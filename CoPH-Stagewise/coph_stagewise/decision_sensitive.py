"""Preregistered decision-sensitive receiver experiment; no runtime learner changes."""
import argparse,copy,datetime,gzip,json,math,os,pickle,platform,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import scipy
from .evaluation import dump,load
from .receiver_reconstruction import fingerprint,file_hash
from .cli import source_hashes,archive_sources
from . import complementary_reconstruction as cr
from . import receiver_corrections as rc
from .cooperative import CommunicationConfig
from .crossing_demo import configuration
from .hazard_demo import communication
from .decision_suite import NODES,EDGES
from .reconstruction_diagnostics import ranking,eligibility
from .route_field import values as route_values
from .learner import Learner
from .model import tensor,features
from .attribution_audits import digest
from .records import Config
from .maze_family import NetworkWorld
from .staggered_demo import execute

SEEDS=tuple(range(79300,79308));XS=(2.,2.5,3.);RATIOS=(.6,.8,.9,1.1,1.25,1.6)
ARMS=('local_only','legacy','corrected_fit','combined')
DIRECT=((1,2),);BYPASS=((1,4),(4,5),(5,2));ESTAR=3.4*(1+24*.02)/24
BOOTSTRAPS=128;ANALYSIS_SEED=179300;ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'results/receiver_corrections_v1'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def patch_integral(a,b,patch):
    a,b=np.asarray(a,float),np.asarray(b,float);length=float(np.linalg.norm(b-a))
    if length==0:return 0.
    center=np.asarray(patch[:2]);sigma=np.asarray(patch[2:4]);d=(b-a)/length
    A=float(np.sum((d/sigma)**2));B=float(np.sum((a-center)*d/sigma**2));C=float(np.sum(((a-center)/sigma)**2))
    k=math.sqrt(A/2)
    return patch[4]*math.exp(-.5*(C-B*B/A))*math.sqrt(math.pi/(2*A))*(math.erf(k*(length+B/A))-math.erf(k*B/A))

def truth_costs(spec):
    def cost(edges):
        return sum(np.linalg.norm(np.asarray(NODES[j])-NODES[i])*(1+24*.02)+24*sum(patch_integral(NODES[i],NODES[j],p) for p in spec['patches']) for i,j in edges)
    direct,bypass=map(cost,(DIRECT,BYPASS));delta=float(direct-bypass)
    return dict(direct=float(direct),bypass=float(bypass),margin=delta,optimal=['direct','bypass'] if abs(delta)<=1e-10 else ['direct' if delta<0 else 'bypass'])

def grid():
    scenes={}
    for x in XS:
        for ratio in RATIOS:
            name=f'x{x:g}_e{ratio:g}';a=ratio*ESTAR/patch_integral(NODES[1],NODES[2],(x,0.,.6,1.,1.))
            scenes[name]=dict(control=False,x=x,ratio=ratio,patches=[[x,0.,.6,1.,a]],starts='complementary')
    scenes['hazard_free']=dict(control=True,x=None,ratio=None,patches=[],starts='complementary')
    scenes['redundant_control']=dict(scenes['x2.5_e0.9'],control=True,starts='redundant')
    return scenes

def specification(scene,seed):
    g=grid()[scene];s=cr.specification(g['starts'],seed)
    s.update(name=scene,scene=scene,patches=copy.deepcopy(g['patches']),decision_sensitive=copy.deepcopy(g));return s

def branch_scores(agent,x,context=None):
    r=ranking(agent,x,context,True);edges={tuple(sorted(e['edge'])):e for e in r['quadrature'] if -1 not in e['edge']}
    def cost(path):return sum(edges[tuple(sorted(e))]['cost'] for e in path)
    direct,bypass=map(cost,(DIRECT,BYPASS));support=[]
    for e in ((1,2),(1,4)):
        row=edges[tuple(sorted(e))];support.append(dict(edge=e,supported=sum(row['route_field']['field_support']),queries=len(row['points'])))
    return dict(direct=float(direct),bypass=float(bypass),margin=float(direct-bypass),support=support,
        eligible_support=all(e['supported']>0 for e in support),quadrature=[edges[tuple(sorted(e))] for e in DIRECT+BYPASS])

def context_margin(context,quadrature):
    costs=[float(np.sum(np.asarray(e['lengths'])*(1+24*route_values(e['points'],context)))) for e in quadrature]
    return costs[0]-sum(costs[1:])

def bootstrap_design(agent):
    """Original local normal equations; fixed overlap rows and shared time-block multiplicities."""
    stages=agent.stages;cfg=agent.cfg;n=len(stages[0].weights);total=n*len(stages);rows=[];targets=[];blocks=[];uids=[]
    for k,s in enumerate(stages):
        for r in s.readings:
            if int(r.uid.split(':')[-1])%5==0:continue
            row=torch.zeros(total,dtype=torch.float64);row[k*n:(k+1)*n]=features(tensor(r.q),s.centers,cfg.rbf_width)
            rows.append(row);targets.append(float(np.log(np.expm1(4*max(r.value,.001)))))
            blocks.append(int(math.floor(r.time/(cfg.dt*cfg.update_every)+1e-9)));uids.append(r.uid)
    overlaps=[]
    for k in range(1,len(stages)):
        left,right=stages[k-1:k+1]
        for q in right.anchors:
            if left.supported(q,cfg) and right.supported(q,cfg):
                row=torch.zeros(total,dtype=torch.float64);row[(k-1)*n:k*n]=features(tensor(q),left.centers,cfg.rbf_width);row[k*n:(k+1)*n]=-features(tensor(q),right.centers,cfg.rbf_width);overlaps.append(2*row)
    return dict(A=torch.stack(rows) if rows else torch.zeros((0,total),dtype=torch.float64),y=tensor(targets),blocks=np.array(blocks),uids=uids,
        overlap=torch.stack(overlaps) if overlaps else torch.zeros((0,total),dtype=torch.float64),prior=torch.cat([s.weights for s in stages]),ridge=cfg.fit_ridge,n=n)

def bootstrap_fit(design,multiplicity):
    A,y,O=design['A'],design['y'],design['overlap'];w=tensor(multiplicity)
    normal=A.T@(w[:,None]*A)+O.T@O+design['ridge']*torch.eye(len(design['prior']),dtype=torch.float64)
    return torch.linalg.solve(normal,A.T@(w*y)+design['ridge']*design['prior'])

def bootstrap_range(agent,x,seed,draws=BOOTSTRAPS):
    began=time.perf_counter();before=digest(agent);d=bootstrap_design(agent);unique=np.unique(d['blocks']);score=branch_scores(agent,x);ctx=copy.deepcopy(agent.navigation)
    if len(unique)<2:return dict(available=False,reason='fewer_than_two_time_blocks',blocks=len(unique),seconds=time.perf_counter()-began)
    def margin(w):
        c=copy.deepcopy(ctx);weights=dict(zip([s.id for s in agent.stages],w.split(d['n'])))
        for s in c['route_field_snapshot']['stages']:s['weights']=weights[s['id']].tolist()
        return context_margin(c,score['quadrature'])
    reference=margin(bootstrap_fit(d,np.ones(len(d['blocks']))));rng=np.random.default_rng(seed);values=[];resamples=[]
    for _ in range(draws):
        selected=rng.choice(unique,size=len(unique),replace=True);counts={int(b):int(np.sum(selected==b)) for b in unique}
        values.append(margin(bootstrap_fit(d,[counts[int(b)] for b in d['blocks']])));resamples.append(selected.tolist())
    perturb=np.asarray(values)-reference;finite=bool(np.isfinite(perturb).all() and np.isfinite(reference));interval=(score['margin']+np.quantile(perturb,[.025,.975])).tolist() if finite else None
    assert digest(agent)==before
    return dict(available=finite,reason='available' if finite else 'nonfinite',interval=interval,straddles=bool(finite and interval[0]<=0<=interval[1]),
        seed=seed,draws=draws,blocks=len(unique),block_ids=unique.tolist(),resamples=resamples,margins=values,reference=reference,committed_margin=score['margin'],
        fitting_measurement_ids=d['uids'],overlap_rows=len(d['overlap']),linear_solves=draws+1,rows=len(d['A']),seconds=time.perf_counter()-began,
        semantics='uncalibrated conditional screening range; frozen support and anchor/unknown fallback; shared time-block multiplicities across stages')

def inspect_snapshot(ctx,packet,seed):
    a=ctx['agents'][1];x=ctx['states'][1];before=digest((a,x));score=branch_scores(a,x);pairs=eligibility(a,packet) if packet else []
    probe=copy.deepcopy(a);event=probe.assimilate(packet) if packet else None
    valid=bool(event and event['reason']=='eligible' and any(e['rectangle'] for e in pairs));uncertainty=bootstrap_range(a,x,seed)
    assert digest((a,x))==before
    return dict(time=ctx['time'],score=score,eligible_anchors=len({e['anchor'] for e in pairs if e['rectangle']}),anchor_eligibility=pairs,
        packet=asdict(packet) if packet else None,event=event,valid_evidence=valid,uncertainty=uncertainty,isolated=True,
        checks=dict(support=score['eligible_support'],evidence=valid,margin=abs(score['margin'])<=.5,uncertainty=bool(uncertainty.get('straddles',False))))

def pair_pass(checks,control,truth_flip):
    required=('support','evidence') if control else ('support','evidence','margin','uncertainty')
    return bool(all(checks.get(k,False) for k in required) and (control or truth_flip))

def selected_scenes(rows,scenes):
    passing=[scene for scene in scenes if len([r for r in rows if r['scene']==scene])==len(SEEDS) and all(r['passed'] for r in rows if r['scene']==scene)]
    grid_pass=[s for s in passing if not scenes[s]['control']]
    return dict(grid=grid_pass,controls=[s for s in passing if scenes[s]['control']] if grid_pass else [],stop=not grid_pass)

class ScreeningComplete(Exception):pass

class ScreenCapture:
    def __init__(self,cfg):self.cfg=cfg;self.cursor=0;self.acquisition=None;self.last=None;self.entry=None;self.export_log=[]
    def boundary(self,ctx):
        edge=cr.branch_entry(ctx['states'][1][:2])
        if edge is not None:self.entry=dict(time=ctx['time'],edge=edge);raise ScreeningComplete()
        if abs(ctx['time']-.96)<1e-9:self.acquisition=copy.deepcopy(ctx)
        tick=next((t for _,i,t in ctx['batch'] if i==1),None)
        if tick is None or tick%self.cfg.update_every or ctx['time']<.96-1e-9:return
        # Shadow only the export protocol state. No export, receipt or communication accounting touches live screening learners.
        sender=copy.deepcopy(ctx['agents'][0]);sender._ledger_turn=self.cursor;self.cursor+=1
        packet=None if any(ctx['samples'][i][-1]['done'] for i in (0,1)) else sender.export(ctx['states'][0].copy(),'1')
        self.export_log.append(dict(time=ctx['time'],turn=self.cursor-1,packet=packet.identity if packet else None,stage=packet.stage if packet else None))
        self.last=(copy.deepcopy(ctx),packet)

def screen_job(task):
    root,key=task;root=Path(root);m=load(root/'manifest.json');spec=m['specs'][key];cfg=Config(**m['config']);comm=CommunicationConfig(**m['communication']);torch.set_num_threads(1);began=time.perf_counter()
    capture=ScreenCapture(cfg)
    try:
        result=None
        try:result=execute(spec,'disabled',comm,audit=False,learner_factory=cr.factory_for(spec,False),world_factory=NetworkWorld,config=cfg,boundary_callback=capture.boundary)
        except ScreeningComplete:pass
        saved={};acq=None;last=None
        if capture.acquisition:
            ctx=capture.acquisition;saved['acquisition']=cr.save_snapshot(root/'snapshots'/f'{key}_acquisition.pkl.gz',ctx);dump(root/'acquisition'/f'{key}.json.gz',cr.prefix(ctx))
            sender=copy.deepcopy(ctx['agents'][0]);packet=sender.export(ctx['states'][0].copy(),'1');acq=inspect_snapshot(ctx,packet,m['bootstrap_seeds'][key]['acquisition'])
        if capture.last and capture.entry:
            ctx,packet=capture.last;saved['prebranch']=cr.save_snapshot(root/'snapshots'/f'{key}_prebranch.pkl.gz',capture.last);last=inspect_snapshot(ctx,packet,m['bootstrap_seeds'][key]['prebranch'])
        g=spec['decision_sensitive'];flip=m['truth_flip'].get(str(g['x']),False);checks=last['checks'] if last else {}
        reasons=[k for k in ('support','evidence')+(() if g['control'] else ('margin','uncertainty')) if not checks.get(k,False)]
        if not g['control'] and not flip:reasons.append('truth_family_no_flip')
        if last is None:reasons.insert(0,'missing_prebranch_snapshot')
        if acq is None:reasons.insert(0,'missing_acquisition_snapshot')
        row=dict(id=key,scene=spec['scene'],seed=spec['seed'],control=g['control'],complete=True,passed=acq is not None and last is not None and pair_pass(checks,g['control'],flip),reasons=reasons,
            acquisition=acq,prebranch=last,truth=m['truth'][spec['scene']],truth_family_flip=flip,entry=capture.entry,snapshots=saved,export_opportunities=capture.export_log,
            screening_outcome=result['agents'] if result else 'stopped_at_first_receiver_branch_entry',seconds=time.perf_counter()-began,
            screening_work=[dict(flow=a.counterfactual_steps,identification=a.identification_steps) for a in capture.last[0]['agents']] if capture.last else None)
    except Exception as e:row=dict(id=key,scene=spec['scene'],seed=spec['seed'],control=spec['decision_sensitive']['control'],complete=False,passed=False,reasons=['screening_exception'],error=str(e),traceback=traceback.format_exc())
    dump(root/'screening'/f'{key}.json.gz',row);return dict(id=key,complete=row['complete'],passed=row['passed'],reasons=row['reasons'])

def make_manifest():
    scenes=grid();specs={f'{scene}_{seed}':specification(scene,seed) for scene in scenes for seed in SEEDS};truth={s:truth_costs(specification(s,SEEDS[0])) for s in scenes}
    flips={str(x):any(truth[f'x{x:g}_e{r:g}']['margin']< -1e-10 for r in RATIOS) and any(truth[f'x{x:g}_e{r:g}']['margin']>1e-10 for r in RATIOS) for x in XS}
    inputs=[ARCHIVE/'manifest.json']+[ARCHIVE/'acquisition'/f'{c}_{s}.json.gz' for c in cr.CASES for s in (79200,79201)]+[ARCHIVE/'episodes'/f'{c}_{s}_{a}.json.gz' for c in cr.CASES for s in (79200,79201) for a in ('local_only','legacy','validation_only')]
    m=dict(schema=1,scenes=scenes,specs=specs,truth=truth,truth_flip=flips,seeds=SEEDS,arms=ARMS,selected_solver=['active_set','diagnostic'],
        config=asdict(replace(configuration(500),route_field_mode='supported_field')),communication=asdict(communication('field')),
        source_hashes=source_hashes(),specification_hash=file_hash(ROOT/'DECISION_SENSITIVE_SPEC.md'),inputs=[dict(path=str(p),sha256=file_hash(p)) for p in inputs],
        numerical=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,torch_config=torch.__config__.show(),import_order='numpy_then_torch',threads=1,environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        bootstrap=dict(draws=BOOTSTRAPS,percentiles=[.025,.975],block='floor(timestamp/update_interval + 1e-9)',conditional_on='committed support and anchor/unknown fallback',calibrated=False),
        bootstrap_seeds={key:dict(acquisition=279300+2*i,prebranch=279301+2*i) for i,key in enumerate(specs)},
        analysis=dict(seed=ANALYSIS_SEED,draws=10000,percentiles=[.025,.975],mcnemar='two-sided exact; zero discordances p=1',alpha=.05,cluster='resample seed IDs jointly across all selected grid scenes',controls_in_primary=False),
        eligibility=dict(time='last pre-entry learner-update opportunity; acquisition descriptive',all_eight_seeds=True,margin=.5,interval_contains_zero=True,controls_exempt=['margin','uncertainty','truth_family_flip']),
        animation=dict(seed=79300,selection='lower-middle eligible grid scene sorted by x then ratio'),estar=ESTAR)
    return m

def freeze(root):
    root=Path(root);torch.set_num_threads(1);m=copy.deepcopy(make_manifest());m['fingerprint']=fingerprint(m)
    if root.exists():
        old=load(root/'manifest.json')
        if old['fingerprint']!=m['fingerprint']:raise ValueError('Frozen source/config/input/environment mismatch; use source archive or a new directory')
        return old
    root.mkdir(parents=True)
    for sub in ('screening','snapshots','acquisition','episodes','status','figures','verification'):(root/sub).mkdir()
    m['frozen_at']=stamp();dump(root/'manifest.json',m);archive_sources(root);shutil.copyfile(ROOT/'DECISION_SENSITIVE_SPEC.md',root/'SPEC.md');return m

def verify_frozen(root):
    return freeze(root)

def execute_tasks(tasks,function,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(function,t) for t in tasks]):print(json.dumps(f.result()),flush=True)

def gate(root,workers=4):
    root=Path(root);m=verify_frozen(root)
    tasks=[(str(root),k) for k in m['specs'] if not (root/'screening'/f'{k}.json.gz').exists()]
    execute_tasks(tasks,screen_job,workers)
    rows=[load(root/'screening'/f'{k}.json.gz') for k in m['specs']];selection=selected_scenes(rows,m['scenes'])
    inventory=[dict(id=r['id'],scene=r['scene'],seed=r['seed'],control=r['control'],complete=r['complete'],passed=r['passed'],reasons=r['reasons'],
        time=(r.get('prebranch') or {}).get('time'),margin=((r.get('prebranch') or {}).get('score') or {}).get('margin'),
        interval=((r.get('prebranch') or {}).get('uncertainty') or {}).get('interval'),truth_family_flip=r.get('truth_family_flip'),
        checks=(r.get('prebranch') or {}).get('checks',{}),source=str(Path('screening')/f"{r['id']}.json.gz")) for r in rows]
    result=dict(manifest_fingerprint=m['fingerprint'],complete=all(r['complete'] for r in rows),rows=inventory,selection=selection,
        row_hashes={k:file_hash(root/'screening'/f'{k}.json.gz') for k in m['specs']})
    if (root/'gate.json').exists():
        if fingerprint(load(root/'gate.json'))!=fingerprint(result):raise ValueError('Frozen gate changed')
    else:dump(root/'gate.json',result)
    seal=dict(manifest_sha256=file_hash(root/'manifest.json'),gate_sha256=file_hash(root/'gate.json'),selection=selection,
        snapshots={p.name:file_hash(p) for p in sorted((root/'snapshots').glob('*.pkl.gz'))})
    if (root/'eligibility_seal.json').exists():
        old=load(root/'eligibility_seal.json');old.pop('sealed_at',None)
        if fingerprint(old)!=fingerprint(seal):raise ValueError('Eligibility seal mismatch')
    else:dump(root/'eligibility_seal.json',dict(seal,sealed_at=stamp()))
    return result

def tests(root):
    root=Path(root);p=root/'verification/tests.json'
    if p.exists():return load(p)
    with (root/'verification/full-tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],stdout=f,stderr=f)
    log=(root/'verification/full-tests.log').read_text();errors=re.findall(r'^(?:ERROR|FAIL): (.+)$',log,re.M)
    okay=r.returncode==0 or len(errors)==1 and 'test_frozen_sources_specs_and_runtime_config' in errors[0]
    result=dict(passed_except_known_hash_error=okay,errors=errors,total=int(re.search(r'Ran (\d+) tests',log).group(1)),returncode=r.returncode)
    dump(p,result);return result

def regressions(root,workers):
    root=Path(root);out=root/'regression';parent=load(root/'manifest.json')
    if not out.exists():
        out.mkdir()
        for sub in ('snapshots','acquisition','episodes','status','verification'):(out/sub).mkdir()
        m=load(ARCHIVE/'manifest.json');m['source_hashes']=parent['source_hashes'];m['parent_manifest']=parent['fingerprint'];m.pop('fingerprint',None);m['fingerprint']=fingerprint(m);dump(out/'manifest.json',m)
        for key in m['specs']:shutil.copyfile(ARCHIVE/'acquisition'/f'{key}.json.gz',out/'acquisition'/f'{key}.json.gz')
    m=load(out/'manifest.json');tasks=[(str(out),k,a) for k in m['specs'] for a in ('local_only','legacy','validation_only') if not (out/'status'/f'{k}_{a}.json').exists()]
    execute_tasks(tasks,rc.episode_job,workers)
    if (out/'baseline_checks.json').exists():return load(out/'baseline_checks.json')
    from .active_set_suite import baseline_compare
    return baseline_compare(out)

def verify_treatment_gate(root,key):
    root=Path(root);m=verify_frozen(root);seal=load(root/'eligibility_seal.json');g=load(root/'gate.json')
    if seal['manifest_sha256']!=file_hash(root/'manifest.json') or seal['gate_sha256']!=file_hash(root/'gate.json'):raise ValueError('Treatment blocked: seal mismatch')
    if not g['complete'] or g['selection']['stop']:raise ValueError('Treatment blocked: incomplete or empty gate')
    if m['specs'][key]['scene'] not in seal['selection']['grid']+seal['selection']['controls']:raise ValueError('Ineligible treatment scene')
    if not load(root/'regression/baseline_checks.json')['passed']:raise ValueError('Baseline reproduction failed')
    if not load(root/'verification/tests.json')['passed_except_known_hash_error']:raise ValueError('Tests failed')
    return m

def treatment_job(task):
    root,key,arm=task;root=Path(root);verify_treatment_gate(root,key);start=stamp();result=rc.episode_job(task)
    result.update(started_at=start,eligibility_sha256=file_hash(root/'eligibility_seal.json'));dump(root/'status'/f'{key}_{arm}.json',result);return result

def primary_analysis(pairs,draws=10000,seed=ANALYSIS_SEED):
    if not pairs:return dict(available=False,reason='no completed primary pairs',pairs=0)
    a=np.array([p['local'] for p in pairs],int);b=np.array([p['corrected'] for p in pairs],int)
    wins=int(np.sum((a==0)&(b==1)));losses=int(np.sum((a==1)&(b==0)));n=wins+losses
    from scipy.stats import binomtest
    pvalue=float(binomtest(wins,n,.5).pvalue) if n else 1.
    ids=sorted(set(p['seed'] for p in pairs));diffs={s:[p['corrected']-p['local'] for p in pairs if p['seed']==s] for s in ids}
    rng=np.random.default_rng(seed);means=[float(np.mean([v for s in rng.choice(ids,len(ids),replace=True) for v in diffs[s]])) for _ in range(draws)]
    ci=np.quantile(means,[.025,.975]).tolist();difference=float(np.mean(b-a))
    return dict(available=True,pairs=len(pairs),seed_clusters=len(ids),local_correct=int(a.sum()),corrected_correct=int(b.sum()),
        both_correct=int(np.sum((a==1)&(b==1))),both_incorrect=int(np.sum((a==0)&(b==0))),corrected_only=wins,local_only=losses,
        difference=difference,mcnemar_p=pvalue,interval=ci,bootstrap_draws=draws,bootstrap_seed=seed,
        positive=bool(difference>0 and ci[0]>0 and pvalue<.05),
        caveat='Nominal pooled exact McNemar does not account for scene/seed dependence; seed-cluster bootstrap has only eight clusters. Eligibility-conditioned development evidence, not generalization.')

def observed_branch(samples):
    for s in samples:
        edge=cr.branch_entry(s['x'][:2])
        if edge:return dict(branch='direct' if edge==[1,2] else 'bypass',time=s['time'],edge=edge)
    return dict(branch=None,time=None,edge=None)

def correct_branch(branch,truth):return branch is not None and branch in truth['optimal']

def report(root):
    root=Path(root);m=load(root/'manifest.json');g=load(root/'gate.json');seal=load(root/'eligibility_seal.json');rows=[];failures=[];incomplete=[dict(id=r['id'],phase='screening',reason=r['reasons'],source=r['source']) for r in g['rows'] if not r['complete']];auditrows=[];lineage=[];controls=[]
    from .receiver_corrections_report import flatten
    eligible=seal['selection']['grid']+seal['selection']['controls']
    expected=[(k,a) for k,s in m['specs'].items() if s['scene'] in eligible for a in ARMS]
    for key,arm in expected:
        p=root/'status'/f'{key}_{arm}.json'
        if not p.exists():incomplete.append(dict(id=f'{key}_{arm}',executed=False,reason='not executed'));continue
        status=load(p)
        if not status['complete']:incomplete.append(status);continue
        r=load(root/'episodes'/status['path']);spec=r['spec'];actual=observed_branch(r['samples'][1]);reasons={}
        for u in r['updates'][1]:
            reason=u['diagnostic']['reason'];reasons[reason]=reasons.get(reason,0)+1
            lineage.append(dict(id=status['id'],time=u['time'],packets=u['packets'],pre_version=u['pre']['version'],post_version=u['post']['version'],accepted=u['diagnostic']['accepted'],reason=reason))
        carrier=r['agents'][1];t=actual['time'];pre=[z for z in r['trajectories'][1] if t is None or z['time']<t]
        planned=pre[-1]['navigation']['rr_route']['best'] if pre else None
        row=dict(id=status['id'],scene=spec['scene'],seed=spec['seed'],arm=arm,control=spec['decision_sensitive']['control'],truth=m['truth'][spec['scene']],actual=actual,
            correct=correct_branch(actual['branch'],m['truth'][spec['scene']]),planned_prebranch=planned,agents=r['agents'],
            receiver_time_risk=carrier['components']['time']+carrier['components']['risk'],receiver_full_cost=carrier['cost'],reasons=reasons,
            acquisition_match=r['acquisition_check'],receiver_diagnostics=r['receiver_diagnostics'],wall_seconds=r['wall_seconds'],audit_seconds=r['audit_seconds'],audit_flow=r['audit_flow'],
            started_at=status['started_at'],episode_sha256=status['sha256'],audit_isolation=all(a['state_unchanged'] and a['measurements_unchanged'] for a in r['receiver_audits']))
        for a in r['receiver_audits']:
            aa=flatten(a,'fresh')
            for b in aa:b.update(episode=status['id'],before_receiver_entry=t is None or a['time']<t)
            auditrows.extend(aa)
        rows.append(row)
        if any(not a['success'] for a in r['agents']):failures.append(row)
    primary=[]
    for key,spec in m['specs'].items():
        rr={r['arm']:r for r in rows if r['scene']==spec['scene'] and r['seed']==spec['seed']}
        if spec['scene'] not in eligible:continue
        if spec['decision_sensitive']['control']:
            if len(rr)==4:controls.append(dict(scene=spec['scene'],seed=spec['seed'],branches={a:r['actual']['branch'] for a,r in rr.items()},failed=len({r['actual']['branch'] for r in rr.values()})>1,unclassified=any(r['actual']['branch'] is None for r in rr.values())))
        elif 'local_only' in rr and 'corrected_fit' in rr:primary.append(dict(scene=spec['scene'],seed=spec['seed'],local=int(rr['local_only']['correct']),corrected=int(rr['corrected_fit']['correct'])))
    analysis=primary_analysis(primary)
    expected_pairs=len(seal['selection']['grid'])*8
    analysis['complete_primary']=len(primary)==expected_pairs
    if len(primary)!=expected_pairs:analysis['positive']=False;analysis['incomplete_warning']='No positive claim while primary episodes are missing'
    for name,data in [('rows',rows),('paired',primary),('analysis',analysis),('controls',controls),('failures',failures),('incomplete',incomplete)]:dump(root/f'{name}.json',data)
    dump(root/'audit_rows.json.gz',auditrows);dump(root/'lineage.json.gz',lineage)
    lines=['# Decision-sensitive receiver reconstruction','',f"Screened {len(g['rows'])}/160 scene-seed pairs. Eligible grid scenes: {len(seal['selection']['grid'])}/18; controls: {len(seal['selection']['controls'])}/2.",'',
        'The gate uses communication-free pre-branch snapshots; acquisition is descriptive. All eight seeds must pass. The bootstrap interval is an uncalibrated conditional screening range, with fixed support and fallback inputs.',
        '', '| Scene | Passing seeds | Truth branch flip | Gate failure reasons |','|---|---:|---|---|']
    from collections import Counter
    for scene,meta in m['scenes'].items():
        rr=[r for r in g['rows'] if r['scene']==scene];counts=Counter(reason for r in rr for reason in r['reasons'])
        lines.append(f"| {scene} | {sum(r['passed'] for r in rr)}/8 | {'exempt' if meta['control'] else m['truth_flip'][str(meta['x'])]} | {dict(counts)} |")
    if seal['selection']['stop']:lines+=['','**Eligibility gate failed: no grid scene qualified. No treatment episodes were run.** Controls were not run independently, and no treatment GIF was manufactured. This fixture family does not establish the intended test; it is not evidence that reconstruction cannot help.']
    else:lines+=['','## Navigation and primary analysis','',f"Completed {len(rows)}/{len(expected)} episodes; goals {sum(a['success'] for r in rows for a in r['agents'])}/{2*len(rows)}. Failed episodes {len(failures)}; incomplete {len(incomplete)}.",'','```json',json.dumps(analysis,indent=2),'```',f"Control branch disagreements: {sum(c['failed'] for c in controls)}/{len(controls)}."]
    test=load(root/'verification/tests.json') if (root/'verification/tests.json').exists() else None
    reg=load(root/'regression/baseline_checks.json') if (root/'regression/baseline_checks.json').exists() else None
    lines+=['','## Verification and interpretation','',f"Full test result: {test}. Archived baseline reproduction: {reg['passed'] if reg else 'not executed'}.",
        'E* is a fixed design reference. Exact route truth includes finite-edge exposure and bypass tails; no amplitude was adjusted to manufacture a flip. Actual branch traversal is separate from planned routing. Local-only fitting still receives shared route anchors.',
        'No adaptive communication, replay, controller tuning, held-out testing or post-outcome fixture search was performed. Shared geometry and seed reuse limit independence; pooled McNemar is nominal and all conclusions are conditional on eligibility.',
        '', '[Manifest](manifest.json) · [Gate: all 160 pairs](gate.json) · [Eligibility seal](eligibility_seal.json) · [Episode rows](rows.json) · [Paired analysis](analysis.json) · [Audits](audit_rows.json.gz) · [Failures](failures.json) · [Incomplete](incomplete.json)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'FINDINGS.md').write_text('\n'.join(lines)+'\n');return rows

def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(root);g=load(root/'gate.json');m=load(root/'manifest.json');fig,axes=plt.subplots(1,2,figsize=(13,5));names=list(m['scenes'])
    for i,name in enumerate(names):
        rr=[r for r in g['rows'] if r['scene']==name and r['margin'] is not None]
        for r in rr:
            interval=r['interval'];axes[0].scatter(r['margin'],i,s=12,c='green' if r['passed'] else 'red')
            if interval:axes[0].plot(interval,[i,i],color='.5',alpha=.25)
    axes[0].axvline(0,c='black',lw=.7);axes[0].axvspan(-.5,.5,alpha=.1);axes[0].set(yticks=range(len(names)),yticklabels=names,xlabel='Signed pre-branch cost margin (uncalibrated ranges)')
    for x in XS:axes[1].plot(RATIOS,[m['truth'][f'x{x:g}_e{r:g}']['margin'] for r in RATIOS],'-o',label=f'x={x:g}')
    axes[1].axhline(0,c='black',lw=.7);axes[1].set(xlabel='Declared E/E*',ylabel='Exact direct minus bypass truth cost');axes[1].legend();fig.tight_layout();fig.savefig(root/'figures/gate_and_truth.png',dpi=150);plt.close(fig)
    gallery=root/'figures/failures';gallery.mkdir(exist_ok=True);(gallery/'index.json').write_text(json.dumps(dict(failures=load(root/'failures.json'),incomplete=load(root/'incomplete.json')),indent=2))
    eligible=g['selection']['grid'];animation=None
    if eligible:
        ordered=sorted(eligible,key=lambda s:(m['scenes'][s]['x'],m['scenes'][s]['ratio']));scene=ordered[(len(ordered)-1)//2];paths={a:root/'episodes'/f'{scene}_79300_{a}.json.gz' for a in ARMS}
        if all(p.exists() for p in paths.values()):
            rr={a:load(p) for a,p in paths.items()};end=max(z['finish_time'] for r in rr.values() for z in r['agents']);times=np.unique(np.r_[np.linspace(0,end,64),.96]);frames=[]
            for now in times:
                fig,axes=plt.subplots(2,4,figsize=(16,7));fig.suptitle(f'{scene} | seed 79300 | t={now:.2f}s | fixed-rate, replay disabled')
                for col,(arm,r) in enumerate(rr.items()):
                    ax=axes[0,col];nodes=np.asarray(NODES)
                    for i,j in EDGES:ax.plot(nodes[[i,j],0],nodes[[i,j],1],c='.8',lw=6)
                    for patch in r['spec']['patches']:ax.scatter(*patch[:2],c='red',alpha=.5)
                    for i,c in enumerate(('tab:blue','tab:orange')):
                        ss=[s for s in r['samples'][i] if s['time']<=now];q=np.array([s['x'][:2] for s in ss]);ax.plot(q[:,0],q[:,1],color=c);ax.scatter(*q[-1],c=c)
                        ts=[t for t in r['trajectories'][i] if t['time']<=now]
                        axes[1,col].plot([t['time'] for t in ts],[t['navigation']['rr_route']['margin'] if t['navigation']['rr_route']['margin'] is not None else np.nan for t in ts],color=c)
                    ax.set(title=arm,xlim=(-1,9),ylim=(-1.5,3),aspect='equal');axes[1,col].set(xlim=(0,end),xlabel='Time',ylabel='Logged best/second path margin (not gate margin)')
                fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
            frames[0].save(root/'figures/comparison.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
            animation=dict(scene=scene,seed=79300,times=times.tolist(),sources={a:file_hash(p) for a,p in paths.items()},frames=Image.open(root/'figures/comparison.gif').n_frames)
    dump(root/'animation.json',animation)
    for name in ('REPORT.md','FINDINGS.md'):
        p=root/name;s=p.read_text();s+='\n![Eligibility and full-route truth](figures/gate_and_truth.png)\n'
        if animation:s+='\n[Synchronized four-arm animation](figures/comparison.gif)\n'
        p.write_text(s)

def suite(root,workers=4):
    root=Path(root);m=freeze(root)
    if not tests(root)['passed_except_known_hash_error']:raise RuntimeError('Tests failed; run blocked')
    if not regressions(root,workers)['passed']:raise RuntimeError('Baseline reproduction failed; run blocked')
    g=gate(root,workers)
    if not g['complete']:report(root);render(root);raise RuntimeError('Screening exceptions preserved; no treatment allowed')
    if not g['selection']['stop']:
        eligible=g['selection']['grid']+g['selection']['controls'];tasks=[(str(root),k,a) for k,s in m['specs'].items() if s['scene'] in eligible for a in ARMS if not (root/'status'/f'{k}_{a}.json').exists()]
        execute_tasks(tasks,treatment_job,workers)
    report(root);render(root)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','gate','suite','render','report'));p.add_argument('--out',type=Path,default=Path('results/decision_sensitive_v1'));p.add_argument('--workers',type=int,default=4);a=p.parse_args(argv)
    if a.command=='freeze':freeze(a.out)
    elif a.command=='gate':gate(a.out,a.workers)
    elif a.command=='suite':suite(a.out,a.workers)
    elif a.command=='report':report(a.out)
    else:render(a.out)
if __name__=='__main__':main()
