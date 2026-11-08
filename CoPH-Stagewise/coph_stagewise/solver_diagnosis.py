"""Offline diagnosis of the bounded receiver solver (receiver_fitting.solve). Evaluator-only.

Every audit receiver is rebuilt as receiver_corrections.audit_snapshot does and updated
with the corrected precision-iterative fit. Each call to `solve` is captured, replayed
bit for bit, traced, compared with a certified reference optimum, and re-solved by
opt-in candidate solvers that report status with the *unchanged* stationarity test,
box and acceptance gates. Nothing here alters operational defaults.
"""
import argparse,copy,glob,json,time,zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.optimize import minimize,Bounds
from . import receiver_fitting as rf
from .receiver_corrections import configure,load_saved_snapshot
from .reconstruction_quality import _local
from .reconstruction_diagnostics import ranking,field_queries
from .route_field import sync
from .attribution_audits import digest
from .evaluation import dump
from .cli import source_hashes

SAVED=Path('results/complementary_reconstruction_v1/snapshots')
FRESH=Path('results/receiver_corrections_v1/snapshots')
TOL=rf.SETTINGS['stationarity_tolerance'];TRUST=rf.SETTINGS['trust'];BUDGET=rf.SETTINGS['max_iterations']
EPS=np.finfo(float).eps


class Stage:
    """The two attributes `solve` reads from a stage."""
    def __init__(self,id,centers):self.id=id;self.centers=centers


# ---------- objective ----------
def objective(w,p):
    delta=w-p['w0'];f=.5*delta@p['H']@delta
    for points,z,L in p['blocks']:
        r=z-rf.vector(w,points,p['stage'],p['width']);y=torch.linalg.solve_triangular(L,r[:,None],upper=False).flatten();f=f+.5*y@y
    return f


def grad(w,p):
    v=w.detach().clone().requires_grad_(True);f=objective(v,p);return float(f),torch.autograd.grad(f,v)[0]


def hessian(w,p):return torch.autograd.functional.hessian(lambda v:objective(v,p),w.detach())


def gauss_newton(w,p):
    """g and N exactly as receiver_fitting.solve builds them."""
    delta=w-p['w0'];rows=[];targets=[]
    for points,z,L in p['blocks']:
        r=z-rf.vector(w,points,p['stage'],p['width']);y=torch.linalg.solve_triangular(L,r[:,None],upper=False).flatten()
        J=torch.autograd.functional.jacobian(lambda v:rf.vector(v,points,p['stage'],p['width']),w)
        rows.append(torch.linalg.solve_triangular(L,J,upper=False));targets.append(y)
    A=torch.cat(rows);y=torch.cat(targets);return p['H']@delta-A.T@y,p['H']+A.T@A,A,y


def stationarity(w,g,p):
    lower,upper=p['w0']-TRUST,p['w0']+TRUST;proj=torch.maximum(lower,torch.minimum(upper,w-g))
    return float(torch.linalg.vector_norm(w-proj,ord=float('inf')))/p['scale']


# ---------- trace of the original solver (verified against it) ----------
def trace(p):
    """Re-execute receiver_fitting.solve's logic, recording every iterate and step."""
    S=rf.SETTINGS;w0=p['w0'];lower,upper=w0-TRUST,w0+TRUST
    def project(v):return torch.maximum(lower,torch.minimum(upper,v))
    def f(v):return float(rf_objective(v,p))
    w=w0.clone();g,N,A,y=gauss_newton(w,p);obj=rf_objective(w,p);steps=[]
    for it in range(BUDGET):
        if stationarity(w,g,p)<=TOL:break
        delta=torch.linalg.solve(N,A.T@y if it==0 else -g);direction=project(w+delta)-w;fallback=False
        clipped=int((torch.abs(direction-delta)>1e-12).sum())
        if float(g@direction)>=0:direction=project(w-g)-w;fallback=True
        predicted=-(float(g@direction)+.5*float(direction@N@direction))
        full=float(obj)-f(project(w+direction));accepted=False;half=0
        for half in range(S['max_halvings']+1):
            step=.5**half;cand=project(w+step*direction);co=rf_objective(cand,p)
            if bool(torch.isfinite(co)) and float(co)<=float(obj+S['armijo']*step*(g@direction)):accepted=True;break
        lam=torch.linalg.eigvalsh(hessian(w,p));lam_gn=torch.linalg.eigvalsh(N)
        steps.append(dict(iteration=it,objective=float(obj),stationarity=stationarity(w,g,p),fallback=fallback,clipped_components=clipped,
            active=int(((w-lower).abs()<1e-9).sum()+((w-upper).abs()<1e-9).sum()),predicted_decrease=predicted,actual_full_decrease=full,
            ratio=full/predicted if predicted>0 else None,halvings=half,accepted=accepted,armijo_term=abs(S['armijo']*.5**half*float(g@direction)),
            precision_floor=EPS*abs(float(obj)),min_hessian_eig=float(lam[0]),min_gn_eig=float(lam_gn[0]),gn_condition=float(lam_gn[-1]/lam_gn[0]),
            residual_norm=float(torch.linalg.vector_norm(y))))
        if not accepted:break
        w=cand;obj=co;g,N,A,y=gauss_newton(w,p)
    return w,steps


def rf_objective(w,p):
    # The exact expression receiver_fitting.solve minimizes (same operation order).
    delta=w-p['w0'];value=.5*delta@p['H']@delta
    for points,z,L in p['blocks']:
        residual=z-rf.vector(w,points,p['stage'],p['width']);y=torch.linalg.solve_triangular(L,residual[:,None],upper=False).flatten();value=value+.5*y@y
    return value


def classify(report,steps):
    """Declared rules, applied in order, for a failed solve."""
    if not steps:return 'no_iterations'
    last=steps[-1]
    if any(s['fallback'] or s['clipped_components'] for s in steps):return 'projection'
    if report['status']=='line_search_failed' and last['armijo_term']<10*last['precision_floor']:return 'floating_point'
    ratios=[s['ratio'] for s in steps if s['ratio'] is not None]
    if ratios and np.median(ratios)<.1:return 'curvature_mismatch'
    if report['status']=='iteration_exhausted':return 'budget'
    return 'other'


# ---------- reference optimum ----------
def reference(p,starts):
    lo=(p['w0']-TRUST).numpy();hi=(p['w0']+TRUST).numpy();bounds=Bounds(lo,hi);best=None;minima=[]
    fun=lambda v:grad(torch.tensor(v,dtype=torch.float64),p)
    for s in starts:
        r=minimize(lambda v:tuple((lambda o:(o[0],o[1].numpy()))(fun(v))),np.clip(s.numpy(),lo,hi),jac=True,method='L-BFGS-B',bounds=bounds,
                   options=dict(maxiter=20000,ftol=1e-16,gtol=1e-13,maxcor=30))
        minima.append(dict(objective=float(r.fun),w=r.x.tolist()))
        if best is None or r.fun<best.fun:best=r
    w=torch.tensor(best.x,dtype=torch.float64)
    tc=minimize(lambda v:float(objective(torch.tensor(v,dtype=torch.float64),p)),best.x,jac=lambda v:grad(torch.tensor(v,dtype=torch.float64),p)[1].numpy(),
                hess=lambda v:hessian(torch.tensor(v,dtype=torch.float64),p).numpy(),method='trust-constr',bounds=bounds,options=dict(maxiter=500,gtol=1e-12,xtol=1e-14))
    _,g=grad(w,p);spread=max(m['objective'] for m in minima)-best.fun
    distinct=sum(np.max(np.abs(np.array(m['w'])-best.x))>1e-4 and m['objective']-best.fun>1e-8 for m in minima)
    return w,dict(objective=float(best.fun),trust_constr_objective=float(tc.fun),stationarity=stationarity(w,g,p),start_spread=float(spread),distinct_local_minima=int(distinct),
                  min_hessian_eig=float(torch.linalg.eigvalsh(hessian(w,p))[0]),active=int(np.sum(np.isclose(best.x,lo,atol=1e-9))+np.sum(np.isclose(best.x,hi,atol=1e-9))))


# ---------- opt-in candidate solvers (same box, tolerance and budget) ----------
def _report(w,p,status,iterations,work):
    obj,g=grad(w,p);st=stationarity(w,g,p)
    if status!='nonfinite' and st<=TOL:status='converged'
    return w,dict(status=status,converged=status=='converged',success=status=='converged',iterations=iterations,objective_initial=p['initial'],objective_final=obj,
        stationarity=st,normalization=p['scale'],total_correction=float(torch.max(torch.abs(w-p['w0']))),history=[],work=work,active_bounds=0,gradient_fallbacks=0,backtracking=0)


def levenberg_marquardt(p):
    w0=p['w0'];lo,hi=w0-TRUST,w0+TRUST;w=w0.clone();mu=1e-3;work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0);status='iteration_exhausted'
    obj=float(rf_objective(w,p))
    for it in range(BUDGET):
        g,N,_,_=gauss_newton(w,p)
        if stationarity(w,g,p)<=TOL:status='converged';break
        for _ in range(30):
            work['linear_solves']+=1;d=torch.linalg.solve(N+mu*torch.diag(torch.diagonal(N)),-g);cand=torch.maximum(lo,torch.minimum(hi,w+d))
            co=float(rf_objective(cand,p));work['objective_evaluations']+=1;pred=-(float(g@(cand-w))+.5*float((cand-w)@N@(cand-w)))
            if np.isfinite(co) and co<obj and pred>0 and (obj-co)/pred>.25:w,obj=cand,co;mu=max(mu/3,1e-12);break
            mu*=10
        else:status='damping_exhausted';break
    return _report(w,p,status,it+1,work)


def trust_region_newton(p):
    """Exact Hessian, eigenvalue-shifted where indefinite, projected onto the box."""
    w0=p['w0'];lo,hi=w0-TRUST,w0+TRUST;w=w0.clone();radius=.1;work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0);status='iteration_exhausted'
    obj=float(rf_objective(w,p))
    for it in range(BUDGET):
        _,g=grad(w,p)
        if stationarity(w,g,p)<=TOL:status='converged';break
        Hx=hessian(w,p);lam=float(torch.linalg.eigvalsh(Hx)[0]);shift=max(0.,-lam)+1e-10;accepted=False
        for _ in range(30):
            work['linear_solves']+=1;d=torch.linalg.solve(Hx+shift*torch.eye(len(w),dtype=torch.float64),-g)
            n=float(torch.linalg.vector_norm(d))
            if n>radius:d=d*radius/n
            cand=torch.maximum(lo,torch.minimum(hi,w+d));step=cand-w;pred=-(float(g@step)+.5*float(step@Hx@step))
            co=float(rf_objective(cand,p));work['objective_evaluations']+=1;rho=(obj-co)/pred if pred>0 else -1
            if rho>.25 or (pred>0 and pred<1e-14*abs(obj) and co<=obj):accepted=True;w,obj=cand,co;radius=min(2*radius,1.) if rho>.75 else radius;break
            radius/=4
        if not accepted:status='trust_region_exhausted';break
    return _report(w,p,status,it+1,work)


def active_set_gauss_newton(p):
    """Bertsekas-style projected Newton: the same Gauss-Newton model, solved on free variables only.

    Variables held at a bound by the gradient stay there; free variables that the model step would
    push out of the box are fixed at that bound and the rest re-solved. The line search runs along
    the projected arc with an Armijo test on the actual projected step. A floating-point floor admits only
    steps that do not increase the objective beyond rounding; status still needs the unchanged test.
    """
    w0=p['w0'];lo,hi=w0-TRUST,w0+TRUST;w=w0.clone();work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0);status='iteration_exhausted'
    obj=float(rf_objective(w,p));it=0
    for it in range(BUDGET):
        g,N,_,_=gauss_newton(w,p)
        if stationarity(w,g,p)<=TOL:status='converged';break
        eps=min(1e-3,float(torch.linalg.vector_norm(w-torch.maximum(lo,torch.minimum(hi,w-g)),ord=float('inf'))))
        binding=((w<=lo+eps)&(g>0))|((w>=hi-eps)&(g<0));fixed=binding.clone();d=torch.zeros_like(w)
        d[binding]=(torch.where(g>0,lo,hi)-w)[binding]
        # Primal active-set refinement: free variables the model step would push out of the box are
        # fixed at that bound and the remaining free subsystem is re-solved.
        for _ in range(len(w)):
            free=~fixed
            if not bool(free.any()):break
            work['linear_solves']+=1;d[free]=torch.linalg.solve(N[free][:,free],-g[free]-N[free][:,fixed]@d[fixed])
            out=free&(((w+d)<lo)|((w+d)>hi))
            if not bool(out.any()):break
            d[out]=(torch.minimum(hi,torch.maximum(lo,w+d))-w)[out];fixed|=out
        # Fall back to the plain free-variable step, then the projected gradient, if not a descent.
        def projected(v):return torch.maximum(lo,torch.minimum(hi,w+v))-w
        if float(g@projected(d))>=0:
            free=~binding;d=torch.zeros_like(w);d[binding]=-g[binding]
            if bool(free.any()):work['linear_solves']+=1;d[free]=torch.linalg.solve(N[free][:,free],-g[free])
            if float(g@projected(d))>=0:d=-g
        accepted=False
        for half in range(40):
            t=.5**half;cand=torch.maximum(lo,torch.minimum(hi,w+t*d));co=float(rf_objective(cand,p));work['objective_evaluations']+=1
            required=-1e-4*float(g@(cand-w));floor=10*EPS*abs(obj)
            if np.isfinite(co) and (obj-co>=required or (required<floor and co<=obj+floor)):accepted=True;break
        if not accepted:status='line_search_failed';break
        w,obj=cand,co
    else:it=BUDGET-1
    return _report(w,p,status,it+1,work)


CANDIDATES=dict(levenberg_marquardt=levenberg_marquardt,trust_region_newton=trust_region_newton,active_set_gauss_newton=active_set_gauss_newton)


# ---------- per-audit work ----------
def score(base,weights,points,truth,support):
    probe=copy.deepcopy(base)
    for s,w in zip(probe.stages,weights):s.weights=w.detach().clone()
    sync(probe,publish=True);pred,_=field_queries(probe,points);mask=support&np.isfinite(pred).all(1)
    return float(np.abs(pred[mask,0]-truth[mask,0]).mean()) if mask.any() else None,mask


def audit(snapshot,meta):
    agent,x,packet,world,tick=snapshot;original=digest((agent,x,world))
    base=copy.deepcopy(agent);base.assimilate(packet)
    quad=ranking(base,x,details=True)['quadrature'];points=np.array([q for e in quad for q in e['points']])
    truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)));initial,_=field_queries(base,points);support=np.isfinite(initial).all(1)
    local,_=_local(copy.deepcopy(base));captured=[];orig=rf.solve
    def capture(stage,w0,H,blocks,width,iterative=True,max_iterations=None):
        w,rep=orig(stage,w0,H,blocks,width,iterative,max_iterations)
        captured.append(dict(stage_id=stage.id,centers=stage.centers.clone(),w0=w0.clone(),H=H.clone(),blocks=[(a.clone(),b.clone(),c.clone()) for a,b,c in blocks],
                             width=width,iterative=iterative,w=w.clone(),report=copy.deepcopy(rep)))
        return w,rep
    rf.solve=capture
    try:a=configure(copy.deepcopy(base),'precision_iterative','legacy');d=a.update(x.copy())
    finally:rf.solve=orig
    out=dict(meta,packet=packet.identity,accepted=bool(d['accepted']),reason=d['reason'],problems=[])
    nofit,mask0=score(base,local,points,truth,support);out['no_fit_error']=nofit
    ids=[s.id for s in base.stages];proposal={c['stage_id']:c['w'] for c in captured};refs={}
    for c in captured:
        p=dict(stage=Stage(c['stage_id'],c['centers']),w0=c['w0'],H=c['H'],blocks=c['blocks'],width=c['width'])
        g0,_,_,_=gauss_newton(p['w0'],p);p['scale']=1+float(torch.linalg.vector_norm(g0,ord=float('inf')));p['initial']=float(rf_objective(p['w0'],p))
        w2,rep2=orig(p['stage'],p['w0'],p['H'],p['blocks'],p['width'],c['iterative'])
        replay=bool(torch.equal(w2,c['w']) and rep2['status']==c['report']['status'] and rep2['iterations']==c['report']['iterations'] and rep2['objective_final']==c['report']['objective_final'])
        wt,steps=trace(p);traced=bool(torch.equal(wt,c['w']))
        rng=np.random.default_rng(zlib.crc32(f"{meta['file']}:{meta.get('sender')}:{c['stage_id']}".encode()))  # reproducible starts
        starts=[p['w0'],c['w']]+[p['w0']+torch.tensor(rng.uniform(-TRUST,TRUST,len(p['w0'])),dtype=torch.float64) for _ in range(4)]
        wr,ref=reference(p,starts);refs[c['stage_id']]=wr
        cand={}
        for name,fn in CANDIDATES.items():
            t0=time.perf_counter();wc,rc=fn(p);cand[name]=dict(status=rc['status'],iterations=rc['iterations'],objective=rc['objective_final'],stationarity=rc['stationarity'],
                gap=rc['objective_final']-ref['objective'],seconds=time.perf_counter()-t0,linear_solves=rc['work']['linear_solves'])
        rep=c['report']
        out['problems'].append(dict(stage=c['stage_id'],status=rep['status'],iterations=rep['iterations'],objective_final=rep['objective_final'],stationarity=rep['stationarity'],
            replay_exact=replay,trace_exact=traced,category=None if rep['success'] else classify(rep,steps),steps=steps,reference=ref,
            gap_to_reference=rep['objective_final']-ref['objective'],field_distance_to_reference=float(torch.max(torch.abs(c['w']-wr))),candidates=cand))
    # Route-query error of the proposal itself, of the reference optimum, and of the rolled-back incumbent.
    def weights_with(sub):return [sub.get(i,w) for i,w in zip(ids,local)]
    out['proposal_error'],_=score(base,weights_with(proposal),points,truth,support)
    out['reference_error'],_=score(base,weights_with(refs),points,truth,support)
    out['committed_error'],_=score(base,[s.weights for s in a.stages],points,truth,support)
    # Candidates through the unchanged update path and gates.
    out['candidate_runs']={}
    for name,fn in {**CANDIDATES,'reference':None}.items():
        def solver(stage,w0,H,blocks,width,iterative=True,max_iterations=None,name=name,fn=fn):
            p=dict(stage=stage,w0=w0,H=H,blocks=blocks,width=width);g0,_,_,_=gauss_newton(w0,p);p['scale']=1+float(torch.linalg.vector_norm(g0,ord=float('inf')));p['initial']=float(rf_objective(w0,p))
            if fn is not None:return fn(p)
            return _report(refs.get(stage.id,w0),p,'reference',0,dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0))
        rf.solve=solver
        try:c=configure(copy.deepcopy(base),'precision_iterative','legacy');dc=c.update(x.copy())
        finally:rf.solve=orig
        err,_=score(base,[s.weights for s in c.stages],points,truth,support)
        diag=getattr(c,'receiver_fit_diagnostic',None) or {}
        out['candidate_runs'][name]=dict(accepted=bool(dc['accepted']),reason=dc['reason'],committed_error=err,solver_success=diag.get('success'))
    assert digest((agent,x,world))==original
    return out


def _job(task):
    torch.set_num_threads(1);path,meta=task;value=load_saved_snapshot(path);out=[]
    if isinstance(value,tuple):return [audit(value,meta)]
    ctx=value;intended=dict(complementary=0,reverse=1,redundant=None)[meta['case']]
    for sender in (0,1):
        exported=copy.deepcopy(ctx['agents'][sender]);packet=exported.export(ctx['states'][sender].copy(),str(1-sender))
        if packet is None:continue
        out.append(audit((ctx['agents'][1-sender],ctx['states'][1-sender],packet,ctx['worlds'][1-sender],8),dict(meta,sender=sender,intended=sender==intended)))
    return out


def jobs():
    tasks=[]
    for scope,root in (('saved',SAVED),('fresh',FRESH)):
        for path in sorted(glob.glob(str(root/'*.pkl.gz'))):
            name=Path(path).name[:-7];parts=name.split('_')
            if scope=='fresh' and parts[2]=='acquisition':continue  # fresh audits are navigation deliveries only
            tasks.append((path,dict(scope=scope,file=name,case=parts[0],seed=int(parts[1]),group='acquisition' if parts[2]=='acquisition' else 'navigation')))
    return tasks


def run(out,workers=40):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);tasks=jobs()
    dump(out/'manifest.json',dict(inputs=[t[0] for t in tasks],settings=rf.SETTINGS,candidates=list(CANDIDATES),source_hashes=source_hashes(),
        semantics='evaluator-only; captured solve inputs; unchanged tolerance, box, budget and gates; candidates are not promoted'))
    with ProcessPoolExecutor(max_workers=workers) as pool:results=[a for batch in pool.map(_job,tasks) for a in batch]
    dump(out/'audits.json.gz',results);return results


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    ap.add_argument('--summarize',action='store_true');args=ap.parse_args(argv)
    if not args.summarize:run(args.out,args.workers)
    from .solver_diagnosis_report import summarize
    summarize(args.out)


if __name__=='__main__':main()
