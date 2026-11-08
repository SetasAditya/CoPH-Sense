"""Opt-in bounded receiver correction. No simulator truth or sensor synthesis."""
import time
import numpy as np
import torch
from .model import features,tensor

SETTINGS=dict(sensor_sigma=.003,trust=.5,max_iterations=20,max_halvings=12,armijo=1e-4,stationarity_tolerance=1e-6,relative_slack=.05,numerical_slack=1e-12)


def precision_information(agent,weights):
    """Frozen joint design; preserve overlap rows and inverse marginal blocks.

    Delta-method precision is an approximation, including near clipped readings.
    The local point estimate and declared regularization remain unchanged.
    """
    cfg=agent.cfg;m=len(weights[0]);n=m*len(weights);rows=[];measurement_rows=0;overlap_rows=0
    for k,(s,w) in enumerate(zip(agent.stages,weights)):
        for r in s.readings:
            if int(r.uid.split(':')[-1])%5==0:continue
            phi=features(tensor(r.q),s.centers,cfg.rbf_width)
            slope=torch.sigmoid(phi@w)/4
            row=torch.zeros(n,dtype=torch.float64);row[k*m:(k+1)*m]=phi*slope/SETTINGS['sensor_sigma']
            rows.append(row);measurement_rows+=1
    for k in range(1,len(agent.stages)):
        left,right=agent.stages[k-1:k+1]
        for q in right.anchors:
            if not left.supported(q,cfg) or not right.supported(q,cfg):continue
            row=torch.zeros(n,dtype=torch.float64)
            row[(k-1)*m:k*m]=features(tensor(q),left.centers,cfg.rbf_width)
            row[k*m:(k+1)*m]=-features(tensor(q),right.centers,cfg.rbf_width)
            rows.append(2*row);overlap_rows+=1
    A=torch.stack(rows) if rows else torch.empty((0,n),dtype=torch.float64)
    normal=A.T@A+cfg.fit_ridge*torch.eye(n,dtype=torch.float64);covariance=torch.linalg.inv(normal)
    information=[torch.linalg.inv(covariance[k*m:(k+1)*m,k*m:(k+1)*m]) for k in range(len(weights))]
    return information,dict(measurement_rows=measurement_rows,overlap_rows=overlap_rows,approximation='frozen delta-method precision; repeated stage evidence and clipped readings retain legacy treatment; not a posterior')


def vector(v,points,stage,width):
    if hasattr(stage,'receiver_compact_centers'):
        from .localized_receiver import vector as compact_vector
        return compact_vector(v,points,stage,width)
    f=features(points,stage.centers,width);raw=f@v;sigmoid=torch.sigmoid(raw)
    df=-(points[:,None,:]-stage.centers)/width**2*f[:,:,None]
    val=torch.nn.functional.softplus(raw)/4
    grad=sigmoid[:,None]/4*torch.einsum('nmd,m->nd',df,v)
    return torch.cat((val[:,None],grad),1).reshape(-1)


def blocks_for(agent,stage):
    blocks=[];lineage=[]
    for message in agent.received.values():
        ids=[k for k,a in enumerate(message.anchors) if stage.contains(a[:2],agent.cfg)]
        if not ids:continue
        indices=[3*k+j for k in ids for j in range(3)]
        C=tensor(np.asarray(message.covariance)[np.ix_(indices,indices)])
        C=C+torch.eye(len(indices),dtype=torch.float64)*(.01+.005*(agent.now-message.timestamp))
        blocks.append((tensor([message.anchors[k][:2] for k in ids]),tensor([message.anchors[k][2:] for k in ids]).reshape(-1),torch.linalg.cholesky(C)))
        lineage.append(dict(packet=message.identity,stage=stage.id,anchor_indices=ids,source_ids=list(message.sources)))
    return blocks,lineage


def _legacy_solve(stage,w0,H,blocks,width,iterative=True,max_iterations=None):
    """Projected Gauss–Newton with a fixed box; report failure rather than convergence."""
    max_iterations=SETTINGS['max_iterations'] if max_iterations is None else max_iterations
    lower=w0-SETTINGS['trust'];upper=w0+SETTINGS['trust'];work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0)
    def project(w):return torch.maximum(lower,torch.minimum(upper,w))
    def terms(w,derivatives):
        delta=w-w0;objective=.5*delta@H@delta;rows=[];targets=[]
        work['objective_evaluations']+=1
        for points,z,L in blocks:
            values=vector(w,points,stage,width);residual=z-values
            y=torch.linalg.solve_triangular(L,residual[:,None],upper=False).flatten()
            objective=objective+.5*y@y;work['residual_rows']+=len(z)
            if derivatives:
                J=torch.autograd.functional.jacobian(lambda v:vector(v,points,stage,width),w)
                rows.append(torch.linalg.solve_triangular(L,J,upper=False));targets.append(y);work['jacobian_rows']+=len(z)
        if not derivatives:return objective
        A=torch.cat(rows);y=torch.cat(targets);g=H@delta-A.T@y
        return objective,g,H+A.T@A,A,y
    history=[];w=w0.clone();status='iteration_exhausted';backtracks=0;fallbacks=0;iterations=0
    obj,g,N,A,y=terms(w,True);scale=1+float(torch.linalg.vector_norm(g,ord=float('inf')))
    def stationarity(v,gradient):return float(torch.linalg.vector_norm(v-project(v-gradient),ord=float('inf')))/scale
    initial=float(obj);pg=stationarity(w,g)
    for it in range(max_iterations if iterative else 1):
        if not bool(torch.isfinite(obj) and torch.isfinite(g).all()):status='nonfinite';break
        if iterative and pg<=SETTINGS['stationarity_tolerance']:status='converged';break
        work['linear_solves']+=1
        # The first legacy-weighted one-step solve preserves operational order.
        delta=torch.linalg.solve(N,A.T@y if it==0 else -g)
        if not iterative:
            w=w+delta.clamp(-SETTINGS['trust'],SETTINGS['trust']);iterations=1;status='single_step';break
        direction=project(w+delta)-w
        if float(g@direction)>=0:
            direction=project(w-g)-w;fallbacks+=1
        accepted=False
        for half in range(SETTINGS['max_halvings']+1):
            step=.5**half;candidate=project(w+step*direction);candidate_obj=terms(candidate,False)
            if bool(torch.isfinite(candidate_obj)) and float(candidate_obj)<=float(obj+SETTINGS['armijo']*step*(g@direction)):
                accepted=True;backtracks+=half;break
        history.append(dict(iteration=it,objective=float(obj),stationarity=pg,accepted=accepted,halvings=half))
        if not accepted:status='line_search_failed';break
        w=candidate;iterations+=1;obj,g,N,A,y=terms(w,True);pg=stationarity(w,g)
    obj,g,_,_,_=terms(w,True);pg=stationarity(w,g)
    if iterative and bool(torch.isfinite(obj) and torch.isfinite(g).all()) and pg<=SETTINGS['stationarity_tolerance']:status='converged'
    if not bool(torch.isfinite(w).all() and torch.isfinite(obj) and torch.isfinite(g).all()):status='nonfinite'
    return w,dict(status=status,converged=status=='converged',success=status in ('converged','single_step'),iterations=iterations,
        objective_initial=initial,objective_final=float(obj),stationarity=pg,normalization=scale,backtracking=backtracks,
        gradient_fallbacks=fallbacks,active_bounds=int(((w-lower).abs()<1e-9).sum()+((w-upper).abs()<1e-9).sum()),
        total_correction=float(torch.max(torch.abs(w-w0))),history=history,work=work)


def fit(agent,weights,cov):
    if getattr(agent,'receiver_localization_mode','global')=='compact':
        from .localized_receiver import fit as compact_fit
        return compact_fit(agent,weights,cov)
    began=time.perf_counter();mode=getattr(agent,'receiver_fit_mode','legacy');validation=getattr(agent,'receiver_validation_mode','legacy')
    precision=mode=='precision_bounded';iterative=precision
    # Evaluator-only factorial switches, never selected by runtime data.
    override=getattr(agent,'receiver_diagnostic_variant',None)
    if override is not None:precision,iterative=override
    info,design=precision_information(agent,weights) if precision and agent.received else (None,None)
    output=[];reports=[];lineage=[];eligible=0
    for k,(stage,w0) in enumerate(zip(agent.stages,weights)):
        blocks,links=blocks_for(agent,stage);lineage+=links
        if not blocks:output.append(w0);continue
        H=info[k] if precision else torch.linalg.inv(cov[k]) if cov[k] is not None else torch.eye(len(w0),dtype=torch.float64)
        eligible+=sum(len(z) for _,z,_ in blocks)
        try:
            solver=getattr(agent,'receiver_solver_mode','projected');search=getattr(agent,'receiver_search_mode','legacy')
            if (solver,search)==('projected','legacy'):
                w,report=solve(stage,w0,H,blocks,agent.cfg.rbf_width,iterative)
            else:
                if mode!='precision_bounded':raise ValueError('nondefault receiver solver requires precision_bounded fitting')
                w,report=solve(stage,w0,H,blocks,agent.cfg.rbf_width,iterative,solver_mode=solver,search_mode=search)
        except (RuntimeError,ValueError) as exc:
            w=w0.clone();report=dict(status='numerical_failure',success=False,converged=False,error=str(exc),stage=stage.id,
                total_correction=0.,work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0))
        report['stage']=stage.id;reports.append(report);output.append(w)
    # Counter records actual residual evaluations separately from the eligibility count.
    agent.message_fit_work+=sum(r['work']['residual_rows'] for r in reports)
    agent.receiver_local_proposal=[w.detach().clone() for w in weights]
    agent.receiver_fit_diagnostic=dict(mode=mode,validation_mode=validation,settings=dict(SETTINGS),eligible_rows=eligible,
        solver_mode=getattr(agent,'receiver_solver_mode','projected'),search_mode=getattr(agent,'receiver_search_mode','legacy'),local_information=design,stages=reports,success=all(r['success'] for r in reports),lineage=lineage,
        local_fields=[w.tolist() for w in weights],proposed_fields=[w.tolist() for w in output],seconds=time.perf_counter()-began)
    def serial(value):
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        if isinstance(value,list):return [serial(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    agent.receiver_fit_diagnostic=serial(agent.receiver_fit_diagnostic)
    agent.reference=[w.clone() for w in output]
    return output,cov


def acceptance(agent,weights):
    diagnostic=getattr(agent,'receiver_fit_diagnostic',None)
    if diagnostic is None:return True,None,None
    d=dict(diagnostic);d['additional_gate_applied']=False
    if not d['success']:return False,'receiver_solver_failure',d
    if d['eligible_rows'] and getattr(agent,'receiver_validation_mode','legacy')=='relative_local':
        count=sum(int(r.uid.split(':')[-1])%5==0 for s in agent.stages for r in s.readings)
        d.update(additional_gate_applied=True,validation_count=count)
        if not count:return False,'receiver_no_reserved_measurements',d
        if getattr(agent,'receiver_localization_mode','global')=='compact':
            local=agent._field_validation(agent.receiver_local_proposal,[None]*len(weights))
            message=agent._field_validation(weights,agent.receiver_localized_proposal)
        else:
            local=agent._field_validation(agent.receiver_local_proposal);message=agent._field_validation(weights)
        threshold=local+SETTINGS['relative_slack']*max(local,SETTINGS['sensor_sigma']**2)+SETTINGS['numerical_slack']
        passed=np.isfinite(message) and message<=threshold
        d.update(local_validation=local,message_validation=message,threshold=threshold,relative_gate_passed=bool(passed))
        if not passed:return False,'receiver_relative_validation',d
    return True,None,d


def solve(stage,w0,H,blocks,width,iterative=True,max_iterations=None,*,solver_mode='projected',search_mode='legacy'):
    """Legacy arithmetic is untouched; opt-in step and search policies are independent."""
    if solver_mode not in ('projected','active_set') or search_mode not in ('legacy','diagnostic'):raise ValueError('receiver solver/search mode')
    if (solver_mode,search_mode)==('projected','legacy'):
        return _legacy_solve(stage,w0,H,blocks,width,iterative,max_iterations)
    if not iterative:raise ValueError('nondefault solver requires iterative precision fitting')
    return _bounded_solve(stage,w0,H,blocks,width,solver_mode,search_mode,max_iterations)


def _bounded_solve(stage,w0,H,blocks,width,solver,search,max_iterations):
    """Port of the diagnosed active-set step; no evaluator or SciPy dependency.

    Diagnostic search additionally permits 40 trials and the recorded rounding
    exception. Its final certification reproduces the diagnostic exact gradient.
    """
    budget=SETTINGS['max_iterations'] if max_iterations is None else max_iterations
    lo,hi=w0-SETTINGS['trust'],w0+SETTINGS['trust'];w=w0.clone()
    work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0)
    history=[];status='iteration_exhausted';refinements=0;fallbacks=0;rounding=0;backtracking=0;iterations=0
    def project(v):return torch.maximum(lo,torch.minimum(hi,v))
    def objective(v):
        d=v-w0;value=.5*d@H@d;work['objective_evaluations']+=1
        for q,z,L in blocks:
            residual=z-vector(v,q,stage,width);y=torch.linalg.solve_triangular(L,residual[:,None],upper=False).flatten()
            value=value+.5*y@y;work['residual_rows']+=len(z)
        return value
    def gn(v):
        rows=[];targets=[]
        for q,z,L in blocks:
            residual=z-vector(v,q,stage,width);y=torch.linalg.solve_triangular(L,residual[:,None],upper=False).flatten()
            J=torch.autograd.functional.jacobian(lambda u:vector(u,q,stage,width),v)
            rows.append(torch.linalg.solve_triangular(L,J,upper=False));targets.append(y)
            work['residual_rows']+=len(z);work['jacobian_rows']+=len(z)
        A=torch.cat(rows);y=torch.cat(targets)
        return H@(v-w0)-A.T@y,H+A.T@A,A,y
    g0,_,_,_=gn(w0);scale=1+float(torch.linalg.vector_norm(g0,ord=float('inf')))
    def stationarity(v,g):return float(torch.linalg.vector_norm(v-project(v-g),ord=float('inf')))/scale
    initial=float(objective(w0));obj=float(objective(w))
    for it in range(budget):
        g,N,A,y=gn(w);st=stationarity(w,g)
        if not bool(torch.isfinite(g).all() and torch.isfinite(N).all()) or not np.isfinite(obj):status='nonfinite';break
        if st<=SETTINGS['stationarity_tolerance']:status='converged';break
        binding=torch.zeros_like(w,dtype=torch.bool);fixed=binding.clone();fallback='none';inner=0
        if solver=='active_set':
            eps=min(1e-3,float(torch.linalg.vector_norm(w-project(w-g),ord=float('inf'))))
            binding=((w<=lo+eps)&(g>0))|((w>=hi-eps)&(g<0));fixed=binding.clone();d=torch.zeros_like(w)
            d[binding]=(torch.where(g>0,lo,hi)-w)[binding]
            for _ in range(len(w)):
                free=~fixed
                if not bool(free.any()):break
                work['linear_solves']+=1;inner+=1
                d[free]=torch.linalg.solve(N[free][:,free],-g[free]-N[free][:,fixed]@d[fixed])
                out=free&(((w+d)<lo)|((w+d)>hi))
                if not bool(out.any()):break
                d[out]=(project(w+d)-w)[out];fixed|=out
            refinements+=inner
            if float(g@(project(w+d)-w))>=0:
                fallback='free_variable';fallbacks+=1;free=~binding;d=torch.zeros_like(w);d[binding]=-g[binding]
                if bool(free.any()):
                    work['linear_solves']+=1;d[free]=torch.linalg.solve(N[free][:,free],-g[free])
                if float(g@(project(w+d)-w))>=0:d=-g;fallback='projected_gradient'
        else:
            work['linear_solves']+=1;delta=torch.linalg.solve(N,A.T@y if it==0 else -g);d=project(w+delta)-w
            if float(g@d)>=0:d=project(w-g)-w;fallback='projected_gradient';fallbacks+=1
        accepted=False;trials=[]
        for half in range(40 if search=='diagnostic' else SETTINGS['max_halvings']+1):
            step=.5**half;cand=project(w+step*d);co=float(objective(cand))
            required=-SETTINGS['armijo']*float(g@(cand-w));floor=10*np.finfo(float).eps*abs(obj)
            strict=np.isfinite(co) and obj-co>=required
            exception=search=='diagnostic' and np.isfinite(co) and required<floor and co<=obj+floor
            accepted=bool(strict or exception)
            trials.append(dict(halvings=half,objective=co,required_decrease=required,actual_decrease=obj-co,rounding_exception=bool(exception and not strict),accepted=accepted))
            if accepted:rounding+=int(exception and not strict);break
        backtracking+=half
        history.append(dict(iteration=it,iterate=w.tolist(),gradient=g.tolist(),objective=obj,stationarity=st,binding=binding.tolist(),fixed=fixed.tolist(),refinement_solves=inner,fallback=fallback,trials=trials,accepted=accepted,step=d.tolist()))
        if not accepted:status='line_search_failed';break
        w,obj=cand,co;iterations+=1
    # Exact-gradient certification matches the archived active-set candidate.
    v=w.detach().clone().requires_grad_(True);f=objective(v);g=torch.autograd.grad(f,v)[0];st=stationarity(w,g)
    work['gradient_rows']=sum(len(z) for _,z,_ in blocks)
    finite=bool(torch.isfinite(w).all() and torch.isfinite(g).all() and torch.isfinite(f))
    if not finite:status='nonfinite'
    elif status not in ('nonfinite','line_search_failed') and st<=SETTINGS['stationarity_tolerance']:status='converged'
    return w,dict(status=status,converged=status=='converged',success=status=='converged',iterations=(it+1 if budget else 0),accepted_steps=iterations,objective_initial=initial,objective_final=float(f),
        stationarity=st,normalization=scale,total_correction=float((w-w0).abs().max()),active_bounds=int(((w-lo).abs()<1e-9).sum()+((w-hi).abs()<1e-9).sum()),
        history=history,work=work,gradient_fallbacks=fallbacks,backtracking=backtracking,active_set_refinements=refinements,rounding_exceptions=rounding,
        solver_mode=solver,search_mode=search,near_tolerance=bool(.5e-6<=st<=2e-6))
