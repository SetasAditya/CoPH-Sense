"""Opt-in centralized native action projection; original Protection is unchanged.

The coordinator pools each agent's own measured q/v and all nominal actions.
No simulator state is read. All six pairs are constrained together, so third
parties share the same feasible action problem. This is not decentralized safety.
"""
import itertools
import time
import numpy as np
import torch
from scipy.optimize import minimize,linprog,nnls
from protection import Protection,horizon_coefficients

SETTINGS=dict(interface='centralized pooled own-observation q/v and all four nominal actions',
    pair_set='all six native pairs, including third parties',separation=.32,physical_contact_distance=.30,
    actuator_bounds=[-1.,1.],prediction_steps=[1,2,3,4],dt=.1,drag=.25,
    initially_unsafe='nondecreasing current radial separation; never claim immediate recovery',
    optimizer='SLSQP convex quadratic/linear constraints',max_iterations=60,maximum_solves=2,lp_max_iterations=200,
    ftol=1e-11,kkt_tolerance=1e-6,feasibility_tolerance=1e-8,float32_feasibility_tolerance=2e-7,
    feasibility_probe='HiGHS linear program only after failed QP; affine feasibility, not global geometric certificate',
    fallback='all-agent component-bounded braking, explicitly uncertified when constraints infeasible',
    contact_dynamics='native contact impulses are not predicted; initial unsafe states explicitly retained')


def constraints(positions,velocities):
    q=np.asarray(positions,dtype=float);v=np.asarray(velocities,dtype=float);rows=[];bounds=[];pairs=[]
    for i,j in itertools.combinations(range(4),2):
        delta=q[i]-q[j];distance=float(np.linalg.norm(delta))
        normal=delta/distance if distance>1e-12 else np.array([1.,0.])
        target=min(SETTINGS['separation'],distance)
        rhs=[]
        for h in SETTINGS['prediction_steps']:
            a,b=horizon_coefficients(h)
            rhs.append((target-distance-a*float(normal@(v[i]-v[j])))/b)
        row=np.zeros((4,2));row[i]=normal;row[j]=-normal
        rows.append(row.reshape(-1));bounds.append(max(rhs))
        pairs.append(dict(agents=[i,j],distance=distance,normal=normal.tolist(),target=target,
            initially_contacting=distance<SETTINGS['physical_contact_distance'],margin_unsafe=distance<SETTINGS['separation'],
            horizon_bounds=rhs,selected_bound=max(rhs),degenerate_normal=distance<=1e-12))
    return np.asarray(rows),np.asarray(bounds),pairs


def certificate(action,point,A,b):
    action=np.asarray(action,float);finite=bool(np.isfinite(action).all())
    if not finite:return dict(finite=False,feasible=False,stationarity=None,minimum_residual=None)
    # Include component box faces in the KKT cone.
    complete=np.vstack((A,np.eye(8),-np.eye(8)));rhs=np.r_[b,np.full(16,-1.)]
    slack=complete@action-rhs;active=slack<=1e-7;g=action-point
    try:
        multiplier=nnls(complete[active].T,g,maxiter=300)[0] if active.any() else np.zeros(0)
        stationarity=float(np.max(abs(g-complete[active].T@multiplier))) if active.any() else float(np.max(abs(g)))
    except (RuntimeError,ValueError):stationarity=None
    return dict(finite=finite,feasible=bool(slack.min()>=-SETTINGS['feasibility_tolerance']),
        stationarity=stationarity,minimum_residual=float(slack.min()),active_faces=int(active.sum()))


def solve(point,A,b):
    point=np.asarray(point,dtype=float).reshape(8);start=np.clip(point,-1.,1.);began=time.perf_counter();attempts=[];lp_status=None
    def objective(x):return .5*float(np.sum((x-point)**2))
    def gradient(x):return x-point
    for attempt in range(SETTINGS['maximum_solves']):
        result=minimize(objective,start,jac=gradient,bounds=[(-1.,1.)]*8,
            constraints=dict(type='ineq',fun=lambda x:A@x-b,jac=lambda x:A),method='SLSQP',
            options=dict(ftol=SETTINGS['ftol'],maxiter=SETTINGS['max_iterations'],disp=False))
        cert=certificate(result.x,point,A,b)
        accepted=bool(result.success and cert['finite'] and cert['feasible'] and cert['stationarity'] is not None and cert['stationarity']<=SETTINGS['kkt_tolerance'])
        attempts.append(dict(success=bool(result.success),status=int(result.status),message=str(result.message),
            iterations=int(result.nit),function_evaluations=int(result.nfev),certificate=cert,accepted=accepted))
        if accepted:
            return result.x,dict(status='certified_optimum',accepted=True,attempts=attempts,lp_status=lp_status,
                affine_infeasibility_certified=False,geometric_infeasibility_certified=False,seconds=time.perf_counter()-began)
        if attempt==0:
            feasibility=linprog(np.zeros(8),A_ub=-A,b_ub=-b,bounds=[(-1.,1.)]*8,method='highs',options=dict(maxiter=SETTINGS['lp_max_iterations']))
            lp_status=dict(success=bool(feasibility.success),status=int(feasibility.status),message=str(feasibility.message))
            if not feasibility.success:
                return None,dict(status='affine_constraints_infeasible' if feasibility.status==2 else 'feasibility_probe_failed',accepted=False,
                    attempts=attempts,lp_status=lp_status,affine_infeasibility_certified=bool(feasibility.status==2),
                    geometric_infeasibility_certified=False,seconds=time.perf_counter()-began)
            start=feasibility.x
    return None,dict(status='numerical_failure',accepted=False,attempts=attempts,lp_status=lp_status,
        affine_infeasibility_certified=False,geometric_infeasibility_certified=False,seconds=time.perf_counter()-began)


class JointProtection(Protection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.initial_unsafe_pairs=None

    def apply(self,nominal):
        point=np.asarray([np.asarray(action,dtype=float).reshape(2) for action in nominal])
        positions=np.array([c['q'] for c in self.context]);velocities=np.array([c['v'] for c in self.context])
        A,b,pairs=constraints(positions,velocities)
        if self.initial_unsafe_pairs is None:self.initial_unsafe_pairs=[p for p in pairs if p['margin_unsafe']]
        result,diagnostic=solve(point.reshape(-1),A,b)
        feasible=result is not None
        if result is None:result=np.clip(-.75*velocities/.1,-1.,1.).reshape(-1)
        applied=np.clip(result.reshape(4,2),-1.,1.).astype(np.float32)
        residual=A@applied.reshape(-1)-b
        float_valid=bool(residual.min()>=-SETTINGS['float32_feasibility_tolerance'])
        if feasible and not float_valid:
            feasible=False;diagnostic=dict(diagnostic,status='float32_feasibility_failure',accepted=False)
            applied=np.clip(-.75*velocities/.1,-1.,1.).astype(np.float32);residual=A@applied.reshape(-1)-b
        diagnostic.update(pair_constraints=pairs,initial_unsafe_pairs=self.initial_unsafe_pairs,
            minimum_applied_constraint_residual=float(residual.min()),float32_feasible=float_valid,
            information_interface=SETTINGS['interface'],geometric_infeasibility_certified=False)
        results=[];logs=[]
        for i,(u,c) in enumerate(zip(applied,self.context)):
            results.append(torch.tensor(u).reshape(1,2))
            logs.append(dict(agent=i,nominal=point[i].tolist(),applied=u.tolist(),intervened=bool(np.linalg.norm(u-point[i])>1e-7),
                feasible=feasible,fallback=None if feasible else 'bounded_braking_uncertified',constraint_residuals=residual.tolist(),
                goal=c['goal'].tolist(),target=c['target'].tolist(),goal_shift=float(np.linalg.norm(c['target']-c['goal'])),
                goal_infeasible=c['goal_infeasible'],projection_candidates=sum(a['function_evaluations'] for a in diagnostic['attempts']),
                unknown_peer_velocities=0,peers=c['peers'],joint=diagnostic))
        return results,logs
