"""Reset-only task allocation and static coverage-compatible targets.

Only native observations and public radii enter these calculations. The target
program uses fixed landmark-pair normals: feasibility is sufficient for true
coverage/separation; failure alone does not prove geometric infeasibility.
"""
import itertools
import time
import numpy as np
from scipy.optimize import minimize


def local_reset_geometry(observation,agent):
    o=np.asarray(observation,dtype=np.float64).reshape(18)
    q=o[:2];positions=np.empty((4,2));positions[agent]=q
    for j,r in zip([j for j in range(4) if j!=agent],o[12:].reshape(3,2)):positions[j]=q+r
    return positions,q+o[4:12].reshape(4,2)


def minimum_assignment(positions,landmarks):
    costs=((positions[:,None]-landmarks[None,:])**2).sum(axis=-1)
    return min(itertools.permutations(range(4)),key=lambda p:(sum(costs[i,p[i]] for i in range(4)),p))


def assignment_consensus(observations):
    proposals=[];geometries=[]
    for i,o in enumerate(observations):
        q,g=local_reset_geometry(o,i);proposals.append(minimum_assignment(q,g));geometries.append((q,g))
    return dict(agreed=all(p==proposals[0] for p in proposals),assignments=[list(p) for p in proposals],
        maximum_position_disagreement=float(max(np.max(np.abs(q-geometries[0][0])) for q,g in geometries)),
        maximum_landmark_disagreement=float(max(np.max(np.abs(g-geometries[0][1])) for q,g in geometries)),
        information='each agent independently uses its raw full native observation, ordered peer identities and landmark identities; not radius-restricted sensing')


def static_targets(landmarks,separation=.32,coverage_radius=.15):
    begin=time.perf_counter();g=np.asarray(landmarks,dtype=np.float64)
    n=len(g);pairs=list(itertools.combinations(range(n),2));A=[];b=[]
    distances=[]
    for i,j in pairs:
        delta=g[i]-g[j];distance=float(np.linalg.norm(delta));distances.append(distance)
        # Identity ordering fixes the degenerate normal. Infeasibility under
        # this normal set is explicitly not treated as a global certificate.
        normal=delta/distance if distance>1e-12 else np.array([1.,0.])
        row=np.zeros((n,2));row[i]=normal;row[j]=-normal
        A.append(row.reshape(-1));b.append(separation-distance)
    A=np.asarray(A);b=np.asarray(b)
    cap=coverage_radius-1e-6  # declared numerical interior margin, not a new success radius
    def fun(x):return .5*float(x@x)
    def jac(x):return x
    def disks(x):return cap*cap-(x.reshape(n,2)**2).sum(axis=1)
    def diskjac(x):
        rows=np.zeros((n,n,2))
        for i in range(n):rows[i,i]=-2*x.reshape(n,2)[i]
        return rows.reshape(n,2*n)
    constraints=[dict(type='ineq',fun=lambda x:A@x-b,jac=lambda x:A),dict(type='ineq',fun=disks,jac=diskjac)]
    result=minimize(fun,np.zeros(2*n),jac=jac,constraints=constraints,method='SLSQP',
        options=dict(ftol=1e-12,maxiter=200,disp=False))
    correction=np.asarray(result.x).reshape(n,2);targets=g+correction
    target_distances=[float(np.linalg.norm(targets[i]-targets[j])) for i,j in pairs]
    residual=float(np.min(A@result.x-b));displacement=np.linalg.norm(correction,axis=1)
    finite=bool(np.isfinite(targets).all());feasible=bool(finite and residual>=-1e-9 and np.max(displacement)<=cap+1e-9 and min(target_distances)>=separation-1e-9)
    certified_pair_conflict=min(distances)<separation-2*coverage_radius-1e-10
    return targets,dict(feasible=feasible,solver_success=bool(result.success),solver_message=str(result.message),
        iterations=int(result.nit),objective=fun(result.x),linear_constraint_residual=residual,
        minimum_landmark_spacing=min(distances),minimum_target_spacing=min(target_distances),
        maximum_target_shift=float(displacement.max()),target_shifts=displacement.tolist(),
        coverage_radius=coverage_radius,target_shift_cap=cap,separation=separation,
        pairwise_necessary_condition_passed=not certified_pair_conflict,
        true_pairwise_infeasibility_certificate=certified_pair_conflict,
        interpretation='feasible target construction is a joint witness; failed fixed-normal solve alone is inconclusive',
        seconds=time.perf_counter()-begin,nfev=int(result.nfev),njev=int(result.njev))


def permute_goal_slots(observations,indices):
    result=np.asarray(observations).copy()
    shape=result.shape;flat=result.reshape(4,18)
    flat[:,4:12]=flat[:,4:12].reshape(4,4,2)[:,list(indices),:].reshape(4,8)
    return result.reshape(shape)
