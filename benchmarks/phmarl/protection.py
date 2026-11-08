"""Shared observation-only native action projection and local goal offsets.

A nominal safety mechanism, not a guarantee: newly visible velocity is unknown,
contact forces are not predicted, and bounded inputs can make constraints infeasible.
"""
import itertools
import numpy as np
import torch

SETTINGS=dict(radius=.15,clearance=.02,peer_radius=.45,dt=.1,drag=.25,
    prediction_steps=[1,2,3,4],near_goal_radius=.35,velocity_initialization='zero_flagged',
    infeasible_action='component_bounded_braking',goal_handling='local_separating_halfplanes')


def project(point,normals,bounds,box=False):
    """Exact Euclidean projection onto feasible 2D halfplanes, A*x>=b.

    Enumerate the interior, single-face projections and pairwise vertices.
    Return None for an empty feasible candidate set; never silently relax bounds.
    """
    p=np.asarray(point,dtype=float)
    A=[np.asarray(n,dtype=float) for n in normals]; b=list(bounds)
    if box:
        A.extend([np.array([1.,0]),np.array([-1.,0]),np.array([0.,1]),np.array([0.,-1])]);b.extend([-1.]*4)
    if not A:return p.copy(),1
    A=np.asarray(A);b=np.asarray(b);candidates=[p]
    for a,rhs in zip(A,b):
        norm=np.dot(a,a)
        if norm>1e-15:candidates.append(p+a*(rhs-np.dot(a,p))/norm)
    for i,j in itertools.combinations(range(len(b)),2):
        a,c=A[i],A[j];det=a[0]*c[1]-a[1]*c[0]
        if abs(det)>1e-12:
            candidates.append(np.array([(b[i]*c[1]-a[1]*b[j])/det,(a[0]*b[j]-b[i]*c[0])/det]))
    C=np.asarray(candidates);feasible=np.all(C@A.T>=b-1e-10,axis=1)&np.isfinite(C).all(axis=1)
    if not feasible.any():return None,len(C)
    C=C[feasible];return C[np.argmin(((C-p)**2).sum(axis=1))],len(candidates)


def horizon_coefficients(steps):
    vcoef=0.;ucoef=0.;v=.75;u=.1
    for _ in range(steps):
        vcoef+=.1*v;ucoef+=.1*u
        v*=.75;u=.75*u+.1
    return vcoef,ucoef


def braking_radius(dt=.1,drag=.25,separation=.32,reaction_steps=1):
    """Smallest interaction radius at which two agents closing head-on at the native top speed can stop.

    Top speed uses |u_x|,|u_y|<=1 (speed .4*sqrt(2)); braking uses only a unit decelerating input along the
    line of approach. Each agent travels one unbraked step per reaction step (first-sighting velocity is
    unknown), then brakes under v <- (1-drag)v - dt.
    """
    v=dt*np.sqrt(2)/drag;travel=reaction_steps*dt*v
    while v>0:travel+=dt*v;v=(1-drag)*v-dt
    return separation+2*travel


def contact_impulses(points,dt=.1,force=100.,margin=1e-3,contact=.30):
    """Velocity change from VMAS's penalty contact force over one native step (public native physics)."""
    points=np.asarray(points,float);dv=np.zeros_like(points)
    for a,b in itertools.combinations(range(len(points)),2):
        delta=points[a]-points[b];dist=float(np.linalg.norm(delta))
        if dist<1e-6 or dist>contact:continue
        pen=np.logaddexp(0.,(contact-dist)/margin)*margin;f=force*delta/dist*pen
        dv[a]+=dt*f;dv[b]-=dt*f
    return dv


def least_violation(normals,bounds):
    """Box-bounded action minimizing the largest constraint violation max_k(b_k - n_k.u)."""
    from scipy.optimize import linprog
    A=np.asarray(normals,float);b=np.asarray(bounds,float)
    res=linprog(c=[0,0,1],A_ub=np.hstack([-A,-np.ones((len(b),1))]),b_ub=-b,bounds=[(-1,1),(-1,1),(0,None)],method='highs')
    return res.x[:2] if res.success else None


class Protection:
    peer_radius=.45
    contact_prediction=False
    least_violation_fallback=False
    def __init__(self,goal_indices=None,fixed_targets=None):
        self.goal_indices=tuple(range(4)) if goal_indices is None else tuple(goal_indices)
        if sorted(self.goal_indices)!=list(range(4)):raise ValueError('Expected a goal permutation')
        self.fixed_targets=None if fixed_targets is None else np.asarray(fixed_targets,dtype=float).copy()
        if self.fixed_targets is not None and (self.fixed_targets.shape!=(4,2) or not np.isfinite(self.fixed_targets).all()):
            raise ValueError('Expected four finite fixed targets')
        self.history=[{} for _ in range(4)];self.step=0;self.context=[]

    def prepare(self,observations):
        processed=[np.asarray(o,dtype=float).reshape(18).copy() for o in observations]
        self.context=[]
        for i,obs in enumerate(processed):
            goal_index=self.goal_indices[i]
            q=obs[:2].copy();v=obs[2:4].copy();goal=q+obs[4+2*goal_index:6+2*goal_index]
            peers=[];current={};normal=[];goal_bounds=[];action_bounds=[];unknown=0
            relative=obs[12:].reshape(3,2);others=[j for j in range(4) if j!=i]
            impulse=np.zeros((4,2))
            if self.contact_prediction:
                points=np.zeros((4,2));points[i]=q
                for rp,j in zip(relative,others):points[j]=q+rp
                impulse=contact_impulses(points)
            v=v+impulse[i]
            for relative_pos,j in zip(relative,others):
                distance=float(np.linalg.norm(relative_pos))
                if distance>self.peer_radius:continue
                p=q+relative_pos;previous=self.history[i].get(j)
                peer_v=((p-previous)/.1 if previous is not None else np.zeros(2))+impulse[j]
                unknown+=int(previous is None);current[j]=p.copy()
                if distance>1e-12:n=-relative_pos/distance
                else:n=np.array([1. if i<j else -1.,0.])
                constraints=[]
                for h in SETTINGS['prediction_steps']:
                    a,b=horizon_coefficients(h)
                    constraints.append((.32-distance-a*np.dot(n,v-peer_v))/(2*b))
                normal.append(n);action_bounds.append(max(constraints))
                goal_bounds.append(np.dot(n,(q+p)/2)+.16)
                peers.append(dict(id=j,position=p.tolist(),velocity=peer_v.tolist(),
                    velocity_available=previous is not None,distance=distance))
            target=goal.copy();goal_infeasible=False;goal_work=0
            if self.fixed_targets is not None and np.linalg.norm(q-goal)<=.35:
                target=self.fixed_targets[i].copy()
            elif self.fixed_targets is None and np.linalg.norm(q-goal)<=.35 and normal:
                proposal,goal_work=project(goal,normal,goal_bounds)
                if proposal is not None:target=proposal
                else:goal_infeasible=True
            obs[4+2*goal_index:6+2*goal_index]=target-q
            self.context.append(dict(agent=i,q=q,v=v,normals=normal,bounds=action_bounds,
                target=target,goal=goal,goal_infeasible=goal_infeasible,goal_candidates=goal_work,
                peers=peers,unknown_peer_velocities=unknown))
            self.history[i]=current
        self.step+=1
        return [o.reshape(1,18).astype(np.float32) for o in processed]

    def apply(self,nominal):
        result=[];logs=[]
        for action,c in zip(nominal,self.context):
            u=np.asarray(action,dtype=float).reshape(2)
            projected,work=project(u,c['normals'],c['bounds'],box=True)
            feasible=projected is not None
            if not feasible and self.least_violation_fallback and c['normals']:projected=least_violation(c['normals'],c['bounds'])
            if projected is None:projected=np.clip(-.75*c['v']/.1,-1,1)
            applied=np.clip(projected,-1,1).astype(np.float32)
            residual=[float(np.dot(n,applied)-b) for n,b in zip(c['normals'],c['bounds'])]
            result.append(torch.tensor(applied).reshape(1,2))
            logs.append(dict(agent=c['agent'],nominal=u.tolist(),applied=applied.tolist(),
                intervened=bool(np.linalg.norm(applied-u)>1e-7),feasible=feasible,
                fallback=None if feasible else 'bounded_braking',constraint_residuals=residual,
                goal=c['goal'].tolist(),target=c['target'].tolist(),goal_shift=float(np.linalg.norm(c['target']-c['goal'])),
                goal_infeasible=c['goal_infeasible'],projection_candidates=work+c['goal_candidates'],
                unknown_peer_velocities=c['unknown_peer_velocities'],peers=c['peers']))
        return result,logs
