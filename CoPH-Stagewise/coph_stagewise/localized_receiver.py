"""Bounded compact receiver residuals; sensor evidence and support remain unchanged.

The local 4x4 field is frozen at its same-update local-only proposal. Received
residuals fit additional compact logit kernels, so their effect is exactly zero
outside the union of radius-.65 anchor neighborhoods. This is an opt-in model
extension, not a claim that the original fitting objective has a better solver.
"""
import copy
import time
from types import SimpleNamespace

import numpy as np
import torch

from .model import compact_features, features, tensor

RADIUS = .65


def validate(residual, cap):
    if residual is None:
        return
    centers=np.asarray(residual.get('centers',[]),dtype=float)
    coefficients=np.asarray(residual.get('coefficients',[]),dtype=float)
    if residual.get('schema')!=1 or residual.get('radius')!=RADIUS:
        raise ValueError('localized_residual_convention')
    if centers.shape!=(len(coefficients),2) or not 0<len(coefficients)<=cap:
        raise ValueError('localized_residual_memory')
    if not np.isfinite(centers).all() or not np.isfinite(coefficients).all() or (np.abs(coefficients)>.5+1e-12).any():
        raise ValueError('localized_residual_bounds')


def vector(coefficients, points, stage, width):
    centers=stage.receiver_compact_centers
    base=features(points,stage.centers,width) @ stage.receiver_base_weights
    f=compact_features(points,centers,RADIUS)
    delta=points[:,None,:]-centers
    radius=torch.linalg.vector_norm(delta,dim=-1)/RADIUS
    df=-20/RADIUS**2*(1-radius).clamp_min(0).pow(3)[:,:,None]*delta
    basef=features(points,stage.centers,width)
    basegrad=torch.einsum('nmd,m->nd',-(points[:,None,:]-stage.centers)/width**2*basef[:,:,None],stage.receiver_base_weights)
    raw=base+f@coefficients
    gradient=(basegrad+torch.einsum('nmd,m->nd',df,coefficients))*torch.sigmoid(raw)[:,None]/4
    return torch.cat((torch.nn.functional.softplus(raw)[:,None]/4,gradient),1).reshape(-1)


def design(agent,weights,blocks):
    """Freeze bounded bases and joint local precision, retaining overlap and ridge."""
    from .receiver_fitting import SETTINGS
    bases=[];views=[];offsets=[0]
    for stage,w0,bb in zip(agent.stages,weights,blocks):
        points=list(dict.fromkeys(tuple(q.tolist()) for q,_,_ in bb for q in q))[:agent.cfg.anchor_cap]
        c=tensor(points).reshape(-1,2)
        bases.append(c)
        views.append(SimpleNamespace(id=stage.id,centers=stage.centers,
            receiver_base_weights=w0.detach().clone(),receiver_compact_centers=c))
        offsets.append(offsets[-1]+len(c))
    n=offsets[-1]
    if not n:
        return views,[],dict(measurement_rows=0,overlap_rows=0,coefficients=0)
    rows=[];measurements=0;overlaps=0
    for k,(stage,w0,centers) in enumerate(zip(agent.stages,weights,bases)):
        if not len(centers):continue
        for reading in stage.readings:
            if int(reading.uid.split(':')[-1])%5==0:continue
            q=tensor(reading.q)
            slope=torch.sigmoid(features(q,stage.centers,agent.cfg.rbf_width)@w0)/4
            row=torch.zeros(n,dtype=torch.float64)
            row[offsets[k]:offsets[k+1]]=compact_features(q,centers,RADIUS)*slope/SETTINGS['sensor_sigma']
            rows.append(row);measurements+=1
    for k in range(1,len(agent.stages)):
        left,right=agent.stages[k-1:k+1]
        for q in right.anchors:
            if not left.supported(q,agent.cfg) or not right.supported(q,agent.cfg):continue
            row=torch.zeros(n,dtype=torch.float64)
            if len(bases[k-1]):row[offsets[k-1]:offsets[k]]=compact_features(tensor(q),bases[k-1],RADIUS)
            if len(bases[k]):row[offsets[k]:offsets[k+1]]=-compact_features(tensor(q),bases[k],RADIUS)
            rows.append(2*row);overlaps+=1
    A=torch.stack(rows) if rows else torch.zeros((0,n),dtype=torch.float64)
    normal=A.T@A+agent.cfg.fit_ridge*torch.eye(n,dtype=torch.float64)
    covariance=torch.linalg.inv(normal)
    information=[torch.linalg.inv(covariance[offsets[k]:offsets[k+1],offsets[k]:offsets[k+1]]) if offsets[k+1]>offsets[k] else None for k in range(len(bases))]
    return views,information,dict(measurement_rows=measurements,overlap_rows=overlaps,coefficients=n,
        approximation='frozen delta-method compact-basis precision; overlap rows and inverse marginal blocks retained; not calibrated posterior')


def fit(agent,weights,covariance):
    from .receiver_fitting import blocks_for, solve, SETTINGS
    began=time.perf_counter()
    blocks=[];lineage=[]
    for stage in agent.stages:
        bb,ll=blocks_for(agent,stage)
        blocks.append(bb);lineage.extend(ll)
    views,information,local=design(agent,weights,blocks)
    residuals=[];reports=[];eligible=0
    for stage,view,bb,H in zip(agent.stages,views,blocks,information or [None]*len(agent.stages)):
        if not bb:
            residuals.append(None)
            continue
        eligible+=sum(len(z) for _,z,_ in bb)
        initial=torch.zeros(len(view.receiver_compact_centers),dtype=torch.float64)
        try:
            coefficients,report=solve(view,initial,H,bb,agent.cfg.rbf_width,
                solver_mode=agent.receiver_solver_mode,search_mode=agent.receiver_search_mode)
        except (RuntimeError,ValueError) as exc:
            coefficients=initial
            report=dict(success=False,status='numerical_failure',converged=False,error=str(exc),
                total_correction=0.,work=dict(residual_rows=0,jacobian_rows=0,linear_solves=0,objective_evaluations=0))
        residual=dict(schema=1,radius=RADIUS,centers=view.receiver_compact_centers.tolist(),
            coefficients=coefficients.detach().tolist(),packets=[r['packet'] for r in lineage if r['stage']==stage.id],
            source_ids=sorted({s for r in lineage if r['stage']==stage.id for s in r['source_ids']}))
        validate(residual,agent.cfg.anchor_cap)
        residuals.append(residual)
        reports.append(dict(report,stage=stage.id,basis='compact_wendland_c2_logit',base_coefficients_fixed=True))
    agent.receiver_localized_proposal=copy.deepcopy(residuals)
    agent.receiver_local_proposal=[w.detach().clone() for w in weights]
    agent.message_fit_work+=sum(r['work']['residual_rows'] for r in reports)
    agent.receiver_fit_diagnostic=dict(mode=agent.receiver_fit_mode,localization_mode='compact',
        validation_mode=agent.receiver_validation_mode,settings=dict(SETTINGS),radius=RADIUS,
        eligible_rows=eligible,local_information=local,stages=reports,success=all(r['success'] for r in reports),lineage=lineage,
        local_fields=[w.tolist() for w in weights],proposed_fields=[w.tolist() for w in weights],
        proposed_residuals=copy.deepcopy(residuals),seconds=time.perf_counter()-began)
    agent.reference=[w.detach().clone() for w in weights]
    return weights,covariance
