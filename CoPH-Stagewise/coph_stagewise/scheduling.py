"""Recipient-value training on advertised information, never evaluator truth.

The labels are bounded-horizon model-based estimates, not oracle mission values.
"""
import copy
from dataclasses import replace
import numpy as np
import torch
from .cooperative import CooperativeLearner,information_gain,field_vector
from .records import Observation,Reading,Transition
from .model import tensor,features,risk,flow


def value_features(packet,status,interval):
    ig,count=information_gain(packet,status)
    if not count:return np.array([1.,0.,0.,0.,0.,packet.bytes()/10000.,interval])
    a=np.array(packet.anchors);b=np.array(status.anchors)
    disagreement=float(np.mean((a[:,2:]-b[:,2:])**2))
    distance=float(np.mean(np.linalg.norm(a[:,:2]-status.q,axis=1)))
    return np.array([1.,ig,disagreement,distance,max(0.,status.timestamp-packet.timestamp),packet.bytes()/10000.,interval])


def reconstruct_advertised_receiver(status,cfg,comm):
    """Build the existing legal-information approximation; no private state input."""
    x=np.array([*status.q,*status.v]);agent=CooperativeLearner(cfg,x,status.goal,
        agent_id=status.sender,communication=comm)
    s=agent.stages[-1];points=np.array([a[:2] for a in status.anchors])
    rs=tuple(Reading(f'model:0:{k*5+1}',status.timestamp,tuple(q),float(a[2])) for k,(q,a) in enumerate(zip(points,status.anchors)))
    agent.observe(Observation(status.timestamp,rs,()))
    s.weights=agent._field_fit()[0][0]
    # Advertised support geometry certifies where this hypothetical reconstruction
    # may be sampled; generated values are never passed to the operational learner.
    extra=tuple(Reading(f'model_support:0:{k*5+1}',status.timestamp,tuple(q),
        float(risk(tensor(q),s.weights,s.centers,cfg.rbf_width))) for k,q in enumerate(status.support_points))
    agent.observe(Observation(status.timestamp,extra,()))
    return agent


def paired_model_label(packet,status,cfg,comm,horizon=8,seed=0):
    """Reconstruct a hypothetical receiver from its advertisement; sample field worlds.

Synthetic observations stay in cloned hypothetical learners. Rejected or unsupported
    branches are explicit missing labels, never zero-benefit training examples.
    """
    if status is None:return dict(valid=False,reason='missing_status',steps=0)
    agent=reconstruct_advertised_receiver(status,cfg,comm)
    x=np.array([*status.q,*status.v]);s=agent.stages[-1]
    points=np.array([a[:2] for a in status.anchors])
    # Sample uncertainty about the field from both legal advertisements, without truth.
    prior=np.array(status.covariance);measurement=np.array(packet.covariance)+np.eye(len(prior))*.01
    post=np.linalg.inv(np.linalg.inv(prior)+np.linalg.inv(measurement))
    z0=np.array(status.anchors)[:,2:].reshape(-1);z1=np.array(packet.anchors)[:,2:].reshape(-1)
    mean=post@(np.linalg.solve(prior,z0)+np.linalg.solve(measurement,z1))
    sample=np.random.default_rng(seed).multivariate_normal(mean,post).reshape(-1,3)
    A=features(tensor(points),s.centers,cfg.rbf_width)
    y=tensor(np.log(np.expm1(4*np.maximum(.001,np.minimum(20,sample[:,0])))))
    environment=torch.linalg.solve(A.T@A+cfg.fit_ridge*torch.eye(len(s.weights),dtype=torch.float64),A.T@y)
    costs={};work=0;replay_work=0;identification_work=0;valid=True
    for deliver in (False,True):
        branch=copy.deepcopy(agent);state=x.copy();total=0.
        if deliver:branch.assimilate(packet)
        branch.update(state)
        for step in range(horizon):
            u,_=branch.action(state);after=flow(tensor(state),tensor(u),branch.damping,cfg.dt).numpy();work+=1
            if not s.supported(after[:2],cfg):valid=False;break
            total+=cfg.dt*(1+4*float(risk(tensor(after[:2]),environment,s.centers,cfg.rbf_width))+.03*float(u@u))
            # Identical index-addressed hypothetical sensors in each branch.
            obs=Observation(status.timestamp+(step+1)*cfg.dt,tuple(
                Reading(f'hypothetical:{step}:{k*5+1}',status.timestamp+(step+1)*cfg.dt,tuple(q),
                        float(risk(tensor(q),environment,s.centers,cfg.rbf_width)))
                for k,q in enumerate(points) if np.linalg.norm(q-after[:2])<=cfg.sensor_radius),())
            branch.observe(obs,Transition(step,obs.time,tuple(state),tuple(u),tuple(after),branch.version))
            state=after
            if np.linalg.norm(state[:2]-branch.goal)<.25:break
            if (step+1)%cfg.update_every==0:branch.update(state)
        replay_work+=branch.counterfactual_steps;identification_work+=branch.identification_steps
        costs['deliver' if deliver else 'withhold']=total
    return dict(valid=valid,reason='supported' if valid else 'domain_exit',steps=work,
                benefit=costs['withhold']-costs['deliver'] if valid else None,
                costs=costs,horizon=horizon,start=x.tolist(),replay_steps=replay_work,identification_work=identification_work)


def fit_predictor(rows):
    valid=[r for r in rows if r['label']['valid']]
    if not valid:return dict(ready=False,reason='no_supported_training_pairs',rows=len(rows))
    X=np.array([r['features'] for r in valid]);y=np.array([r['label']['benefit'] for r in valid])
    scale=np.maximum(np.std(X,axis=0),1.);Z=X/scale
    coefficients=np.linalg.solve(Z.T@Z+.1*np.eye(Z.shape[1]),Z.T@y)
    residual=y-Z@coefficients
    return dict(ready=True,coefficients=coefficients.tolist(),scale=scale.tolist(),
                residual_rmse=float(np.sqrt(np.mean(residual**2))),rows=len(rows),valid_rows=len(valid),
                label='8-step legal-information paired continuation; not full-mission oracle')


def predict(model,features):
    if not model.get('ready'):return None
    return float(np.dot(np.array(features)/model['scale'],model['coefficients']))
