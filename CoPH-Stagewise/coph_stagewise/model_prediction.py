"""Legal-information frozen environments and paired value predictions."""
import copy
import time
from dataclasses import replace
import numpy as np
import torch
from .model import tensor,flow,balance,features
from .records import Observation,Reading
from .stage_prediction import FrozenStages,DomainExit
from .receiver_summary import reconstruct_receiver
from .scheduling import reconstruct_advertised_receiver
from .continuations import continuation


class ModelEnvironment:
    def __init__(self,agent,seed,cross_stage=False):
        self.cfg=agent.cfg;self.seed=seed;self.time=agent.now;self.cross_stage=cross_stage
        frozen=FrozenStages.freeze(agent)
        self.atlas=frozen if cross_stage else replace(frozen,stages=(frozen.stages[-1],))
        self.index=len(self.atlas.stages)-1;self._damping=agent.damping
        self.context_jumps=[];self.frozen_prediction=True

    def sample_shared_field(self,packet,status,seed):
        """One legal common-anchor Gaussian proxy draw, shared by both branches.

This preserves the earlier scheduler's Gaussian approximation; it is not a
calibrated posterior over the whole atlas. Value and gradient residuals transport
the draw into supported local bases, without changing domain certificates.
"""
        if status is None:return False
        pa=np.array(packet.anchors);sa=np.array(status.anchors)
        if pa.shape!=sa.shape or not np.allclose(pa[:,:2],sa[:,:2]):return False
        prior=np.array(status.covariance);noise=np.array(packet.covariance)+np.eye(len(prior))*.01
        precision=np.linalg.inv(prior)+np.linalg.inv(noise)
        covariance=np.linalg.inv(precision)
        mean=covariance@(np.linalg.solve(prior,sa[:,2:].reshape(-1))+np.linalg.solve(noise,pa[:,2:].reshape(-1)))
        rng=np.random.default_rng(np.random.SeedSequence([seed,314159]))
        draw=rng.multivariate_normal(mean,covariance).reshape(-1,3);draw[:,0]=np.maximum(.001,draw[:,0])
        updated=[];self.field_draw_work=0
        for stage in self.atlas.stages:
            ids=[k for k,q in enumerate(pa[:,:2]) if stage.supported(q,self.cfg)]
            if not ids:updated.append(stage);continue
            points=tensor(pa[ids,:2]);centers=tensor(stage.centers);weights=tensor(stage.weights)
            def vector(w):
                f=features(points,centers,self.cfg.rbf_width);raw=f@w
                df=-(points[:,None,:]-centers)/self.cfg.rbf_width**2*f[:,:,None]
                grad=torch.sigmoid(raw)[:,None]/4*torch.einsum('nmd,m->nd',df,w)
                return torch.cat((torch.nn.functional.softplus(raw)[:,None]/4,grad),1).reshape(-1)
            J=torch.autograd.functional.jacobian(vector,weights)
            self.field_draw_work+=len(ids)*3*len(weights)
            indices=[3*k+j for k in ids for j in range(3)]
            C=tensor(covariance[np.ix_(indices,indices)])+torch.eye(len(indices),dtype=torch.float64)*.01
            L=torch.linalg.cholesky(C);A=torch.linalg.solve_triangular(L,J,upper=False)
            y=torch.linalg.solve_triangular(L,(tensor(draw[ids].reshape(-1))-vector(weights))[:,None],upper=False).flatten()
            delta=torch.linalg.solve(A.T@A+self.cfg.fit_ridge*torch.eye(len(weights),dtype=torch.float64),A.T@y)
            updated.append(replace(stage,weights=tuple((weights+delta.clamp(-.5,.5)).tolist())))
        self.atlas=replace(self.atlas,stages=tuple(updated))
        self.sampled_anchors=draw.tolist();return True

    def prepare(self,x):
        if not self.atlas.stages[self.index].supported(x[:2],self.cfg):raise DomainExit('unsupported_state')
        if self.cross_stage:
            nxt=self.atlas.advance(x[:2],self.index)
            if nxt!=self.index:
                # Environmental shaping only; physical state and kinetic energy stay fixed.
                jump=float(self.atlas.value(tensor(x[:2]),nxt)-self.atlas.value(tensor(x[:2]),self.index))
                self.context_jumps.append(dict(from_stage=self.atlas.stages[self.index].id,to_stage=self.atlas.stages[nxt].id,
                                               environmental_jump=jump,physical_jump=0.))
            self.index=nxt

    def supported(self,q):
        try:self.atlas.index_at(q);return True
        except DomainExit:return False

    def risk_truth(self,q):
        # Interface name denotes environment evaluation, not access to hidden scene truth.
        q=np.asarray(q);values=[]
        for point in q.reshape(-1,2):
            index=self.index if self.atlas.stages[self.index].supported(point,self.cfg) else self.atlas.index_at(point)
            values.append(float(self.atlas.value(tensor(point),index)))
        return np.asarray(values).reshape(q.shape[:-1])

    def step(self,x,u):
        after=flow(tensor(x),tensor(u),self._damping,self.cfg.dt).numpy()
        if not self.atlas.stages[self.index].supported(after[:2],self.cfg):raise DomainExit('unsupported_motion')
        a=x[:2];delta=after[:2]-a;collision=False
        for ox,oy,r in self.atlas.stages[self.index].obstacles:
            c=np.array([ox,oy]);f=np.clip(np.dot(c-a,delta)/(np.dot(delta,delta)+1e-20),0,1)
            collision |= np.linalg.norm(a+f*delta-c)<=r+.12
        cost=self.cfg.dt*(1+4*float(self.risk_truth(after[:2]))+.03*float(u@u))
        return after,bool(collision),cost,balance(tensor(x),tensor(u),self._damping,self.cfg.dt)

    def sense(self,x,tick):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,tick,7919]))
        offsets=np.array([(a,b) for a in np.linspace(-1.,1.,5) for b in np.linspace(-1.,1.,5) if a*a+b*b<=1.])*self.cfg.sensor_radius
        points=x[:2]+offsets;noise=rng.normal(0,.003,len(points))
        readings=tuple(Reading(f'hypothetical_sensor:{self.seed}:{tick}:{j}',tick*self.cfg.dt,tuple(q),
                              float(max(.001,self.risk_truth(q)+noise[j]))) for j,q in enumerate(points) if self.supported(q))
        obstacles=set(o for s in self.atlas.stages for o in s.obstacles)
        visible=tuple(sorted(o for o in obstacles if np.linalg.norm(x[:2]-o[:2])-o[2]<=self.cfg.sensor_radius))
        return Observation(tick*self.cfg.dt,readings,visible)


def predict_pair(packet,status,cfg,comm,summary=None,cross_stage=False,horizons=(8,16,32),seed=0,
                 environment_receiver=None):
    """No private receiver or simulator input: environment_receiver must be a reconstructed model.

The default environment is reconstructed from the same legal advertisement as
its learner. Evaluators may supply a fixed reconstructed model to hold the world
constant across receiver-summary comparisons; never used by runtime scheduling.
"""
    started=time.perf_counter()
    if summary is not None:
        a=reconstruct_receiver(summary,packet.sender,summary.timestamp,cross_stage)
        state=np.array(summary.state);tick=int(round(summary.timestamp/cfg.dt))
    elif status is not None:
        a=reconstruct_advertised_receiver(status,cfg,comm)
        a.cfg=replace(a.cfg,replay_domain='retained' if cross_stage else 'single')
        state=np.array([*status.q,*status.v]);tick=int(round(status.timestamp/cfg.dt))
    else:return dict(rows=[dict(horizon=h,valid=False,reason='missing_status',benefit=None) for h in horizons],branches={},seconds=0.)
    world=ModelEnvironment(environment_receiver or a,seed,cross_stage)
    if not world.sample_shared_field(packet,status,seed):
        return dict(rows=[dict(horizon=h,valid=False,reason='missing_common_uncertainty',benefit=None) for h in horizons],branches={},seconds=0.)
    result=continuation(a,state,world,packet,tick,horizons)
    result['sampled_anchors']=world.sampled_anchors
    result['field_draw_work']=world.field_draw_work
    result['continuation_seconds']=result['seconds'];result['seconds']=time.perf_counter()-started
    result['approximation']='hypothetical history and validation; not exact receiver learning'
    return result


def variant_prediction(packet,status,summary,cfg,comm,variant,seed=0,horizons=(8,16,32)):
    started=time.perf_counter()
    if variant not in ('baseline','compact','cross_stage','both'):raise ValueError('invalid variant')
    cross=variant in ('cross_stage','both')
    if variant=='baseline':return predict_pair(packet,status,cfg,comm,seed=seed,horizons=horizons)
    if summary is None:return dict(rows=[dict(horizon=h,valid=False,reason='missing_summary',benefit=None) for h in horizons],branches={},seconds=0.)
    a=reconstruct_receiver(summary,packet.sender,summary.timestamp,cross)
    if variant=='cross_stage':
        # Stage geometry/history is necessary for this ablation, but learned policy
        # and plant initialization remain the legacy advertisement defaults.
        a.policy=tensor(cfg.policy_initial);a.damping=cfg.damping_initial
    world=ModelEnvironment(a,seed,cross)
    if not world.sample_shared_field(packet,status,seed):
        return dict(rows=[dict(horizon=h,valid=False,reason='missing_common_uncertainty',benefit=None) for h in horizons],branches={},seconds=0.)
    result=continuation(a,np.array(summary.state),world,packet,int(round(summary.timestamp/cfg.dt)),horizons)
    result['sampled_anchors']=world.sampled_anchors
    result['field_draw_work']=world.field_draw_work
    result['continuation_seconds']=result['seconds'];result['seconds']=time.perf_counter()-started
    result['approximation']='synthetic history and validation; not measured evidence'
    return result
