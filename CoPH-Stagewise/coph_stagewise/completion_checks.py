"""Controlled completion and reciprocal causal checks (no evaluation-seed tuning)."""
import copy,hashlib,pickle
import numpy as np
import torch
from dataclasses import asdict
from .records import Config,Transition
from .learner import Learner
from .cooperative import CooperativeLearner
from .cooperative_evaluation import CooperativeWorld,specification
from .continuations import continuation
from .evaluation import dump


def controlled_checks(out=None):
    torch.set_num_threads(1);missions=[]
    cfg=Config(controller_mode='routed',shaping_mode='progress',replay_domain='retained',replay_objective='mission',replay_horizon_mode='supported_boundary',steps=240)
    for name,obstacles in [('disk',[(2.,0.,.35)]),('passage',[(2.,-.85,.35),(2.,.85,.35)])]:
        for mode in ('frozen','full'):
            spec=dict(name=name,seed=91,initial=[0.,0.,0.,0.],goal=[4.,0.],damping=.8,patches=[[2.,.5,.6,.5,1.5]],obstacles=obstacles,dynamic=False)
            w=CooperativeWorld(spec,cfg);a=Learner(cfg,spec['initial'],spec['goal'],mode=mode);x=np.array(spec['initial']);a.observe(w.sense(x,0));records=[];updates=[];hit=False
            for tick in range(cfg.steps):
                if mode=='full' and tick%cfg.update_every==0:updates.append(a.update(x))
                u,_=a.action(x);y,hit,_,_=w.step(x,u);records.append(dict(tick=tick,x=x.tolist(),after=y.tolist(),u=u.tolist(),navigation=copy.deepcopy(a.last_control)))
                a.observe(w.sense(y,tick+1),Transition(tick,(tick+1)*cfg.dt,tuple(x),tuple(u),tuple(y),a.version));x=y
                if hit or np.linalg.norm(x[:2]-a.goal)<.25:break
                a.advance(x)
            missions.append(dict(name=name,mode=mode,config=asdict(cfg),spec=spec,success=bool(np.linalg.norm(x[:2]-a.goal)<.25 and not hit),collision=bool(hit),goal_gap=float(np.linalg.norm(x[:2]-a.goal)),trajectory=records,updates=updates))
    spec=specification('complementary',34,'smoke');w=CooperativeWorld(spec,cfg);agents=[];states=[]
    for i in range(2):
        x=np.array([.1,.2*i,0.,0.]);a=CooperativeLearner(cfg,x,(6.,0.),agent_id=str(i))
        for tick in range(10):a.observe(w.sense(x,tick))
        states.append(x);agents.append(a)
    packets=[a.export(states[i],str(1-i)) for i,a in enumerate(agents)];audits=[]
    for i,a in enumerate(agents):
        before=pickle.dumps(a);state=states[i].copy();r=continuation(a,state,w,packets[1-i],9,(8,16,32))
        assert before==pickle.dumps(a);np.testing.assert_array_equal(state,states[i])
        d=r['branches']['deliver'];b=r['branches']['withhold'];field_change=float(np.linalg.norm(np.array(d['initial_model']['fields'])-b['initial_model']['fields']));force_change=float(np.linalg.norm(np.array(d['initial_control'])-b['initial_control']))
        r.update(receiver=i,field_revision=field_change,control_revision=force_change,operational_state_unchanged=True,operational_sha256=hashlib.sha256(before).hexdigest())
        assert field_change>1e-10 and force_change>1e-10
        audits.append(r)
    result=dict(missions=missions,audits=audits,passed=all(m['success'] for m in missions),scope='controlled development; true-world continuations are evaluator-only and never training records')
    if out is not None:dump(out,result)
    assert result['passed'],[(m['name'],m['mode'],m['goal_gap']) for m in missions if not m['success']]
    return result
