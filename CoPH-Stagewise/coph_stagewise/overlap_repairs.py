"""Frozen controlled tests of opt-in supported scalar-potential stage blending."""
import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .records import Config,Reading,new_stage
from .learner import Learner
from .model import tensor,flow,balance
from .stage_prediction import FrozenStages,DomainExit


def fixture(mode):
    cfg=Config(controller_mode='protected',shaping_mode='progress',replay_domain='retained',replay_objective='mission',stage_blend_mode=mode)
    a=Learner(cfg,(0.,0.,0.,0.),(4.,0.),mode='evidence');a.stages.append(new_stage(1,(2.1,0.),(1.06,0.,.4,0.),a.goal,cfg))
    for k,s in enumerate(a.stages):
        s.weights=tensor(np.linspace(-1.,1.,16)+2*k)
        s.readings=[Reading('overlap:%d'%(5*i+1),i*.12,(float(x),float(y)),.1) for i,(x,y) in enumerate((x,y) for x in np.linspace(.6,1.5,10) for y in (-.5,0.,.5))]
    return a,FrozenStages.freeze(a)


def evaluate(mode):
    a,atlas=fixture(mode);rows=[];eps=1e-5;p=a.policy
    for x in np.linspace(.65,1.45,81):
        q=tensor([x,.03]).requires_grad_();index=int(x>=1.05);energy=atlas.energy(q,index,p);gradient=torch.autograd.grad(energy,q)[0]
        finite=tensor([(float(atlas.energy(q.detach()+tensor(np.eye(2)[k]*eps),index,p))-float(atlas.energy(q.detach()-tensor(np.eye(2)[k]*eps),index,p)))/(2*eps) for k in range(2)])
        state=torch.cat((q.detach(),tensor([.3,0.])));u,nominal=atlas.control(state,index,p,.85,{},0)
        row=dict(x=float(x),energy=float(energy),force=(-gradient).tolist(),applied=u.detach().tolist(),nominal=nominal.detach().tolist(),
            finite_difference_error=float(torch.linalg.vector_norm(gradient-finite)),support=[s.supported(q.detach().numpy(),a.cfg) for s in atlas.stages])
        if mode=='smooth':
            members,weights=atlas.blend_weights(q);row.update(members=[s.id for s in members],weights=weights.detach().tolist())
        rows.append(row)
    fixed=tensor([1.06,.03,.3,0.]);u0,p0=atlas.control(fixed,0,p,.85,{},0);u1,p1=atlas.control(fixed,1,p,.85,{},0)
    handoff=dict(energy_jump=float(atlas.energy(fixed[:2],1,p)-atlas.energy(fixed[:2],0,p)),nominal_force_jump=float(torch.linalg.vector_norm(p1-p0)),
        applied_force_jump=float(torch.linalg.vector_norm(u1-u0)),physical_jump=0.)
    limiting=[]
    for delta in (.1,.05,.01,.005,.001,.0001):
        q0=tensor([1.05-delta,.03]);q1=tensor([1.05+delta,.03]);q0.requires_grad_();q1.requires_grad_();e0=atlas.energy(q0,0,p);e1=atlas.energy(q1,1,p)
        f0=-torch.autograd.grad(e0,q0)[0];f1=-torch.autograd.grad(e1,q1)[0]
        limiting.append(dict(delta=delta,energy_difference=float(e1-e0),force_difference=float(torch.linalg.vector_norm(f1-f0))))
    def replay(policy,record=False):
        x=tensor([.95,0.,.5,0.]);index=0;cost=tensor(0.);paths=[];context={}
        for step in range(8):
            index=atlas.advance(x[:2].detach().numpy(),index);u,nominal=atlas.control(x,index,policy,.85,context,step)
            y=flow(x,u,.85,a.cfg.dt);cost=cost+a.cfg.dt*(1+4*atlas.value(y[:2],index)+.03*u.square().sum())
            if record:paths.append(dict(step=step,stage=atlas.stages[index].id,x=x.detach().tolist(),after=y.detach().tolist(),u=u.detach().tolist(),
                nominal=nominal.detach().tolist(),energy=balance(x.detach(),u.detach(),.85,a.cfg.dt),control=copy.deepcopy(context['last_control'])))
            x=y
        return cost,paths
    pp=p.clone().requires_grad_();loss,_=replay(pp);g=torch.autograd.grad(loss,pp)[0];finite=tensor([(float(replay(p+tensor(np.eye(2)[k]*eps))[0])-float(replay(p-tensor(np.eye(2)[k]*eps))[0]))/(2*eps) for k in range(2)])
    _,path=replay(p,True)
    # Execution from the same frozen complete state must use identical controls.
    live=copy.deepcopy(a);live.damping=.85;parity=[]
    for t in path:
        control,_=live.action(np.array(t['x']));parity.append(float(np.linalg.norm(control-np.array(t['u']))))
    frontier_rejected=False
    if mode=='smooth':
        try:atlas.energy(tensor([2.8,1.8]),1,p)
        except DomainExit:frontier_rejected=True
    from .maze_learning import MazeLearner
    from .maze_visuals import landscape
    a.local_geometry_reports={};a.remote_geometry_reports={};logged=MazeLearner.control_model(a)
    state=np.array([1.06,.03,.3,0.]);_,nominal=a.action(state)
    H,F,supported=landscape(logged,np.array([state[:2]]),state[2:],a.cfg)
    logged_error=float(np.linalg.norm(F[0]-1.4*state[2:]-nominal))
    return dict(mode=mode,profile=rows,fixed_state_handoff=handoff,limiting=limiting,replay=dict(steps=8,gradient=g.tolist(),finite_difference=finite.tolist(),
        max_error=float(torch.linalg.vector_norm(g-finite)),trajectory=path,execution_control_errors=parity),unsupported_frontier_rejected=frontier_rejected,
        logged_landscape_nominal_error=logged_error,logged_model=logged,
        qualification='Legacy execution always uses the latest retained stage; historical restart can select the earlier stage. Smooth mode is index-invariant on a fixed certified atlas.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    files=[Path(__file__),*[Path(__file__).with_name(f) for f in ('stage_prediction.py','navigation.py','model.py','records.py','learner.py','maze_learning.py','staggered_demo.py','maze_visuals.py')]]
    protocol=dict(modes=['legacy','smooth'],queries=81,interval=[.65,1.45],handoff=[1.06,.03,.3,0.],finite_difference=1e-5,replay_steps=8,
        config=asdict(fixture('smooth')[0].cfg),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=1,dtype='float64',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        claims='Fixed-membership conservative blending only; no global smoothness, stability, passivity or radial-progress guarantee.')
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2));archive=out/'sources';archive.mkdir()
    for p in files:(archive/p.name).write_bytes(p.read_bytes())
    started=time.perf_counter();results=[evaluate(mode) for mode in protocol['modes']];(out/'records.json').write_text(json.dumps(results,indent=2))
    blend=results[1];assert max(r['finite_difference_error'] for r in blend['profile'])<1e-6
    assert blend['fixed_state_handoff']['energy_jump']==0. and blend['fixed_state_handoff']['nominal_force_jump']==0.
    assert blend['replay']['max_error']<1e-6 and max(blend['replay']['execution_control_errors'])<1e-12 and blend['unsupported_frontier_rejected']
    assert all(r['logged_landscape_nominal_error']<1e-12 for r in results)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for r in results:
        xs=[p['x'] for p in r['profile']];axes[0].plot(xs,[p['energy'] for p in r['profile']],label=r['mode']);axes[1].plot(xs,[p['force'][0] for p in r['profile']],label=r['mode'])
        axes[2].loglog([p['delta'] for p in r['limiting']],[p['force_difference'] for p in r['limiting']],'o-',label=r['mode'])
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    axes[0].set(xlabel='Overlap x',ylabel='Navigation potential');axes[1].set(xlabel='Overlap x',ylabel='Conservative x force');axes[2].set(xlabel='Distance to switch on each side',ylabel='Force difference')
    fig.tight_layout();fig.savefig(out/'overlap_comparison.png',dpi=170);fig.savefig(out/'overlap_comparison.pdf');plt.close(fig)
    lines=['# Supported scalar-potential blending','', 'The opt-in smooth mode blends complete navigation potentials over retained stages already supported by local measurements. Compact cubic rectangular weights are normalized and differentiated along with every stage potential. All members use the same target and sensed geometry. Legacy remains the default.','',
        '| Mode | Fixed-state shaping jump | Nominal force jump | Applied force jump | Replay gradient error |','|---|---:|---:|---:|---:|']
    for r in results:
        h=r['fixed_state_handoff'];lines.append('| %s | %.6g | %.6g | %.6g | %.3g |'%(r['mode'],h['energy_jump'],h['nominal_force_jump'],h['applied_force_jump'],r['replay']['max_error']))
    lines+=['','Smooth replay/execution controls agree at every step of the eight-step crossing; unsupported frontier is rejected. Legacy controls can differ because the active historical model and latest execution model differ at the same state. This controlled result is not a navigation-success or mission-cost comparison.','',
        'Continuity is conditional on fixed supported membership and frozen models/target/geometry. The hard nearest-measurement support cutoff, stage creation/eviction, model commits, and waypoint changes can still introduce context jumps. Spatial blending adds gradient terms, so the original per-stage radial progress bound does not automatically transfer. General switched-system stability and passivity remain unproved. No measurement or replay support is created by blending.','',
        'The production potential/force and the unchanged safety projection/guards are shared by execution and replay. Baseline/candidate policy optimization uses the same frozen stage collection. Safety guard evaluations retain their existing flow accounting; field blending records separate field/partition/certification work. Rich logged models retain bounded per-stage bases and support so their scalar landscapes and nominal forces can be reproduced; both legacy and smooth nominal-landscape errors are below 1e-12.','', 'Runtime %.3f s; frozen sources and all query/control records included.'%(time.perf_counter()-started)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
