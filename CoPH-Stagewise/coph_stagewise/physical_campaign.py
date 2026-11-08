"""Controlled accounting, differentiation and overlap diagnostics; no controller changes."""
import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from .model import tensor, flow, balance, potential, risk, command
from .records import Config, Reading, new_stage
from .learner import Learner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accounting():
    rows=[]
    for dt in (.24,.12,.06,.03,.015):
        for damping in (.15,.85,1.5):
            for label,u in [('free',[0.,0.]),('held',[1.2,-.7]),('brake',[-2.,.5])]:
                x=tensor([.3,-.2,.8,-.1]); control=tensor(u)
                b=balance(x,control,damping,dt)
                # Compare the numerical integral of the true held-input power.
                ts=np.linspace(0.,dt,1001)
                states=torch.stack([flow(x,control,damping,float(t)) for t in ts])
                work=float(np.trapezoid((states[:,2:]*control).sum(-1).numpy(),ts))
                heat=float(np.trapezoid((damping*states[:,2:].square().sum(-1)).numpy(),ts))
                rows.append(dict(dt=dt,damping=damping,case=label,**b,
                    quadrature_work_error=abs(work-b['work']),quadrature_heat_error=abs(heat-b['heat'])))
    return rows


def derivative_checks():
    cfg=Config(controller_mode='protected',shaping_mode='progress',replay_domain='retained',replay_objective='mission')
    s=new_stage(0,(0.,0.),(0.,0.,0.,0.),(3.,0.),cfg)
    weights=tensor(np.linspace(-1.,1.,16));policy=tensor([1.2,2.])
    obstacles=((1.,.8,.25),);target=tensor([2.,0.]);q=tensor([.2,.1]).requires_grad_()
    energy=potential(q,target,weights,s.centers,cfg.rbf_width,policy,obstacles,cfg)
    grad=torch.autograd.grad(energy,q)[0].detach().numpy();eps=1e-5
    fd=np.array([(float(potential(q.detach()+tensor(np.eye(2)[i]*eps),target,weights,s.centers,cfg.rbf_width,policy,obstacles,cfg))-
        float(potential(q.detach()-tensor(np.eye(2)[i]*eps),target,weights,s.centers,cfg.rbf_width,policy,obstacles,cfg)))/(2*eps) for i in range(2)])
    # Frozen target and context; held-input integrator receives shaping exactly once.
    def replay(p,dt,steps):
        x=tensor([.2,.1,.15,0.]);cost=tensor(0.)
        for _ in range(steps):
            u,_=command(x,target,weights,s.centers,cfg.rbf_width,p,obstacles,cfg=cfg)
            x=flow(x,u,.85,dt)
            cost=cost+dt*((x[:2]-target).square().sum()+.1*u.square().sum()+risk(x[:2],weights,s.centers,cfg.rbf_width))
        return cost,x
    rows=[]
    for steps in (1,8,16,32):
        p=policy.clone().requires_grad_();c,_=replay(p,.12,steps);g=torch.autograd.grad(c,p)[0].numpy()
        finite=np.array([(float(replay(policy+tensor(np.eye(2)[i]*eps),.12,steps)[0])-float(replay(policy-tensor(np.eye(2)[i]*eps),.12,steps)[0]))/(2*eps) for i in range(2)])
        rows.append(dict(steps=steps,gradient=g.tolist(),finite_difference=finite.tolist(),max_error=float(np.max(abs(g-finite)))))
    # Context changes are evaluated at fixed physical state, never actuator work.
    old=float(energy);new=float(potential(q.detach(),target,weights+.15,s.centers,cfg.rbf_width,policy,obstacles,cfg))
    return dict(potential_gradient=grad.tolist(),finite_difference=fd.tolist(),max_force_error=float(np.max(abs(grad-fd))),
        replay=rows,context_change=dict(shaping_jump=new-old,physical_jump=0.,actuator_work=0.))


def overlap_checks():
    rows=[];profiles=[];cfg=Config()
    for count in (2,4,8,12):
        learner=Learner(cfg,[0.,0.,0.,0.],[5.,0.],mode='evidence')
        right=new_stage(1,(2.1,0.),(1.2,0.,0.,0.),(5.,0.),cfg);learner.stages.append(right)
        # Exact controlled sensor readings: different local coverage of one smooth field.
        uid=0
        for stage in learner.stages:
            for x in np.linspace(stage.center[0]-1.45,stage.center[0]+1.45,16):
                for y in (-.5,0.,.5):
                    value=.04+.6*np.exp(-.5*((x-1.15)/.55)**2-.5*((y-.2)/.8)**2)
                    stage.readings.append(Reading('physical:%d'%uid,uid*.12,(float(x),y),float(value)));uid+=1
        # Fit using production overlap rows, on supported points only.
        right.anchors=tuple((float(x),0.) for x in np.linspace(.61,1.49,count))
        weights,_=learner._field_fit()
        for stage,w in zip(learner.stages,weights):stage.weights=w
        delta_values=[];delta_grad=[]
        for x in np.linspace(.61,1.49,101):
            q=tensor([x,0.]).requires_grad_();values=[];grads=[]
            for stage in learner.stages:
                val=risk(q,stage.weights,stage.centers,cfg.rbf_width)
                values.append(float(val));grads.append(torch.autograd.grad(val,q,retain_graph=True)[0].numpy())
            dv=values[1]-values[0];dg=grads[1]-grads[0]
            delta_values.append(dv);delta_grad.append(np.linalg.norm(dg))
            profiles.append(dict(anchors=count,x=float(x),values=values,gradients=[g.tolist() for g in grads],supported=all(s.supported(q.detach().numpy(),cfg) for s in learner.stages)))
        rows.append(dict(anchors=count,queries=101,max_value_gap=float(np.max(abs(np.array(delta_values)))),
            rms_value_gap=float(np.sqrt(np.mean(np.square(delta_values)))),max_gradient_gap=float(max(delta_grad)),
            rms_gradient_gap=float(np.sqrt(np.mean(np.square(delta_grad))))))
    return dict(summary=rows,profiles=profiles,interpretation='Finite latent-value alignment does not impose derivative continuity; this is a controlled fitted-field diagnostic, not a mission outcome.')


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args(argv)
    out=args.out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    sources=[Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('records.py'),Path(__file__).with_name('learner.py')]
    manifest=dict(protocol='physical campaign v1',dt=[.24,.12,.06,.03,.015],damping=[.15,.85,1.5],overlap_anchors=[2,4,8,12],
        replay_steps=[1,8,16,32],finite_difference=1e-5,config=asdict(Config()),device='cpu',dtype='float64',threads=1,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sources={p.name:digest(p) for p in sources},
        restrictions='No changes to overlap algorithm, controller, fitting or safety; numerical checks do not prove stability or safety.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));source=out/'sources';source.mkdir()
    for p in sources:(source/p.name).write_bytes(p.read_bytes())
    began=time.perf_counter();result=dict(accounting=accounting(),derivatives=derivative_checks(),overlap=overlap_checks())
    result['seconds']=time.perf_counter()-began
    (out/'records.json').write_text(json.dumps(result,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for d in manifest['damping']:
        rr=[r for r in result['accounting'] if r['damping']==d and r['case']=='held']
        axes[0].loglog([r['dt'] for r in rr],[max(abs(r['residual']),1e-18) for r in rr],'o-',label='d=%s'%d)
    axes[0].set(xlabel='Held step (s)',ylabel='Absolute energy residual');axes[0].legend()
    for count in manifest['overlap_anchors']:
        rr=[r for r in result['overlap']['profiles'] if r['anchors']==count]
        axes[1].plot([r['x'] for r in rr],[r['values'][1]-r['values'][0] for r in rr],label=str(count)+' anchors')
        axes[2].plot([r['x'] for r in rr],[np.linalg.norm(np.array(r['gradients'][1])-r['gradients'][0]) for r in rr],label=str(count))
    axes[1].set(xlabel='Overlap x',ylabel='Right minus left field');axes[1].legend(fontsize=7)
    axes[2].set(xlabel='Overlap x',ylabel='Gradient / nominal force mismatch')
    fig.tight_layout();fig.savefig(out/'accounting_overlap.png',dpi=170);fig.savefig(out/'accounting_overlap.pdf');plt.close(fig)
    maxres=max(abs(r['residual']) for r in result['accounting']);der=result['derivatives']
    report=f'''# Physical consistency campaign

All 45 held-input cases completed. Maximum exact physical accounting residual: {maxres:.3g}. Numerical quadrature is an independent approximate check, not the production work formula.

Potential finite-difference error: {der['max_force_error']:.3g}. Maximum multistep policy-gradient error: {max(r['max_error'] for r in der['replay']):.3g}. Context changes alter shaping energy at fixed physical state and do not create kinetic change or actuator work.

Production local RBF fitting was tested with 2, 4, 8 and 12 supported overlap rows. Value and gradient discrepancies remain in `records.json`; increasing value anchors does not impose gradient continuity. These are controlled field evaluations, not stage-switching stability or protected-execution guarantees. The replay derivative check freezes context and does not exercise safety active-set switching.

See `accounting_overlap.png`, immutable `manifest.json`, per-query records and source copies. Runtime: {result['seconds']:.2f} s. All results are CPU float64, single thread.
'''
    (out/'REPORT.md').write_text(report);print(report)


if __name__=='__main__':main()
