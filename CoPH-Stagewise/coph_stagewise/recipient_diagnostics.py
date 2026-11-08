"""Evaluator-only factorial diagnosis of recipient-value prediction errors.

Private snapshots and simulator truth never enter scheduling or training. This is
not a replacement value predictor: its frozen model is the incumbent receiver's
current stage estimate, shared exactly between the two receiver-state conditions.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from .cooperative import CommunicationConfig, advertise
from .cooperative_evaluation import execute, specification
from .scheduling import reconstruct_advertised_receiver
from .records import Observation, Transition
from .model import risk, tensor
from .evaluation import dump
from .cli import source_hashes, archive_sources


from .stage_prediction import DomainExit
from .continuations import continuation


class FrozenWorld:
    """Frozen reconstructed plant/field, with the execution sensor/noise convention."""
    def __init__(self, agent, seed):
        self.cfg=agent.cfg;self.seed=seed;self.time=agent.now
        self.stage=copy.deepcopy(agent.stages[-1])
        self._damping=agent.damping;self._obstacles=tuple(self.stage.obstacles)

    def supported(self,q):
        return bool(self.stage.supported(np.asarray(q),self.cfg))

    def risk_truth(self,q):
        q=np.asarray(q)
        if not all(self.supported(p) for p in q.reshape(-1,2)):raise DomainExit()
        return risk(tensor(q),self.stage.weights,self.stage.centers,self.cfg.rbf_width).detach().numpy()

    def step(self,x,u):
        from .world import World
        return World.step(self,x,u)

    def sense(self,x,tick):
        # Use the same offsets and indexed noise as execution, but never invent
        # observations outside the frozen model's certified spatial support.
        from .records import Reading
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,tick,7919]))
        offsets=np.array([(a,b) for a in np.linspace(-1.,1.,5) for b in np.linspace(-1.,1.,5)
                          if a*a+b*b<=1.])*self.cfg.sensor_radius
        points=x[:2]+offsets;noise=rng.normal(0,.003,len(points))
        readings=tuple(Reading(f'{self.seed}:{tick}:{j}',tick*self.cfg.dt,tuple(q),
                       float(max(.001,self.risk_truth(q)+noise[j])))
                       for j,q in enumerate(points) if self.supported(q))
        visible=tuple(o for o in self._obstacles if np.linalg.norm(x[:2]-o[:2])-o[2]<=self.cfg.sensor_radius)
        return Observation(tick*self.cfg.dt,readings,visible)



def diagnose(agent,state,world,packet,tick,horizons=(8,16,32)):
    status=advertise(copy.deepcopy(agent),state,packet.sender,[z[:2] for z in packet.anchors])
    frozen=FrozenWorld(agent,world.seed)
    receivers={'exact':copy.deepcopy(agent)}
    if status is not None:receivers['advertised']=reconstruct_advertised_receiver(status,agent.cfg,agent.communication)
    cells={}
    for receiver,a in receivers.items():
        for environment,w in [('true',world),('frozen',frozen)]:
            cells[receiver+'_'+environment]=continuation(a,state,w,packet,tick,horizons)
    return dict(packet=packet.identity,recipient=agent.agent_id,stage=agent.stages[-1].id,
                tick=tick,start=state.tolist(),horizons=list(horizons),cells=cells,
                advertised_missing=status is None,packet_payload=asdict(packet),
                receiver_damping=agent.damping,receiver_policy=agent.policy.tolist(),
                retained_stages=[s.id for s in agent.stages],status_bytes=0 if status is None else status.bytes())


def summarize(cases):
    rows=[]
    for case in cases:
        for h in case['horizons']:
            values={k:next(r for r in v['rows'] if r['horizon']==h) for k,v in case['cells'].items()}
            for condition in ('exact_frozen','advertised_frozen','advertised_true'):
                target=values['exact_true'];estimate=values.get(condition)
                valid=target['valid'] and estimate is not None and estimate['valid']
                rows.append(dict(scene=case['scene'],seed=case['seed'],packet=case['packet'],horizon=h,
                    condition=condition,valid=valid,truth=target['benefit'],
                    prediction=estimate['benefit'] if estimate else None,
                    error=estimate['benefit']-target['benefit'] if valid else None))
    summary=[]
    for h in sorted({r['horizon'] for r in rows}):
        # Common validity prevents an easier subset from making one method look better.
        eligible={r['packet']+':'+r['scene']+':'+str(r['seed']) for r in rows if r['horizon']==h}
        for r in rows:
            if r['horizon']==h and not r['valid']:eligible.discard(r['packet']+':'+r['scene']+':'+str(r['seed']))
        for c in ('exact_frozen','advertised_frozen','advertised_true'):
            subset=[r for r in rows if r['horizon']==h and r['condition']==c]
            matched=[r for r in subset if r['packet']+':'+r['scene']+':'+str(r['seed']) in eligible]
            errors=np.array([r['error'] for r in matched])
            nonzero=[r for r in matched if abs(r['truth'])>1e-6 and abs(r['prediction'])>1e-6]
            summary.append(dict(horizon=h,condition=c,total=len(subset),valid=sum(r['valid'] for r in subset),
                common_valid=len(matched),mae=float(np.mean(abs(errors))) if len(errors) else None,
                sign_pairs=len(nonzero),sign_agreement=float(np.mean([np.sign(r['truth'])==np.sign(r['prediction']) for r in nonzero])) if nonzero else None))
    return rows,summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--seed-base',type=int,default=60000)
    ap.add_argument('--scenes',nargs='+',default=['complementary','early_warning','reverse_warning','redundant','irrelevant'])
    args=ap.parse_args(argv)
    if args.out.exists():ap.error('output exists; choose a new directory')
    if args.seeds<1:ap.error('seeds must be positive')
    torch.set_num_threads(1);args.out.mkdir(parents=True)
    specs=[specification(s,args.seed_base+k,'development') for s in args.scenes for k in range(args.seeds)]
    manifest=dict(specs=specs,horizons=[8,16,32],arm='reciprocal_fixed',communication=asdict(CommunicationConfig()),
        source_hashes=source_hashes(),status='development diagnostic; not training or held-out evaluation',
        design='receiver exact/advertised x environment true/frozen; common frozen incumbent current-stage model',
        sensor='same indexed sensor noise; frozen model omits unsupported sensor locations and rejects unsupported motion',
        selection='first eligible delivered packet per visited receiver stage',numpy=np.__version__,torch=torch.__version__)
    dump(args.out/'manifest.json',manifest);archive_sources(args.out)
    cases=[];errors=[];started=time.perf_counter()
    for spec in specs:
        def capture(a,x,w,m,t):
            result=diagnose(a,x,w,m,t);result.update(scene=spec['name'],seed=spec['seed'])
            cases.append(result)
            dump(args.out/'cases.json.gz',cases)
        try:
            result=execute(spec,'reciprocal_fixed',diagnostic_callback=capture)
            dump(args.out/f"episode_{spec['name']}_{spec['seed']}.json.gz",result)
            print(spec['name'],spec['seed'],'cases',len(cases),flush=True)
        except Exception as exc:
            import traceback
            errors.append(dict(scene=spec['name'],seed=spec['seed'],error=str(exc),traceback=traceback.format_exc()))
            dump(args.out/'errors.json',errors)
    rows,summary=summarize(cases);dump(args.out/'rows.json',rows);dump(args.out/'summary.json',summary)
    text=['# Recipient prediction diagnostic','',
          'Development evidence. Exact receiver snapshots and true worlds are evaluator-only; scheduling and training are unchanged.',
          'The frozen environment uses the incumbent receiver’s current-stage field and damping. This isolates receiver reconstruction under a common model; it does not reproduce the legacy sampled-label estimator.',
          '',f'Cases: {len(cases)}. Incomplete episodes: {len(errors)}. Wall time: {time.perf_counter()-started:.1f} seconds.','',
          '| Horizon | Condition | Valid / total | Common valid | Common MAE | Nonzero sign agreement (n) |',
          '|---|---|---|---|---|---|']
    for r in summary:
        mae='unavailable' if r['mae'] is None else f"{r['mae']:.6f}"
        sign='unavailable' if r['sign_agreement'] is None else f"{100*r['sign_agreement']:.1f}%"
        text.append(f"| {r['horizon']} | {r['condition']} | {r['valid']}/{r['total']} | {r['common_valid']} | {mae} | {sign} ({r['sign_pairs']}) |")
    text+=['','MAE uses the same valid cases across all conditions at each horizon. Sign agreement excludes benefits within 1e-6 of zero. Domain exits are missing predictions, never zero benefits.',
           'Comparing advertised_true with exact_true isolates receiver-state approximation under the true simulator. Comparing exact_frozen with exact_true measures reconstructed-world error. These effects can interact; they are not an additive decomposition.',
           'Horizons share continuation prefixes. Different horizons estimate different truncated benefits; longer predictions are not automatically more accurate. Unsupported longer rollouts remain excluded.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,h in zip(axes,(8,16,32)):
        for condition in ('exact_frozen','advertised_frozen','advertised_true'):
            subset=[r for r in rows if r['horizon']==h and r['condition']==condition and r['valid']]
            ax.scatter([r['truth'] for r in subset],[r['prediction'] for r in subset],s=15,alpha=.6,label=condition)
        low,high=ax.get_xlim();ax.plot([low,high],[low,high],'k:',linewidth=1)
        ax.set_title(f'{h} steps');ax.set_xlabel('Exact receiver / true-world benefit');ax.set_ylabel('Predicted benefit')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(args.out/'prediction_scatter.png',dpi=160);plt.close(fig)
    text+=['','![Matched prediction diagnostic](prediction_scatter.png)']
    (args.out/'REPORT.md').write_text('\n'.join(text)+'\n')
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
