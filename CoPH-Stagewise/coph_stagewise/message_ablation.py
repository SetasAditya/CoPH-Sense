"""Leave-one-out ablation of the packet m_{i->j} on the graded-hazard study.

Each arm removes or degrades one component of the full learned-field packet and keeps
everything else (channel, schedule, validation, ledger rotation) as in hazard_demo:

- field: the full packet (reference);
- no_gradient: anchor gradients zeroed and marked unobserved;
- no_covariance: bootstrap covariance replaced by a unit, uninformative diagonal;
- no_sender_state: sender q, p, v, H and V_task zeroed (the receiver's validation still passes);
- no_support: support points removed, so the route store must reject the packet;
- newest_only: only the newest stage is exported (no ledger rotation);
- anchors_2 / anchors_1: fewer anchors per packet;
- quantized: values thresholded at tau=0.25 (occupancy), for reference;
- compact: no_gradient, no_covariance and no_sender_state together;
- none: no communication.
"""
import argparse,functools
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .crossing_demo import configuration
from .maze_family import NetworkWorld
from .cooperative import CommunicationConfig
from .staggered_demo import execute
from .sharing import FieldLearner,LEDGER_MODES,LedgerExport,GRADIENT_UNKNOWN,quantized
from .hazard_demo import SCENES,specification
from .evaluation import dump,load
from .cli import source_hashes,archive_sources

SEEDS=(79000,79001,79002,79003)


class NoGradient(LedgerExport,FieldLearner):
    def export(self,x,recipient):
        m=super().export(x,recipient)
        if m is None:return None
        n=len(m.anchors);cov=np.array(m.covariance)
        for k in range(n):
            for j in (1,2):i=3*k+j;cov[i,:]=0;cov[:,i]=0;cov[i,i]=GRADIENT_UNKNOWN
        return replace(m,anchors=tuple((*a[:3],0.,0.) for a in m.anchors),covariance=tuple(map(tuple,cov)))


class NoCovariance(LedgerExport,FieldLearner):
    def export(self,x,recipient):
        m=super().export(x,recipient)
        return None if m is None else replace(m,covariance=tuple(map(tuple,np.eye(3*len(m.anchors)))))


class NoSenderState(LedgerExport,FieldLearner):
    def export(self,x,recipient):
        m=super().export(x,recipient)
        return None if m is None else replace(m,q=(0.,0.),p=(0.,0.),v=(0.,0.),H=0.,V_task=0.)


class NoSupport(LedgerExport,FieldLearner):
    def export(self,x,recipient):
        m=super().export(x,recipient)
        return None if m is None else replace(m,support_points=())


class Compact(LedgerExport,FieldLearner):
    def export(self,x,recipient):
        m=super().export(x,recipient)
        if m is None:return None
        n=len(m.anchors);cov=np.diag([1.,GRADIENT_UNKNOWN,GRADIENT_UNKNOWN]*n)
        return replace(m,anchors=tuple((*a[:3],0.,0.) for a in m.anchors),covariance=tuple(map(tuple,cov)),q=(0.,0.),p=(0.,0.),v=(0.,0.),H=0.,V_task=0.)


class Quantized(LedgerExport,FieldLearner):
    def export(self,x,recipient):
        m=super().export(x,recipient)
        return None if m is None else quantized(m,.25)


# Module-level names keep the classes picklable for the runner's packet audits.
LEDGER_FIELD=LEDGER_MODES['field']
# label: (learner class, anchors per packet, channel)
ARMS=dict(none=(LEDGER_FIELD,4,'disabled'),field=(LEDGER_FIELD,4,'reciprocal_fixed'),no_gradient=(NoGradient,4,'reciprocal_fixed'),
          no_covariance=(NoCovariance,4,'reciprocal_fixed'),no_sender_state=(NoSenderState,4,'reciprocal_fixed'),
          no_support=(NoSupport,4,'reciprocal_fixed'),newest_only=(FieldLearner,4,'reciprocal_fixed'),
          anchors_2=(LEDGER_FIELD,2,'reciprocal_fixed'),anchors_1=(LEDGER_FIELD,1,'reciprocal_fixed'),
          quantized=(Quantized,4,'reciprocal_fixed'),compact=(Compact,4,'reciprocal_fixed'))


def run(job):
    spec,arm,root=job;torch.set_num_threads(1);cls,anchors,channel=ARMS[arm]
    network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges'])))
    factories=[functools.partial(cls,network=network,risk_weight=spec['risk_weights'][i],unknown_risk=spec['unknown_risk']) for i in range(2)]
    def factory(cfg,x,goal,agent_id,communication):return factories[int(agent_id)](cfg,x,goal,agent_id=agent_id,communication=communication)
    comm=CommunicationConfig(byte_budget=4000000,ttl=300.,anchors=anchors)
    r=execute(spec,channel,comm,learner_factory=factory,world_factory=NetworkWorld,config=configuration(spec['steps']))
    r['arm']=arm;r['channel']=channel;dump(Path(root)/'episodes'/f"{spec['seed']}_{arm}.json.gz",r)
    return dict(scene=spec['scene'],seed=spec['seed'],arm=arm,goals=sum(a['success'] for a in r['agents']),
                carrier=r['agents'][1]['components']['time']+r['agents'][1]['components']['risk'])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    ap.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS),help='subset of arms to run')
    args=ap.parse_args(argv);args.out.mkdir(parents=True,exist_ok=False);jobs=[]
    for scene in SCENES:
        root=args.out/scene;(root/'episodes').mkdir(parents=True);specs=[specification(scene,seed=s) for s in SEEDS]
        dump(root/'manifest.json',dict(specs=specs,arms={k:dict(learner=v[0].__name__,anchors=v[1],channel=v[2]) for k,v in ARMS.items()},
             export='retained-stage rotation (LedgerExport) except newest_only',source_hashes=source_hashes()))
        archive_sources(root);jobs+=[(spec,arm,str(root)) for spec in specs for arm in args.arms]
    done=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):done.append(f.result());dump(args.out/'completion.json',done);print(done[-1],flush=True)


if __name__=='__main__':main()
