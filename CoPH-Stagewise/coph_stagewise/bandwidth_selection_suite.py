"""Receiver-relevant versus rotation payload selection under per-agent byte ceilings.

Graded-hazard scenes (communication changes the carrier's cost there), communication_navigation's
fixed-rate link and compact codec, unchanged mechanisms; only the per-agent byte ceiling varies.
Advertisements are charged to the receiver as the mechanism specifies.
Pre-declared gate: at the two tightest ceilings pooled, receiver_relevant carrier cost lower than
rotation with a paired 95% interval excluding zero. Disabled and 4 MB arms give the headroom.
"""
import argparse,functools,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from . import communication_navigation as cn
from .cooperative import CommunicationConfig
from .sharing import LEDGER_MODES
from .hazard_demo import SCENES,specification
from .evaluation import dump

SEEDS=(86100,86101);BUDGETS=(8000,16000,32000,4000000)


def job(task):
    scene,seed,arm,selector,budget,out=task;torch.set_num_threads(1);spec=specification(scene,seed=seed)
    cn.CommunicationConfig=lambda byte_budget,ttl:CommunicationConfig(byte_budget=budget,ttl=ttl)
    def factory(spec):
        network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges'])))
        def make(cfg,x,goal,agent_id,communication):
            return LEDGER_MODES['field'](cfg,x,goal,mode='evidence',agent_id=agent_id,communication=communication,network=network,
                risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'])
        return make
    cn.factory=factory
    r=cn.execute(spec,arm,selector);key=f'{scene}_{seed}_{arm}_{selector}_{budget}'
    sent=[e for e in r['events'] if e.get('reason')=='sent'];refused=[e.get('reason') for e in r['events'] if e.get('reason') not in ('sent',None)]
    row=dict(key=key,scene=scene,seed=seed,arm=arm,selector=selector,budget=budget,goals=sum(a['success'] for a in r['agents']),
        collisions=sum(a['collision'] for a in r['agents']),carrier_cost=r['agents'][1]['cost'],team_cost=sum(a['cost'] for a in r['agents']),
        packets=len(sent),bytes={k:sum(a['bytes'][k] for a in r['agents']) for k in ('status','payload','acknowledgment')},
        refusals={k:refused.count(k) for k in set(refused)})
    dump(Path(out)/f'{key}.json',row);return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    args=ap.parse_args();out=args.out
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);dump(out/'manifest.json',dict(scenes=list(SCENES),seeds=SEEDS,budgets=BUDGETS,gate=__doc__))
    jobs=[(s,k,'disabled','rotation',4000000,str(out)) for s in SCENES for k in SEEDS]
    jobs+=[(s,k,'reciprocal_fixed',c,b,str(out)) for s in SCENES for k in SEEDS for c in cn.SELECTORS for b in BUDGETS]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job,j) for j in jobs]):rows.append(f.result());print(len(rows),len(jobs),rows[-1]['key'],flush=True)
    dump(out/'rows.json',rows);c={(r['scene'],r['seed'],r['arm'],r['selector'],r['budget']):r for r in rows}
    table={};rng=np.random.RandomState(186100)
    for b in BUDGETS:
        for sel in cn.SELECTORS:
            rr=[c[s,k,'reciprocal_fixed',sel,b] for s in SCENES for k in SEEDS]
            table[f'{sel}_{b}']=dict(carrier_cost=float(np.mean([r['carrier_cost'] for r in rr])),packets=float(np.mean([r['packets'] for r in rr])),
                bytes={k:float(np.mean([r['bytes'][k] for r in rr])) for k in ('status','payload','acknowledgment')},goals=sum(r['goals'] for r in rr),collisions=sum(r['collisions'] for r in rr))
    table['disabled']=dict(carrier_cost=float(np.mean([c[s,k,'disabled','rotation',4000000]['carrier_cost'] for s in SCENES for k in SEEDS])))
    d=np.array([c[s,k,'reciprocal_fixed','rotation',b]['carrier_cost']-c[s,k,'reciprocal_fixed','receiver_relevant',b]['carrier_cost'] for s in SCENES for k in SEEDS for b in BUDGETS[:2]])
    m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1)
    res=dict(table=table,tight_rotation_minus_relevant=dict(mean=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]),
             gate=dict(passed=bool(np.percentile(m,2.5)>0)))
    dump(out/'summary.json',res);print(json.dumps(res,indent=1))


if __name__=='__main__':main()
