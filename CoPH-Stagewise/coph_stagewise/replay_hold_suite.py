"""Stage-transition 'hold' repair for route-candidate replay, with a pre-declared stop rule.

Arms reuse route_decision_suite unchanged except for the prediction transition rule.
Pre-declared gate (written into the manifest before running):
  (1) repair: hold raises valid saved-snapshot batches for current and historical starts;
  (2) benefit: history_terminal_hold lowers mean team cost versus graph with a paired 95%
      interval excluding zero on the fresh seeds, without fewer goals or more collisions.
If (1) passes and (2) fails, historical replay is reported as a negative navigation result
and no further replay repair is attempted on these scenes. Forced-branch arms measure the
best achievable branch, i.e. the headroom any route decision could recover.
"""
import argparse,copy,gzip,json,pickle,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from . import route_decision_suite as rds
from .route_decision import RouteDecisionOptions,evaluate_routes
from .campaign_reconstruction import specification,NODES,EDGES,CASES,entered
from .crossing_demo import configuration
from .hazard_demo import communication
from .maze_family import NetworkWorld
from .staggered_demo import execute
from .attribution_audits import digest
from .evaluation import dump

SEEDS=tuple(range(86000,86008))
ARMS=dict(graph=(rds.OPTIONS['graph'],'strict'),prefix_terminal=(rds.OPTIONS['prefix_terminal'],'strict'),
          history_terminal=(rds.OPTIONS['history_terminal'],'strict'),prefix_terminal_hold=(rds.OPTIONS['prefix_terminal'],'hold'),
          history_terminal_hold=(rds.OPTIONS['history_terminal'],'hold'))
# Forced-branch references: the route is fixed to one branch for the whole mission.
FORCED=dict(force_upper=(0,1,2,4,5),force_lower=(0,1,3,4,5))


def audit_job(path):
    torch.set_num_threads(1)
    with gzip.open(path,'rb') as f:ctx,packet=pickle.load(f)
    agent=copy.deepcopy(ctx['agents'][1]);x=np.asarray(ctx['states'][1]).copy()
    agent.assimilate(packet);agent.mode='evidence';agent.update(x);out={}
    for mode in ('strict','hold'):
        for name in ('prefix_terminal','history_terminal'):
            a=copy.deepcopy(agent);a.stage_transition_mode=mode;r=evaluate_routes(a,x,rds.OPTIONS[name])
            out[f'{name}_{mode}']=dict(reason=r['reason'],selected=r['selected'],flow=r['flow_work'],
                invalid=[c.get('reason') for c in r['candidates'] if not c['valid']])
    return dict(snapshot=Path(path).name,results=out)


class Forced(rds.LedgerRouteLearner):
    def update(self,x):
        report=super().update(x);self.navigation.update(route_candidate_schema=1,route_candidate_nodes=list(self.forced));return report


def episode_job(job):
    case,seed,name,out=job;torch.set_num_threads(1)
    spec=specification(case,seed);cfg=replace(configuration(spec['steps']),route_field_mode='supported_field')
    forced=FORCED.get(name);options,mode=(rds.OPTIONS['graph'],'strict') if forced else ARMS[name]
    def factory(cfg,x,goal,agent_id,communication):
        a=(Forced if forced else rds.LedgerRouteLearner)(cfg,x,goal,agent_id=agent_id,communication=communication,mode='evidence',
            network=(NODES,EDGES),risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'],
            receiver_fit_mode='precision_bounded',receiver_validation_mode='relative_local',receiver_solver_mode='active_set',receiver_search_mode='diagnostic',
            route_options=options)
        a.stage_transition_mode=mode
        if forced:a.forced=forced;a.navigation.update(route_candidate_schema=1,route_candidate_nodes=list(forced))
        return a
    records=execute(spec,'reciprocal_fixed',communication('field'),audit=False,learner_factory=factory,world_factory=NetworkWorld,config=cfg)
    dump(Path(out)/'episodes'/f'{case}_{seed}_{name}.json.gz',records)
    decisions=[u['diagnostic']['route_decision'] for updates in records['updates'] for u in updates if 'route_decision' in u['diagnostic']]
    branch=[next((entered(p['x'][:2]) for p in t if entered(p['x'][:2]) is not None),None) for t in records['samples']]
    return dict(case=case,seed=seed,arm=name,agents=[dict(success=a['success'],collision=a['collision'],cost=a['cost']) for a in records['agents']],
        branch=branch,route_updates=len(decisions),valid=sum(d['reason']=='supported_route_choice' for d in decisions),
        reasons=sorted({c.get('reason') for d in decisions for c in d.get('candidates',[]) if not c['valid']}-{None}),
        flow=sum(d['flow_work'] for d in decisions))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);(out/'episodes').mkdir();torch.set_num_threads(1)
    dump(out/'manifest.json',dict(seeds=SEEDS,cases=CASES,arms=list(ARMS)+list(FORCED),gate=__doc__,sources=rds.hashes()))
    inputs=sorted(rds.SOURCE.glob('*.pkl.gz'));assert len(inputs)==24
    jobs=[(c,s,n,str(out)) for c in CASES for s in SEEDS for n in list(ARMS)+list(FORCED)];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        audits=[f.result() for f in as_completed([pool.submit(audit_job,p) for p in inputs])];dump(out/'audits.json',audits)
        rows=[]
        for f in as_completed([pool.submit(episode_job,j) for j in jobs]):
            rows.append(f.result());dump(out/'rows.json',rows);print(len(rows),len(jobs),rows[-1]['case'],rows[-1]['arm'],flush=True)
    valid={k:sum(a['results'][k]['reason']=='supported_route_choice' for a in audits) for k in audits[0]['results']}
    cost={(r['case'],r['seed'],r['arm']):sum(a['cost'] for a in r['agents']) for r in rows}
    rng=np.random.RandomState(186000);summary={};paired={}
    for n in list(ARMS)+list(FORCED):
        rr=[r for r in rows if r['arm']==n]
        summary[n]=dict(goals=sum(a['success'] for r in rr for a in r['agents']),collisions=sum(a['collision'] for r in rr for a in r['agents']),
            mean_team_cost=float(np.mean([cost[r['case'],r['seed'],n] for r in rr])),valid_decisions=sum(r['valid'] for r in rr),
            decisions=sum(r['route_updates'] for r in rr),flow=sum(r['flow'] for r in rr),invalid_reasons=sorted({x for r in rr for x in r['reasons']}))
        d=np.array([cost[c,s,n]-cost[c,s,'graph'] for c in CASES for s in SEEDS]);m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1)
        paired[n]=dict(mean=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],better=int((d<-1e-9).sum()),worse=int((d>1e-9).sum()))
    oracle={c:float(np.mean([min(cost[c,s,'force_upper'],cost[c,s,'force_lower']) for s in SEEDS])) for c in CASES}
    graph={c:float(np.mean([cost[c,s,'graph'] for s in SEEDS])) for c in CASES}
    h=paired['history_terminal_hold']
    gate=dict(repair=valid['history_terminal_hold']>valid['history_terminal_strict'] and valid['prefix_terminal_hold']>=valid['prefix_terminal_strict'],
        benefit=h['interval95'][1]<0 and summary['history_terminal_hold']['goals']>=summary['graph']['goals'] and summary['history_terminal_hold']['collisions']<=summary['graph']['collisions'])
    res=dict(snapshot_valid=valid,summary=summary,paired_vs_graph=paired,graph_team_cost=graph,best_branch_team_cost=oracle,
             headroom={c:graph[c]-oracle[c] for c in CASES},gate=gate,seconds=time.perf_counter()-t)
    dump(out/'summary.json',res);print(json.dumps(res,indent=1))


if __name__=='__main__':main()
