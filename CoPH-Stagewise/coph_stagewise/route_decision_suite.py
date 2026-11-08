"""Development mechanism tests; preserved symmetric fork, no sensitivity claims."""
import argparse,copy,gzip,hashlib,json,os,pickle,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from .route_decision import RouteDecisionLearner,RouteDecisionOptions,evaluate_routes
from .sharing import LedgerExport
from .campaign_reconstruction import specification,NODES,EDGES,CASES,entered
from .crossing_demo import configuration
from .hazard_demo import communication
from .maze_family import NetworkWorld
from .staggered_demo import execute
from .attribution_audits import digest
from .evaluation import dump,load

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/current_campaign_v1/reconstruction/snapshots'
OPTIONS=dict(graph=RouteDecisionOptions(),prefix_only=RouteDecisionOptions(mode='prefix',terminal='none'),
    prefix_terminal=RouteDecisionOptions(mode='prefix'),history_terminal=RouteDecisionOptions(mode='prefix',starts='historical'))
SEEDS=(82400,82401)

class LedgerRouteLearner(LedgerExport,RouteDecisionLearner):pass


def hashes():
    names=['route_decision.py','route_decision_suite.py','maze_learning.py','maze_geometry.py','maze_family.py','model.py','learner.py',
        'cooperative.py','records.py','navigation.py','stage_prediction.py','receiver_fitting.py','localized_receiver.py','receiver_summary.py',
        'route_field.py','terrain_warning.py','sharing.py','staggered_demo.py','diagnostics.py','receiver_fitting_active_set.py']
    return {name:hashlib.sha256((ROOT/'coph_stagewise'/name).read_bytes()).hexdigest() for name in names if (ROOT/'coph_stagewise'/name).exists()}


def audit_job(path):
    torch.set_num_threads(1)
    with gzip.open(path,'rb') as f:ctx,packet=pickle.load(f)
    original=digest(ctx);agent=copy.deepcopy(ctx['agents'][1]);x=np.asarray(ctx['states'][1]).copy()
    event=agent.assimilate(packet);agent.mode='evidence';agent.update(x)
    measured=digest([(s.readings,s.transitions) for s in agent.stages]);navigation=digest(agent.navigation)
    results={}
    for name,options in OPTIONS.items():
        a=copy.deepcopy(agent);results[name]=evaluate_routes(a,x,options)
        assert digest([(s.readings,s.transitions) for s in a.stages])==measured
        assert digest(a.navigation)==navigation
        assert results[name]['flow_work']<=1024
    assert digest(ctx)==original
    return dict(snapshot=Path(path).name,packet=packet.identity,arrival=event,results=results,
        operational_state_unchanged=True,measurements_unchanged=True)


def episode_job(job):
    case,seed,name,out=job;torch.set_num_threads(1)
    spec=specification(case,seed);cfg=replace(configuration(spec['steps']),route_field_mode='supported_field')
    def factory(cfg,x,goal,agent_id,communication):
        return LedgerRouteLearner(cfg,x,goal,agent_id=agent_id,communication=communication,mode='evidence',
            network=(NODES,EDGES),risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'],
            receiver_fit_mode='precision_bounded',receiver_validation_mode='relative_local',receiver_solver_mode='active_set',receiver_search_mode='diagnostic',
            route_options=OPTIONS[name])
    records=execute(spec,'reciprocal_fixed',communication('field'),audit=False,learner_factory=factory,world_factory=NetworkWorld,config=cfg)
    filename=f'{case}_{seed}_{name}.json.gz';dump(Path(out)/'episodes'/filename,records)
    decisions=[u['diagnostic']['route_decision'] for updates in records['updates'] for u in updates if 'route_decision' in u['diagnostic']]
    firstbranch=[]
    for trajectory in records['samples']:
        firstbranch.append(next((entered(p['x'][:2]) for p in trajectory if entered(p['x'][:2]) is not None),None))
    return dict(case=case,seed=seed,mode=name,path='episodes/'+filename,agents=records['agents'],first_actual_branch=firstbranch,
        route_updates=len(decisions),valid_decisions=sum(d['reason']=='supported_route_choice' for d in decisions),
        excluded_decisions=[d['reason'] for d in decisions if d['selected'] is None],
        decision_flow=sum(d['flow_work'] for d in decisions),maximum_update_flow=max(a['max_update_work'] for a in records['agents']))


def render_report(out,rows,audits):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    fig,axes=plt.subplots(1,3,figsize=(13,4));summary={}
    for i,name in enumerate(OPTIONS):
        rr=[r for r in rows if r['mode']==name];agents=[a for r in rr for a in r['agents']]
        summary[name]=dict(episodes=len(rr),goals=sum(a['success'] for a in agents),collisions=sum(a['collision'] for a in agents),timeouts=sum(a['timeout'] for a in agents),
            mean_cost=float(np.mean([a['cost'] for a in agents])),valid_decisions=sum(r['valid_decisions'] for r in rr),
            all_decisions=sum(r['route_updates'] for r in rr),flow=sum(r['decision_flow'] for r in rr))
        for ax,key in zip(axes,['goals','mean_cost','flow']):ax.bar(i,summary[name][key]);ax.set_ylabel(key)
    for ax in axes:ax.set_xticks(range(4));ax.set_xticklabels(list(OPTIONS),rotation=30,ha='right')
    fig.tight_layout();fig.savefig(out/'figures/outcomes.png',dpi=150);plt.close(fig)
    failures=[dict(case=r['case'],seed=r['seed'],mode=r['mode'],agent=a['agent'],collision=a['collision'],timeout=a['timeout'],goal_gap=a['goal_gap'],path=r['path']) for r in rows for a in r['agents'] if not a['success'] or a['collision']]
    dump(out/'failure_inventory.json',failures)
    inventory=[]
    for r in failures:
        record=load(out/r['path']);fig,ax=plt.subplots(figsize=(5,3));nodes=np.asarray(NODES)
        for i,j in EDGES:ax.plot(*nodes[[i,j]].T,color='.8')
        for samples in record['samples']:ax.plot(*np.asarray([p['x'][:2] for p in samples]).T)
        ax.set_aspect('equal');ax.set_title(f"{r['case']} {r['seed']} {r['mode']} agent {r['agent']}",fontsize=8)
        name=f"failure_{r['case']}_{r['seed']}_{r['mode']}_{r['agent']}.png";fig.tight_layout();fig.savefig(out/'figures'/name,dpi=100);plt.close(fig);inventory.append(name)
    paired=[]
    for case in CASES:
        for seed in SEEDS:
            ref=next(r for r in rows if r['case']==case and r['seed']==seed and r['mode']=='graph')
            for r in rows:
                if r['case']==case and r['seed']==seed:
                    paired.append(dict(case=case,seed=seed,mode=r['mode'],team_cost_difference=sum(a['cost'] for a in r['agents'])-sum(a['cost'] for a in ref['agents']),branch_changed=r['first_actual_branch']!=ref['first_actual_branch']))
    dump(out/'results.json',dict(rows=rows,summary=summary,paired=paired))
    for case in CASES:
        rr=[load(out/next(r['path'] for r in rows if r['case']==case and r['seed']==82400 and r['mode']==name)) for name in OPTIONS]
        fig,axes=plt.subplots(1,4,figsize=(14,4));panels=[]
        maximum=max(p['time'] for r in rr for samples in r['samples'] for p in samples)
        for ax,name,r in zip(axes,OPTIONS,rr):
            nodes=np.asarray(NODES)
            for i,j in EDGES:ax.plot(*nodes[[i,j]].T,color='.85')
            lines=[ax.plot([],[])[0] for _ in range(2)];dots=[ax.plot([],[],'o')[0] for _ in range(2)]
            ax.set(xlim=(-.5,5.5),ylim=(-1.2,1.2),aspect='equal');panels.append((ax,name,r,lines,dots))
        indices=[]
        def update(now):
            for ax,name,r,lines,dots in panels:
                for i in range(2):
                    s=[p for p in r['samples'][i] if p['time']<=now+1e-9]
                    if not s:continue
                    points=np.asarray([p['x'][:2] for p in s]);lines[i].set_data(*points.T);dots[i].set_data([points[-1,0]],[points[-1,1]])
                ax.set_title(f'{name}\nt={now:.1f}s',fontsize=9)
        times=np.linspace(0,maximum,60);fig.suptitle(case+'; unchanged symmetric fork, seed 82400');fig.tight_layout()
        FuncAnimation(fig,update,frames=times,interval=100).save(out/'figures'/f'{case}.gif',writer=PillowWriter(fps=10));plt.close(fig)
        indices.append(dict(case=case,seed=82400,times=times.tolist(),semantics='last logged state at or before each shared timestamp'))
        dump(out/'figures'/f'{case}_animation.json',indices)
    valid={name:sum(a['results'][name]['reason']=='supported_route_choice' for a in audits) for name in OPTIONS}
    text='# Route-candidate prediction repair\n\n24 preserved snapshots and 24 new development episodes (48 independent-agent missions) on unchanged symmetric-fork cases, two previously used seeds. These are mechanism/readiness comparisons, not a replacement for failed sensitivity gates or held-out evidence. No fixture/hazard/controller tuning. All arms retain local and received fitting, measured validation, reconciliation, rollback, fixed-rate rotation and support rules. Policy-coefficient replay is disabled to isolate discrete route decisions.\n\n'
    text+='Prefixes use execution forces, clipping, safety, integration and support. Eight-step horizons and candidate/start sets are frozen within each batch. A batch with any unsupported candidate retains the incumbent. Current/historical modes share a 1,024-flow update ceiling including guards; actual work is reported. Continuation is approximate production graph length+risk at 1 m/s with unchanged unknown prior, no terminal effort estimate. It does not certify unseen field or geometry and is not an HJB value. The production incumbent connector is frozen before enumerating bounded simple paths.\n\n'
    text+='| Arm | Goals /12 | Collisions | Timeouts | Valid route decisions | Flow work |\n|---|---:|---:|---:|---:|---:|\n'
    for name,s in summary.items():text+=f"| {name} | {s['goals']} | {s['collisions']} | {s['timeouts']} | {s['valid_decisions']}/{s['all_decisions']} | {s['flow']} |\n"
    text+='\nValid saved-snapshot candidate batches /24: '+str(valid)+'. All exclusions and terminal/prefix components are retained in audits.json.gz. Actual branch changes versus graph: '+str(sum(r['branch_changed'] for r in paired if r['mode']!='graph'))+' /18 paired comparisons. Do not equate candidate evaluation or lower predicted cost with demonstrated historical-replay navigation benefit.\n\n'
    text+='All failures remain in failure_inventory.json and figures. Timing, risk, effort, failure costs, packets, model versions, route decisions and per-control protections are logged. Truth occurs only in simulator/evaluation, not planning. Original experimental records and gate outcomes remain unchanged.\n'
    (out/'REPORT.md').write_text(text)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Refuse existing output')
    torch.set_num_threads(1);sources=hashes();inputs=sorted(SOURCE.glob('*.pkl.gz'));assert len(inputs)==24
    manifest=dict(options={k:asdict(v) for k,v in OPTIONS.items()},cases=CASES,seeds=SEEDS,steps=500,snapshot_count=24,
        sources=sources,inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        development_only=True,sensitivity_gate='previous failed gate preserved; these engineering comparisons make no decision-sensitive treatment claim',
        scenarios=[specification(c,s) for c in CASES for s in SEEDS],numpy=np.__version__,torch=torch.__version__,threads=1,workers=args.workers)
    out.mkdir(parents=True);(out/'episodes').mkdir();(out/'figures').mkdir();(out/'sources').mkdir();dump(out/'manifest.json',manifest)
    for name in sources:(out/'sources'/name).write_bytes((ROOT/'coph_stagewise'/name).read_bytes())
    audits=[];rows=[];begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(audit_job,p) for p in inputs]):audits.append(future.result());dump(out/'audit_progress.json',dict(complete=len(audits)))
        dump(out/'audits.json.gz',audits)
        jobs=[(c,s,name,str(out)) for c in CASES for s in SEEDS for name in OPTIONS]
        for future in as_completed([pool.submit(episode_job,j) for j in jobs]):
            rows.append(future.result());dump(out/'completion.json',rows);print(len(rows),'/24',rows[-1]['case'],rows[-1]['mode'],flush=True)
    if hashes()!=sources:raise RuntimeError('Source changed; preserve results as incomplete, do not interpret')
    render_report(out,rows,audits);dump(out/'execution.json',dict(episodes=len(rows),snapshots=len(audits),seconds=time.perf_counter()-begin,source_fingerprints_verified=True))

if __name__=='__main__':main()
