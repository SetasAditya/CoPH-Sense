"""Frozen development campaign for packet representations, reciprocity and faults.

Runtime learners remain unchanged. All codecs use the established packet envelope;
costmap means a regular .5 m lattice of locally reconstructed scalar values, not a
new native compressed raster transport. Evaluator truth never enters exporters.
"""
import argparse, copy, hashlib, json, os, platform, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from .evaluation import dump, load
from .cooperative import CommunicationConfig, bootstrap_field, field_vector
from .crossing_demo import configuration
from .hazard_demo import specification as hazard_spec, SCENES
from .sharing import LEDGER_MODES, LedgerExport, FieldLearner, GRADIENT_UNKNOWN
from .message_ablation import Compact, Quantized
from .maze_family import grid_maze, specification as maze_spec, NetworkWorld, learner_for
from .staggered_demo import execute
from .route_oracle import predict, shortest
from . import complementary_reconstruction as complementary

ROOT=Path(__file__).resolve().parents[1]
REP_ARMS=('none','field','raw','costmap','compact','binary')
CHANNELS=('disabled','one_way_01','one_way_10','reciprocal_fixed')
BUDGETS=(120000,4000000)
LAYOUT_SEEDS=tuple(range(89500,89508))
NOISE_SEED=89400


class CostmapLearner(FieldLearner):
    """Bounded regular-lattice scalar costmap; unchanged evidence validation."""
    def export(self,x,recipient):
        packet=super().export(x,recipient)
        if packet is None:return None
        s=self.stages[-1]
        result=bootstrap_field(s,self.cfg,self.communication,self.now,12345+int(self.now*1000))
        if result is None:return None
        mean,_,_=result
        points=[(float(a),float(b)) for a in np.arange(s.center[0]-self.cfg.stage_width/2,s.center[0]+self.cfg.stage_width/2+1e-9,.5)
                for b in np.arange(s.center[1]-self.cfg.stage_height/2,s.center[1]+self.cfg.stage_height/2+1e-9,.5) if s.supported((a,b),self.cfg)]
        if not points:return packet
        ids=np.unique(np.linspace(0,len(points)-1,min(len(points),self.communication.anchors)).round().astype(int))
        points=np.asarray(points)[ids];values=field_vector(mean,s.centers,points,self.cfg.rbf_width).reshape(-1,3)[:,0]
        anchors=tuple((float(q[0]),float(q[1]),float(v),0.,0.) for q,v in zip(points,values))
        cov=np.diag([.0025,GRADIENT_UNKNOWN,GRADIENT_UNKNOWN]*len(anchors))
        self.export_work+=len(points)
        return replace(packet,anchors=anchors,covariance=tuple(map(tuple,cov)),uncertainty='regular .5m lattice graded costmap; scalar values only')


class LedgerCostmap(LedgerExport,CostmapLearner):pass


def source_hashes():
    # Other campaign workers may add modules while this suite runs. Freeze only
    # existing operational sources plus this runner, never unrelated worker files.
    paths=[p for p in (ROOT/'coph_stagewise').glob('*.py') if not p.name.startswith('campaign')]+[Path(__file__)]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def protocol(suite):
    base=dict(schema=1,suite=suite,development_only=True,device='cpu',dtype='float64',threads=1,
              controller=asdict(configuration(1800)),policy_replay=False,communication_interval_seconds=.96,
              ttl=300.,support_unchanged=True,source_hashes=source_hashes(),python=sys.version,
              numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),
              import_order=['numpy','torch'],thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    if suite=='representation':
        specs=[hazard_spec(scene,NOISE_SEED) for scene in SCENES]
        base.update(specs=specs,arms=list(REP_ARMS),budgets=list(BUDGETS),expected_episodes=48,
            animation_scene='double_mild',animation_seed=NOISE_SEED,
            seed_semantics='four designed layouts; one noise seed, no statistical generalization claim',
            codecs='field 4 anchors; raw 12 readings; costmap <=12 .5m grid scalar samples; compact joint neutralization; binary threshold .25; same JSON envelope',
            limitations=['costmap is a bounded sampled grid in the shared envelope, not an optimized native raster codec',
                          'same budget ceilings and opportunities, actual bytes/compute need not match'])
    elif suite=='reciprocity':
        specs=[];catalogue=[]
        for k,seed in enumerate(LAYOUT_SEEDS):
            # Include every draw, including unreachable truth layouts. No oracle screen.
            m=grid_maze(np.random.default_rng(seed),nx=3,ny=2,sx=5.,sy=4.,loops=2,doors=k%3,layout='crossing' if k%2==0 else 'relay',stagger=5.)
            s=maze_spec(m,f'unselected_{seed}',seed,1800);s['experiment']='unselected development layouts; no outcome or oracle-benefit screening'
            # Horizon from surveyed graph travel scale, not hidden doors or predicted benefit.
            length=sum(np.linalg.norm(np.asarray(m['nodes'][a])-m['nodes'][b]) for a,b in m['edges'])
            s['steps']=int(np.ceil((2*length/.25+max(s['start_seconds'])+30)/.12));specs.append(s)
            reachable=all(shortest(s['network_nodes'],s['network_edges'],s['obstacles'],s['initials'][i],s['goals'][i]) is not None for i in (0,1))
            catalogue.append(dict(seed=seed,selected=True,truth_reachable=reachable,doors=len(m['doors']),
                                  oracle={a:predict(s,a) for a in CHANNELS} if reachable else None))
        base.update(specs=specs,catalogue=catalogue,arms=list(CHANNELS),expected_episodes=32,
                    animation_seed=LAYOUT_SEEDS[0],byte_budget=4000000,
                    seed_semantics='eight independent layouts; one indexed sensor stream per layout; all generated layouts retained')
    else:
        base.update(cases=['complementary','reverse','redundant'],seeds=[89600,89601],expected_contexts=6,
            acquisition_seconds=.96,faults=['clean','duplicate','echo','stale','reordered','biased','correlated'],
            bias=.1,correlation='same source IDs in newer summary; compare sequential replacement versus clean',
            limitations=['snapshot audits establish interface/model behavior, not mission robustness',
                         'covariance is not calibrated; bias robustness is measured, not assumed'])
    return base


def freeze(root,suite):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/'manifest.json';m=protocol(suite)
    if p.exists():
        old=load(p)
        if old['fingerprint']!=fingerprint(m):raise ValueError('Source/config/environment fingerprint changed; refuse resume')
        return old
    m=dict(protocol=m,fingerprint=fingerprint(m),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with p.open('x') as f:json.dump(m,f,indent=2)
    (root/'source_archive').mkdir()
    for name in m['protocol']['source_hashes']:
        (root/'source_archive'/name).write_bytes((ROOT/'coph_stagewise'/name).read_bytes())
    (root/'episodes').mkdir();return load(p)


def rep_factory(spec,arm):
    cls={'none':LEDGER_MODES['field'],'field':LEDGER_MODES['field'],'raw':LEDGER_MODES['raw'],
         'costmap':LedgerCostmap,'compact':Compact,'binary':Quantized}[arm]
    network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges'])))
    def factory(cfg,x,goal,agent_id,communication):
        return cls(cfg,x,goal,mode='evidence',agent_id=agent_id,communication=communication,network=network,
                   risk_weight=spec['risk_weights'][int(agent_id)],unknown_risk=spec['unknown_risk'])
    return factory


def episode_metrics(r):
    agents=[]
    for i,a in enumerate(r['agents']):
        points=np.asarray([s['x'][:2] for s in r['samples'][i]])
        agents.append(dict(a,path_length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                           time_risk=a['components']['time']+a['components']['risk'],total_bytes=sum(a['bytes'].values())))
    return dict(agents=agents,goals=sum(a['success'] for a in agents),collisions=sum(a['collision'] for a in agents),
        timeouts=sum(a['timeout'] for a in agents),team_time_risk=sum(a['time_risk'] for a in agents),
        team_path=sum(a['path_length'] for a in agents),bytes=sum(a['total_bytes'] for a in agents),wall_seconds=r['wall_seconds'],
        audit_seconds=r['audit_seconds'],max_update_work=max(a['max_update_work'] for a in agents),
        delivered=len([e for e in r['events'] if e['kind']=='delivered']),
        budget_skips=len([e for e in r['events'] if e['kind']=='budget_skip']))


def run_episode(job):
    root,suite,spec,arm,budget=job;torch.set_num_threads(1)
    key=f"{spec['name']}_{spec['seed']}_{arm}_{budget}";path=Path(root)/'episodes'/f'{key}.json.gz'
    if path.exists():return dict(key=key,suite=suite,scene=spec['name'],seed=spec['seed'],arm=arm,budget=budget,**episode_metrics(load(path)))
    started=time.perf_counter()
    try:
        comm=CommunicationConfig(byte_budget=budget,ttl=300.,anchors=12 if arm in ('raw','costmap') else 4)
        factory=rep_factory(spec,arm) if suite=='representation' else learner_for(spec)
        if suite=='reciprocity':
            old=factory
            def factory(*a,**kw):return old(*a,mode='evidence',**kw)
        channel=('disabled' if arm=='none' else 'reciprocal_fixed') if suite=='representation' else arm
        r=execute(spec,channel,comm,audit=False,learner_factory=factory,world_factory=NetworkWorld,config=configuration(spec['steps']))
        r['arm']=arm;r['campaign_suite']=suite;dump(path,r)
        return dict(key=key,suite=suite,scene=spec['name'],seed=spec['seed'],arm=arm,budget=budget,**episode_metrics(r))
    except Exception as exc:
        return dict(key=key,suite=suite,scene=spec['name'],seed=spec['seed'],arm=arm,budget=budget,
                    error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)


def fault_variants(packet,receiver):
    # Changed packets are exclusively evaluator branch inputs.
    return dict(clean=[packet],duplicate=[packet,packet],echo=[replace(packet,sources=(receiver.agent_id+':echo',))],
        stale=[replace(packet,expires=receiver.now-.01)],
        reordered=[packet,replace(packet,identity=packet.identity+':old',timestamp=packet.timestamp-.01)],
        biased=[replace(packet,identity=packet.identity+':biased',anchors=tuple((*a[:2],a[2]+.1,*a[3:]) for a in packet.anchors))],
        correlated=[packet,replace(packet,identity=packet.identity+':correlated',timestamp=packet.timestamp+1e-10)])


def fault_suite(root,m):
    from .attribution_audits import digest
    rows=[];torch.set_num_threads(1)
    for case in m['cases']:
        for seed in m['seeds']:
            spec=complementary.specification(case,seed);ctx=complementary.acquire(spec,configuration(500),CommunicationConfig(byte_budget=4000000,ttl=300.))
            for sender in (0,1):
                receiver=1-sender;s=copy.deepcopy(ctx['agents'][sender]);packet=s.export(ctx['states'][sender],str(receiver));a=ctx['agents'][receiver];x=ctx['states'][receiver]
                before=digest(a);base_measurements=digest([(st.readings,st.transitions) for st in a.stages]);base_x=x.copy()
                if packet is None:
                    rows.append(dict(case=case,seed=seed,sender=sender,error='no_packet'));continue
                for name,packets in fault_variants(packet,a).items():
                    branch=copy.deepcopy(a);t=time.perf_counter();events=[branch.assimilate(p) for p in packets]
                    proposal,unused=branch._field_fit();proposed=[w.tolist() for w in proposal]
                    d=branch.update(x.copy());u,_=branch.action(x.copy())
                    immutable=base_measurements==digest([(st.readings,st.transitions) for st in branch.stages])
                    rows.append(dict(case=case,seed=seed,sender=sender,receiver=receiver,fault=name,
                        events=events,accepted=d['accepted'],reason=d['reason'],version=branch.version,
                        received_count=len(branch.received),remote_summaries=len(branch.remote_terrain),remote_cells=len(branch.remote_cells),
                        received_sources={k:list(v.sources) for k,v in branch.received.items()},
                        weights=[st.weights.tolist() for st in branch.stages],proposed_weights=proposed,
                        control=u.tolist(),measurement_immutable=immutable,physical_state_unchanged=bool(np.array_equal(base_x,x)),
                        operational_unchanged=digest(a)==before,work=d['counterfactual_steps'],
                        message_fit_work=branch.message_fit_work-a.message_fit_work,seconds=time.perf_counter()-t))
                    assert immutable and digest(a)==before and np.array_equal(base_x,x) and d['counterfactual_steps']<=1024
    dump(Path(root)/'audit_rows.json',rows);return rows


def suite(root,suite,workers=8,smoke=False):
    frozen=freeze(root,suite);m=frozen['protocol'];root=Path(root)
    if suite=='faults':fault_suite(root,m);return report(root,suite)
    specs=m['specs'][:1] if smoke else m['specs'];arms=m['arms']
    jobs=[(str(root),suite,s,a,b) for s in specs for a in arms for b in (m['budgets'] if suite=='representation' else [m['byte_budget']])]
    done={}
    old=root/'completion.json'
    if old.exists():done={r['key']:r for r in load(old)}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run_episode,j) for j in jobs]):
            row=f.result();done[row['key']]=row;dump(old,list(done.values()));print({k:row.get(k) for k in ('key','goals','collisions','timeouts','error','wall_seconds')},flush=True)
    return report(root,suite)


def report(root,suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);m=load(root/'manifest.json')['protocol'];figs=root/'figures';figs.mkdir(exist_ok=True)
    if suite=='faults':
        rows=load(root/'audit_rows.json');valid=[r for r in rows if 'fault' in r];summary=[]
        for name in m['faults']:
            rs=[r for r in valid if r['fault']==name]
            summary.append(dict(fault=name,audits=len(rs),eligible=sum(e['reason']=='eligible' for r in rs for e in r['events']),
                                rejected=sum(e['reason']!='eligible' for r in rs for e in r['events']),
                                unchanged=sum(r['operational_unchanged'] and r['measurement_immutable'] and r['physical_state_unchanged'] for r in rs)))
        fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(len(summary));ax.bar(xx,[r['eligible'] for r in summary],label='eligible');ax.bar(xx,[r['rejected'] for r in summary],bottom=[r['eligible'] for r in summary],label='rejected');ax.set_xticks(xx,[r['fault'] for r in summary],rotation=25);ax.set_ylabel('packet validations');ax.legend();fig.tight_layout();fig.savefig(figs/'packet_faults.png');plt.close(fig)
        allsafe=all(r['operational_unchanged'] and r['measurement_immutable'] and r['physical_state_unchanged'] for r in valid)
        findings=f"{len(valid)} snapshot audits; branch/measurement/physical isolation passed: {allsafe}. Bias and correlation are legal payloads; acceptance does not imply accuracy or calibrated uncertainty. These are model/interface audits, not navigation robustness trials."
        dump(root/'summary.json',dict(suite=suite,rows=summary,isolation_passed=allsafe,complete=len(valid)==84))
        lines=['# Packet fault audits','',findings,'','| Fault | Audits | Eligible packets | Rejected packets | Isolation passed |','|---|---:|---:|---:|---:|']+[f"| {r['fault']} | {r['audits']} | {r['eligible']} | {r['rejected']} | {r['unchanged']} |" for r in summary]
    else:
        rows=load(root/'completion.json') if (root/'completion.json').exists() else [];good=[r for r in rows if 'error' not in r]
        failed=[r for r in rows if 'error' in r or r.get('goals')!=2];dump(root/'failure_inventory.json',failed)
        expected=m['expected_episodes'];summary=dict(suite=suite,expected_episodes=expected,completed_episodes=len(good),errors=len(rows)-len(good),goals=sum(r['goals'] for r in good),collisions=sum(r['collisions'] for r in good),timeouts=sum(r['timeouts'] for r in good),complete=len(good)==expected)
        lines=[f'# {suite.capitalize()} campaign','',f"Completed {len(good)}/{expected} episodes; {summary['goals']}/{2*len(good)} goals; {summary['collisions']} collisions; {summary['timeouts']} timeouts. All failures/incomplete runs retained.",'',m['seed_semantics'],'']
        if suite=='representation':
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for arm in m['arms']:
                means=[];bytes_=[]
                for budget in m['budgets']:
                    rs=[r for r in good if r['arm']==arm and r['budget']==budget]
                    means.append(np.mean([r['agents'][1]['time_risk'] for r in rs]) if rs else np.nan);bytes_.append(np.mean([r['bytes'] for r in rs]) if rs else np.nan)
                axes[0].plot(bytes_,means,'o-',label=arm)
            axes[0].set(xlabel='Actual team bytes',ylabel='Mean carrier time + weighted risk');axes[0].legend(fontsize=8)
            for k,budget in enumerate(m['budgets']):
                rs=[r for r in good if r['budget']==budget];axes[1].scatter([r['bytes'] for r in rs],[r['wall_seconds'] for r in rs],label=f'ceiling {budget:,}')
            axes[1].set(xlabel='Actual team bytes',ylabel='Wall seconds');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figs/'budget_curves.png');plt.close(fig)
            lines+=['Same envelope comparisons; costmap is a .5m supported regular lattice with up to 12 scalar points. No claim against an optimized raster codec. Binary threshold .25. Policy replay disabled in every arm.','']
        else:
            paired=[]
            for spec in m['specs']:
                rs={r['arm']:r for r in good if r['seed']==spec['seed']}
                if set(rs)==set(CHANNELS):
                    c={a:r['team_path'] for a,r in rs.items()};interaction=c['one_way_01']+c['one_way_10']-c['disabled']-c['reciprocal_fixed']
                    paired.append(dict(seed=spec['seed'],interaction_path=interaction,reciprocal_saving=c['disabled']-c['reciprocal_fixed'],all_complete=all(r['goals']==2 for r in rs.values())))
            dump(root/'paired_comparisons.json',paired)
            fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar([str(r['seed']) for r in paired],[r['reciprocal_saving'] for r in paired]);axes[0].set_ylabel('Off − reciprocal team path (m)');axes[0].tick_params(axis='x',rotation=35);axes[1].bar([str(r['seed']) for r in paired],[r['interaction_path'] for r in paired]);axes[1].set_ylabel('Interaction in path saving (m)');axes[1].tick_params(axis='x',rotation=35)
            for ax in axes:ax.axhline(0,color='black',lw=.7)
            fig.tight_layout();fig.savefig(figs/'reciprocity_distribution.png');plt.close(fig)
            lines+=['All eight generated layouts included without oracle selection. Unreachable layouts and failed missions are retained; path savings on incomplete missions do not establish navigation benefit. Policy replay disabled.','']
            summary['paired']=paired
        dump(root/'summary.json',summary)
        lines+=['| Scene | Arm | Budget | Goals | Collisions | Timeouts | Bytes | Wall seconds |','|---|---|---:|---:|---:|---:|---:|---:|']+[f"| {r['scene']} | {r['arm']} | {r['budget']} | {r.get('goals','ERROR')} | {r.get('collisions','—')} | {r.get('timeouts','—')} | {r.get('bytes','—')} | {r['wall_seconds']:.2f} |" for r in rows]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return summary if suite!='faults' else findings


def render(root,suite):
    from .crossing_visuals import world_panel
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(root);m=load(root/'manifest.json')['protocol']
    if suite=='faults':return report(root,suite)
    spec=next(s for s in m['specs'] if s['seed']==m['animation_seed'] and (suite!='representation' or s['scene']==m['animation_scene']))
    arms=('binary','field','raw','costmap') if suite=='representation' else CHANNELS
    budget=4000000;episodes=[]
    for arm in arms:
        key=f"{spec['name']}_{spec['seed']}_{arm}_{budget}";p=root/'episodes'/f'{key}.json.gz'
        if not p.exists():return
        episodes.append(load(p))
    end=max(a['finish_time'] for r in episodes for a in r['agents']);frames=[]
    for now in np.linspace(0,end,60):
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,r in zip(axes.flat,episodes):world_panel(ax,r,float(now));ax.set_title(f"{r['arm']} · {sum(a['success'] for a in r['agents'])}/2 final goals")
        fig.suptitle(f"{suite} · {spec['name']} · t={now:.1f}s; logged states",fontsize=12);fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    (root/'figures').mkdir(exist_ok=True);frames[0].save(root/'figures'/'selected_comparison.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['freeze','suite','report','render']);ap.add_argument('--suite',choices=['representation','reciprocity','faults'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--smoke',action='store_true');args=ap.parse_args(argv)
    if args.command=='freeze':freeze(args.out,args.suite)
    elif args.command=='suite':suite(args.out,args.suite,args.workers,args.smoke)
    elif args.command=='report':report(args.out,args.suite)
    else:render(args.out,args.suite)


if __name__=='__main__':main()
