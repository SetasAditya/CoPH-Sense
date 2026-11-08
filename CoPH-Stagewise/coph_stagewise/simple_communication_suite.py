"""Frozen factorial static hazard test of two simple communication repairs."""
import argparse,copy,hashlib,json,platform,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from . import communication_navigation as cn
from .simple_communication import SimpleRepairLink,SimpleOptions
from .hazard_demo import SCENES,specification
from .evaluation import dump,load
from .communication_repair_suite import DEPENDENCIES
from .communication_reconciliation import episode_row

ROOT=Path(__file__).resolve().parents[1]
ARMS={'rotation':SimpleOptions(),'existing_selector':SimpleOptions(),'ack_only':SimpleOptions(True,False),
      'compact_status_only':SimpleOptions(False,True),'both':SimpleOptions(True,True)}
SEEDS=(86100,86101);BUDGETS=(8000,16000,32000,4000000)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);names=tuple(dict.fromkeys(DEPENDENCIES+('simple_communication.py','simple_communication_suite.py','communication_navigation.py','staggered_demo.py','communication_reconciliation.py')))
    manifest=dict(version=1,scenes=list(SCENES),seeds=list(SEEDS),budgets=list(BUDGETS),arms={k:asdict(v) for k,v in ARMS.items()},
        specs=[specification(scene,seed=seed) for scene in SCENES for seed in SEEDS],expected_episodes=168,disabled='once per scene/seed at4MB',
        animation=dict(scene='double_mild',seed=86100,budgets=[8000,4000000],arms=['disabled']+list(ARMS)),
        accounting='corrected declared risk weights; primary time+weighted risk; FULL adds effort+failure',
        design='ack_only changes acknowledgment/novelty interface only; compact_status_only changes lossless cached/delta status coding AND payload-reserving admission/fallback; both combines them',
        admission='compact-status arms reserve each endpoint normal local payload before status charging; unsent ads not charged; no private receiver measurements enter sender selection',
        fixed_cadence=True,adaptive_rate=False,baseline_reuse=False,
        computation='new link handlers include failed admission, preview exports, codecs, scoring and fallback; receiver-local preview/ad work attributed to receiver; baseline handler counters may underlog failed offers, full episode wall time always logged',
        sources={n:sha(ROOT/'coph_stagewise'/n) for n in names},runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),device='cpu',dtype='float64',threads=1))
    manifest=json.loads(json.dumps(manifest));p=out/'manifest.json'
    if p.exists():
        if load(p)!=manifest:raise ValueError('frozen source/config changed')
        return manifest
    with p.open('x') as f:json.dump(manifest,f,indent=2)
    for sub in ('source_archive','episodes','errors','figures'):(out/sub).mkdir()
    for n in names:(out/'source_archive'/n).write_bytes((ROOT/'coph_stagewise'/n).read_bytes())
    return manifest


def job(task):
    spec,name,budget,out=task;torch.set_num_threads(1);root=Path(out);channel='disabled' if name=='disabled' else 'reciprocal_fixed';selector='rotation' if name in ('disabled','rotation') else 'receiver_relevant';key=f"{spec['scene']}_{spec['seed']}_{name}_{budget}";path=root/'episodes'/f'{key}.json.gz'
    if path.exists():
        r=load(path);row=episode_row(r,key,budget);row.update(r['simple_work']);return row
    old_link,old_factory=cn.FixedRateRepairLink,cn.factory;instances=[];agents=[]
    def link_factory(*args,**kwargs):
        link=SimpleRepairLink(*args,simple_options=ARMS.get(name,SimpleOptions()),**kwargs);instances.append(link);return link
    def factory(spec):
        base=old_factory(spec)
        def make(*args,**kwargs):
            a=base(*args,**kwargs);agents.append(a);return a
        return make
    try:
        cn.FixedRateRepairLink=link_factory;cn.factory=factory
        r=cn.execute(spec,channel,selector,byte_budget=budget,accounting_mode='corrected');r['selector']=name
        work=dict(condition=name,export_work=sum(a.export_work for a in agents),message_fit_work=sum(a.message_fit_work for a in agents),
            total_offer_seconds=sum(l.total_offer_seconds for l in instances),
            logged_channel_processing_seconds=sum(a['bytes']['processing_seconds'] for a in r['agents']),
            baseline_failed_offer_processing_unavailable=name in ('rotation','existing_selector','disabled'))
        r['simple_work']=work;dump(path,r);row=episode_row(r,key,budget);row.update(work);return row
    except Exception as exc:
        result=dict(key=key,scene=spec['scene'],seed=spec['seed'],condition=name,budget=budget,error=repr(exc),traceback=traceback.format_exc());dump(root/'errors'/f'{key}.json',result);return result
    finally:cn.FixedRateRepairLink=old_link;cn.factory=old_factory


def suite(out,workers=16):
    m=freeze(out);out=Path(out);tasks=[(s,'disabled',4000000,str(out)) for s in m['specs']]+[(s,a,b,str(out)) for s in m['specs'] for a in ARMS for b in BUDGETS];rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):
            row=f.result();rows.append(row);dump(out/'rows.json',rows);print(len(rows),'/168',row['key'],row.get('goals'),row.get('error'),flush=True)
    return report(out)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(out);rows=load(root/'rows.json');good=[r for r in rows if 'error' not in r];failures=[r for r in rows if r.get('goals')!=2];paired=[]
    for scene in SCENES:
        for seed in SEEDS:
            for budget in BUDGETS:
                group={r['selector']:r for r in good if r['scene']==scene and r['seed']==seed and r['budget']==budget}
                for name in ARMS:
                    for reference in ('rotation','existing_selector'):
                        if name in group and reference in group:
                            a,b=group[name],group[reference];paired.append(dict(scene=scene,seed=seed,budget=budget,arm=name,reference=reference,
                                primary_difference=a['carrier_time_risk']-b['carrier_time_risk'],full_difference=a['carrier_cost']-b['carrier_cost'],
                                team_bytes_difference=sum(a['bytes_team'].values())-sum(b['bytes_team'].values()),wall_difference=a['wall_seconds']-b['wall_seconds']))
    metrics=[]
    for budget in BUDGETS:
        for name in ARMS:
            rr=[r for r in good if r['selector']==name and r['budget']==budget]
            if not rr:continue
            metrics.append(dict(arm=name,budget=budget,episodes=len(rr),goals=sum(r['goals'] for r in rr),primary=float(np.mean([r['carrier_time_risk'] for r in rr])),full=float(np.mean([r['carrier_cost'] for r in rr])),
                bytes_team={k:float(np.mean([r['bytes_team'][k] for r in rr])) for k in ('status','payload','acknowledgment')},
                wall_seconds=float(np.mean([r['wall_seconds'] for r in rr])),export_work=sum(r['export_work'] for r in rr)))
    dump(root/'paired_rows.json',paired);dump(root/'failure_inventory.json',failures)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ARMS:
        rs=[r for r in metrics if r['arm']==name];axes[0].plot([r['budget'] for r in rs],[r['primary'] for r in rs],'o-',label=name);axes[1].plot([r['budget'] for r in rs],[sum(r['bytes_team'].values()) for r in rs],'o-',label=name)
    axes[0].set(xscale='log',xlabel='Per-agent ceiling',ylabel='Mean carrier time + weighted risk');axes[1].set(xscale='log',yscale='log',xlabel='Per-agent ceiling',ylabel='Actual mean team bytes')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'figures/cost_bytes.png');plt.close(fig)
    verdict=dict(component='communication',complete=len(rows)==168 and len(good)==168,mechanism_correct=not any('error' in r for r in rows),
        performance_demonstrated=any(p['arm']=='both' and p['reference']=='rotation' and p['primary_difference']< -1e-8 for p in paired),
        metrics=metrics,goals=sum(r['goals'] for r in good),collisions=sum(r['collisions'] for r in good),timeouts=sum(r['timeouts'] for r in good),
        limitations=['development: four designed scenes and two sensor-noise seeds','status coding and reserve admission are bundled in one overhead option','no adaptive rates','baseline failed-offer processing counters underlog; full wall time preserved','positive case differences do not establish generalization'],
        artifacts=['REPORT.md','paired_rows.json','failure_inventory.json','figures/cost_bytes.png','manifest.json'])
    dump(root/'VERDICT.json',verdict)
    lines=['# Simple communication fixes','',f"Episodes {len(good)}/168; agent goals {verdict['goals']}/{2*len(good)}; collisions {verdict['collisions']}; timeouts {verdict['timeouts']}.",
        '', 'All five arms share fixed cadence, packet schema, legacy receiver fitting, evidence-only learning, routing and safety. Rotation and existing selector are rerun. Ack-only changes route-used acknowledgment and actual-anchor novelty only. Compact-status-only bundles lossless full/delta advertisements with payload-reserving admission and normal-rotation fallback. Both combines those independent options.',
        '', '| Budget / agent | Arm | Episodes | Mean carrier primary | Mean carrier FULL | Team status | Team payload | Team ack | Mean wall seconds |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['budget']:,} | {r['arm']} | {r['episodes']} | {r['primary']:.4f} | {r['full']:.4f} | {r['bytes_team']['status']:.1f} | {r['bytes_team']['payload']:.1f} | {r['bytes_team']['acknowledgment']:.1f} | {r['wall_seconds']:.3f} |" for r in metrics]
    lines+=['', 'Negative paired differences favor the candidate. Compare each candidate against rotation and existing-selector separately in paired_rows.json; winning against the existing broken/expensive selector is not automatically winning against rotation. Noise seeds and repeated budgets are not independent terrain samples.',
        '', 'New handler timers include failed admissions, local preview exports, codec/scoring and fallback work. Receiver-local reserve export and advertisement coding are assigned to receiver processing. Baseline counters may omit unsuccessful offers; complete episode wall time and export/fit counts remain available. No GPU is used.',
        '', '[Cost and byte curves](figures/cost_bytes.png). Full episode records, failures, immutable sources/configuration and endpoint-separated paired rows are retained.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return verdict


def render(out):
    import matplotlib.pyplot as plt
    from PIL import Image
    from .crossing_visuals import background
    root=Path(out);inventory=[]
    for budget in (8000,4000000):
        rr=[]
        for name in ('disabled',)+tuple(ARMS):
            b=4000000 if name=='disabled' else budget;p=root/'episodes'/f'double_mild_86100_{name}_{b}.json.gz'
            if not p.exists():continue
            rr.append((name,load(p)))
        if len(rr)!=6:continue
        end=max(a['finish_time'] for _,r in rr for a in r['agents']);frames=[];metadata=[]
        for now in np.linspace(0,end,30):
            fig,axes=plt.subplots(2,3,figsize=(12,7),dpi=80);states=[]
            for ax,(name,r) in zip(axes.flat,rr):
                background(ax,r['spec']);ax.set_title(name)
                for i in (0,1):
                    indices=[k for k,s in enumerate(r['samples'][i]) if s['time']<=now+1e-9]
                    if indices:
                        k=indices[-1];p=np.asarray([s['x'][:2] for s in r['samples'][i][:k+1]]);ax.plot(*p.T,color=('#0072B2','#D55E00')[i]);ax.scatter(*p[-1],color=('#0072B2','#D55E00')[i],s=20);states.append(dict(arm=name,agent=i,index=k,time=r['samples'][i][k]['time']))
                ax.text(.02,.02,f"FINAL primary {r['agents'][1]['time_risk_cost']:.2f}",transform=ax.transAxes,color='white',fontsize=8)
            fig.suptitle(f'Double hazard · seed86100 · {budget:,}/agent · t={now:.1f}s · fixed communication cadence');fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig);metadata.append(dict(time=float(now),states=states))
        p=root/'figures'/f'double_mild_86100_{budget}.gif';frames[0].save(p,save_all=True,append_images=frames[1:],duration=150,loop=0)
        dump(root/'figures'/f'{p.stem}_frames.json',metadata);inventory.append(dict(budget=budget,path=str(p.relative_to(root)),frames=30,timestamps_verified=all(s['time']<=f['time']+1e-9 for f in metadata for s in f['states'])))
    dump(root/'animation_inventory.json',inventory);return inventory


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','suite','report','render']);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    if a.command=='freeze':freeze(a.out)
    elif a.command=='suite':suite(a.out,a.workers)
    elif a.command=='report':report(a.out)
    else:render(a.out)
