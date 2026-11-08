"""Frozen four-controller comparison; original complex scenes/costs are immutable."""
import argparse,hashlib,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .complex_environments import ARMS,SCENES,world_axes,paths,COLORS
from .cooperative import CommunicationConfig
from .cooperative_evaluation import execute
from .evaluation import dump,load
from .records import Config
from .cli import source_hashes,archive_sources

MODES=('legacy','geometry','protected','routed')

def cells(profile):
    if profile=='smoke':return [(m,a) for m in MODES for a in ('disabled','reciprocal_fixed')]
    return [('routed',a) for a in ARMS]+[(m,a) for m in ('geometry','protected') for a in ('disabled','reciprocal_fixed')]

def metrics(r,mode,label):
    ts=[t for seq in r['trajectories'] for t in seq];us=[u for seq in r['updates'] for u in seq]
    live=[t.get('navigation',{}) for t in ts]
    replay=[m for u in us for c in u['replay_calls'] for m in c.get('controls',[])]
    pilots=[m for u in us for p in u['diagnostic'].get('replay_plan',[]) for m in p.get('pilot_controls',[])]
    diagnostic=[m for u in us for m in u['diagnostic'].get('diagnostic_controls',[])]
    all_controls=live+replay+pilots+diagnostic
    for u in us:
        d=u['diagnostic'];work=d.get('counterfactual_steps',0)
        assert work<=1024,('budget',work)
        assert work==d.get('replay_pilot_steps',0)+sum(c['steps'] for c in u['replay_calls'])+d.get('diagnostic_guard_steps',0),('unaccounted_work',d)
    components={k:sum(a['components'][k] for a in r['agents']) for k in ('time','risk','effort','failure')}
    assert abs(sum(components.values())-r['summary']['team_cost'])<1e-8
    if mode in ('protected','routed'):
        assert all(np.linalg.norm(t['u'])<=r['config']['force_limit']+1e-8 for t in ts)
    return dict(r['summary'],mode=mode,arm=label,components=components,
        collisions=sum(a['collision'] for a in r['agents']),timeouts=sum(a['timeout'] for a in r['agents']),
        goals=sum(a['success'] for a in r['agents']),goal_gaps=[a['goal_gap'] for a in r['agents']],
        max_stage=max(t['stage'] for t in ts),stage_traversals=[len(set(t['stage'] for t in seq))-1 for seq in r['trajectories']],
        interventions=sum(m.get('reason','nominal')!='nominal' for m in live),
        infeasible=sum(m.get('infeasible',False) for m in live),brakes=sum(m.get('reason') in ('brake','geometry_overflow') for m in live),
        route_changes=sum(m.get('route_changed',False) for m in live),
        route_expansions=sum(m.get('route_expansions',0) for m in all_controls),projection_candidates=sum(m.get('projection_candidates',0) for m in all_controls),
        guard_flows=sum(m.get('guard_steps',0) for m in all_controls),execution_guard_flows=sum(m.get('guard_steps',0) for m in live),
        max_update_work=max((u['diagnostic'].get('counterfactual_steps',0) for u in us),default=0),
        budget_exhaustions=sum(u['diagnostic'].get('replay_budget_exhausted',False) for u in us),
        replay_handoffs=sum(len(c.get('stage_events',[])) for u in us for c in u['replay_calls']),
        safety_work=sum(t.get('safety_work',0) for t in ts),route_shaping_jumps=sum(m.get('route_shaping_jump',0) for m in live),
        max_energy_residual=max(a['max_energy_residual'] for a in r['agents']),
        identification_work=sum(a['identification_work'] for a in r['agents']),
        path_lengths=[a['path_length'] for a in r['agents']],near_stationary_seconds=sum(a['near_stationary_seconds'] for a in r['agents']))

def job(args):
    spec,mode,label,root,comm=args;torch.set_num_threads(1)
    filename=f"{spec['name']}_{spec['seed']}_{mode}_{label}.json.gz";path=Path(root)/filename
    item=dict(scene=spec['name'],seed=spec['seed'],mode=mode,arm=label,path=filename,complete=False)
    try:
        arm,horizon=ARMS[label]
        r=execute(spec,arm,CommunicationConfig(**comm),audits=False,replay_domain='retained',horizon_mode=horizon,replay_cap=1024,controller_mode=mode)
        r['comparison_arm']=label;r['controller_mode']=mode;row=metrics(r,mode,label)
        dump(path,r);item.update(complete=True,metrics=row)
    except Exception as exc:
        item.update(error=str(exc),traceback=traceback.format_exc());dump(path,item)
    return item

def frame(rs,tick):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    from PIL import Image
    tick=min(tick,max(t['tick'] for r in rs.values() for seq in r['trajectories'] for t in seq))
    fig,axes=plt.subplots(2,2,figsize=(12,7));spec=rs['routed_reciprocal_fixed']['spec']
    for ax,key in zip(axes.flat,('legacy_disabled','routed_disabled','routed_reciprocal_fixed')):
        r=rs[key];world_axes(ax,spec);paths(ax,r,tick)
        for i,seq in enumerate(r['trajectories']):
            visible=[t for t in seq if t['tick']<=tick]
            if not visible:continue
            t=visible[-1];n=t.get('navigation',{});q=np.array(t['x'][:2]);c=COLORS[i]
            if n.get('route'):
                p=np.array(n['route']);ax.plot(p[:,0],p[:,1],':',color=c,lw=1);ax.scatter(*n['waypoint'],marker='x',color=c,s=18)
            if n.get('nominal') is not None:
                for v,color in ((n['nominal'],'orange'),(t['u'],c)):
                    v=np.array(v)*.18;ax.arrow(*q,*v,width=.012,color=color,alpha=.8,length_includes_head=True)
        ax.set_title(key.replace('_',' '),fontsize=10)
    ax=axes[1,1];r=rs['routed_reciprocal_fixed'];lo,hi,bot,top=spec['bounds']
    for i,c in enumerate(COLORS):
        updates=[u for u in r['updates'][i] if u['tick']<=tick]
        if not updates:continue
        u=updates[-1];snap=u['post']
        for ox,oy,rad in snap.get('geometry',[]):ax.add_patch(Circle((ox,oy),rad+.17,fill=False,color=c,alpha=.6))
        for st in snap['stages']:
            b=st['bounds'];ax.add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],fill=False,color=c,alpha=.15))
            for fx,fy,rad,_ in st.get('footprints',[])[::12]:ax.add_patch(Circle((fx,fy),rad,fill=False,color=c,alpha=.08))
            for cf in st['counterfactuals']:
                p=np.array(cf['states']);ax.plot(p[:,0],p[:,1],'--',color=c,alpha=.7)
        ax.text(.02,.98-i*.09,f'Agent {i}: model {snap["version"]}, stage {snap["stages"][-1]["id"]}, snapshot {u["tick"]*.12:.2f}s',transform=ax.transAxes,color=c,fontsize=7,va='top')
    paths(ax,r,tick)
    for packet in r['packets']:
        if 0<=tick*.12-packet['timestamp']<.96:
            p=np.array(packet['anchors']);ax.scatter(p[:,0],p[:,1],marker='+',s=20,color='purple')
    ax.set(xlim=(lo,hi),ylim=(bot,top),aspect='equal',title='Sensed inflated geometry; retained replay dashed')
    fig.suptitle(f'{spec["name"]} | seed {spec["seed"]} | t={(tick+1)*.12:.2f}s\nOrange: nominal; blue/green: executed. Dotted routes; purple: exchanged anchors. Terminal outcomes retained.',fontsize=10)
    fig.tight_layout();fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return im

def report(root,legacy_root,animate=False):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    completed=load(root/'completion.json');rows=[c['metrics'] for c in completed if c['complete']]
    dump(root/'rows.json',rows);figdir=root/'figures';figdir.mkdir(exist_ok=True);gallery=figdir/'failures';gallery.mkdir(exist_ok=True)
    selected={};failures=[]
    for c in completed:
        if not c['complete']:continue
        if c['seed']==61000 or c['metrics']['failures']:
            r=load(root/'episodes'/c['path'])
            if c['seed']==61000:selected.setdefault(c['scene'],{})[c['mode']+'_'+c['arm']]=r
            if c['metrics']['failures']:
                name=c['path'].replace('.json.gz','.png');failures.append(dict(c,figure=name))
                fig,ax=plt.subplots(figsize=(10,3));world_axes(ax,r['spec']);paths(ax,r,10**9)
                ax.set_title(f"{c['scene']} {c['seed']} {c['mode']} {c['arm']} | goals {c['metrics']['goals']}/2; collisions {c['metrics']['collisions']}; timeouts {c['metrics']['timeouts']}",fontsize=9)
                fig.tight_layout();fig.savefig(gallery/name,dpi=95);plt.close(fig)
    dump(root/'failures.json',failures)
    (gallery/'index.html').write_text('<h1>Every failed episode (including safe timeouts)</h1>'+''.join(f'<p>{c["figure"]}<br><img width="850" src="{c["figure"]}"></p>' for c in failures))
    paired=[];lookup={(r['scene'],r['seed'],r['mode'],r['arm']):r for r in rows}
    for r in rows:
        base=lookup.get((r['scene'],r['seed'],r['mode'],'disabled'))
        if base:paired.append(dict(scene=r['scene'],seed=r['seed'],mode=r['mode'],arm=r['arm'],cost_difference=r['team_cost']-base['team_cost'],goal_difference=r['goals']-base['goals']))
    dump(root/'paired.json',paired)
    lines=['# Navigation repair: frozen complex-scene evaluation','',
        'Within-mission development evidence; static independent agents, fixed-rate environmental communication. Original scenes, horizons, penalties and byte ceilings are unchanged. Safe stops and timeouts are failures. Legacy results are preserved as separate references.',
        f'Completed {len(rows)}/{len(completed)} runs. Incomplete records remain in completion.json.',
        '\n| Scene | Controller | Arm | Runs | Goals / agents | Collisions | Timeouts | Mean mission cost | Mean bytes | Max update work |',
        '|---|---|---|---|---|---|---|---|---|---|']
    groups=sorted({(r['scene'],r['mode'],r['arm']) for r in rows})
    for scene,mode,arm in groups:
        rs=[r for r in rows if (r['scene'],r['mode'],r['arm'])==(scene,mode,arm)]
        lines.append(f'| {scene} | {mode} | {arm} | {len(rs)} | {sum(r["goals"] for r in rs)}/{2*len(rs)} | {sum(r["collisions"] for r in rs)} | {sum(r["timeouts"] for r in rs)} | {np.mean([r["team_cost"] for r in rs]):.3f} | {np.mean([r["bytes"] for r in rs]):.0f} | {max(r["max_update_work"] for r in rs)} |')
    lines+=['','Communication benefits must be assessed within the same controller (paired.json). Navigation gains relative to legacy are a common-controller intervention, not evidence of cooperation. Safety projection is not a guarantee against unseen obstacles or out-of-range dynamics. Risk-shaped potentials can still stall along feasible routes.','',
        'The 1,024-flow update ceiling includes pilots, guard flows, gradient evaluation, candidate validation, and diagnostic controls. Routing expansions and projection candidates are separate work counters; actual computation differs across arms. Route and stage energy jumps are separate from actuator/clipping/safety work.','',
        '[Every failed episode](figures/failures/index.html). Machine-readable rows include interventions, infeasibility, stage traversal, route changes, safety work, bytes, computation and terminal outcomes.']
    for scene,rs in selected.items():
        legacy_path=legacy_root/'episodes'/f'{scene}_61000_disabled.json.gz'
        if 'legacy_disabled' not in rs and legacy_path.exists():rs['legacy_disabled']=load(legacy_path)
        keys=('legacy_disabled','routed_disabled','routed_reciprocal_fixed')
        if all(k in rs for k in keys):
            im=frame(rs,10**9);im.save(figdir/f'{scene}_paired.png')
            if animate:
                end=max(t['tick'] for k in keys for seq in rs[k]['trajectories'] for t in seq)
                frames=[frame(rs,int(t)) for t in sorted(set(np.linspace(0,end,36).astype(int)))]
                frames[0].save(figdir/f'{scene}.gif',save_all=True,append_images=frames[1:],duration=max(100,int((end+1)*120/len(frames))),loop=0)
                lines+=['',f'![{scene}](figures/{scene}.gif)']
    labels=sorted({(r['mode'],r['arm']) for r in rows});fig,axes=plt.subplots(1,3,figsize=(16,5))
    for i,(mode,arm) in enumerate(labels):
        rs=[r for r in rows if (r['mode'],r['arm'])==(mode,arm)]
        axes[0].scatter([i]*len(rs),[r['team_cost'] for r in rs],s=9)
        axes[1].bar(i,sum(r['goals'] for r in rs),color='green');axes[1].bar(i,sum(r['collisions'] for r in rs),bottom=sum(r['goals'] for r in rs),color='red');axes[1].bar(i,sum(r['timeouts'] for r in rs),bottom=sum(r['goals']+r['collisions'] for r in rs),color='gray')
        axes[2].scatter([i]*len(rs),[r['wall_seconds'] for r in rs],s=9)
    for ax in axes:ax.set_xticks(range(len(labels)),[m+'\n'+a for m,a in labels],rotation=70,fontsize=7)
    for ax,title in zip(axes,('Mission cost','Agent goals (green), collisions (red), timeouts (gray)','Episode computation (seconds)')):ax.set_title(title,fontsize=9)
    fig.tight_layout();fig.savefig(figdir/'summary.png',dpi=130);plt.close(fig)
    lines+=['','![Costs, outcomes and computation](figures/summary.png)'];(root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reference',type=Path,default=Path('results/complex_fixed_v1'));ap.add_argument('--workers',type=int,default=16);ap.add_argument('--resume',action='store_true');ap.add_argument('--render-only',action='store_true');args=ap.parse_args(argv)
    source=source_hashes()
    if not args.out.exists():
        args.out.mkdir(parents=True);old=load(args.reference/'manifest.json')
        manifest=dict(specs=old['specs'],communication=old['communication'],cells={p:cells(p) for p in ('smoke','development')},
            source_hashes=source,reference=str(args.reference.resolve()),reference_manifest_sha256=hashlib.sha256((args.reference/'manifest.json').read_bytes()).hexdigest(),
            animation_seed=61000,controller_configs={m:asdict(Config(controller_mode=m,replay_domain='retained',replay_objective='mission',replay_horizon_mode='supported_boundary',replay_update_cap=1024)) for m in MODES},
            reserved_validation=old['reserved_validation'],reserved_test=old['reserved_test'])
        dump(args.out/'manifest.json',manifest);archive_sources(args.out)
    elif not (args.resume or args.render_only):ap.error('use a fresh output directory or --resume')
    manifest=load(args.out/'manifest.json')
    if not args.render_only and source!=manifest['source_hashes']:ap.error('source changed since manifest freeze; use a fresh output directory')
    started=time.perf_counter()
    for profile in ('smoke','development'):
        root=args.out/profile;(root/'episodes').mkdir(parents=True,exist_ok=True)
        complete=load(root/'completion.json') if (root/'completion.json').exists() else []
        if not args.render_only:
            done={(c['scene'],c['seed'],c['mode'],c['arm']) for c in complete}
            jobs=[(s,m,a,str(root/'episodes'),manifest['communication']) for s in manifest['specs'][profile] for m,a in manifest['cells'][profile] if (s['name'],s['seed'],m,a) not in done]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(job,j) for j in jobs]
                for f in as_completed(futures):
                    c=f.result();complete.append(c);dump(root/'completion.json',complete)
                    print(profile,len(complete),c['scene'],c['seed'],c['mode'],c['arm'],c['complete'],c.get('error',''),flush=True)
        report(root,Path(manifest['reference'])/profile,animate=profile=='development')
        if any(not c['complete'] for c in complete):raise SystemExit('Incomplete runs preserved; resolve before progressing')
    dump(args.out/'completion.json',dict(seconds_this_invocation=time.perf_counter()-started,complete=True))
    (args.out/'REPORT.md').write_text('# Navigation fix evaluation\n\n[Development results and GIFs](development/REPORT.md)\n\n[Smoke](smoke/REPORT.md)\n')

if __name__=='__main__':main()
