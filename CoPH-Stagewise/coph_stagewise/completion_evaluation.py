"""Completion-first frozen evaluation; held-out runs require the complete development gate."""
import argparse,hashlib,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .complex_environments import SCENES,scene,world_axes,paths,COLORS
from .cooperative import CommunicationConfig
from .cooperative_evaluation import execute
from .navigation_evaluation import metrics
from .evaluation import dump,load
from .records import Config
from .cli import source_hashes,archive_sources

CELLS=[(m,a) for m in ('protected','routed') for a in ('disabled','reciprocal_fixed')]

def job(args):
    spec,mode,arm,root,comm=args;torch.set_num_threads(1)
    path=Path(root)/f"{spec['name']}_{spec['seed']}_{mode}_{arm}.json.gz"
    item=dict(scene=spec['name'],seed=spec['seed'],mode=mode,arm=arm,path=path.name,complete=False)
    try:
        r=execute(spec,arm,CommunicationConfig(**comm),audits=False,replay_domain='retained',horizon_mode='supported_boundary',replay_cap=1024,controller_mode=mode,shaping_mode='progress')
        r['comparison_arm']=arm;r['controller_mode']=mode;row=metrics(r,mode,arm)
        row['stagnation_replans']=sum(t.get('navigation',{}).get('stagnation',False) for seq in r['trajectories'] for t in seq)
        row['shaping_mode']='progress';dump(path,r);item.update(complete=True,metrics=row)
    except Exception as exc:item.update(error=str(exc),traceback=traceback.format_exc());dump(path,item)
    return item

def gate(completed,expected):
    groups=[]
    for mode,arm in CELLS:
        cs=[c for c in completed if (c['mode'],c['arm'])==(mode,arm)]
        rs=[c['metrics'] for c in cs if c['complete']]
        goals=sum(r['goals'] for r in rs);collisions=sum(r['collisions'] for r in rs)
        groups.append(dict(mode=mode,arm=arm,episodes=len(rs),required_episodes=expected,goals=goals,required_goals=expected*2,collisions=collisions,passed=len(rs)==expected and goals==expected*2 and collisions==0))
    return dict(passed=all(g['passed'] for g in groups),groups=groups)

def frame(results,tick):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    fig,axes=plt.subplots(2,2,figsize=(12,7));spec=next(iter(results.values()))['spec']
    for ax,(mode,arm) in zip(axes.flat,CELLS):
        r=results[(mode,arm)];world_axes(ax,spec);paths(ax,r,tick)
        for i,seq in enumerate(r['trajectories']):
            ts=[t for t in seq if t['tick']<=tick]
            if not ts:continue
            t=ts[-1];n=t.get('navigation',{});q=np.array(t['x'][:2]);c=COLORS[i]
            if n.get('route'):
                p=np.array(n['route']);ax.plot(p[:,0],p[:,1],':',color=c);ax.scatter(*n['waypoint'],marker='x',color=c,s=15)
            for v,color in ((n.get('nominal',[0,0]),'orange'),(t['u'],c)):
                ax.arrow(*q,*(np.array(v)*.18),width=.015,color=color,length_includes_head=True)
            updates=[u for u in r['updates'][i] if u['tick']<=tick]
            if updates:
                snap=updates[-1]['post']
                for ox,oy,rad in snap.get('geometry',[]):ax.add_patch(Circle((ox,oy),rad+.17,fill=False,color=c,alpha=.3))
                for st in snap['stages']:
                    for cf in st['counterfactuals']:
                        p=np.array(cf['states']);ax.plot(p[:,0],p[:,1],'--',color=c,alpha=.5)
        for p in r['packets']:
            if 0<=tick*.12-p['timestamp']<.96:
                z=np.array(p['anchors']);ax.scatter(z[:,0],z[:,1],marker='+',color='purple',s=20)
        ax.set_title(f'{mode} progress | {arm}',fontsize=10)
    end=max(t['tick'] for r in results.values() for seq in r['trajectories'] for t in seq)
    fig.suptitle(f'{spec["name"]} | seed {spec["seed"]} | t={min(tick+1,end+1)*.12:.2f}s\nOrange nominal / blue-green applied; dotted route, dashed replay; inflated sensed disks; purple packets.',fontsize=10)
    fig.tight_layout();fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return im

def report(root,legacy,animate=False):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    completed=load(root/'completion.json');rows=[c['metrics'] for c in completed if c['complete']];dump(root/'rows.json',rows)
    figdir=root/'figures';gallery=figdir/'failures';gallery.mkdir(parents=True,exist_ok=True);failures=[];selected={}
    for c in completed:
        if not c['complete']:failures.append(c);continue
        if c['seed']==61000 or c['metrics']['goals']<2:
            r=load(root/'episodes'/c['path'])
            if c['seed']==61000:selected.setdefault(c['scene'],{})[(c['mode'],c['arm'])]=r
            if c['metrics']['goals']<2:
                name=c['path'].replace('.json.gz','.png');failures.append(dict(c,figure=name));fig,ax=plt.subplots(figsize=(10,3));world_axes(ax,r['spec']);paths(ax,r,10**9)
                ax.set_title(f"{c['scene']} {c['seed']} {c['mode']} {c['arm']}: goals {c['metrics']['goals']}/2, collisions {c['metrics']['collisions']}, timeouts {c['metrics']['timeouts']}",fontsize=9)
                fig.tight_layout();fig.savefig(gallery/name,dpi=95);plt.close(fig)
    dump(root/'failures.json',failures)
    (gallery/'index.html').write_text('<h1>All failures and incomplete episodes</h1>'+''.join(f'<p>{c["path"]}<br>'+ (f'<img width="850" src="{c["figure"]}">' if 'figure' in c else c.get('error','incomplete'))+'</p>' for c in failures))
    reference=load(legacy/'rows.json') if (legacy/'rows.json').exists() else []
    paired=[]
    for r in rows:
        old=next((b for b in reference if (b['scene'],b['seed'],b['mode'],b['arm'])==(r['scene'],r['seed'],r['mode'],r['arm'])),None)
        off=next((b for b in rows if (b['scene'],b['seed'],b['mode'],b['arm'])==(r['scene'],r['seed'],r['mode'],'disabled')),None)
        paired.append(dict(scene=r['scene'],seed=r['seed'],mode=r['mode'],arm=r['arm'],goals=r['goals'],legacy_goals=old['goals'] if old else None,communication_goal_difference=r['goals']-off['goals'] if off else None,secondary_cost_difference=r['team_cost']-off['team_cost'] if off else None))
    dump(root/'paired.json',paired)
    lines=['# Completion-first navigation results','',f'Completed {len(rows)}/{len(completed)} episodes. All failures and incomplete runs are preserved. Within-mission adaptation only; static independent agents, unchanged sensing, horizons, goal threshold, costs and fixed-rate communication.', '', '| Controller | Communication | Goals / agents | Collisions | Timeouts | Stagnation replans | Max flow work | Mean bytes | Mean seconds |','|---|---|---|---|---|---|---|---|---|']
    for mode,arm in CELLS:
        rs=[r for r in rows if (r['mode'],r['arm'])==(mode,arm)]
        if not rs:continue
        lines.append(f'| {mode} | {arm} | {sum(r["goals"] for r in rs)}/{2*len(rs)} | {sum(r["collisions"] for r in rs)} | {sum(r["timeouts"] for r in rs)} | {sum(r["stagnation_replans"] for r in rs)} | {max(r["max_update_work"] for r in rs)} | {np.mean([r["bytes"] for r in rs]):.0f} | {np.mean([r["wall_seconds"] for r in rs]):.2f} |')
    lines+=['','Completion is the external readiness gate; learning still uses its original cost objective. The radial gradient bound applies to the nominal potential at a fixed target, not to safety-projected control or unknown environments. Dispersion, safety stops and lower cost are not success criteria.', '', 'Rows include goal gaps, stage traversal, interventions, infeasibility, near-stationary time, energy residuals, computation and secondary costs. paired.json separates common-controller improvements from within-controller communication comparisons. No adaptive scheduling or predictor training is performed.', '', '[All failed episodes](figures/failures/index.html)']
    for name,rs in selected.items():
        if len(rs)!=4:continue
        frame(rs,10**9).save(figdir/f'{name}_paired.png')
        if animate:
            end=max(t['tick'] for r in rs.values() for seq in r['trajectories'] for t in seq)
            frames=[frame(rs,int(t)) for t in sorted(set(np.linspace(0,end,32).astype(int)))];frames[0].save(figdir/f'{name}.gif',save_all=True,append_images=frames[1:],duration=max(100,int((end+1)*120/len(frames))),loop=0)
            lines+=['',f'![{name}](figures/{name}.gif)']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for i,(mode,arm) in enumerate(CELLS):
        rs=[r for r in rows if (r['mode'],r['arm'])==(mode,arm)]
        if not rs:continue
        goals=sum(r['goals'] for r in rs);hits=sum(r['collisions'] for r in rs);timeouts=sum(r['timeouts'] for r in rs)
        axes[0].bar(i,goals,color='green');axes[0].bar(i,hits,bottom=goals,color='red');axes[0].bar(i,timeouts,bottom=goals+hits,color='gray')
        axes[1].scatter([i]*len(rs),[max(r['goal_gaps']) for r in rs],s=10);axes[2].scatter([i]*len(rs),[r['wall_seconds'] for r in rs],s=10)
    for ax,title in zip(axes,['Goals / collisions / timeouts','Maximum terminal goal gap','Computation seconds']):ax.set_title(title);ax.set_xticks(range(4),[m+'\n'+a for m,a in CELLS],rotation=30,fontsize=7)
    fig.tight_layout();fig.savefig(figdir/'summary.png',dpi=120);plt.close(fig);lines+=['','![Summary](figures/summary.png)'];(root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reference',type=Path,default=Path('results/complex_fixed_v1'));ap.add_argument('--legacy',type=Path,default=Path('results/navigation_fix_v3'));ap.add_argument('--workers',type=int,default=32);ap.add_argument('--resume',action='store_true');ap.add_argument('--render-only',action='store_true');args=ap.parse_args(argv)
    hashes=source_hashes()
    if not args.out.exists():
        old=load(args.reference/'manifest.json');args.out.mkdir(parents=True)
        specs=dict(old['specs']);specs['validation']=[scene(n,s) for n in SCENES for s in range(110000,110008)]
        manifest=dict(specs=specs,cells=CELLS,communication=old['communication'],source_hashes=hashes,reference=str(args.reference.resolve()),legacy=str(args.legacy.resolve()),reference_sha256=hashlib.sha256((args.reference/'manifest.json').read_bytes()).hexdigest(),animation_seed=61000,validation_seeds=list(range(110000,110008)),reserved_final_test=old['reserved_test'],controllers={m:asdict(Config(controller_mode=m,shaping_mode='progress',replay_domain='retained',replay_objective='mission',replay_horizon_mode='supported_boundary',replay_update_cap=1024)) for m in ('protected','routed')})
        dump(args.out/'manifest.json',manifest);archive_sources(args.out)
    elif not(args.resume or args.render_only):ap.error('fresh output directory required')
    manifest=load(args.out/'manifest.json')
    if not args.render_only and hashes!=manifest['source_hashes']:ap.error('source changed after freeze')
    started=time.perf_counter()
    if not args.render_only and not (args.out/'controlled_checks.json').exists():
        from .completion_checks import controlled_checks
        controlled_checks(args.out/'controlled_checks.json')
    for profile in ('smoke','development','validation'):
        if profile=='validation':
            g=gate(load(args.out/'development'/'completion.json'),32);dump(args.out/'development_gate.json',g)
            if not g['passed']:
                dump(args.out/'validation_status.json',dict(executed=False,reason='development_gate_failed',seeds=manifest['validation_seeds']));break
        root=args.out/profile;(root/'episodes').mkdir(parents=True,exist_ok=True);completed=load(root/'completion.json') if (root/'completion.json').exists() else []
        if not args.render_only:
            done={(c['scene'],c['seed'],c['mode'],c['arm']) for c in completed};cells=CELLS if profile!='validation' else [(m,a) for m,a in CELLS if m=='routed']
            jobs=[(s,m,a,str(root/'episodes'),manifest['communication']) for s in manifest['specs'][profile] for m,a in cells if (s['name'],s['seed'],m,a) not in done]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(job,j) for j in jobs]
                for f in as_completed(futures):
                    c=f.result();completed.append(c);dump(root/'completion.json',completed);print(profile,len(completed),c['scene'],c['seed'],c['mode'],c['arm'],c.get('metrics',{}).get('goals'),c.get('error',''),flush=True)
        report(root,Path(manifest['legacy'])/profile,animate=profile=='development')
        if any(not c['complete'] for c in completed):raise SystemExit('incomplete runs preserved; resolve before progression')
        if profile=='validation':dump(args.out/'validation_status.json',dict(executed=True,passed=len(completed)==64 and all(c['metrics']['goals']==2 and c['metrics']['collisions']==0 for c in completed)))
    dump(args.out/'completion.json',dict(seconds_this_invocation=time.perf_counter()-started,complete=True))
    (args.out/'REPORT.md').write_text('# Completion-first cooperative navigation\n\n[Development results and synchronized GIFs](development/REPORT.md)\n\n[Smoke](smoke/REPORT.md)\n\nRead development_gate.json and validation_status.json for readiness. Failed development blocks held-out execution and any working-version claim.\n')

if __name__=='__main__':main()
