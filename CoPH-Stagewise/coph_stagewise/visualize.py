"""Offline artifacts only. Truth panels never enter learner state."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .evaluation import load,ARMS,dump
from .scenarios import ScenarioWorld
from .records import Config


def episodes(root):
    return sorted((Path(root)/'episodes').glob('*.json.gz'))


def common_probe_error(a,b):
    if a['probe_points']!=b['probe_points']:raise ValueError('probe grids differ')
    mask=np.array(a['final_probes']['support']) & np.array(b['final_probes']['support'])
    def mean(d,key):
        v=[v for v,ok in zip(d['final_probes'][key],mask) if ok and v is not None]
        return float(np.mean(v)) if v else None
    return dict(count=int(mask.sum()),left_coverage=float(np.mean(a['final_probes']['support'])),right_coverage=float(np.mean(b['final_probes']['support'])),left_mse=mean(a,'squared_error'),right_mse=mean(b,'squared_error'),
                left_gradient_mse=mean(a,'gradient_squared_error'),right_gradient_mse=mean(b,'gradient_squared_error'))


def background(ax,d):
    if 'spec' not in d:
        ax.text(.5,.9,'Truth geometry not recorded (legacy)',transform=ax.transAxes,ha='center');return
    spec=d['spec'];world=ScenarioWorld(spec,Config());lo,hi,bot,top=spec['bounds']
    xx,yy=np.meshgrid(np.linspace(lo,hi,140),np.linspace(bot,top,55))
    zz=world.risk_truth(np.stack([xx,yy],-1))
    ax.pcolormesh(xx,yy,zz,shading='auto',cmap='YlOrRd',vmin=0,vmax=1.2,alpha=.6)
    for x,y,r in spec['obstacles']:ax.add_patch(plt.Circle((x,y),r,color='black'))
    ax.plot(*spec['goal'],'*',color='green',ms=12);ax.set_xlim(lo,hi);ax.set_ylim(bot,top)
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x');ax.set_ylabel('y')


def frame(d,index):
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    trajectory=d.get('trajectory',[])
    if not trajectory:
        for ax in axs.flat:ax.text(.5,.5,'Execution states unavailable',ha='center',transform=ax.transAxes)
        return fig
    index=min(index,len(trajectory)-1)
    xy=np.array([r['x'][:2] for r in trajectory[:index+1]])
    actual=np.array(trajectory[index].get('after',trajectory[index]['x']))
    background(axs[0,0],d);axs[0,0].plot(xy[:,0],xy[:,1],color='navy');axs[0,0].plot(*actual[:2],'bo')
    radius=d.get('config',{}).get('sensor_radius',1.25)
    axs[0,0].add_patch(plt.Circle(actual[:2],radius,fill=False,ls=':',color='blue'))
    tick=trajectory[index]['tick'];available=[u for u in d.get('updates',[]) if 'post' in u and u['tick']<=tick]
    update=available[-1] if available else None
    if update:
        for s in update['post']['stages']:
            lo,hi,bot,top=s['bounds'];axs[0,0].add_patch(plt.Rectangle((lo,bot),hi-lo,top-bot,fill=False,alpha=.45))
        points=np.array(d['probe_points']);values=np.array([np.nan if v is None else v for v in update['probes']['values']])
        axs[0,1].scatter(points[:,0],points[:,1],c=values,vmin=0,vmax=1.2,cmap='YlOrRd',s=12)
        mask=np.array(update['probes']['support']);g=np.array([v or [0,0] for v in update['probes']['gradients']])
        axs[0,1].quiver(points[mask,0],points[mask,1],-g[mask,0],-g[mask,1],scale=8,width=.004)
        axs[0,1].text(.02,.98,f"snapshot tick {update['tick']}, version {update['post']['version']}; supported {update['probes']['coverage']:.0%}\nArrows: negative risk gradient (not total control)",transform=axs[0,1].transAxes,va='top',fontsize=8)
        axs[0,1].plot(*actual[:2],'bo')
        calls=[c for c in update['replay_calls'] if c['trajectories']]
        background(axs[1,0],d)
        for j,c in enumerate(([calls[0],calls[-1]] if calls else [])):
            for k,t in enumerate(c['trajectories']):
                path=np.array(t['states']);axs[1,0].plot(path[:,0],path[:,1],ls='--' if j==0 else '-',
                    color='purple' if j==0 else 'green',label=('incumbent' if j==0 else 'last candidate') if k==0 else None)
                axs[1,0].plot(*path[0,:2],'x',color='black')
        if calls:axs[1,0].legend(fontsize=8)
        else:axs[1,0].text(.5,.5,'No valid replay recorded',transform=axs[1,0].transAxes,ha='center')
        times=[u['tick'] for u in available];err=[u['probes']['mse'] for u in available]
        axs[1,1].plot(times,err,'o-',ms=3,label='field MSE')
        axs[1,1].scatter([u['tick'] for u in available if not u['diagnostic']['accepted']],
                        [u['probes']['mse'] or 0 for u in available if not u['diagnostic']['accepted']],marker='x',c='red',label='not committed')
        twin=axs[1,1].twinx();twin.plot([r['tick'] for r in trajectory[:index+1]],np.cumsum([r['cost'] for r in trajectory[:index+1]]),c='gray',label='actual accumulated cost');twin.set_ylabel('actual running cost')
        axs[1,1].legend(fontsize=8);axs[1,1].set_xlabel('physical tick')
    else:
        for ax in (axs[0,1],axs[1,0],axs[1,1]):ax.text(.5,.5,'Snapshot / replay unavailable\nnot inferred from final model',ha='center',transform=ax.transAxes)
    for ax,title in zip(axs.flat,['Evaluator truth and executed motion','Agent field estimate and support','Hypothetical trajectories, not executed history','Within-mission learning diagnostics']):ax.set_title(title,fontsize=10)
    if 'spec' in d:
        bounds=d['spec']['bounds'];axs[0,1].set_xlim(bounds[:2]);axs[0,1].set_ylim(bounds[2:]);axs[0,1].set_aspect('equal',adjustable='box')
    summary=d.get('summary',{});fig.suptitle(f"{summary.get('scene','legacy')} | {summary.get('arm',summary.get('mode',''))} | seed {summary.get('seed')} | tick {tick} | final outcome: {'SUCCESS' if summary.get('success') else 'FAILURE'}",fontsize=11)
    return fig


def render_main(argv):
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--frames',type=int,default=28);args=p.parse_args(argv)
    root=args.run;out=root/'figures';out.mkdir(exist_ok=True)
    if args.frames<1:p.error('frames must be positive')
    manifest=load(root/'manifest.json');files=episodes(root)
    if not files:
        # Legacy diagnostics deliberately omit unavailable fields.
        files=sorted(root.glob('*_seed*.json'))
    failure_dir=out/'failures';failure_dir.mkdir(exist_ok=True);gallery=[]
    for path in files:
        d=load(path);s=d['summary']
        if not s['success']:
            fig=frame(d,len(d['trajectory'])-1);name=path.name.split('.json')[0]+'.png'
            fig.savefig(failure_dir/name,dpi=85);plt.close(fig);gallery.append((name,s))
        if s.get('seed')==manifest.get('animation_seed') and s.get('arm')==manifest.get('animation_arm'):
            images=[]
            for idx in np.unique(np.linspace(0,len(d['trajectory'])-1,min(args.frames,len(d['trajectory']))).astype(int)):
                fig=frame(d,int(idx));fig.canvas.draw()
                images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'));plt.close(fig)
            images[0].save(out/f"{s['scene']}_seed{s['seed']}.gif",save_all=True,append_images=images[1:],duration=150,loop=0)
    html=['<html><body><h1>All recorded failures</h1><p>No failures removed or substituted.</p>']
    for name,s in gallery:html.append(f'<h2>{s.get("scene")} / {s.get("arm",s.get("mode"))} / seed {s["seed"]}</h2><img width="900" src="{name}">')
    html.append('</body></html>');(failure_dir/'index.html').write_text('\n'.join(html))
    print(f'Rendered {len(gallery)} failures and predeclared animation episodes',flush=True)


def report_main(argv):
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args(argv)
    root=args.run;manifest=load(root/'manifest.json');out=root/'figures';out.mkdir(exist_ok=True)
    rows=[];audit_rows=[];common=[];paired=[]
    # Stream compressed records; do not hold the complete experiment in memory.
    by_key={}
    for path in episodes(root):
        d=load(path);s=d['summary'];rows.append(s);key=(s['scene'],s['seed'])
        by_key.setdefault(key,{})[s['arm']]=path
        for a in d['audits']:
            audit_rows.append(dict(scene=s['scene'],seed=s['seed'],arm=s['arm'],stage=a['stage'],valid=a['valid'],
                predicted=a['predicted_improvement'],realized=a['realized_improvement']))
    for (scene,seed),files in by_key.items():
        if 'frozen' not in files:continue
        base=load(files['frozen'])
        for arm,path in files.items():
            d=load(path);common.append(dict(scene=scene,seed=seed,arm=arm,**common_probe_error(d,base)))
            paired.append(dict(scene=scene,seed=seed,arm=arm,difference=d['summary']['mission_cost']-base['summary']['mission_cost']))
    for name,data in [('summary',rows),('audits',audit_rows),('common_probes',common),('paired',paired)]:
        dump(root/f'{name}.json',data)
    flat=[]
    for r in rows:flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|{f'cost_{k}':v for k,v in r['components'].items()})
    if flat:
        with (root/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    scenes=list(dict.fromkeys(s['name'] for s in manifest['scenarios']))
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,metric,title in [(axs[0],'mission_cost','Complete cost'),(axs[1],'success','Success fraction')]:
        matrix=np.full((len(ARMS),len(scenes)),np.nan)
        for i,arm in enumerate(ARMS):
            for j,scene in enumerate(scenes):
                group=[r[metric] for r in rows if r['arm']==arm and r['scene']==scene]
                if group:matrix[i,j]=np.mean(group)
        im=ax.imshow(matrix,aspect='auto');fig.colorbar(im,ax=ax);ax.set_xticks(range(len(scenes)),scenes,rotation=60,ha='right');ax.set_yticks(range(len(ARMS)),ARMS);ax.set_title(title)
    fig.savefig(out/'performance.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,arm in enumerate(ARMS):
        delta=[r['difference'] for r in paired if r['arm']==arm]
        if delta:axs[0].scatter(np.full(len(delta),i),delta,s=10,alpha=.5)
        group=[r for r in rows if r['arm']==arm]
        counts=[sum(r['success'] for r in group),sum(r['collision'] for r in group),sum(r['timeout'] for r in group)]
        bottom=0
        for count,color,label in zip(counts,('green','red','orange'),('success','collision','timeout')):
            axs[1].bar(i,count,bottom=bottom,color=color,label=label if i==0 else None);bottom+=count
    axs[0].axhline(0,color='black',ls='--');axs[0].set_ylabel('Paired complete cost minus frozen');axs[1].legend()
    for ax in axs:ax.set_xticks(range(len(ARMS)),ARMS,rotation=60,ha='right')
    fig.savefig(out/'ablations.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for i,arm in enumerate(ARMS):
        group=[r for r in rows if r['arm']==arm]
        if not group:continue
        bottom=0
        for k,color in zip(('time','risk','effort','failure'),('steelblue','orange','green','red')):
            value=np.mean([r['components'][k] for r in group]);axs[0].bar(i,value,bottom=bottom,color=color,label=k if i==0 else None);bottom+=value
        axs[1].bar(i,np.mean([r['useful_replay_steps'] for r in group]),color='steelblue')
        axs[1].bar(i,np.mean([r['padding_steps'] for r in group]),bottom=np.mean([r['useful_replay_steps'] for r in group]),color='gray')
        axs[2].bar(i,np.mean([r['wall_seconds'] for r in group]))
    for ax,title in zip(axs,['Mission cost components','Replay work (gray = padding)','Episode wall time, seconds']):ax.set_xticks(range(len(ARMS)),ARMS,rotation=65,ha='right');ax.set_title(title)
    axs[0].legend();fig.savefig(out/'cost_compute.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    valid=[a for a in audit_rows if a['valid']]
    if valid:
        x=[a['predicted'] for a in valid];y=[a['realized'] for a in valid];ax.scatter(x,y,s=9,alpha=.5)
        lo=min(x+y);hi=max(x+y);ax.plot([lo,hi],[lo,hi],'k--')
    else:ax.text(.5,.5,'No domain-valid audits',transform=ax.transAxes,ha='center')
    ax.set_xlabel('Predicted improvement');ax.set_ylabel('Realized simulator improvement');ax.set_title('Matched replay objective; not mission cost')
    fig.savefig(out/'replay_calibration.png',dpi=140);plt.close(fig)
    # Predeclared paired trajectory and per-update learning plots.
    for scene in scenes:
        files=by_key.get((scene,manifest['animation_seed']),{})
        if not files:continue
        fig,ax=plt.subplots(figsize=(9,3),layout='constrained')
        first=load(next(iter(files.values())));background(ax,first)
        for arm in ('frozen','evidence','full','current_replay'):
            if arm in files:
                d=load(files[arm]);xy=np.array([r['x'][:2] for r in d['trajectory']]);ax.plot(xy[:,0],xy[:,1],label=arm)
                for tick in range(0,len(xy),40):
                    ax.plot(*xy[tick],'o',ms=3);ax.annotate(str(tick),xy[tick],fontsize=6)
        ax.legend(fontsize=8);ax.set_title(f'{scene}: paired trajectories, seed {manifest["animation_seed"]}');fig.savefig(out/f'{scene}_paired.png',dpi=130);plt.close(fig)
        if 'full' in files:
            d=load(files['full']);fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained');u=d['updates'];ticks=[x['tick'] for x in u]
            for key in ('mse','gradient_mse'):axs[0,0].plot(ticks,[x['probes'][key] for x in u],label=key)
            axs[0,0].legend();axs[0,0].set_title('Field errors on supported probes')
            axs[0,1].plot(ticks,[x['post']['damping'] for x in u]);axs[0,1].axhline(d['spec']['damping'],ls='--',color='black');axs[0,1].set_title('Estimated / true damping')
            axs[1,0].plot(ticks,[x['post']['policy'] for x in u]);axs[1,0].set_title('Policy coefficients; x = rejected update')
            rejected=[x for x in u if not x['diagnostic']['accepted']]
            axs[1,0].scatter([x['tick'] for x in rejected],[x['post']['policy'][0] for x in rejected],marker='x',color='red')
            axs[1,1].plot(ticks,[x['diagnostic'].get('replay_before') for x in u],label='before')
            axs[1,1].plot(ticks,[x['diagnostic'].get('replay_after') for x in u],label='after');axs[1,1].legend();axs[1,1].set_title('Replay cost (batch changes over time)')
            fig.savefig(out/f'{scene}_learning.png',dpi=130);plt.close(fig)
    lines=['# Diagnostic evaluation report','',f"Profile: **{manifest['profile']}**. Completed **{len(rows)}/{manifest['expected_episodes']}** episodes.",
        '', 'These are within-mission adaptation measurements, not across-episode learning curves. All scenarios and animation seeds were fixed before execution. Failures and invalid audits are retained.','',
        '| Scenario | Arm | Success / completed | Mean cost | Paired delta vs frozen |','|---|---|---:|---:|---:|']
    for scene in scenes:
        for arm in ARMS:
            g=[r for r in rows if r['scene']==scene and r['arm']==arm];diff=[r['difference'] for r in paired if r['scene']==scene and r['arm']==arm]
            if g:lines.append(f"| {scene} | {arm} | {sum(r['success'] for r in g)}/{len(g)} | {np.mean([r['mission_cost'] for r in g]):.3f} | {np.mean(diff):+.3f} |")
            else:lines.append(f'| {scene} | {arm} | MISSING | — | — |')
    significant=[a for a in valid if abs(a['predicted'])>1e-9 or abs(a['realized'])>1e-9]
    agreement=np.mean([np.sign(a['predicted'])==np.sign(a['realized']) for a in significant]) if significant else None
    lines.extend(['','## Mechanism diagnostics',''])
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        if not group:continue
        lines.append(f"- {arm}: {sum(r['success'] for r in group)}/{len(group)} successes; mean stationary time {np.mean([r['near_stationary_seconds'] for r in group]):.2f} s; field coverage {np.mean([r['coverage'] for r in group]):.1%}; mean damping error {np.mean([r['damping_error'] for r in group]):.4f}; {sum(r['domain_failures'] for r in group)} rejected replay batches from domain support; {sum(r['accepted_updates'] for r in group)} committed updates.")
    full=[r for r in rows if r['arm']=='full'];frozen=[r for r in rows if r['arm']=='frozen']
    if full and frozen:
        deltas=[r['difference'] for r in paired if r['arm']=='full']
        lines.append(f"\nFull replay has mean paired mission-cost change {np.mean(deltas):+.3f} versus frozen (positive is worse). This is the measured outcome, not an optimization target used to select examples.")
    long=[r for r in rows if r['scene']=='long_corridor']
    if long:lines.append(f"\nLong-corridor geometry spans at least eight nominal stages; policies actually visited {min(r['visited_stages'] for r in long)}–{max(r['visited_stages'] for r in long)}. Geometry alone does not establish that the memory ablation was exercised; inspect visited and peak retained stage counts.")
    if rows:lines.append(f"\nMaximum exact physical energy residual: {max(r['max_energy_residual'] for r in rows):.3g}. Physical accounting correctness does not imply mission improvement.")
    lines.extend(['','## Replay validity and interpretation','',f'Domain-valid audits: {len(valid)}/{len(audit_rows)}. Nonzero-effect sign agreement: {agreement}.',
        'Audit work is reported separately and cannot enter fitting. Both branches share starts, horizons and objective coefficients; their model and true fields differ. Invalid branches are not silently replaced.',
        '', 'The current-state comparator has the paired full arm’s episode-level replay-transition cap. Unused budget is executed as explicitly labeled non-learning padding; compare useful work as well as charged work. This is not a claim of equal optimizer progress or equal latency.',
        '', 'Common-probe comparisons and coverage are in `common_probes.json`; empty intersections remain missing. Final retained-domain probes do not describe the entire traversed map.',
        '', '## Artifacts','', '[Performance](figures/performance.png) · [Paired ablations](figures/ablations.png) · [Cost and computation](figures/cost_compute.png) · [Replay calibration](figures/replay_calibration.png) · [All failures](figures/failures/index.html)',
        '', 'Each scenario also has predeclared paired trajectories, full-arm learning traces and an animation. Missing snapshots in historical runs are labeled unavailable. See `metrics.csv`, `paired.json`, `audits.json`, and the immutable manifest for numerical evidence.',
        '', '## Limitations','', f"Executed attraction target: {manifest['config'].get('navigation_target','stage')}. Replay and its audits retain stage-local targets; audit calibration is not a validation of the mission-target controller. Failure prices and replay settings were not tuned. Scene seeds measure environment variation, not training-population uncertainty. No cooperative, across-episode or confirmatory superiority claim is supported."])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Reported {len(rows)} episodes, {len(valid)}/{len(audit_rows)} valid audits',flush=True)
