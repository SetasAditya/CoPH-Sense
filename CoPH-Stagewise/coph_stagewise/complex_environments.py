"""Frozen static stress environments and fixed-frequency cooperative comparisons."""
import argparse,hashlib,json,time,traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .cooperative import CommunicationConfig
from .cooperative_evaluation import specification,execute,CooperativeWorld
from .evaluation import dump,load
from .records import Config
from .cli import source_hashes,archive_sources

SCENES=('hazard_slalom','blocked_risk_passage','asymmetric_coverage','long_obstacle_corridor')
ARMS={'disabled':('disabled','supported_boundary'),'one_way_01':('one_way_01','supported_boundary'),
      'one_way_10':('one_way_10','supported_boundary'),'evidence_only':('evidence_only','supported_boundary'),
      'reciprocal_fixed':('reciprocal_fixed','supported_boundary'),'short_replay':('reciprocal_fixed','fixed')}
COLORS=('tab:blue','tab:green')

def scene(name,seed):
    if name not in SCENES:raise ValueError(name)
    s=specification('complementary',seed,'development');rng=np.random.default_rng(seed)
    jitter=float(rng.uniform(-.12,.12));length=20. if name=='long_obstacle_corridor' else 10.
    s.update(name=name,initials=[[0.,-.65,0.,0.],[0.,.65,0.,0.]],goals=[[length,-.65],[length,.65]],
        bounds=[-1.5,length+1.5,-3.,3.],steps=int(np.ceil(180*length/6)),dynamic=False,channel='normal')
    if name=='hazard_slalom':
        s['obstacles']=[[2.7,-.65,.5],[5.2,.65,.55],[7.6,-.55,.5]]
        s['patches']=[[2.5,1.+jitter,.8,.6,1.1],[5.,-1.,.9,.65,1.3],[7.8,1.,.8,.6,1.2]]
    elif name=='blocked_risk_passage':
        s['obstacles']=[[3.8,-1.1,.6],[3.8,1.1,.6],[6.6,0.,.65],[7.,1.8,.45]]
        s['patches']=[[3.8,jitter,.7,.75,1.6],[6.8,-1.25,.9,.5,1.3],[8.3,.6,.7,.8,.8]]
    elif name=='asymmetric_coverage':
        s['initials']=[[1.4,-1.1,0.,0.],[0.,1.1,0.,0.]];s['goals']=[[length,.8],[length,-.8]]
        s['obstacles']=[[3.,0.,.65],[5.5,1.15,.55],[7.5,-.9,.55]]
        s['patches']=[[2.5,-.9+jitter,.7,.5,1.5],[4.7,.6,.9,.75,1.2],[7.,-.9,.8,.55,1.4],[8.8,1.,.7,.5,.9]]
    else:
        s['obstacles']=[[x,(-1)**k*.8,.5] for k,x in enumerate(np.arange(2.8,19.,2.6))]
        s['patches']=[[x,(-1)**(k+1)*.9+jitter,.8,.6,1.+.15*(k%3)] for k,x in enumerate(np.arange(2.4,19.,2.6))]
    s.update(initial=s['initials'][0],goal=s['goals'][0]);return s

def job(args):
    spec,label,root,comm=args;torch.set_num_threads(1);path=Path(root)/f"{spec['name']}_{spec['seed']}_{label}.json.gz"
    try:
        arm,horizon=ARMS[label]
        r=execute(spec,arm,CommunicationConfig(**comm),audits=False,replay_domain='retained',horizon_mode=horizon,replay_cap=1024)
        r['comparison_arm']=label;dump(path,r)
        return dict(scene=spec['name'],seed=spec['seed'],arm=label,path=path.name,complete=True)
    except Exception as e:
        dump(path,dict(error=str(e),traceback=traceback.format_exc()))
        return dict(scene=spec['name'],seed=spec['seed'],arm=label,path=path.name,complete=False,error=str(e))

def world_axes(ax,spec):
    from matplotlib.patches import Circle
    w=CooperativeWorld(spec,Config());lo,hi,bot,top=spec['bounds']
    xx,yy=np.meshgrid(np.linspace(lo,hi,150),np.linspace(bot,top,65))
    z=w.risk_truth(np.column_stack((xx.ravel(),yy.ravel()))).reshape(xx.shape)
    ax.imshow(z,extent=(lo,hi,bot,top),origin='lower',cmap='YlOrRd',vmin=0,vmax=2.,alpha=.65)
    for x,y,r in spec['obstacles']:ax.add_patch(Circle((x,y),r,color='slategray'));ax.add_patch(Circle((x,y),r+.12,fill=False,ls=':',color='gray',lw=.6))
    for i,c in enumerate(COLORS):ax.scatter(*spec['goals'][i],marker='*',color=c,s=65)
    ax.set(xlim=(lo,hi),ylim=(bot,top),aspect='equal');return xx,yy

def paths(ax,result,tick):
    for i,c in enumerate(COLORS):
        tr=[r for r in result['trajectories'][i] if r['tick']<=tick]
        if not tr:continue
        p=np.array([tr[0]['x']]+[r['after'] for r in tr]);ax.plot(p[:,0],p[:,1],color=c,lw=1.6)
        ax.scatter(*p[-1,:2],color=c,s=24)
        if len(tr)==len(result['trajectories'][i]):
            status='goal' if result['agents'][i]['success'] else 'collision' if result['agents'][i]['collision'] else 'timeout'
            ax.annotate(status,p[-1,:2],fontsize=6,color=c,xytext=(3,5),textcoords='offset points')

def comparison_frame(results,tick):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    from PIL import Image
    from .model import tensor,risk
    fig,axes=plt.subplots(2,2,figsize=(12,7));spec=results['reciprocal_fixed']['spec']
    for ax,label in zip(axes.flat,('disabled','evidence_only','reciprocal_fixed')):
        r=results[label];world_axes(ax,spec);paths(ax,r,tick)
        tr=[t for ts in r['trajectories'] for t in ts if t['tick']<=tick]
        cumulative=sum(t['cost'] for t in tr)
        for i,ts in enumerate(r['trajectories']):
            if ts[-1]['tick']<=tick and not r['agents'][i]['success']:cumulative+=40.
        ax.set_title(f'{label}: accrued mission cost {cumulative:.2f}',fontsize=10)
        if label!='disabled':
            for p in r['packets']:
                if 0<=(tick*.12-p['timestamp'])<.96:
                    a=np.array(p['anchors']);ax.scatter(a[:,0],a[:,1],marker='+',color='purple',s=15)
    ax=axes[1,1];r=results['reciprocal_fixed'];lo,hi,bot,top=spec['bounds']
    xx,yy=np.meshgrid(np.linspace(lo,hi,100),np.linspace(bot,top,45));q=np.column_stack((xx.ravel(),yy.ravel()))
    for i,c in enumerate(COLORS):
        us=[u for u in r['updates'][i] if u['tick']<=tick]
        if not us:continue
        u=us[-1];snap=u['post'];s=snap['stages'][-1];samples=np.array([z['q'] for z in s['readings'] if int(z['uid'].split(':')[-1])%5]);b=s['bounds']
        mask=(xx>=b[0])&(xx<=b[1])&(yy>=b[2])&(yy<=b[3])
        if len(samples):mask&=(np.min(np.sum((q[:,None,:]-samples[None,:,:])**2,axis=2),axis=1).reshape(xx.shape)<=.65**2)
        else:mask[:]=False
        v=risk(tensor(q),tensor(s['weights']),tensor(s['centers']),r['config']['rbf_width']).numpy().reshape(xx.shape)
        if mask.any():ax.contour(xx,yy,np.where(mask,v,np.nan),levels=5,colors=c,linewidths=.8)
        for st in snap['stages']:
            b=st['bounds'];ax.add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],fill=False,color=c,alpha=.2))
            for cf in st['counterfactuals']:
                p=np.array(cf['states']);ax.plot(p[:,0],p[:,1],color=c,alpha=.6,ls='--')
        for ox,oy,rad in s['obstacles']:ax.add_patch(Circle((ox,oy),rad,fill=False,color=c,alpha=.5))
        ax.text(.02,.97-i*.07,f'agent {i}: stage {s["id"]}, model {snap["version"]}; snapshot t={u["tick"]*.12:.2f}s',transform=ax.transAxes,color=c,fontsize=7,va='top')
    ax.set(xlim=(lo,hi),ylim=(bot,top),aspect='equal',title='Supported field estimates; retained replay dashed')
    fig.suptitle(f'{spec["name"]} | seed {spec["seed"]} | t={(tick+1)*.12:.2f}s\nBlue/green: independent agents; purple +: sent anchors; gray: obstacles',fontsize=11)
    fig.tight_layout();fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return im

def report(root,completion,animation_seed,animate=False):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    rows=[];selected={};failures=[];figdir=root/'figures';figdir.mkdir(exist_ok=True)
    (figdir/'failures').mkdir(exist_ok=True)
    for item in completion:
        if not item['complete']:continue
        r=load(root/'episodes'/item['path']);row=dict(r['summary'],arm=item['arm'],collision=sum(a['collision'] for a in r['agents']),timeout=sum(a['timeout'] for a in r['agents']),
            components={k:sum(a['components'][k] for a in r['agents']) for k in ('time','risk','effort','failure')},
            max_stage=max(t['stage'] for ts in r['trajectories'] for t in ts),
            accepted=sum(u['diagnostic']['accepted'] for us in r['updates'] for u in us),
            handoffs=sum(len(c.get('stage_events',[])) for us in r['updates'] for u in us for c in u['replay_calls']),
            force_changes=[u['diagnostic'].get('force_change',0) for us in r['updates'] for u in us if u['phase']=='message'])
        rows.append(row)
        assert abs(row['team_cost']-sum(row['components'].values()))<1e-8
        assert all(u['diagnostic'].get('counterfactual_steps',0)<=1024 for us in r['updates'] for u in us)
        if item['seed']==animation_seed:selected.setdefault(item['scene'],{})[item['arm']]=r
        if row['failures']:
            failures.append(item)
            f,ax=plt.subplots(figsize=(10,3.5));world_axes(ax,r['spec']);paths(ax,r,10**9)
            ax.set_title(f"{item['scene']} seed {item['seed']} {item['arm']} | {row['failures']} failed agents")
            f.tight_layout();f.savefig(figdir/'failures'/f"{item['scene']}_{item['seed']}_{item['arm']}.png",dpi=110);plt.close(f)
    dump(root/'rows.json',rows);dump(root/'failures.json',failures)
    base={(r['scene'],r['seed']):r for r in rows if r['arm']=='disabled'}
    paired=[dict(scene=r['scene'],seed=r['seed'],arm=r['arm'],difference=r['team_cost']-base[r['scene'],r['seed']]['team_cost']) for r in rows if (r['scene'],r['seed']) in base]
    dump(root/'paired.json',paired)
    lines=['# Complex static environments: fixed-frequency cooperation','',
      'Development evidence. Identical scenes, seeds, horizons and learner settings across arms. Transmission opportunities every 0.96 seconds; the same 2,000,000-byte per-agent ceiling is frozen before runs to avoid early budget depletion in long scenes. Per-byte costs, sensing and learning rules are unchanged. Actual traffic differs between arms. No receiver-benefit predictor controls communication.',
      'These are independent plants sharing environmental information. There is no inter-agent collision model or joint mechanical planner. Scalar-potential navigation can become trapped by obstacles; no global route planner is added.',
      f'\nCompleted {sum(c["complete"] for c in completion)}/{len(completion)} episodes. Failed-agent episodes remain in the gallery.',
      '\n| Scene | Arm | Episodes | Team successes | Collisions / timeouts | Mean cost | Paired cost vs off | Mean bytes |', '|---|---|---|---|---|---|---|---|']
    for scene_name in SCENES:
        for arm in ARMS:
            rs=[r for r in rows if r['scene']==scene_name and r['arm']==arm];ps=[p['difference'] for p in paired if p['scene']==scene_name and p['arm']==arm]
            if not rs:continue
            lines.append(f"| {scene_name} | {arm} | {len(rs)} | {sum(r['team_success'] for r in rs)} | {sum(r['collision'] for r in rs)} / {sum(r['timeout'] for r in rs)} | {np.mean([r['team_cost'] for r in rs]):.4f} | {np.mean(ps):+.4f} | {np.mean([r['bytes'] for r in rs]):.0f} |")
    for name,rs in selected.items():
        f,axes=plt.subplots(2,3,figsize=(14,7))
        for ax,arm in zip(axes.flat,ARMS):
            if arm not in rs:continue
            r=rs[arm];world_axes(ax,r['spec']);paths(ax,r,10**9);ax.set_title(f"{arm}: cost {r['summary']['team_cost']:.2f}",fontsize=9)
        f.suptitle(f'{name}: preselected seed {animation_seed}; identical world scales');f.tight_layout();f.savefig(figdir/f'{name}_paired.png',dpi=130);plt.close(f)
        if animate and all(k in rs for k in ('disabled','evidence_only','reciprocal_fixed')):
            end=max(len(ts) for arm in ('disabled','evidence_only','reciprocal_fixed') for ts in rs[arm]['trajectories'])-1
            ticks=sorted(set(np.linspace(0,end,min(42,end+1)).astype(int)));frames=[comparison_frame(rs,int(t)) for t in ticks]
            frames[0].save(figdir/f'{name}.gif',save_all=True,append_images=frames[1:],duration=max(100,int((end+1)*120/len(ticks))),loop=0)
            lines += ['',f'![{name}](figures/{name}.gif)']
        lines += ['',f'![{name} paired](figures/{name}_paired.png)']
    f,axes=plt.subplots(1,2,figsize=(13,4));names=list(ARMS)
    for i,arm in enumerate(names):
        ys=[p['difference'] for p in paired if p['arm']==arm];axes[0].scatter([i]*len(ys),ys,s=13,alpha=.6)
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set_ylabel('Paired mission cost minus communication off')
    bottom=np.zeros(len(names))
    for key in ('time','risk','effort','failure'):
        values=np.array([np.mean([r['components'][key] for r in rows if r['arm']==a]) for a in names]);axes[1].bar(range(len(names)),values,bottom=bottom,label=key);bottom+=values
    for ax in axes:ax.set_xticks(range(len(names)),names,rotation=25,fontsize=7)
    axes[1].legend(fontsize=7);axes[1].set_ylabel('Mean cost components');f.tight_layout();f.savefig(figdir/'costs.png',dpi=140);plt.close(f)
    lines+=['','![Cost comparison](figures/costs.png)','','[Every failed episode](figures/failures/index.html). Missing/incomplete runs remain in completion.json. Animations use logged positions and the latest preceding model snapshot; snapshot times are labeled. Finished agents are held at their final positions. Actual sensing is local; true maps appear only in evaluator visualizations.']
    (figdir/'failures'/'index.html').write_text('<h1>All failed episodes</h1>'+''.join(f'<p>{x["scene"]} {x["seed"]} {x["arm"]}<br><img width="850" src="{x["scene"]}_{x["seed"]}_{x["arm"]}.png"></p>' for x in failures))
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args(argv)
    if args.out.exists():ap.error('use a fresh output directory')
    args.out.mkdir(parents=True);comm=asdict(CommunicationConfig(byte_budget=2_000_000))
    specs={p:[scene(s,61000+k) for s in SCENES for k in range(n)] for p,n in [('smoke',2),('development',8)]}
    dump(args.out/'manifest.json',dict(specs=specs,arms=ARMS,communication=comm,source_hashes=source_hashes(),animation_seed=61000,
        config=asdict(Config(replay_objective='mission',replay_domain='retained',replay_horizon_mode='supported_boundary',replay_update_cap=1024)),
        horizon_rule='ceil(180 * goal_x / 6)',reserved_validation=[100000,199999],reserved_test=[200000,299999],status='development evidence; smoke repeats first two development seeds'))
    archive_sources(args.out);started=time.perf_counter();completed={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for profile in ('smoke','development'):
            root=args.out/profile;(root/'episodes').mkdir(parents=True);cs=[]
            for item in pool.map(job,[(s,a,str(root/'episodes'),comm) for s in specs[profile] for a in ARMS]):
                cs.append(item);print(profile,item['scene'],item['seed'],item['arm'],item['complete'],item.get('error',''),flush=True)
            dump(root/'completion.json',cs);completed[profile]=cs
            report(root,cs,61000,animate=profile=='development')
            if not all(c['complete'] for c in cs):break
    dump(args.out/'completion.json',dict(profiles=completed,seconds=time.perf_counter()-started))
    (args.out/'REPORT.md').write_text('# Fixed-frequency complex environments\n\n[Development results and GIFs](development/REPORT.md)\n\n[Smoke](smoke/REPORT.md)\n')
    if any(not c['complete'] for cs in completed.values() for c in cs):raise SystemExit(1)

if __name__=='__main__':main()
