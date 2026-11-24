"""Evidence-grounded scenario media. Rendering never updates a learner.

Run from CoPH-Stagewise with the trusted original results directory supplied via
--archive. MP4s, GIF previews, source hashes and per-frame provenance are emitted
into the project's static gallery. Unsupported surfaces and absent legacy signals
remain explicitly unavailable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image
from . import scenario_logs as logs
from .records import Config
from .maze_visuals import landscape
from .terrain_warning import terrain_values

BLUE='#2474c8'; ORANGE='#d67532'; TEAL='#008b87'; PURPLE='#8654ac'
INK='#1b2c42'; MUTED='#66778a'; GRID='#e3e9ef'; COLORS=[BLUE,TEAL,PURPLE,ORANGE]
FPS=10
TITLES={'dynamics':'Adapting to a dynamics change','hazards':'One field, different preferences',
        'reciprocity':'A warning changes the next discovery','encounters':'Clearing a symmetric encounter',
        'uncertainty':'Knowing when a route choice is uncertain'}


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def setup():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
      'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
      'animation.ffmpeg_path':'/usr/bin/ffmpeg' if Path('/usr/bin/ffmpeg').exists() else 'ffmpeg',
      'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b2becb',
      'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED})


def figure(title,subtitle):
    fig=plt.figure(figsize=(12.8,8),dpi=100,facecolor='#f7f9fc')
    gs=fig.add_gridspec(2,2,left=.065,right=.97,top=.80,bottom=.10,hspace=.46,wspace=.32)
    fig.text(.055,.955,'CoPH  /  OBSERVE · REVISE · ACT',fontsize=10,color=TEAL,weight='bold')
    fig.text(.055,.906,title,fontsize=21,weight='bold')
    fig.text(.055,.865,subtitle,fontsize=10,color=MUTED)
    return fig,gs


def axis(fig,gs,k,title,three=False):
    ax=fig.add_subplot(gs[k],projection='3d' if three else None)
    ax.set_title(title,loc='left',pad=8,color=INK,weight='bold')
    if not three:ax.grid(color=GRID,lw=.6,alpha=.8);ax.set_axisbelow(True)
    return ax


def world3d(ax,bounds,network=None):
    if network:
        nodes,edges=network
        for i,j in edges:
            p=np.asarray([nodes[i],nodes[j]],float)
            ax.plot(p[:,0],p[:,1],zs=0,color='#dbe4ed',lw=14,alpha=.9)
            ax.plot(p[:,0],p[:,1],zs=.002,color='#abbac9',lw=.7,ls=':')
    ax.set(xlim=bounds[:2],ylim=bounds[2:4],zlim=(0,.8),xlabel='x (m)',ylabel='y (m)')
    ax.set_zticks([]);ax.view_init(elev=42,azim=-63);ax.set_box_aspect((1.6,1,.18))
    for a in [ax.xaxis,ax.yaxis,ax.zaxis]:a.pane.fill=False
    ax.grid(False)


def footer(fig,text):
    fig.text(.055,.035,text,fontsize=9,color=MUTED)


def absence(ax,message):
    ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color=MUTED,fontsize=11)
    ax.set_xticks([]);ax.set_yticks([])


def indices(count,frames,extra=()):
    return sorted(set(np.rint(np.linspace(0,count-1,frames)).astype(int))|set(int(x) for x in extra if 0<=x<count))


def render_video(stem,draw,steps,out,records,poster_index=None):
    """Every frame is drawn from an explicit logged index or saved-state query."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gifs=[];entries=[];writer=FFMpegWriter(fps=FPS,codec='libx264',bitrate=1300,
        extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    # Writer can receive independent figure canvases through its fig attribute.
    first,record=draw(steps[0]);poster_index=len(steps)//2 if poster_index is None else poster_index
    with writer.saving(first,str(out/f'{stem}.mp4'),100):
        for n,step in enumerate(steps):
            if n:fig,record=draw(step)
            else:fig=first
            writer.fig=fig;writer.grab_frame()
            if n%2==0 or n==len(steps)-1:
                fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                im=Image.fromarray(rgb).resize((768,480),Image.Resampling.LANCZOS).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)
                gifs.append(im)
            if n==poster_index:fig.savefig(out/f'{stem}.png',dpi=100)
            entries.append({'video_frame':n,'video_seconds':n/FPS,**record})
            plt.close(fig)
    gifs[0].save(out/f'{stem}.gif',save_all=True,append_images=gifs[1:],duration=200,loop=0,optimize=False)
    for ext in ['mp4','png','gif']:
        path=out/f'{stem}.{ext}';records['artifacts'][path.name]={'sha256':logs.source_record(path)['sha256'],'bytes':path.stat().st_size}
    records['frames'][stem]=entries
    write(out/f'{stem}.frames.json',entries)
    print(f'{stem}: {len(steps)} frames',flush=True)


def dynamics(log):
    names=list(log['arms']);names.sort(key=lambda n:0 if 'online' in n else 1 if 'known' in n else 2)
    return_floor=min(float(log['arms'][n]['arrays']['rewards'][200:].reshape(200,4).mean(1).sum()) for n in names)*1.08
    styles={names[0]:(BLUE,'Online CoPH'),names[1]:(ORANGE,'Fixed drag 0.25'),names[2]:(MUTED,'Matched phMARL')}
    def draw(step):
        frame=logs.drag_frame(log,step);now=frame['time'];fig,gs=figure(TITLES['dynamics'],f'Native VMAS · seed {log["seed"]} · logged t = {now:.1f} s · drag 0.25 → {log["after_drag"]:.2f} at 20 s')
        ax=axis(fig,gs,0,'WORLD  ·  planar simulation in oblique view',True);world3d(ax,[-1.5,1.5,-1.5,1.5])
        for name in names[:2]:
            col,label=styles[name];q=log['arms'][name]['arrays']['positions'][:step+1]
            for i in range(4):
                ax.plot(q[:,i,0],q[:,i,1],zs=.035,color=col,lw=1.3,ls='-' if name==names[0] else '--',alpha=.8)
            ax.scatter(q[-1,:,0],q[-1,:,1],zs=.04,c=col,s=32,label=label)
        goals=np.asarray(frame['arms'][names[0]]['goals']);ax.scatter(goals[:,0],goals[:,1],zs=.02,c='#47566b',marker='*',s=75)
        ax.legend(loc='upper left',frameon=False)
        ax=axis(fig,gs,1,'MODEL  ·  committed drag estimates')
        for name in names[:2]:
            col,label=styles[name];z=log['arms'][name]['drag_estimate'][:step+1]
            ax.plot(log['arms'][name]['times'][:step+1],np.median(z,axis=1),color=col,label=label,lw=2)
            if name==names[0]:ax.fill_between(log['arms'][name]['times'][:step+1],z.min(1),z.max(1),color=col,alpha=.12)
        ax.plot([0,20,20,40],[.25,.25,log['after_drag'],log['after_drag']],color=INK,ls=':',label='True drag (evaluation)')
        ax.axvline(20,color=MUTED,ls='--',lw=.7);ax.set(xlim=(0,40),ylim=(0,.52),xlabel='Mission time (s)',ylabel='Native velocity-loss parameter');ax.legend(frameon=False,loc='upper left')
        ax=axis(fig,gs,2,'LEARNING → CONTROL  ·  agent 0 executed command')
        if step<400:
            k=step;u=frame['arms'][names[0]]['control'][0];nom=frame['arms'][names[0]]['nominal'][0]
            ax.bar(np.arange(2)-.17,nom,width=.34,color='#93b9df',label='Nominal');ax.bar(np.arange(2)+.17,u,width=.34,color=BLUE,label='Applied')
            ax.set_xticks([0,1],['x input','y input']);ax.set_ylim(-1.55,1.55);ax.set_ylabel('Native component-bounded input');ax.legend(frameon=False)
            updates=sum(u['step']<=step for u in log['arms'][names[0]]['updates'])
            ax.text(.02,.94,f'{updates} logged per-agent identification updates',va='top',transform=ax.transAxes,color=MUTED,fontsize=9)
        else:absence(ax,'Recorded horizon reached\nNo further control is shown')
        ax=axis(fig,gs,3,'OUTCOME  ·  post-change cumulative mean-agent return')
        for name in names:
            col,label=styles[name];r=log['arms'][name]['arrays']['rewards'].reshape(400,4).mean(1)
            tt=np.arange(200,step+1)*.1
            if step>=200:ax.plot(tt,np.r_[0,np.cumsum(r[200:step])],color=col,label=label,lw=1.8)
        ax.set(xlim=(20,40),ylim=(return_floor,5),xlabel='Mission time (s)',ylabel='Return since intervention')
        if step>=200:ax.legend(frameon=False,loc='lower left')
        footer(fig,'Same starts, goals, allocation and protection. Blue band: min–max across four agents. Native velocities were not retained; no inferred velocity is displayed.')
        return fig,{'logged_step':int(step),'mission_seconds':now,'true_drag':frame['true_drag'],'state_sha256':hashlib.sha256(log['arms'][names[0]]['arrays']['positions'][step].tobytes()).hexdigest()}
    return draw


def stagewise(log,kind):
    primary='reciprocal_fixed' if kind=='reciprocity' else 'field';reference='disabled' if kind=='reciprocity' else 'decision'
    ep=log['episodes'][primary];ref=log['episodes'][reference];who=0 if kind=='reciprocity' else 1;cfg=Config(**ep['config'])
    end=max(a['finish_time'] for a in ep['agents']);nodes,edges=ep['spec']['network_nodes'],ep['spec']['network_edges']
    def draw(now):
        frame=logs.stagewise_frame(ep,now);fig,gs=figure(TITLES[kind],f'Logged t = {now:.2f} s · seed {log["seed"]} · '+('reciprocal fixed-rate exchange' if kind=='reciprocity' else 'scout weight 0.25 · carrier weight 6 · carrier starts at 25 s'))
        ax=axis(fig,gs,0,'WORLD  ·  truth view + logged execution',True);world3d(ax,[-1,13,-1,8.5],(nodes,edges))
        for patch in ep['spec'].get('patches',[]):
            x,y,sx,sy,amp=patch;X,Y=np.meshgrid(np.linspace(x-2*sx,x+2*sx,28),np.linspace(y-1.2,y+1.2,24))
            R=amp*np.exp(-.5*(((X-x)/sx)**2+((Y-y)/sy)**2))
            ax.plot_surface(X,Y,np.zeros_like(X),facecolors=plt.cm.YlOrRd(R/max(amp,1e-12)),alpha=.45,shade=False)
        for ob in ep['spec'].get('obstacles',[]):
            x,y,rad=ob[:3];dy=(ob[4] if len(ob)==5 else rad)
            ax.plot([x,x],[y-dy,y+dy],zs=.15,color='#a9383b',lw=5)
        for i in range(2):
            sam=[s for s in ep['samples'][i] if s['time']<=now+1e-10];ghost=[s for s in ref['samples'][i] if s['time']<=now+1e-10]
            if sam:
                q=np.asarray([s['x'][:2] for s in sam]);ax.plot(q[:,0],q[:,1],zs=.03,color=COLORS[i],lw=2);ax.scatter(*q[-1],zs=.06,c=COLORS[i],s=38,label=f'Agent {i}')
            if ghost:
                q=np.asarray([s['x'][:2] for s in ghost]);ax.plot(q[:,0],q[:,1],zs=.015,color=MUTED,ls='--',alpha=.45,lw=1)
            item=frame['agents'][i]
            if item.get('active') and item.get('model'):
                q=np.asarray(item['state'][:2]);target=np.asarray(item['model']['target']);ax.scatter(*target,zs=.06,marker='x',c=COLORS[i],s=30)
                if item.get('control') is not None:ax.quiver(q[0],q[1],.08,item['control'][0]*.25,item['control'][1]*.25,0,color=COLORS[i],linewidth=1.5)
        recent=[e for e in frame['events'] if e['kind']=='delivered' and now-e['time']<=1.2]
        for event in recent:
            a,b=np.asarray(event['sender_q']),np.asarray(event['receiver_q']);ax.plot([a[0],b[0]],[a[1],b[1]],zs=.3,color=PURPLE,lw=1.8,ls=':')
        ax.legend(loc='upper left',frameon=False,ncol=2)
        model=frame['agents'][who].get('model');item=frame['agents'][who]
        ax=axis(fig,gs,1,f'MODEL  ·  agent {who} shaping Hamiltonian',True)
        surface_record=None
        if model is not None:
            c=np.asarray(model['center']);X,Y=np.meshgrid(np.linspace(c[0]-1.5,c[0]+1.5,33),np.linspace(c[1]-2,c[1]+2,33));points=np.c_[X.ravel(),Y.ravel()]
            # The model is the control-time snapshot; use its original pre-step velocity.
            row=ep['trajectories'][who][item['trajectory_index']];H,F,supported=landscape(model,points,np.asarray(row['x'][2:]),cfg)
            if supported.any():
                Z=np.where(supported,H,np.nan).reshape(X.shape);ax.plot_surface(X,Y,Z,cmap='viridis',vmin=0,vmax=12,linewidth=0,antialiased=False,alpha=.94)
            ax.set(xlim=(c[0]-1.5,c[0]+1.5),ylim=(c[1]-2,c[1]+2),zlim=(0,12),xlabel='x (m)',ylabel='y (m)',zlabel='K(v logged) + V(q) (J)');ax.view_init(elev=34,azim=-60)
            ax.text2D(.01,.95,f'Version {model["version"]} · stage {model["stage"]} · model t={row["time"]:.2f} s'+(' · finished' if not item['active'] else ''),transform=ax.transAxes,fontsize=8,color=MUTED)
            surface_record={'version':model['version'],'stage':model['stage'],'supported':int(supported.sum()),'queries':len(points),'finite':bool(np.isfinite(H[supported]).all()),'control_time':row['time'],'max_supported_H':float(H[supported].max()) if supported.any() else None,'model_sha256':hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()}
        else:ax.text2D(.1,.5,'Agent not active; no model snapshot yet',transform=ax.transAxes,color=MUTED)
        ax=axis(fig,gs,2,'EVIDENCE → ACTION  ·  force components')
        if model is not None and item['active']:
            nav=item['navigation'];labels=['Attraction','Soft shaping','Damping','Nominal','Applied']
            v=np.asarray(ep['trajectories'][who][item['trajectory_index']]['x'][2:]);forces=[nav.get('attraction_force'),nav.get('shaping_force'),(-1.4*v).tolist(),item.get('nominal'),item.get('control')]
            for k,force in enumerate(forces):
                if force is not None:ax.bar(k,float(np.linalg.norm(force)),color=[TEAL,PURPLE,MUTED,'#93b9df',BLUE][k])
            ax.set_xticks(range(5),labels,rotation=12);ax.set(ylim=(0,3.8),ylabel='Force magnitude (N)')
            deliveries=[e for e in frame['events'] if e['kind']=='delivered' and e['receiver']==who]
            accepted=sum(bool(e.get('accepted')) for e in deliveries)
            ax.text(.01,.95,f'{len(deliveries)} deliveries to agent {who}; {accepted} accepted model updates',transform=ax.transAxes,va='top',fontsize=8,color=MUTED)
        elif model is not None:absence(ax,'Agent finished; no subsequent command\nLast logged control model is retained above')
        else:absence(ax,'Carrier not active until 25 s\nPeer evidence can already be delivered')
        ax=axis(fig,gs,3,'OUTCOME  ·  '+('distance to goal' if kind=='reciprocity' else 'carrier cumulative weighted risk'))
        if kind=='reciprocity':
            for i in range(2):
                sam=[s for s in ep['samples'][i] if s['time']<=now+1e-10];ghost=[s for s in ref['samples'][i] if s['time']<=now+1e-10]
                for ss,col,ls in [(sam,COLORS[i],'-'),(ghost,MUTED,'--')]:
                    if ss:
                        q=np.asarray([s['x'][:2] for s in ss]);ax.plot([s['time'] for s in ss],np.linalg.norm(q-np.asarray(ep['spec']['goals'][i]),axis=1),color=col,ls=ls,lw=1.5)
            ax.set(xlim=(0,max(end,max(a['finish_time'] for a in ref['agents']))),ylim=(0,16),ylabel='Euclidean goal gap (m)',xlabel='Mission time (s)')
        else:
            for e,col,label in [(ep,BLUE,'Shared field'),(ref,ORANGE,"Scout's binary advice")]:
                seq=[r for r in e['trajectories'][1] if r['end_time']<=now+1e-10]
                if seq:ax.plot([r['end_time'] for r in seq],np.cumsum([r['components']['risk'] for r in seq]),color=col,label=label,lw=1.8)
            ax.set(xlim=(0,max(end,max(a['finish_time'] for a in ref['agents']))),ylim=(0,70),ylabel='Logged weighted risk component',xlabel='Mission time (s)')
            if now>25.:ax.legend(frameon=False)
        current=float(now);ax.axvline(current,color=PURPLE,lw=.7,ls=':')
        footer(fig,'Dashed world paths: matched reference. World truth is evaluator-only. Surface is the exact logged control model; unsupported regions masked. Applied arrows: 0.25 m/N.')
        return fig,{'mission_seconds':float(now),'agent_records':[{'agent':a['agent'],'trajectory_index':a.get('trajectory_index'),'logged_time':a.get('logged_time'),'model_version':a.get('model_version')} for a in frame['agents']],'surface':surface_record,'delivered_packets':len([e for e in frame['events'] if e['kind']=='delivered'])}
    return draw,end


def encounters(log):
    times=log['times']
    def draw(sample):
        fr=logs.encounter_frame(log,sample);now=fr['time'];fig,gs=figure(TITLES['encounters'],f'Native four-agent swap · seed {log["seed"]} · logged t = {now:.1f} s · programmed yielding/protection')
        ax=axis(fig,gs,0,'WORLD  ·  logged planar encounter',True);world3d(ax,[-1.5,1.5,-1.5,1.5])
        for layer,col,label in [('base',ORANGE,'Base protection'),('final',BLUE,'Final layer')]:
            rows=log['arms'][layer];q=rows['positions'][:sample+1]
            for i in range(4):ax.plot(q[:,i,0],q[:,i,1],zs=.03,color=col,lw=1.5,ls='--' if layer=='base' else '-')
            ax.scatter(q[-1,:,0],q[-1,:,1],zs=.04,c=col,s=38,label=label)
        goals=np.asarray(log['arms']['final']['goals']);ax.scatter(goals[:,0],goals[:,1],zs=.02,c=INK,marker='*',s=80);ax.legend(frameon=False,loc='upper left')
        ax=axis(fig,gs,1,'EXECUTION CONTEXT  ·  retained and unavailable signals')
        absence(ax,'Same starts, goals and nominal controller\n\nFinal layer: braking range + contact prediction\n+ least-violation fallback + deterministic priority\n\nForces, yielding events and model revisions\nwere not retained in this archive.')
        ax=axis(fig,gs,2,'INTERACTION  ·  sampled minimum pair separation')
        for layer,col,label in [('base',ORANGE,'Base'),('final',BLUE,'Final')]:
            q=log['arms'][layer]['positions'];sep=np.min(np.linalg.norm(q[:,np.array([0,0,0,1,1,2])]-q[:,np.array([1,2,3,2,3,3])],axis=-1),axis=1)
            ax.plot(times[:sample+1],sep[:sample+1],color=col,label=label)
        ax.axhline(.30,color='#b13838',ls=':',label='Contact distance');ax.set(xlim=(0,40),ylim=(0,2),xlabel='Recorded mission time (s)',ylabel='Separation (m)');ax.legend(frameon=False)
        ax=axis(fig,gs,3,'OUTCOME  ·  distance to assigned goals')
        for layer,col,label in [('base',ORANGE,'Base'),('final',BLUE,'Final')]:
            q=log['arms'][layer]['positions'];g=np.asarray(log['arms'][layer]['goals']);gap=np.linalg.norm(q-g[None],axis=-1).max(1)
            ax.plot(times[:sample+1],gap[:sample+1],color=col,label=label)
        ax.axhline(.15,color=TEAL,ls=':',label='Success radius');ax.set(xlim=(0,40),ylim=(0,2.2),xlabel='Recorded mission time (s)',ylabel='Maximum assigned goal gap (m)');ax.legend(frameon=False)
        footer(fig,'Recorded positions are subsampled after integration: t = 0, 0.1, 0.5, …, 39.7 s. Sampled separation is not a continuous safety certificate. No learned yielding is claimed.')
        return fig,{'mission_seconds':now,'logged_sample':int(sample),'time_semantics':'post-integration subsamples; no interpolation'}
    return draw


def uncertainty(saved,geometry):
    rows=saved['rows'];queries=geometry['queries'];n=len(queries)
    first=queries[0]['reps']['field']['anchors'];a=np.asarray(first);bounds=[a[:,0].min()-.7,a[:,0].max()+.7,a[:,1].min()-.7,a[:,1].max()+.7]
    XX,YY=np.meshgrid(np.linspace(bounds[0],bounds[1],40),np.linspace(bounds[2],bounds[3],40));points=np.c_[XX.ravel(),YY.ravel()]
    supported=np.linalg.norm(points[:,None]-a[None,:,:2],axis=-1).min(1)<=.65
    risk=terrain_values(points,list(map(tuple,a)),unknown=.02).reshape(XX.shape)
    def draw(pair):
        q=queries[pair];r=rows[pair];fig,gs=figure(TITLES['uncertainty'],f'Saved-state query {pair+1}/{n} · {saved["group"]} · query sequence, not mission time')
        ax=axis(fig,gs,0,'DECISION VIEW  ·  two supported candidate paths')
        ax.contourf(XX,YY,np.ma.masked_where(~supported.reshape(XX.shape),risk),levels=12,cmap='YlGnBu',alpha=.65)
        ax.scatter(a[:,0],a[:,1],c=INK,s=25,marker='s',label='Evidence anchors')
        for i,path in enumerate(q['paths']):ax.plot(path[:,0],path[:,1],color=[BLUE,ORANGE][i],lw=3,label=f'Path {i+1}')
        ax.set(xlim=bounds[:2],ylim=bounds[2:],xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');ax.legend(frameon=False)
        ax=axis(fig,gs,1,'MODEL  ·  supported shared-field route estimate',True)
        ax.plot_surface(XX,YY,np.where(supported.reshape(XX.shape),risk,np.nan),cmap='viridis',linewidth=0,vmin=0,vmax=.6)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='Estimated risk',zlim=(0,.6));ax.view_init(elev=34,azim=-62)
        ax.text2D(.01,.95,'Value–gradient anchor scorer, not physical terrain height',transform=ax.transAxes,color=MUTED,fontsize=8)
        ax=axis(fig,gs,2,'ESTIMATE → CHOICE  ·  signed path-cost difference')
        estimates=[q['reps'][name]['estimated'][0]-q['reps'][name]['estimated'][1] for name in ['field','raw','costmap']]
        ax.barh([0,1,2],estimates,color=[BLUE,MUTED,ORANGE]);ax.axvline(0,color=INK,lw=.8);truth=q['true'][0]-q['true'][1];ax.axvline(truth,color=TEAL,ls=':',lw=2,label='True difference (evaluation)')
        labels=['field','raw','costmap']
        for k,name in enumerate(labels):
            c=r['certificate'][name]
            ax.text(.02,k,'sample-certified' if c['certified'] else 'flagged',transform=ax.get_yaxis_transform(),va='center',fontsize=8,color=INK,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1.5})
        ax.set_yticks([0,1,2],labels);ax.set_xlabel('Cost(path 1) − cost(path 2)');ax.legend(frameon=False,loc='lower right',fontsize=8)
        ax=axis(fig,gs,3,'OUTCOME  ·  regret on the displayed query')
        names=['field','raw','costmap'];ax.bar(names,[r['decision'][name]['regret'] for name in names],color=[BLUE,MUTED,ORANGE]);ax.set_ylabel('True chosen cost − best true candidate cost');ax.set_ylim(0,max(.4,max(r['decision'][name]['regret'] for name in names)*1.15))
        footer(fig,'Original paths regenerated from frozen RNG and checked against archived costs. Certificate flags are archived. Sampling distributions were not retained and are not fabricated.')
        return fig,{'query_index':int(pair),'group':saved['group'],'paths_regenerated':True,'costs_verified':True,'physical_time':None,'certified':{name:r['certificate'][name]['certified'] for name in names}}
    return draw


def build(archive,site,frames=80,only=None):
    setup();archive=Path(archive).resolve();site=Path(site).resolve();out=site/'dist/assets';out.mkdir(parents=True,exist_ok=True)
    provenance={'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'extractor_sha256':hashlib.sha256(Path(logs.__file__).read_bytes()).hexdigest(),'version':1,'renderer':'scenario_presentation.py','physical_simulations':'planar','frames':{},'artifacts':{},'sources':{},'new_training_or_navigation_runs':False}
    jobs=list(TITLES) if only is None else only
    for name in jobs:
        if name=='dynamics':
            data=logs.load_dynamics(archive);draw=dynamics(data);steps=indices(401,frames,[199,200,208,216,224,232,240]);provenance['sources'][name]=data['sources']
        elif name in ['hazards','reciprocity']:
            data=logs.load_hazards(archive) if name=='hazards' else logs.load_crossing(archive);draw,end=stagewise(data,name)
            episode=data['episodes']['field' if name=='hazards' else 'reciprocal_fixed'];ref=data['episodes']['decision' if name=='hazards' else 'disabled'];end=max(end,max(a['finish_time'] for a in ref['agents']))
            important=[e['time'] for e in episode['events'] if e['kind']=='delivered' and e.get('force_change',0)>.1][:8]
            steps=sorted(set(np.linspace(0,end,frames).tolist()+important+([25.] if name=='hazards' else [4.8,21.12])))
            provenance['sources'][name]=data['sources']
        elif name=='encounters':
            data=logs.load_encounters(archive);draw=encounters(data);steps=indices(101,frames,[0,1,100]);provenance['sources'][name]=data['sources']
        else:
            data=logs.load_uncertainty(archive,pairs=12);geo=logs.regenerate_uncertainty_geometry(archive,pairs=12);draw=uncertainty(data,geo);steps=[k for k in range(12) for _ in range(5)];provenance['sources'][name]=data['sources']+[geo['source']]
        render_video(name,draw,steps,out,provenance)
        provenance['sources'][name].append({'selection':data.get('selection',data.get('semantics','manifest-prescribed example'))})
        write(out/'provenance.json',provenance)
    return provenance


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--site',type=Path,required=True);ap.add_argument('--frames',type=int,default=80);ap.add_argument('--only',nargs='+',choices=list(TITLES));args=ap.parse_args(argv)
    if args.frames<12:ap.error('--frames must be at least 12')
    build(args.archive,args.site,args.frames,args.only)

if __name__=='__main__':main()
