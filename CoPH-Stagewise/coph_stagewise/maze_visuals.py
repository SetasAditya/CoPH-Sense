"""Read-only displays of exact logged controller models, not reconstructed beliefs."""
from pathlib import Path
import numpy as np
import torch
from .evaluation import load,dump
from .records import Config
from .model import tensor,shaping_parameters
from .maze_geometry import NODES,EDGES,CORRIDOR_RADIUS,distance_to_network


def landscape(model,points,velocity,cfg):
    if model.get('stage_blend_mode')=='smooth' or model.get('local_residual') is not None:
        return extended_landscape(model,points,velocity,cfg)
    q=tensor(points).requires_grad_(True);weights=tensor(model['weights']);centers=tensor(model['centers']);policy=tensor(model['policy'])
    d2=(q-tensor(model['target'])).square().sum(-1)
    features=torch.exp(-.5*((q[:,None,:]-centers)/cfg.rbf_width).square().sum(-1))
    shaping=policy[0]*torch.nn.functional.softplus(features@weights)/4
    obs=model['geometry']
    if obs:
        if any(len(z)==5 for z in obs):
            from .shapes import surface_distance
            clearance=surface_distance(q,obs)-.12
        else:
            o=tensor(obs);clearance=torch.linalg.vector_norm(q[:,None,:]-o[None,:,:2],dim=-1)-o[:,2]-.12
        shaping=shaping+policy[1]*(torch.nn.functional.softplus((.35-clearance)*6).square()/36).sum(-1)
    params=shaping_parameters(weights,cfg.rbf_width,obs,cfg)
    V=.5*cfg.shaping_k*d2+params['alpha']*d2/(d2+cfg.shaping_ell**2)*shaping
    force=-torch.autograd.grad(V.sum(),q)[0]
    support=np.asarray(model['support']);pts=np.asarray(points);center=np.asarray(model['center'])
    supported=(np.abs(pts[:,0]-center[0])<=cfg.stage_width/2)&(np.abs(pts[:,1]-center[1])<=cfg.stage_height/2)
    if len(support):supported &= np.min(np.linalg.norm(pts[:,None,:]-support[None,:,:],axis=-1),axis=1)<=.65
    else:supported[:]=False
    return (V.detach().numpy()+.5*np.dot(velocity,velocity)),force.detach().numpy(),supported


def extended_landscape(model,points,velocity,cfg):
    """Exact logged opt-in scalar blend/residual; legacy rendering is unchanged."""
    from .model import risk
    q=tensor(points).requires_grad_(True);policy=tensor(model['policy']);obs=model['geometry'];target=tensor(model['target'])
    d2=(q-target).square().sum(-1);barrier=q[:,0]*0.
    if obs:
        if any(len(z)==5 for z in obs):
            from .shapes import surface_distance
            clearance=surface_distance(q,obs)-.12
        else:
            o=tensor(obs);clearance=torch.linalg.vector_norm(q[:,None,:]-o[None,:,:2],dim=-1)-o[:,2]-.12
        barrier=policy[1]*(torch.nn.functional.softplus((.35-clearance)*6).square()/36).sum(-1)
    stages=model.get('blend_models',[model]);energies=[];certificates=[];raw=[];points=np.asarray(points)
    for stage in stages:
        weights=tensor(stage['weights']);centers=tensor(stage['centers']);residual=stage.get('local_residual')
        shaping=policy[0]*risk(q,weights,centers,cfg.rbf_width,residual=residual)+barrier
        if cfg.shaping_mode=='progress':
            parameters=shaping_parameters(weights,cfg.rbf_width,obs,cfg,residual=residual)
            energies.append(.5*cfg.shaping_k*d2+parameters['alpha']*d2/(d2+cfg.shaping_ell**2)*shaping)
        else:energies.append(.5*d2+shaping)
        support=np.asarray(stage['support']);center=np.asarray(stage['center']);inside=(np.abs(points[:,0]-center[0])<=cfg.stage_width/2+1e-10)&(np.abs(points[:,1]-center[1])<=cfg.stage_height/2+1e-10)
        if len(support):inside &= np.min(np.linalg.norm(points[:,None,:]-support[None,:,:],axis=-1),axis=1)<=.65
        else:inside[:]=False
        certificates.append(inside)
        z=((tensor([cfg.stage_width/2,cfg.stage_height/2])-(q-tensor(center)).abs())/max(min(cfg.stage_width,cfg.stage_height)*min(cfg.overlap,.5),1e-12)).clamp(0.,1.)
        raw.append((z.square()*(3-2*z)).prod(-1)*tensor(inside.astype(float)))
    supported=np.any(certificates,axis=0)
    if model.get('stage_blend_mode')=='smooth':
        raw=torch.stack(raw);total=raw.sum(0);members=tensor(np.sum(certificates,axis=0));singleton=members==1
        weights=torch.where(singleton[None,:],tensor(np.asarray(certificates,float)),raw/total.clamp_min(1e-20))
        supported &= singleton.detach().numpy()|(total.detach().numpy()>1e-20)
        V=(weights*torch.stack(energies)).sum(0)
    else:V=energies[0]
    force=-torch.autograd.grad(V.sum(),q)[0]
    return V.detach().numpy()+.5*np.dot(velocity,velocity),force.detach().numpy(),supported


def maze_background(ax):
    from matplotlib.patches import Rectangle,Circle
    ax.set_facecolor('#303741')
    for i,j in EDGES:
        a,b=np.asarray(NODES[i]),np.asarray(NODES[j]);lo=np.minimum(a,b)
        horizontal=abs(a[1]-b[1])<1e-9
        corner=lo-[0,CORRIDOR_RADIUS] if horizontal else lo-[CORRIDOR_RADIUS,0]
        width=abs(b[0]-a[0]) if horizontal else 2*CORRIDOR_RADIUS
        height=2*CORRIDOR_RADIUS if horizontal else abs(b[1]-a[1])
        ax.add_patch(Rectangle(corner,width,height,facecolor='#e9ebed',edgecolor='none',zorder=0))
        for z in (a,b):ax.add_patch(Circle(z,CORRIDOR_RADIUS,facecolor='#e9ebed',edgecolor='none',zorder=0))
        ax.plot([a[0],b[0]],[a[1],b[1]],'--',color='#b8bfc6',lw=.7,zorder=1)
    # Exact union of the surveyed corridor capsules used by the simulator.
    ax.set(xlim=(-1,13),ylim=(-1,7),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
    ax.scatter(0,0,marker='s',color='#555',s=80);ax.text(0,-.55,'START',ha='center',fontsize=9)
    ax.scatter(12,0,marker='*',color='#43a047',s=150);ax.text(12,-.55,'GOAL',ha='center',fontsize=9)


def record_at(r,i,now):
    records=r['trajectories'][i];ids=[k for k,t in enumerate(records) if t['time']<=now+1e-9]
    return records[ids[-1]] if ids else None


def frame(r,now):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,FancyArrowPatch
    from PIL import Image
    cfg=Config(**r['config']);fig=plt.figure(figsize=(17,11));gs=fig.add_gridspec(3,3,width_ratios=[1.15,1.15,1],height_ratios=[1,1,.8])
    world=fig.add_subplot(gs[:2,:2]);maze_background(world);colors=['#0072B2','#009E73'];labels=['Scout A0','Carrier A1'];records=[record_at(r,i,now) for i in range(2)]
    discovered=set()
    for i in range(2):
        for event in r['geometry_events'][i]:
            if event['time']<=now and event['kind']=='locally_sensed':discovered.add(tuple(event['obstacle']))
    for ox,oy,rad in r['spec']['obstacles']:
        sensed=(ox,oy,rad) in discovered
        world.add_patch(Circle((ox,oy),rad,color='#d64545' if sensed else '#8e949c',zorder=3))
    for k,(gx,gy,_) in enumerate(r['spec']['gate_groups']):world.text(gx+.45,gy+.85,f'G{k+1}',color='#8b2424',fontsize=10,weight='bold')
    if r['spec'].get('patches'):
        xx,yy=np.meshgrid(np.linspace(-1,13,180),np.linspace(-1,7,100))
        truth=np.zeros_like(xx)
        for px,py,sx,sy,amp in r['spec']['patches']:
            truth+=amp*np.exp(-.5*(((xx-px)/sx)**2+((yy-py)/sy)**2))
        inside=distance_to_network(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)<=CORRIDOR_RADIUS
        world.contourf(xx,yy,np.ma.masked_where(~inside,truth),levels=[.1,.5,1,2,4.1],cmap='YlOrRd',alpha=.5,zorder=2)
    for i in range(2):
        ss=[s for s in r['samples'][i] if s['time']<=now]
        if not ss:
            world.text(0,.35,'Carrier starts at 5 s',fontsize=9);continue
        p=np.asarray([s['x'][:2] for s in ss]);q=p[-1];world.plot(*p.T,color=colors[i],lw=2.5,zorder=5);world.scatter(*q,color=colors[i],s=70,zorder=7)
        status='GOAL' if ss[-1]['done'] and r['agents'][i]['success'] else 'FAILED' if ss[-1]['done'] else ''
        world.annotate(labels[i]+' '+status,q,xytext=(5,12+10*i),textcoords='offset points',color=colors[i],fontsize=10,weight='bold',zorder=9)
        t=records[i]
        if t:
            path=np.asarray(t['navigation'].get('route',[]))
            if len(path):world.plot(*path.T,color=colors[i],alpha=.4,ls=':',lw=2)
            target=t['control_model']['target'];world.scatter(*target,color=colors[i],marker='x',s=60)
    for event in [e for e in r['events'] if e['kind']=='delivered' and 0<=now-e['time']<=1.]:
        world.add_patch(FancyArrowPatch(event['sender_q'],event['receiver_q'],arrowstyle='->',mutation_scale=18,color='#a140b0',connectionstyle='arc3,rad=.18',lw=2,zorder=10))
    world.set_title('Cooperative maze | world truth, executed paths and current plans\nGray gate = not yet sensed; red section = sensed. Closures are hidden from agents.',fontsize=12)
    if r['spec'].get('patches'):world.set_title('Soft-risk maze | evaluator risk truth, executed paths and plans\nAll corridors traversable. Risk truth is not available to learners.',fontsize=12)
    force_colors=['#c62828','#ed8b23','#808080','#151515'];force_labels=['attraction','soft shaping','damping','applied']
    for i in range(2):
        ax=fig.add_subplot(gs[i,2]);t=records[i]
        if not t:ax.text(.5,.5,'Carrier inactive until 5 s',ha='center');ax.axis('off');continue
        model=t['control_model'];center=np.asarray(model['center']);xs=np.linspace(center[0]-1.5,center[0]+1.5,45);ys=np.linspace(center[1]-2,center[1]+2,55);xx,yy=np.meshgrid(xs,ys);points=np.c_[xx.ravel(),yy.ravel()]
        H,F,mask=landscape(model,points,np.asarray(t['x'][2:]),cfg);mask &= distance_to_network(points)<=CORRIDOR_RADIUS-.12;values=np.ma.masked_where(~mask.reshape(xx.shape),H.reshape(xx.shape))
        ax.set_facecolor('#b7b7b7');im=ax.pcolormesh(xx,yy,values,vmin=0,vmax=8,cmap='viridis',shading='auto');fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='H(q, p at logged step) [J]')
        ax.contour(xx,yy,values,levels=[.1,.25,.5,1,2,4,6],colors='white',linewidths=.45,alpha=.5)
        ids=np.flatnonzero(mask)[::70];ax.quiver(points[ids,0],points[ids,1],F[ids,0],F[ids,1],color='white',alpha=.4,angles='xy',scale_units='xy',scale=4,width=.004)
        q=np.asarray(t['x'][:2]);vectors=[t['navigation'].get('attraction_force',[0,0]),t['navigation'].get('shaping_force',[0,0]),-1.4*np.asarray(t['x'][2:]),t['u']]
        for v,col,label in zip(vectors,force_colors,force_labels):ax.quiver(*q,*v,color=col,angles='xy',scale_units='xy',scale=1,label=label,width=.009,zorder=8)
        for ox,oy,rad in model['geometry']:ax.add_patch(Circle((ox,oy),rad,color='#d64545',alpha=.7))
        ax.scatter(*q,color=colors[i],edgecolor='white',s=35,zorder=9);ax.scatter(*model['target'],marker='x',color='red',s=35)
        ax.set(xlim=(xs[0],xs[-1]),ylim=(ys[0],ys[-1]));ax.set_aspect('equal');ax.set_title(f'{labels[i]} Hamiltonian slice\nstage {model["stage"]}, model v{model["version"]}, logged {t["time"]:.2f}s',fontsize=10)
        if i==0:ax.legend(loc='upper left',fontsize=7,ncol=2)
    forceax=fig.add_subplot(gs[2,0]);energyax=fig.add_subplot(gs[2,1]);eventax=fig.add_subplot(gs[2,2]);eventax.axis('off')
    for i in range(2):
        ts=[t for t in r['trajectories'][i] if t['time']<=now]
        if not ts:continue
        times=[t['time'] for t in ts];style='-' if i==0 else '--'
        for key,col,label in [('attraction_force',force_colors[0],'attraction'),('shaping_force',force_colors[1],'soft shaping')]:forceax.plot(times,[np.linalg.norm(t['navigation'].get(key,[0,0])) for t in ts],style,color=col,lw=1,label=label if i==0 else None)
        forceax.plot(times,[np.linalg.norm(t['u']) for t in ts],style,color='black',label='applied' if i==0 else None)
        # Only evaluate exact logged state, never invent between-update model history.
        H=[];K=[]
        for t in ts[::max(1,len(ts)//100)]:
            h,_,_=landscape(t['control_model'],np.array([t['x'][:2]]),np.array(t['x'][2:]),cfg);H.append((t['time'],h[0]));K.append((t['time'],.5*np.dot(t['x'][2:],t['x'][2:])))
        energyax.plot(*np.array(H).T,style,color=colors[i],label=labels[i]+' H');energyax.plot(*np.array(K).T,':',color=colors[i],alpha=.65,label=labels[i]+' kinetic')
    end=max(a['finish_time'] for a in r['agents']);forceax.set(xlim=(0,end),xlabel='Global time (s)',ylabel='Force (N)',title='Forces: scout solid, carrier dashed');forceax.legend(fontsize=7,ncol=3)
    energyax.set(xlim=(0,end),xlabel='Global time (s)',ylabel='Energy (J)',title='Shaping H versus physical kinetic energy');energyax.legend(fontsize=7,ncol=2)
    events=sorted([dict(e,agent=i) for i,ee in enumerate(r['geometry_events']) for e in ee if e['time']<=now],key=lambda e:e['time'])
    lines=['DISCOVERY → REPORT → REROUTE','']
    seen=set();selected=[]
    for e in reversed(events):
        key=(e['agent'],e['kind'],round(e['obstacle'][0],1))
        if key not in seen:seen.add(key);selected.append(e)
        if len(selected)==5:break
    for e in reversed(selected):lines.append(f'{e["time"]:6.2f}s A{e["agent"]} '+('sensed' if e['kind']=='locally_sensed' else 'received')+f' gate x={e["obstacle"][0]:.1f}')
    if r.get('attribution_arm'):
        lines+=['', 'RECENT PACKET DELIVERIES']
        for e in [e for e in r['events'] if e['kind']=='delivered' and e['time']<=now][-3:]:
            used=[k.replace('field_assimilation','field').replace('route_anchors','route') for k,v in e.get('components',{}).items() if v['used']]
            lines.append(f"{e['time']:.2f}s A{e['sender']}→A{e['receiver']}: {','.join(used) or e['reason']}")
    lines+=['','Gray landscape: unsupported or wall.','White −∇V arrows: 0.25 m per N.','Colored force arrows: 1 m per N.','H = kinetic + navigation shaping;','shaping jumps are not actuator work.']
    eventax.text(0,1,'\n'.join(lines),va='top',fontsize=9,family='monospace')
    fig.suptitle(f'Scout–carrier maze | global time {now:.2f} s | second agent starts at 5.00 s',fontsize=16);fig.tight_layout(rect=[0,0,1,.97]);fig.canvas.draw()
    image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return image


def metrics(r):
    gates=[]
    for gx,gy,_ in r['spec']['gate_groups']:
        per=[]
        for i in range(2):
            events=[e for e in r['geometry_events'][i] if abs(e['obstacle'][0]-gx)<.01 and abs(e['obstacle'][1]-gy)<=.6]
            sensed=min((e['time'] for e in events if e['kind']=='locally_sensed'),default=None)
            received=min((e['time'] for e in events if e['kind']=='received_geometry'),default=None)
            per.append(dict(agent=i,sensed=sensed,received=received,remote_before_local=received is not None and (sensed is None or received<sensed)))
        gates.append(dict(gate=[gx,gy],agents=per))
    return dict(seed=r['spec']['seed'],arm=r['arm'],agents=r['agents'],gates=gates,goals=sum(a['success'] for a in r['agents']),collisions=sum(a['collision'] for a in r['agents']),timeouts=sum(a['timeout'] for a in r['agents']),bytes=sum(sum(a['bytes'].values()) for a in r['agents']),wall_seconds=r['wall_seconds'])


def report(root):
    root=Path(root);(root/'figures').mkdir(exist_ok=True);rows=[];failures=[];selected=None
    for p in sorted((root/'episodes').glob('*.json.gz')):
        r=load(p);row=metrics(r);rows.append(row)
        if row['goals']<2:failures.append(dict(file=p.name,metrics=row))
        if r['spec']['seed']==75000 and r['arm']=='reciprocal_fixed':selected={k:r[k] for k in ('config','spec','agents','geometry_events','trajectories','samples','events')}
        del r
    dump(root/'rows.json',rows);dump(root/'failures.json',failures)
    end=max(a['finish_time'] for a in selected['agents']);events=sorted({e['time'] for ee in selected['geometry_events'] for e in ee})
    times=sorted({0.,5.,*np.linspace(0,end,90).tolist(),*[min(end,max(0,t+d)) for t in events for d in (-.3,0,.6)]})
    from PIL import Image
    frames=[frame(selected,t).convert('P',palette=Image.Palette.ADAPTIVE,colors=256) for t in times];frames[0].save(root/'figures/cooperative_maze.gif',save_all=True,append_images=frames[1:],duration=180,loop=0)
    first_warning=min((e['time'] for es in selected['geometry_events'] for e in es if e['kind']=='received_geometry'),default=0.)
    key_frame=min(range(len(times)),key=lambda k:abs(times[k]-(first_warning+.6)))
    frames[key_frame].save(root/'figures/first_warning.png')
    frames[-1].save(root/'figures/final.png');dump(root/'figures/frames.json',dict(times=times,seed=75000,arm='reciprocal_fixed',interpretation='Boundary-state logs, compressed playback; no interpolated model states.'))
    lines=['# Cooperative maze with hidden closures','','Single-run visualization shows both agents, discovered gates, plans, exact logged H slices, forces, and separate physical kinetic energy. Communication-off runs remain diagnostic only.','','| Seed | Arm | Goals | Collisions | Timeouts | Bytes |','|---|---|---|---|---|---|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["arm"]} | {r["goals"]}/2 | {r["collisions"]} | {r["timeouts"]} | {r["bytes"]} |')
    lines+=['','![Maze](figures/cooperative_maze.gif)','','This is a static development demonstration with surveyed corridors and unknown physical closures. Geometry reports are a new explicit packet capability, separate from environmental field estimates and local measurements. No inter-agent contact dynamics or adaptive communication rates. Unsupported landscape regions are gray. Remote geometry does not certify environmental replay support. All failures remain in failures.json and full episode records.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
