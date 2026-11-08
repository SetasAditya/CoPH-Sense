"""Read-only rendering of logged control-time fields, force components and packets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .evaluation import load
from .records import Config
from .maze_visuals import landscape,record_at
from .model import potential,tensor


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def network_distance(points,spec):
    nodes=np.asarray(spec['network_nodes']);distance=np.full(len(points),np.inf)
    for i,j in spec['network_edges']:
        a=nodes[i];d=nodes[j]-a;t=np.clip((points-a)@d/(d@d),0,1)
        distance=np.minimum(distance,np.linalg.norm(points-a-t[:,None]*d,axis=1))
    return distance


def validate(r,cfg):
    rows=[]
    for agent,records in enumerate(r['trajectories']):
        for t in records:
            if 'control_model' not in t or 'navigation' not in t:
                rows.append(dict(agent=agent,time=t['time'],available=False,reason='missing_logged_control_context'));continue
            m=t['control_model'];q=tensor(t['x'][:2]).requires_grad_(True)
            v=potential(q,tensor(m['target']),tensor(m['weights']),tensor(m['centers']),cfg.rbf_width,tensor(m['policy']),m['geometry'],cfg)
            conservative=-torch.autograd.grad(v,q)[0].detach().numpy();damping=-1.4*np.asarray(t['x'][2:])
            nominal=conservative+damping;logged=np.asarray(t['navigation'].get('nominal',t['nominal']))
            attraction=cfg.shaping_k*(np.asarray(m['target'])-t['x'][:2]);shaping=conservative-attraction
            rows.append(dict(agent=agent,time=t['time'],tick=t['tick'],stage=m['stage'],version=m['version'],available=True,
                nominal_error=float(np.linalg.norm(nominal-logged)),potential=float(v),kinetic=float(.5*np.dot(t['x'][2:],t['x'][2:])),
                attraction_error=float(np.linalg.norm(attraction-np.asarray(t['navigation']['attraction_force']))),
                shaping_error=float(np.linalg.norm(shaping-np.asarray(t['navigation']['shaping_force']))),
                unclipped=bool(np.linalg.norm(nominal)<cfg.force_limit-1e-8)))
    available=[r for r in rows if r['available']]
    assert available and max(r['nominal_error'] for r in available)<1e-9
    assert max(r['shaping_error'] for r in available)<1e-9
    return rows


def draw(r,cfg,now,validated):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle,FancyArrowPatch
    from PIL import Image
    colors=['#0072B2','#009E73'];labels=['Scout A0','Carrier A1'];spec=r['spec'];nodes=np.asarray(spec['network_nodes']);rad=spec['corridor_radius']
    fig=plt.figure(figsize=(15,9));gs=fig.add_gridspec(3,3,height_ratios=[1.1,1.1,.8]);world=fig.add_subplot(gs[:2,:2]);world.set_facecolor('#343a40')
    for i,j in spec['network_edges']:
        a,b=nodes[i],nodes[j];lo=np.minimum(a,b);horizontal=abs(a[1]-b[1])<1e-9
        origin=lo-[0,rad] if horizontal else lo-[rad,0]
        world.add_patch(Rectangle(origin,abs(b[0]-a[0]) if horizontal else 2*rad,2*rad if horizontal else abs(b[1]-a[1]),color='#eceff1',zorder=0))
        for p in (a,b):world.add_patch(Circle(p,rad,color='#eceff1',zorder=0))
    gx,gy=np.meshgrid(np.linspace(nodes[:,0].min()-rad,nodes[:,0].max()+rad,130),np.linspace(nodes[:,1].min()-rad,nodes[:,1].max()+rad,100));pts=np.c_[gx.ravel(),gy.ravel()];truth=np.full(len(pts),.02)
    for x,y,sx,sy,amp in spec['patches']:truth+=amp*np.exp(-.5*(((pts[:,0]-x)/sx)**2+((pts[:,1]-y)/sy)**2))
    mask=network_distance(pts,spec)>rad
    world.pcolormesh(gx,gy,np.ma.masked_where(mask.reshape(gx.shape),truth.reshape(gx.shape)),cmap='YlOrRd',vmin=.02,vmax=.3,alpha=.65,shading='auto',zorder=1)
    records=[record_at(r,i,now) for i in range(2)]
    for i in range(2):
        ss=[s for s in r['samples'][i] if s['time']<=now+1e-9]
        if not ss:continue
        xy=np.asarray([s['x'][:2] for s in ss]);world.plot(*xy.T,color=colors[i],lw=2.2,zorder=3);world.scatter(*xy[-1],c=colors[i],s=50,zorder=4)
        status='GOAL' if ss[-1]['done'] and r['agents'][i]['success'] else 'ACTIVE'
        world.annotate(labels[i]+' '+status,xy[-1],xytext=(5,10+12*i),textcoords='offset points',color=colors[i],fontsize=9)
        if records[i]:
            tr=records[i];path=np.asarray(tr['navigation'].get('route',[]))
            if len(path):world.plot(*path.T,color=colors[i],ls=':',alpha=.7)
            world.scatter(*tr['control_model']['target'],marker='x',color=colors[i],s=50)
    for e in r['events']:
        if e['kind']=='delivered' and 0<=now-e['time']<=1.5:
            world.add_patch(FancyArrowPatch(e['sender_q'],e['receiver_q'],arrowstyle='->',mutation_scale=15,color='#9c27b0',connectionstyle='arc3,rad=.12',lw=1.7,zorder=5))
            anchors=np.asarray(e.get('anchors',[]))
            if len(anchors):world.scatter(anchors[:,0],anchors[:,1],s=24,marker='+',color='#9c27b0',zorder=6)
    world.set(xlim=(-1,13),ylim=(-1,8.5),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Evaluator world risk, executed paths, logged routes and exchanged anchors\nSurveyed corridors; soft hazards are traversable; carrier starts at 25 s')
    for i,tr in enumerate(records):
        ax=fig.add_subplot(gs[i,2]);ax.set_facecolor('#b8b8b8')
        if tr is None:ax.text(.5,.5,'Inactive; no control-time field yet',ha='center',transform=ax.transAxes);ax.axis('off');continue
        m=tr['control_model'];center=np.asarray(m['center']);xs=np.linspace(center[0]-1.5,center[0]+1.5,36);ys=np.linspace(center[1]-2,center[1]+2,40);xx,yy=np.meshgrid(xs,ys);points=np.c_[xx.ravel(),yy.ravel()]
        H,F,supported=landscape(m,points,np.asarray(tr['x'][2:]),cfg);K=.5*np.dot(tr['x'][2:],tr['x'][2:]);V=H-K
        supported &= network_distance(points,spec)<=rad-.12;values=np.ma.masked_where(~supported.reshape(xx.shape),V.reshape(xx.shape));maximum=max(.05,float(np.max(V[supported]))) if supported.any() else 1.
        im=ax.pcolormesh(xx,yy,values,shading='auto',cmap='viridis',vmin=0,vmax=maximum);fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Navigation V (J)')
        ids=np.flatnonzero(supported)[::65]
        ax.quiver(points[ids,0],points[ids,1],F[ids,0],F[ids,1],color='white',alpha=.45,angles='xy',scale_units='xy',scale=4,width=.004)
        q=np.asarray(tr['x'][:2]);vectors=[tr['navigation']['attraction_force'],tr['navigation']['shaping_force'],-1.4*np.asarray(tr['x'][2:]),tr['u']]
        for vector,color,label in zip(vectors,['#e53935','#fb8c00','#999999','#111111'],['attraction','soft shaping','damping','applied']):ax.quiver(*q,*vector,color=color,angles='xy',scale_units='xy',scale=1,width=.008,label=label)
        ax.scatter(*q,s=25,color=colors[i],edgecolor='white');target=m['target'];ax.scatter(*target,marker='x',color='#e53935',s=35)
        ax.set(xlim=(xs[0],xs[-1]),ylim=(ys[0],ys[-1]),aspect='equal')
        ax.set_title('%s V | stage %d, v%d\nlogged %.2f s; K=%.3f J'%(labels[i],m['stage'],m['version'],tr['time'],K),fontsize=9)
        if i==0:ax.legend(loc='upper left',fontsize=6,ncol=2)
    force=fig.add_subplot(gs[2,0]);energy=fig.add_subplot(gs[2,1]);communication=fig.add_subplot(gs[2,2]);end=max(a['finish_time'] for a in r['agents'])
    for i in range(2):
        ts=[t for t in r['trajectories'][i] if t['time']<=now];style='-' if i==0 else '--'
        if not ts:continue
        for key,color,label in [('attraction_force','#e53935','attraction'),('shaping_force','#fb8c00','soft shaping')]:force.plot([t['time'] for t in ts],[np.linalg.norm(t['navigation'][key]) for t in ts],style,c=color,label=label if i==0 else None,lw=1)
        force.plot([t['time'] for t in ts],[np.linalg.norm(t['u']) for t in ts],style,c='black',label='applied' if i==0 else None,lw=1)
        vv=[v for v in validated if v['agent']==i and v['time']<=now and v['available']]
        energy.plot([v['time'] for v in vv],[v['potential'] for v in vv],style,c=colors[i],label=labels[i]+' V')
        energy.plot([v['time'] for v in vv],[v['kinetic'] for v in vv],':',c=colors[i],label=labels[i]+' K')
    force.set(xlim=(0,end),xlabel='Global time (s)',ylabel='Force norm (N)',title='Scout solid; carrier dashed');force.legend(fontsize=6,ncol=3)
    energy.set(xlim=(0,end),xlabel='Global time (s)',ylabel='Energy (J)',title='Physical K and navigation V separately');energy.legend(fontsize=6,ncol=2)
    delivered=[e for e in r['events'] if e['kind']=='delivered' and e['time']<=now]
    for sender in (0,1):
        e=[v for v in delivered if int(v['sender'])==sender];communication.scatter([v['time'] for v in e],[sender]*len(e),s=[24 if v.get('accepted') else 10 for v in e],c=colors[sender],marker='|')
    communication.axvline(now,c='black',lw=.8);communication.set(xlim=(0,end),ylim=(-.5,1.5),yticks=[0,1],yticklabels=['A0→A1','A1→A0'],xlabel='Global time (s)',title='Delivered packets; larger = accepted update')
    fig.suptitle('Logged cooperative Hamiltonian pipeline | t=%.2f s\nGray field cells: unsupported or outside surveyed free corridor. Field panels use the last logged control context.'%now,fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93]);fig.canvas.draw();image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    episode=root/'episodes/hazard_double_mild_89400_field_4000000.json.gz';r=load(episode);cfg=Config(**r['config']);out=root/'figures';out.mkdir(exist_ok=True)
    end=max(a['finish_time'] for a in r['agents']);times=np.linspace(0.,end,60).tolist()
    files=[Path(__file__),Path(__file__).with_name('maze_visuals.py'),Path(__file__).with_name('model.py')]
    protocol=dict(input_sha256=digest(episode),episode=episode.name,frames=60,frame_rule='uniform closed interval [0,max logged finish_time]',times=times,
        static_frame_index=30,source_hashes={p.name:digest(p) for p in files},render_only=True)
    (out/'hamiltonian_render_protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter();torch.set_num_threads(1)
    rows=validate(r,cfg);frames=[];checks=[]
    for k,now in enumerate(times):
        frames.append(draw(r,cfg,now,rows));selected=[]
        for i in range(2):
            tr=record_at(r,i,now);selected.append(None if tr is None else dict(agent=i,time=tr['time'],tick=tr['tick'],stage=tr['control_model']['stage'],version=tr['control_model']['version']))
            if tr is not None:assert tr['time']<=now+1e-9
        checks.append(dict(frame=k,global_time=now,selected_records=selected))
    frames[30].save(out/'hamiltonian_pipeline.png');frames[0].save(out/'hamiltonian_pipeline.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
    available=[v for v in rows if v['available']]
    verification=dict(protocol=protocol,frames=checks,records_checked=len(rows),available=len(available),missing=len(rows)-len(available),
        max_nominal_error=max(v['nominal_error'] for v in available),max_shaping_error=max(v['shaping_error'] for v in available),
        unclipped_records=sum(v['unclipped'] for v in available),seconds=time.perf_counter()-start,operational_input_unchanged=digest(episode)==protocol['input_sha256'],
        limitations=['Uniform frames may omit short communication events; timeline retains all delivered events up to the displayed time.',
            'Panels show exact current-stage logged navigation shaping, not a global posterior or remote anchor route-cost surface.',
            'After goal completion the last logged control context is held and its timestamp displayed; no post-terminal field is inferred.',
            'K is physical kinetic energy; V is navigation shaping. Their plotted sum is not claimed as a physical conservation law.',
            'No invented replay paths: policy replay is disabled in this representation experiment.'])
    (out/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps({k:v for k,v in verification.items() if k not in ('protocol','frames')},indent=2))


if __name__=='__main__':main()
