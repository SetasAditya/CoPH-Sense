"""Read-only displays of logged maze episodes, driven entirely by the scene spec.
Door truth is evaluator-only; it is drawn, never given to learners."""
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .route_oracle import true_distance,shortest

COLORS=('#0072B2','#D55E00')
ARMS=('disabled','one_way_01','one_way_10','reciprocal_fixed')
ARM_LABELS=dict(disabled='Communication OFF',one_way_01='One-way A → B',one_way_10='One-way B → A',reciprocal_fixed='Reciprocal communication')


def doors_of(spec):
    return [tuple(o) for o in spec['obstacles'] if len(o)==5]


def door_name(spec,obstacle):
    """'door k' for a slab in the spec, None for anything else."""
    key=tuple(obstacle);doors=doors_of(spec)
    return f'door {doors.index(key)+1}' if key in doors else None


def path_length(samples):
    p=np.array([s['x'][:2] for s in samples]);return float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)))


def door_events(r):
    out={}
    for i,events in enumerate(r['geometry_events']):
        for e in events:
            name=door_name(r['spec'],e['obstacle'])
            if name is None:continue
            key=(name,i,'sensed' if e['kind']=='locally_sensed' else 'received');out[key]=min(out.get(key,np.inf),e['time'])
    return out


def metrics(r):
    spec=r['spec'];events=door_events(r);agents=[]
    for i,a in enumerate(r['agents']):
        length=path_length(r['samples'][i]);best=shortest(spec['network_nodes'],spec['network_edges'],doors_of(spec),spec['initials'][i],spec['goals'][i])
        agents.append(dict(agent=i,success=a['success'],collision=a['collision'],timeout=a['timeout'],finish_time=a['finish_time'],path=length,wasted=length-best,shortest=best,
            doors={name:dict(sensed=events.get((name,i,'sensed')),received=events.get((name,i,'received'))) for name in {k[0] for k in events}},bytes=sum(a['bytes'].values())))
    return dict(seed=spec['seed'],arm=r['arm'],agents=agents,goals=sum(a['success'] for a in agents),team_wasted=sum(a['wasted'] for a in agents),
                team_finish=max(a['finish_time'] for a in agents),bytes=sum(a['bytes'] for a in agents),wall_seconds=r['wall_seconds'])


def background(ax,spec,fontsize=8.5):
    from matplotlib.patches import Polygon,Circle
    nodes=np.asarray(spec['network_nodes'],float);r=spec['corridor_radius'];ax.set_facecolor('#303741')
    for i,j in spec['network_edges']:
        a,b=nodes[i],nodes[j];d=b-a;n=np.array([-d[1],d[0]])/np.linalg.norm(d)*r
        ax.add_patch(Polygon([a+n,b+n,b-n,a-n],facecolor='#e9ebed',edgecolor='none',zorder=0))
        for z in (a,b):ax.add_patch(Circle(z,r,facecolor='#e9ebed',edgecolor='none',zorder=0))
    lo,hi=nodes.min(0)-r-1.,nodes.max(0)+r+1.
    if spec.get('patches'):
        # Evaluator-only true risk, drawn inside the corridors.
        from .maze_geometry import distance_to_network
        xs=np.linspace(lo[0],hi[0],240);ys=np.linspace(lo[1],hi[1],160);xx,yy=np.meshgrid(xs,ys);pts=np.c_[xx.ravel(),yy.ravel()]
        risk=sum(a*np.exp(-.5*((pts[:,0]-x)/sx)**2-.5*((pts[:,1]-y)/sy)**2) for x,y,sx,sy,a in spec['patches'])
        inside=distance_to_network(pts,spec['network_nodes'],spec['network_edges'])<=r
        ax.pcolormesh(xx,yy,np.ma.masked_where(~inside|(risk<.02),risk).reshape(xx.shape),cmap='Reds',vmin=0,vmax=.6,shading='auto',zorder=1)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]+.6));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    marks={}
    for i in (0,1):
        marks.setdefault(tuple(spec['initials'][i][:2]),[]).append(f'{"AB"[i]} start');marks.setdefault(tuple(spec['goals'][i][:2]),[]).append(f'{"AB"[i]} goal')
    for q,labels in marks.items():ax.text(q[0],q[1]-r-.15,' · '.join(labels),ha='center',va='top',fontsize=fontsize,color='white',zorder=2)


def at(samples,now):
    ids=[k for k,s in enumerate(samples) if s['time']<=now+1e-9];return samples[:ids[-1]+1] if ids else []


def world_panel(ax,r,now,ghost=None):
    from matplotlib.patches import Polygon,Circle,FancyArrowPatch
    spec=r['spec'];background(ax,spec);events=door_events(r)
    for o in doors_of(spec):
        name=door_name(spec,o);known=[(i,kind,events[(name,i,kind)]) for i in (0,1) for kind in ('sensed','received') if events.get((name,i,kind),np.inf)<=now]
        c=np.asarray(o[:2]);h=np.asarray(o[3:5]);u=h/np.linalg.norm(h);n=np.array([-u[1],u[0]])*o[2];e=h+u*o[2]
        ax.add_patch(Polygon([c-e+n,c+e+n,c+e-n,c-e-n],facecolor='#c62828' if known else '#8e949c',edgecolor='black',lw=.6,zorder=4))
        knowers={}
        for i,kind,t in sorted(known,key=lambda z:z[2]):knowers.setdefault(i,(kind,t))
        label=[f'{"AB"[i]} {"saw" if k=="sensed" else "told"} {t:.0f}s' for i,(k,t) in sorted(knowers.items())]
        ax.text(*(c+[0,spec['corridor_radius']+.12]),f'{name}: '+(', '.join(label) if label else 'hidden'),ha='center',va='bottom',fontsize=7.5,
                color='#ffffff',zorder=6,bbox=dict(boxstyle='round,pad=.2',facecolor='#c62828' if known else '#59606a',edgecolor='none'))
    for i in (0,1):
        if ghost is not None:
            gs=at(ghost['samples'][i],now)
            if gs:g=np.array([s['x'][:2] for s in gs]);ax.plot(*g.T,color=COLORS[i],lw=1.4,ls=(0,(2,2)),alpha=.45,zorder=3)
        ss=at(r['samples'][i],now)
        if not ss or now<spec['start_seconds'][i]:continue
        p=np.array([s['x'][:2] for s in ss]);q=p[-1];ax.plot(*p.T,color=COLORS[i],lw=2.6,zorder=5)
        ax.add_patch(Circle(q,1.25,fill=False,ls='--',lw=.8,color=COLORS[i],alpha=.6,zorder=5))
        ax.scatter(*q,color=COLORS[i],s=60,edgecolor='white',zorder=7)
        done=ss[-1]['done'];status=' ✓ goal' if done and r['agents'][i]['success'] else ' ✗' if done else ''
        ax.annotate('AB'[i]+status,q,xytext=(0,12) if done else (6,6),textcoords='offset points',ha='center' if done else 'left',color=COLORS[i],fontsize=11,weight='bold',zorder=9)
        controls=[t for t in r['trajectories'][i] if t['time']<=now]
        if controls and not done:
            route=np.asarray(controls[-1]['navigation'].get('route',[]))
            if len(route):ax.plot(*route.T,color=COLORS[i],ls=':',lw=1.6,alpha=.8,zorder=4)
    packets={e['packet']:e for e in r['events'] if e['kind']=='delivered'}
    for i,events in enumerate(r['geometry_events']):
        for e in events:
            name=door_name(spec,e['obstacle'])
            if e['kind']!='received_geometry' or name is None or not 0<=now-e['time']<=5:continue
            d=packets.get(e['packet'])
            if d is None:continue
            ax.add_patch(FancyArrowPatch(d['sender_q'],d['receiver_q'],arrowstyle='-|>',mutation_scale=22,color='#8e24aa',lw=2.6,connectionstyle='arc3,rad=-.25',zorder=8))
            mid=(np.asarray(d['sender_q'])+np.asarray(d['receiver_q']))/2
            ax.text(*mid+[0,.5],f'{"AB"[d["sender"]]} → {"AB"[i]}: "{name} is closed"',ha='center',fontsize=9,color='white',weight='bold',zorder=9,
                    bbox=dict(boxstyle='round,pad=.3',facecolor='#8e24aa',edgecolor='none'))


def wasted_so_far(r,i,now):
    ss=at(r['samples'][i],now)
    if len(ss)<2:return 0.
    spec=r['spec'];goal=spec['goals'][i];start=spec['initials'][i][:2]
    return path_length(ss)-(true_distance(spec,start,goal)-true_distance(spec,ss[-1]['x'][:2],goal))


def frame(off,on,now,end,predictions):
    import matplotlib.pyplot as plt
    from PIL import Image
    spec=on['spec'];fig=plt.figure(figsize=(16,9.6));gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],hspace=.18,wspace=.06)
    for k,(r,ghost) in enumerate(((off,None),(on,off))):
        ax=fig.add_subplot(gs[0,k]);world_panel(ax,r,now,ghost)
        word='wasted' if not spec.get('patches') else 'extra distance'
        tally='   '.join(f'{"AB"[i]} {word} {wasted_so_far(r,i,now):4.1f} m' for i in (0,1))
        ax.set_title(f'{ARM_LABELS.get(r["arm"],r["arm"])}\n{tally}',fontsize=12,color='#222')
    ax=fig.add_subplot(gs[1,0]);top=0
    for r,style,label in ((off,'--',ARM_LABELS.get(off['arm'],off['arm'])),(on,'-',ARM_LABELS.get(on['arm'],on['arm']))):
        for i in (0,1):
            ss=[s for s in at(r['samples'][i],now)[::4] if s['time']>=spec['start_seconds'][i]]
            if ss:
                y=[true_distance(spec,s['x'][:2],spec['goals'][i]) for s in ss];top=max(top,max(y))
                ax.plot([s['time'] for s in ss],y,style,color=COLORS[i],lw=2,label=f'{"AB"[i]} {label}')
    for i,events in enumerate(on['geometry_events']):
        for e in events:
            if e['kind']=='received_geometry' and door_name(spec,e['obstacle']) and e['time']<=now:ax.axvline(e['time'],color='#8e24aa',lw=1,alpha=.6)
    start=max(true_distance(spec,spec['initials'][i][:2],spec['goals'][i]) for i in (0,1))
    ax.axvline(now,color='black',lw=.8);ax.set(xlim=(0,end),ylim=(0,max(top,start)*1.15+1),xlabel='time (s)',ylabel='true remaining distance to goal (m)')
    ax.set_title(f'Dashed: {ARM_LABELS.get(off["arm"],off["arm"])}, solid: {ARM_LABELS.get(on["arm"],on["arm"])}.  Rising = moving away from the goal.  Purple: a warning arrives.',fontsize=10)
    ax.legend(fontsize=8,ncol=2,loc='upper right');ax.grid(alpha=.25)
    ax=fig.add_subplot(gs[1,1]);ax.axis('off')
    lines=[]
    if 'wasted' in str(predictions):
        lines.append(f'{"":14}{"OFF":>10}{"reciprocal":>12}   oracle final (OFF/recip.)')
        for i in (0,1):
            p=[predictions[a][str(i)]['wasted'] for a in ('disabled','reciprocal_fixed')]
            lines.append(f'{"AB"[i]+" wasted (m)":14}{wasted_so_far(off,i,now):10.1f}{wasted_so_far(on,i,now):12.1f}{p[0]:6.1f}/{p[1]:.1f}')
    else:
        # Graded-hazard runs: mission cost so far (time + weighted risk) against the oracle.
        lines.append(f'{"":14}{off["arm"]:>10}{on["arm"]:>12}   oracle final')
        for i in (0,1):
            so_far=[sum(t['components']['time']+t['components']['risk'] for t in r['trajectories'][i] if t['time']<=now) for r in (off,on)]
            lines.append(f'{"AB"[i]+" cost":14}{so_far[0]:10.1f}{so_far[1]:12.1f}   {predictions[off["arm"]][i]["cost"]:.0f}/{predictions[on["arm"]][i]["cost"]:.0f}')
    lines+=['','Event log (reciprocal run)'];log=[]
    for i,events in enumerate(on['geometry_events']):
        for e in events:
            name=door_name(spec,e['obstacle'])
            if name and e['time']<=now:log.append((e['time'],f'{e["time"]:6.1f}s  {"AB"[i]} '+(f'sees {name}' if e['kind']=='locally_sensed' else f'is told: {name} closed')))
    lines+=[t for _,t in sorted(log)[-7:]] or ['  (no door known yet)']
    lines+=['','Doors are hidden from both agents until sensed or reported.','Sensor range: dashed circle (1.25 m); no door is visible from a junction.',
            'Faint dashed paths: the same agent with communication OFF.','Independent plants: agents may pass through each other.']
    if max(spec['start_seconds'])>0:lines.append(f'Start times: A {spec["start_seconds"][0]:.0f} s, B {spec["start_seconds"][1]:.0f} s.')
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=10,family='monospace')
    fig.suptitle(f'{spec["name"]}  |  t = {now:5.1f} s  |  same seed, same sensors, only communication differs',fontsize=15,y=.985)
    fig.canvas.draw();image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return image


def summary_figure(rows,off,on,path,predictions):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,5.2),gridspec_kw=dict(width_ratios=[1,1,.9]))
    end=max(a['finish_time'] for r in (off,on) for a in r['agents'])
    for ax,(r,ghost) in zip(axes,((off,None),(on,off))):
        world_panel(ax,r,end,ghost);m=metrics(r)
        ax.set_title(f'{ARM_LABELS[r["arm"]]}: A {m["agents"][0]["path"]:.1f} m, B {m["agents"][1]["path"]:.1f} m traveled',fontsize=11)
    ax=axes[2];arms=[a for a in ARMS if a in predictions];x=np.arange(len(arms))
    ax.bar(x-.2,[sum(v['wasted'] for v in predictions[a].values()) for a in arms],.38,color='white',edgecolor='#555',hatch='//',label='route oracle')
    for k,arm in enumerate(arms):
        values=[r['team_wasted'] for r in rows if r['arm']==arm and r['goals']==2]
        if not values:continue
        ax.bar(k+.2,np.mean(values),.38,color='#8e24aa' if arm!='disabled' else '#777',label='measured' if k==0 else None)
        ax.scatter([k+.2]*len(values),values,color='black',s=12,zorder=3)
    ax.set_xticks(x,[ARM_LABELS[a].replace('Communication ','comm ').replace(' communication','') for a in arms],fontsize=9)
    ax.set(ylabel='team detour (m)',title='Team detour = traveled − true shortest route\n(completed episodes only)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)


def report(root,frames_count=110,pair=None):
    root=Path(root);(root/'figures').mkdir(exist_ok=True);manifest=load(root/'manifest.json');rows=[];episodes={}
    pair=tuple(pair or manifest.get('animation_arms',('disabled','reciprocal_fixed')))
    for p in sorted((root/'episodes').glob('*.json.gz')):
        r=load(p)
        if 'agents' not in r:continue
        rows.append(metrics(r))
        if r['spec']['seed']==manifest['animation_seed'] and r['arm'] in pair:episodes[r['arm']]=r
    dump(root/'rows.json',rows);pred=manifest['predictions']
    if len(episodes)<2:return
    off,on=(episodes[a] for a in pair);spec=on['spec']
    if 'wasted' in str(pred):summary_figure(rows,off,on,root/'figures/summary.png',pred)
    end=max(a['finish_time'] for r in (off,on) for a in r['agents'])
    keys=sorted({e['time'] for r in (off,on) for es in r['geometry_events'] for e in es if door_name(spec,e['obstacle'])})
    times=sorted({*np.linspace(0,end,frames_count).round(2).tolist(),*[round(min(end,t+d),2) for t in keys for d in (0.,1.,2.5)]})
    from PIL import Image
    frames=[frame(off,on,t,end,pred) for t in times]
    stem='episode' if pair==('disabled','reciprocal_fixed') else '_vs_'.join(pair)
    frames[0].save(root/'figures'/f'{stem}.gif',save_all=True,append_images=[f.convert('P',palette=Image.Palette.ADAPTIVE,colors=256) for f in frames[1:]],duration=160,loop=0)
    # Without door warnings, the key moment is just after the later agent can first hear the other.
    warned=min((e['time'] for es in on['geometry_events'] for e in es if e['kind']=='received_geometry' and door_name(spec,e['obstacle'])),default=max(spec['start_seconds'])+2.)
    frames[min(range(len(times)),key=lambda k:abs(times[k]-(warned+2.5)))].save(root/'figures'/f'{stem}_first_warning.png' if stem!='episode' else root/'figures/first_warning.png')
    frames[-1].save(root/'figures'/f'{stem}_final.png' if stem!='episode' else root/'figures/final.png');dump(root/'figures/frames.json',dict(times=times,seed=manifest['animation_seed'],playback='logged 0.12 s boundary states, compressed time'))
    if 'wasted' not in str(pred):return  # graded-hazard runs write their own tables
    delay=manifest['communication'].get('delay_ticks',0)*manifest['config']['dt']
    lines=[f'# {spec["name"]}: results','',f'Development run. Message delay {delay:.2f} s. Doors are hidden from both agents. Oracle predictions were frozen in `manifest.json` before any episode ran.','',
           '| Seed | Arm | Goals | Collisions | A path (m) | B path (m) | A detour, meas / oracle | B detour, meas / oracle | A finish (s) | B finish (s) | Bytes |','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows,key=lambda r:(r['seed'],ARMS.index(r['arm']))):
        a,b=r['agents'];p=pred[r['arm']]
        lines.append(f'| {r["seed"]} | {r["arm"]} | {r["goals"]}/2 | {sum(z["collision"] for z in r["agents"])} | {a["path"]:.1f} | {b["path"]:.1f} | {a["wasted"]:.1f} / {p["0"]["wasted"]:.1f} | {b["wasted"]:.1f} / {p["1"]["wasted"]:.1f} | {a["finish_time"]:.1f} | {b["finish_time"]:.1f} | {r["bytes"]} |')
    team={a:sum(v['wasted'] for v in pred[a].values()) for a in pred}
    lines+=['','## Saving from communication (team detour vs. no communication)','','| Seed | Arm | Measured | Oracle |','|---|---|---|---|']
    for seed in sorted({r['seed'] for r in rows}):
        w={r['arm']:r['team_wasted'] for r in rows if r['seed']==seed}
        for arm in ARMS[1:]:
            if arm in w and 'disabled' in w:lines.append(f'| {seed} | {arm} | {w["disabled"]-w[arm]:.1f} m | {team["disabled"]-team[arm]:.1f} m |')
    lines+=['','![summary](figures/summary.png)','','![episode](figures/episode.gif)','',
            'Detour is the measured path length minus the true shortest centerline route with every door known; corner cutting biases it low. The oracle assumes equal constant speed, instant turns and the declared delay, and ignores the planner\'s terrain costs. Agents are independent plants with no inter-agent contact dynamics.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
