"""Read-only attribution analysis and displays derived from logged models."""
from pathlib import Path
import csv
import json
import numpy as np
from .evaluation import load,dump
from .attribution import ARMS
from .maze_geometry import NODES,EDGES,CORRIDOR_RADIUS,distance_to_network
from .maze_visuals import maze_background,landscape,frame
from .records import Config

PAIRS=(('local_evidence','local_historical'),('local_evidence','geometry_only'),
       ('geometry_only','remote_route'),('remote_route','reconstruction'),
       ('reconstruction','current_replay'),('current_replay','historical_replay'),
       ('local_historical','historical_replay'))


def routes(r, i):
    points=np.array([s['x'][:2] for s in r['samples'][i]])
    nodes=np.array(NODES);edge_ids=[];edge_steps=[]
    for a,b in zip(points[:-1],points[1:]):
        mid=(a+b)/2;candidates=[]
        for ei,(u,v) in enumerate(EDGES):
            delta=nodes[v]-nodes[u];length=np.linalg.norm(delta);t=(mid-nodes[u])@delta/length**2
            if .2/length<t<1-.2/length:
                distance=np.linalg.norm(mid-(nodes[u]+t*delta))
                if distance<.7:candidates.append((distance,ei,float((b-a)@delta/length)))
        if candidates:
            _,edge,signed=min(candidates);edge_steps.append((edge,signed))
            if not edge_ids or edge_ids[-1]!=edge:edge_ids.append(edge)
    first={};retreat=0.
    for edge,step in edge_steps:
        if abs(step)<1e-10:continue
        first.setdefault(edge,np.sign(step))
        if step*first[edge]<0:retreat+=abs(step)
    blocked=[]
    for gx,gy,_ in r['spec']['gate_groups']:
        for ei,(u,v) in enumerate(EDGES):
            a,b=nodes[u],nodes[v];t=np.clip((np.array([gx,gy])-a)@(b-a)/np.sum((b-a)**2),0,1)
            if np.linalg.norm(np.array([gx,gy])-a-t*(b-a))<.01:blocked.append(ei)
    return dict(edge_sequence=edge_ids,blocked_branch_entries=sum(e in blocked for e in edge_ids),
        retreat_distance=retreat,path_length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))


def summarize(r):
    agents=[]
    for i,a in enumerate(r['agents']):
        records=r['trajectories'][i];stalls=0
        for k in range(16,len(records)):
            q=np.array(records[k]['x'][:2]);old=np.array(records[k-16]['x'][:2]);target=np.array(records[k]['control_model']['target'])
            if (np.linalg.norm(q-old)<.05 and np.linalg.norm(old-target)-np.linalg.norm(q-target)<.05
                and np.linalg.norm(q-np.array(r['spec']['goals'][i]))>=.25):stalls+=1
        updates=r['updates'][i]
        agents.append(dict(a,**routes(r,i),stall_windows=stalls,
            route_changes=sum(t['navigation'].get('route_changed',False) for t in records),
            replay_eligible=sum(e['eligible'] for u in updates for e in u['diagnostic'].get('replay_eligibility',[])),
            replay_excluded=sum(not e['eligible'] for u in updates for e in u['diagnostic'].get('replay_eligibility',[])),
            replay_domain_failures=sum(e['reason'] not in ('supported','unsupported_entry') for u in updates for e in u['diagnostic'].get('replay_eligibility',[])),
            policy_gradient_nonzero=sum(u['diagnostic'].get('policy_gradient_norm',0)>1e-10 for u in updates),
            accepted_updates=sum(u['diagnostic']['accepted'] for u in updates),
            rejected_updates=sum(not u['diagnostic']['accepted'] for u in updates),
            update_seconds=sum(u['diagnostic']['wall_seconds'] for u in updates)))
    warning=[]
    for i,events in enumerate(r['geometry_events']):
        for gx,gy,_ in r['spec']['gate_groups']:
            related=[e for e in events if abs(e['obstacle'][0]-gx)<.01 and abs(e['obstacle'][1]-gy)<=.6]
            received=min((e['time'] for e in related if e['kind']=='received_geometry'),default=None)
            sensed=min((e['time'] for e in related if e['kind']=='locally_sensed'),default=None)
            warning.append(dict(agent=i,gate=[gx,gy],received=received,sensed=sensed,
                lead_seconds=sensed-received if sensed is not None and received is not None else None,
                warned_without_local_discovery=received is not None and sensed is None))
    events=[e for e in r['events'] if e['kind']=='delivered']
    audit_rows=[]
    for audit in r['audits']:
        audit_rows.append({k:audit[k] for k in ('packet','receiver','stage','time','field_delta','control_delta','H_delta',
            'realized_improvement','source_ids','geometry_source_ids','seconds','truth_steps')})
    return dict(scene=r['scene'],seed=r['spec']['seed'],arm=r['attribution_arm'],agents=agents,warnings=warning,
        goals=sum(a['success'] for a in agents),collisions=sum(a['collision'] for a in agents),timeouts=sum(a['timeout'] for a in agents),
        deliveries=len(events),eligible_deliveries=sum(e['reason']=='eligible' for e in events),audits=audit_rows,
        audit_seconds=r['audit_seconds'],wall_seconds=r['wall_seconds'],bytes=sum(sum(a['bytes'].values()) for a in agents),
        audit_replay_work=sum(b['total_replay_work'] for a in r['audits'] for b in a['branches'].values()),
        audit_truth_steps=sum(a['truth_steps'] for a in r['audits']),
        components={key:sum(e.get('components',{}).get(key,{}).get('used',False) for e in events)
                    for key in ('geometry','route_anchors','field_assimilation')})


def trajectory_figure(r,path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6));maze_background(ax)
    from matplotlib.patches import Circle
    for x,y,rad in r['spec']['obstacles']:ax.add_patch(Circle((x,y),rad,color='red'))
    for i,color in enumerate(('tab:blue','tab:green')):
        q=np.array([s['x'][:2] for s in r['samples'][i]])
        ax.plot(*q.T,color=color,label=f"A{i}: {'GOAL' if r['agents'][i]['success'] else 'COLLISION' if r['agents'][i]['collision'] else 'TIMEOUT'}")
    ax.legend();ax.set_title(f"{r['scene']} | {r['attribution_arm']} | seed {r['spec']['seed']}")
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def paired_probes(left,right):
    out=[]
    for i in (0,1):
        a={u['tick']:u['evaluation'] for u in left[i]};b={u['tick']:u['evaluation'] for u in right[i]}
        for tick in sorted(a.keys()&b.keys()):
            x,y=a[tick],b[tick];mask=np.array(x['support']) & np.array(y['support']);n=int(mask.sum())
            metric=dict(agent=i,tick=tick,common_probes=n,coverage_left=x['coverage'],coverage_right=y['coverage'])
            for key in ('squared_error','gradient_squared_error'):
                metric[key+'_left']=float(np.mean(np.array(x[key],dtype=float)[mask])) if n else None
                metric[key+'_right']=float(np.mean(np.array(y[key],dtype=float)[mask])) if n else None
            out.append(metric)
    return out


def report(root):
    root=Path(root);m=load(root/'manifest.json');figdir=root/'figures';figdir.mkdir(exist_ok=True)
    gallery=figdir/'failures';gallery.mkdir(exist_ok=True)
    rows=[];failures=[];lineage=[];probe_groups={};completed=set()
    for path in sorted((root/'status').glob('*.json')):
        status=load(path);completed.add(status['id'])
        if not status['complete']:failures.append(status);continue
        r=load(root/'episodes'/status['path']);row=summarize(r);rows.append(row)
        probe_groups.setdefault((row['scene'],row['seed']),{})[row['arm']]=[
            [dict(tick=u['tick'],evaluation=u['evaluation']) for u in us] for us in r['updates']]
        packets={p['identity']:p for p in r['packets']}
        for e in r['events']:
            if e['kind']!='delivered':continue
            packet=packets[e['packet']]
            lineage.append(dict(scene=row['scene'],seed=row['seed'],arm=row['arm'],**e,
                source_ids=packet['sources'],geometry_source_ids=[g[-1] for g in packet['geometry']]))
        if row['goals']<2:
            failures.append(dict(id=status['id'],path=status['path'],agents=row['agents']))
            trajectory_figure(r,gallery/f"{status['id']}.png")
        del r
    expected=[f'{scene}_{seed}_{arm}' for scene in ('blocked_maze','soft_risk_maze') for seed in m['development_seeds'] for arm in ARMS]
    incomplete=[key for key in expected if key not in completed]
    pairs=[];index={(r['scene'],r['seed'],r['arm']):r for r in rows}
    for (scene,seed),group in probe_groups.items():
        for left,right in PAIRS:
            if left not in group or right not in group:continue
            a,b=index[scene,seed,left],index[scene,seed,right]
            differences={k:sum(v[k] for v in b['agents'])-sum(v[k] for v in a['agents']) for k in
                ('cost','path_length','retreat_distance','blocked_branch_entries','stall_windows','replay_work')}
            pairs.append(dict(scene=scene,seed=seed,left=left,right=right,differences=differences,
                goal_difference=b['goals']-a['goals'],common_probes=paired_probes(group[left],group[right])))
    dump(root/'rows.json',rows);dump(root/'paired_comparisons.json',pairs)
    dump(root/'packet_lineage.json.gz',lineage);dump(root/'failures.json',failures)
    dump(root/'incomplete.json',incomplete)
    dump(root/'audit_rows.json',[dict(scene=r['scene'],seed=r['seed'],arm=r['arm'],**a) for r in rows for a in r['audits']])
    with (root/'metrics.csv').open('w',newline='') as f:
        keys=['scene','seed','arm','goals','collisions','timeouts','deliveries','eligible_deliveries','bytes','wall_seconds','audit_seconds']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in rows)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8));labels=list(ARMS)
    for scene,color in [('blocked_maze','tab:blue'),('soft_risk_maze','tab:orange')]:
        selected=[[r for r in rows if r['scene']==scene and r['arm']==arm] for arm in labels]
        for ax,key,title in [(axes[0,0],'goals','Agent goals / episode'),(axes[0,1],'bytes','Actual bytes / episode'),
                            (axes[1,0],'wall_seconds','Wall seconds / episode'),(axes[1,1],'timeouts','Timeouts / episode')]:
            ax.plot(range(len(labels)),[np.mean([r[key] for r in group]) if group else np.nan for group in selected],'-o',label=scene,color=color)
            ax.set_title(title);ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'completion_and_computation.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for scene,color in [('blocked_maze','tab:blue'),('soft_risk_maze','tab:orange')]:
        for j,(left,right) in enumerate(PAIRS):
            pp=[p for p in pairs if p['scene']==scene and p['left']==left and p['right']==right]
            axes[0].scatter([j]*len(pp),[p['differences']['cost'] for p in pp],color=color,alpha=.6)
            common=[z for p in pp for z in p['common_probes'] if z['common_probes']]
            if common:
                weights=np.array([z['common_probes'] for z in common])
                axes[1].scatter(j,np.average([z['squared_error_right']-z['squared_error_left'] for z in common],weights=weights),color=color)
        audits=[a for r in rows if r['scene']==scene for a in r['audits'] if a['realized_improvement']['32'] is not None]
        axes[2].scatter([a['control_delta'] for a in audits],[a['realized_improvement']['32'] for a in audits],color=color,alpha=.4,label=scene)
    for ax in axes[:2]:
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(len(PAIRS)),[f'{a}→{b}' for a,b in PAIRS],rotation=55,ha='right')
    axes[0].set_title('Paired mission-cost difference (secondary)')
    axes[1].set_title('Field MSE difference on common supported probes')
    axes[2].set(xlabel='Immediate applied-control revision (N)',ylabel='True-world 32-step benefit',title='Packet audits: decision change versus outcome');axes[2].legend()
    fig.tight_layout();fig.savefig(figdir/'paired_effects.png',dpi=130);plt.close(fig)
    lines=['# Cooperative mechanism attribution','',f"Completed episodes: {len(rows)}/112. Failed execution attempts: {sum(not f.get('complete',True) for f in failures)}. Unfinished/unexecuted: {len(incomplete)}.",
        '',m['seed_semantics'],'','All evidence is within-mission development evidence. Two fixed scenes do not establish broad maze generalization.',
        '', '| Scene | Arm | Episodes | Goals | Collisions | Timeouts |', '|---|---|---:|---:|---:|---:|']
    for scene in ('blocked_maze','soft_risk_maze'):
        for arm in ARMS:
            rr=[r for r in rows if r['scene']==scene and r['arm']==arm]
            lines.append(f"| {scene} | {arm} | {len(rr)} | {sum(r['goals'] for r in rr)}/{2*len(rr)} | {sum(r['collisions'] for r in rr)} | {sum(r['timeouts'] for r in rr)} |")
    lines+=['','## Mechanism comparisons','',
        'Differences below are right minus left on matched completed episodes. No significance claim is made; zero and adverse differences are retained.', '',
        '| Scene | Comparison | Paired episodes | Δ goals | Δ blocked-branch entries | Δ retreat (m) | Δ cost (secondary) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scene in ('blocked_maze','soft_risk_maze'):
        for left,right in PAIRS:
            pp=[p for p in pairs if p['scene']==scene and p['left']==left and p['right']==right]
            if not pp:continue
            mean=lambda k:np.mean([p['differences'][k] for p in pp])
            lines.append(f"| {scene} | {left} → {right} | {len(pp)} | {sum(p['goal_difference'] for p in pp):+d} | {mean('blocked_branch_entries'):+.3f} | {mean('retreat_distance'):+.3f} | {mean('cost'):+.3f} |")
    lines+=['','Geometry sharing is isolated by local_evidence → geometry_only. Route-anchor use is isolated by geometry_only → remote_route. Supported reconstruction adds to that path in remote_route → reconstruction. Historical-start value is evaluated by current_replay → historical_replay; local_evidence → local_historical checks replay without communication.',
        '', 'A route change alone is not evidence of policy learning. Inspect field/control audit deltas, component eligibility, nonzero policy gradients, and common-supported errors together with the paired outcomes. Missing common support remains missing.',
        '', 'Replay coverage and denominator totals:', '', '| Arm | Eligible starts | Excluded starts | Domain failures | Nonzero-gradient updates | Audits | Nonzero field/control audits |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];aa=[a for r in rr for a in r['agents']];au=[a for r in rr for a in r['audits']]
        lines.append(f"| {arm} | {sum(a['replay_eligible'] for a in aa)} | {sum(a['replay_excluded'] for a in aa)} | {sum(a['replay_domain_failures'] for a in aa)} | {sum(a['policy_gradient_nonzero'] for a in aa)} | {len(au)} | {sum(a['field_delta']>1e-10 for a in au)}/{sum(a['control_delta']>1e-10 for a in au)} |")
    lines+=['','Branch entry means entering the interior of a surveyed edge containing an evaluator-known gate. Retreat is travel opposite the first observed direction on an edge. Stalls count overlapping 16-step windows with displacement and target progress below 0.05 m outside the goal. These definitions are diagnostics, not hidden learner inputs.',
        '', 'Warnings without subsequent local discovery have missing lead time and are counted separately. Every eligible packet is charged even when its payload channel is disabled. Equal byte ceilings do not imply equal bytes or compute.',
        '', 'Audits use identical complete starting snapshots and indexed sensor noise, actual retained evidence for the initial update, and branch-only simulated future observations. Intermediate decomposition is ordered and includes field-induced route changes; safety modification is shown separately. True-world branches never feed learning or scheduling.',
        '', 'See `failures.json`, `incomplete.json`, `packet_lineage.json.gz`, `audit_rows.json`, `paired_comparisons.json`, and `figures/failures/`. No incomplete or failed episode is dropped.',
        '', '![Completion and computation](figures/completion_and_computation.png)',
        '', '![Paired mechanism effects](figures/paired_effects.png)',
        '', '## Current evidence assessment','']
    for scene in ('blocked_maze','soft_risk_maze'):
        for left,right,label in [('local_evidence','geometry_only','Geometry sharing'),('remote_route','reconstruction','Field assimilation beyond route anchors'),('current_replay','historical_replay','Historical replay beyond the current state')]:
            pp=[p for p in pairs if p['scene']==scene and p['left']==left and p['right']==right]
            if not pp:
                lines.append(f'- {scene}, {label}: insufficient completed pairs.');continue
            goals=sum(p['goal_difference'] for p in pp);retreat=np.mean([p['differences']['retreat_distance'] for p in pp]);cost=np.mean([p['differences']['cost'] for p in pp])
            lines.append(f'- {scene}, {label}: {len(pp)} pairs, goal difference {goals:+d}, mean retreat difference {retreat:+.3f} m, secondary cost difference {cost:+.3f}. '+
                ('Completion advantage observed.' if goals>0 else 'Completion disadvantage observed.' if goals<0 else 'No completion advantage observed.')+
                (' Full eight-seed comparison remains incomplete.' if len(pp)<8 else ''))
    lines+=['','A beneficial route/geometry effect does not establish a learned-policy benefit. Effects without eligible supported replay cannot support a historical-learning claim. No penalties, support rules, or scene parameters were tuned to obtain these results.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def render(root):
    from PIL import Image
    import matplotlib.pyplot as plt
    root=Path(root);m=load(root/'manifest.json');out=root/'figures';out.mkdir(exist_ok=True)
    artifacts=[]
    for scene in ('blocked_maze','soft_risk_maze'):
        path=root/'episodes'/f"{scene}_{m['animation_seed']}_historical_replay.json.gz"
        if not path.exists():artifacts.append(dict(scene=scene,reason='selected_episode_unavailable'));continue
        r=load(path);end=max(a['finish_time'] for a in r['agents'])
        times=np.unique(np.r_[np.linspace(0,end,100),[a['time'] for a in r['audits']],end])
        frames=[frame(r,float(t)).resize((1190,770)).convert('P',palette=Image.Palette.ADAPTIVE,colors=256) for t in times]
        frames[0].save(out/f'{scene}.gif',save_all=True,append_images=frames[1:],duration=180,loop=0)
        frames[-1].save(out/f'{scene}_final.png')
        artifacts.append(dict(scene=scene,seed=m['animation_seed'],arm='historical_replay',times=times.tolist(),frames=len(frames)))
        # Predetermined first eligible audit; never choose a favorable effect.
        if r['audits']:
            a=r['audits'][0];fig,axes=plt.subplots(1,3,figsize=(15,5));cfg=Config(**r['config'])
            center=np.array(a['start'][:2]);xx,yy=np.meshgrid(np.linspace(center[0]-1.5,center[0]+1.5,45),np.linspace(center[1]-2,center[1]+2,55));points=np.c_[xx.ravel(),yy.ravel()]
            for ax,key in zip(axes[:2],('withhold','deliver')):
                b=a['branches'][key];H,F,mask=landscape(b['model'],points,np.array(a['start'][2:]),cfg)
                mask &= distance_to_network(points)<=CORRIDOR_RADIUS-.12
                ax.pcolormesh(xx,yy,np.ma.masked_where(~mask.reshape(xx.shape),H.reshape(xx.shape)),vmin=0,vmax=8,cmap='viridis',shading='auto')
                ax.set_facecolor('gray');ax.scatter(*center,color='red');ax.quiver(*center,*b['control'],color='red',scale=1,scale_units='xy',angles='xy');ax.set_aspect('equal');ax.set_title(f'{key}: v{b["version"]}, accepted={b["accepted"]}')
            maze_background(axes[2])
            for key,color in [('withhold','tab:blue'),('deliver','tab:orange')]:
                c=a['branches'][key]['checkpoints']['32']
                if c is not None:
                    q=np.array(c['path']);axes[2].plot(q[:,0],q[:,1],color=color,label=key)
            axes[2].legend();axes[2].set_title('Matched true-world 32-step continuations')
            fig.suptitle(f"First eligible packet {a['packet']} | fixed initial state | Δcontrol={a['control_delta']:.3g}")
            fig.tight_layout();fig.savefig(out/f'{scene}_first_packet.png',dpi=140);plt.close(fig)
    dump(out/'animation_frames.json',artifacts)
