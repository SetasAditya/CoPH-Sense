"""Diagnostic summaries and pictures from saved operational/evaluator records."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .evaluation import load,dump
from .decision_trace import path_reach
from .maze_geometry import NODES,EDGES,network,distance_to_network


def branch_sequence(records,nodes,edges):
    nodes=np.asarray(nodes);out=[]
    for t in records:
        q=np.asarray(t['x'][:2]);candidates=[]
        for ei,(a,b) in enumerate(edges):
            d=nodes[b]-nodes[a];length=np.linalg.norm(d);z=(q-nodes[a])@d/(d@d)
            if .25/length<z<1-.25/length:
                distance=np.linalg.norm(q-nodes[a]-z*d)
                if distance<.25:candidates.append((distance,ei))
        if candidates:
            edge=min(candidates)[1]
            if not out or edge!=out[-1]:out.append(edge)
    return out


def episode_summary(r):
    cfg=r['config'];context={'network':(r['spec'].get('network_nodes',NODES),r['spec'].get('network_edges',EDGES))}
    nodes,edges=network(context);updates=[u for group in r['updates'] for u in group];calls=[];plans=[]
    for u in updates:
        d=u['diagnostic'];calls.extend(d.get('decision_replay',[]));plans.extend(d.get('replay_plan',[]))
    audits=r['audits'];legacy=not any('policy_trace' in u['diagnostic'] for u in updates)
    counts=dict(model_accepted=sum(u['diagnostic']['accepted'] for u in updates),
        nonzero_gradients=sum(u['diagnostic'].get('policy_gradient_norm',0)>1e-10 for u in updates),
        committed_policy_changes=sum(bool(np.linalg.norm(np.asarray(u['post']['policy'])-u['pre']['policy'])>1e-10) for u in updates),
        backtracking_accepted=None if legacy else sum(u['diagnostic']['policy_trace']['backtracking_accepted'] for u in updates),
        budget_exhausted=sum(u['diagnostic'].get('replay_budget_exhausted',False) for u in updates),
        unsupported_entries=sum(not p['eligible'] for p in plans),replay_plans=len(plans),
        calls=len(calls) if not legacy else None,valid_calls=sum(c['valid'] for c in calls) if not legacy else None,
        returned_paths=sum(len(c['paths']) for c in calls) if not legacy else None,
        decision_reaching_paths=sum(p['reach']['reaches_decision'] for c in calls for p in c['paths']) if not legacy else None)
    warning=[]
    for a in audits:
        states=a.get('decision_intermediates',[])
        decisions={}
        for name,b in a['branches'].items():
            d=b.get('decision_audit')
            if d:
                refs=d['route_references'];best=lambda k:refs[k]['best']['nodes'] if refs[k]['best'] else None
                decisions[name]=dict(estimated=best('estimated'),supported_truth=best('supported_truth'),full_truth=best('full_truth'),
                    supported_reference_changes_route=best('estimated')!=best('supported_truth'),full_reference_changes_route=best('estimated')!=best('full_truth'),
                    sensitivity=d['sensitivity'],forces=d['forces'],margin=refs['estimated']['margin'])
        warning.append(dict(packet=a['packet'],receiver=a['receiver'],stage=a['stage'],time=a['time'],
            field_delta=a['field_delta'],control_delta=a['control_delta'],source_ids=a['source_ids'],
            route_effects=[dict(label=s['label'],route=s['route']) for s in states],references=decisions,
            realized_improvement=a['realized_improvement']))
    rows=[]
    for i,a in enumerate(r['agents']):
        ts=r['trajectories'][i];stalls=0
        for k in range(16,len(ts)):
            q=np.asarray(ts[k]['x'][:2]);old=np.asarray(ts[k-16]['x'][:2]);target=np.asarray(ts[k]['control_model']['target'])
            stalls+=bool(np.linalg.norm(q-old)<.05 and np.linalg.norm(old-target)-np.linalg.norm(q-target)<.05 and np.linalg.norm(q-r['spec']['goals'][i])>=.25)
        rows.append(dict(a,branch_sequence=branch_sequence(ts,nodes,edges),stall_windows=stalls,
            planned_route_changes=sum(t['navigation'].get('route_changed',False) for t in ts)))
    return dict(case=r.get('case',r.get('scene',r['spec']['name'])),seed=r['spec']['seed'],arm=r.get('method',r.get('attribution_arm',r['arm'])),
        regression=r.get('regression',False),agents=rows,counts=counts,audits=warning,
        diagnostic_seconds=sum(u['diagnostic'].get('decision_seconds',0) for u in updates)+sum(t['navigation'].get('decision_work',{}).get('seconds',0) for group in r['trajectories'] for t in group),
        diagnostic_route_quadrature_points=sum(t['navigation'].get('decision_work',{}).get('quadrature_points',0) for group in r['trajectories'] for t in group),
        sensitivity_guard_flows=sum(a.get('decision_guard_flows',0) for a in audits),
        audit_seconds=r['audit_seconds'],wall_seconds=r['wall_seconds'],
        unavailable=['route margins','unconstrained/bounded policy proposals','backtracking outcomes','rejected replay paths','oracle rankings','policy sensitivities'] if legacy else [])


def analyze(reference,out):
    reference=Path(reference);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];files={}
    for status in sorted((reference/'status').glob('*.json')):
        s=load(status)
        if not s['complete']:rows.append(dict(id=s['id'],error=s.get('error')));continue
        path=reference/'episodes'/s['path'];sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==s['sha256'];files[s['path']]=sha
        r=load(path);rows.append(episode_summary(r));print(s['id'],flush=True)
    dump(out/'legacy_analysis.json',dict(reference=str(reference.resolve()),source_manifest=load(reference/'manifest.json')['fingerprint'],episodes=len(rows),rows=rows,files=files,
         semantics='Read-only analysis. Missing legacy instrumentation remains unavailable; no inferred route scores or proposal decisions.'))


def report(root):
    import matplotlib.pyplot as plt
    root=Path(root);m=load(root/'manifest.json');fig=root/'figures';fig.mkdir(exist_ok=True);failure_dir=fig/'failures';failure_dir.mkdir(exist_ok=True)
    rows=[];failures=[];lineage=[];traces=[];completed=set();regressions=[]
    for f in sorted((root/'status').glob('*.json')):
        s=load(f);completed.add(s['id'])
        if not s['complete']:failures.append(s);continue
        p=root/'episodes'/s['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256'];r=load(p)
        row=episode_summary(r)
        if r['regression']:regressions.append(dict(id=s['id'],**r['regression_comparison']))
        else:rows.append(row)
        for i,group in enumerate(r['updates']):
            for u in group:traces.append(dict(episode=s['id'],agent=i,time=u['time'],tick=u['tick'],packets=u['packets'],
                model_before=u['pre']['version'],model_after=u['post']['version'],policy=u['diagnostic'].get('policy_trace'),
                active_packets=u['diagnostic'].get('messages',[]),
                field_revision_norm=float(np.linalg.norm(np.asarray([s['weights'] for s in u['post']['stages']])-np.asarray([s['weights'] for s in u['pre']['stages']]))),
                damping_revision=u['post']['damping']-u['pre']['damping'],control_record_key=dict(time=u['time'],agent=i,version=u['post']['version']),
                replay=u['diagnostic'].get('decision_replay'),plan=u['diagnostic'].get('replay_plan'),reason=u['diagnostic']['reason']))
        packets={p['identity']:p for p in r['packets']}
        for e in r['events']:
            if e['kind']=='delivered':lineage.append(dict(episode=s['id'],**e,source_ids=packets[e['packet']]['sources']))
        if not all(a['success'] for a in r['agents']):
            failures.append(dict(id=s['id'],agents=r['agents']));draw_episode(r,failure_dir/f"{s['id']}.png")
    expected=[f'{c}_{s}_{a}' for c in m['cases'] for s in m['seeds'] for a in m['arms']]+[f"{x['case']}_{x['seed']}_{x['arm']}" for x in m['regressions']]
    incomplete=sorted(set(expected)-completed)
    dump(root/'rows.json',rows);dump(root/'decision_traces.json.gz',traces);dump(root/'packet_lineage.json.gz',lineage)
    dump(root/'failures.json',failures);dump(root/'incomplete.json',incomplete);dump(root/'regressions.json',regressions)
    audits=[dict(case=r['case'],seed=r['seed'],arm=r['arm'],**a) for r in rows for a in r['audits']];dump(root/'audit_rows.json',audits)
    fig1,axes=plt.subplots(1,3,figsize=(14,4))
    for case in m['cases']:
        data=[a for a in audits if a['case']==case]
        margins=[b['margin'] for a in data for b in a['references'].values() if b['margin'] is not None]
        axes[0].plot(sorted(margins),label=case)
        rr=[r for r in rows if r['case']==case]
        for r in rr:axes[1].scatter(r['counts']['returned_paths'] or 0,r['counts']['decision_reaching_paths'] or 0,label=case if r==rr[0] else None)
        for a in data:
            if 'withhold' not in a['references'] or 'deliver' not in a['references']:continue
            b,d=[a['references'][k]['forces'] for k in ('withhold','deliver')]
            axes[2].scatter(np.linalg.norm(np.asarray(d['nominal'])-b['nominal']),np.linalg.norm(np.asarray(d['applied'])-b['applied']),s=9,alpha=.4)
    axes[0].set(title='Route-score margins',xlabel='Sorted audit branch',ylabel='Second − best score');axes[0].legend()
    axes[1].set(title='Replay decision reach',xlabel='Returned replay paths',ylabel='Paths entering an outgoing branch');axes[1].legend()
    axes[2].set(title='Force transfer after a packet',xlabel='Nominal-control change',ylabel='Applied-control change')
    fig1.tight_layout();fig1.savefig(fig/'decision_bottlenecks.png',dpi=140);plt.close(fig1)
    lines=['# Decision bottleneck diagnosis','',f'Fork episodes completed: {len(rows)}/30. Instrumentation regressions: {sum(r["passed"] for r in regressions)}/2 passed. Incomplete records: {len(incomplete)}.',
        '', 'Two sensor-noise seeds on three fixed fork cases; development evidence only. No settings tuned to obtain a route change.',
        '', 'Regression acceptance is numerical (atol 1e-12, rtol 1e-10), with exact identities, schedules, decisions and work counts. Bitwise results and raw serialization-byte differences remain explicit in regressions.json; numerical equivalence is not byte-for-byte parity.',
        '', '| Case | Arm | Goals | Collisions | Timeouts | Policy changes / updates with nonzero gradient | Decision-reaching / returned replay paths |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case in m['cases']:
        for arm in m['arms']:
            group=[r for r in rows if r['case']==case and r['arm']==arm];aa=[a for r in group for a in r['agents']]
            lines.append(f'| {case} | {arm} | {sum(a["success"] for a in aa)}/{len(aa)} | {sum(a["collision"] for a in aa)} | {sum(a["timeout"] for a in aa)} | {sum(r["counts"]["committed_policy_changes"] for r in group)}/{sum(r["counts"]["nonzero_gradients"] for r in group)} | {sum(r["counts"]["decision_reaching_paths"] or 0 for r in group)}/{sum(r["counts"]["returned_paths"] or 0 for r in group)} |')
    refs=[b for a in audits for b in a['references'].values()];sens=[s for b in refs for s in b['sensitivity']]
    lines+=['','## Where influence is lost','',
        f'- Supported exact-risk route reference changes the selected route in {sum(b["supported_reference_changes_route"] for b in refs)}/{len(refs)} audited branches. Full-edge exact-risk reference changes it in {sum(b["full_reference_changes_route"] for b in refs)}/{len(refs)}. These are ranking references, not navigation outcomes.',
        f'- Active-set changes occur in {sum(s["active_set_changed"] for s in sens)}/{len(sens)} policy finite-difference comparisons. Derivatives at those points are not treated as smooth sensitivities.',
        f'- {sum(r["counts"]["unsupported_entries"] for r in rows)}/{sum(r["counts"]["replay_plans"] for r in rows)} recorded replay starts are unsupported; {sum(r["counts"]["budget_exhausted"] for r in rows)} updates report budget exhaustion.',
        '- Reported model acceptance, backtracking acceptance and committed policy change are separate. A zero policy change is retained even when reconstruction commits.',
        '- Graph route costs retain fixed risk weight 4; policy coefficients affect local shaping. A nonzero local force derivative is not a derivative of the discrete route choice.',
        '- Replay path counts include baseline, gradient and candidate evaluations, not independent experimental samples. Missing paths after domain rejection are not assigned zero length or success.',
        '- Attribution of route/field/policy effects follows the logged ordering and is not unique. Both live agent states and measured stores are isolated from evaluator branches.',
        '', '## Evidence and limitations','',
        'Timing/support, reconstruction, unchanged route ranking, replay reach, rejected/small updates and projection may coexist. Inspect per-packet route references and decomposition together; aggregates alone do not identify a unique cause. No architecture change is made by this suite.',
        '', 'Operational flow counts retain the 1,024 ceiling. Sensitivity guard flows and audit wall time are separate. Missing legacy fields remain unavailable in the saved-analysis artifact.',
        '', '![Decision diagnostics](figures/decision_bottlenecks.png)','',
        'See `rows.json`, `audit_rows.json`, `decision_traces.json.gz`, `packet_lineage.json.gz`, `regressions.json`, `failures.json`, and `incomplete.json`. Every failed mission receives a figure under `figures/failures/`.']
    for c in m['cases']:lines.append(f'\n[{c} animation](figures/{c}.gif)')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def draw_episode(r,path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));world_axes(ax,r)
    for i,color in enumerate(('tab:blue','tab:green')):
        q=np.asarray([s['x'][:2] for s in r['samples'][i]]);ax.plot(*q.T,color=color,label=f'A{i}: '+('GOAL' if r['agents'][i]['success'] else 'COLLISION' if r['agents'][i]['collision'] else 'TIMEOUT'))
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def world_axes(ax,r):
    nodes=np.asarray(r['spec']['network_nodes']);edges=r['spec']['network_edges']
    for a,b in edges:ax.plot(*nodes[[a,b]].T,color='#e3e3e3',lw=20,zorder=0)
    for a,b in edges:ax.plot(*nodes[[a,b]].T,color='gray',lw=.6,zorder=1)
    for x,y,sx,sy,amp in r['spec']['patches']:
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x,y),4*sx,4*sy,color='orange',alpha=.35))
    ax.set(xlim=(-.5,8.5),ylim=(-1.,2.7),aspect='equal')


def render(root):
    import matplotlib.pyplot as plt
    from PIL import Image
    from .maze_visuals import landscape
    from .records import Config
    root=Path(root);m=load(root/'manifest.json');figdir=root/'figures';meta=[]
    for case in m['cases']:
        p=root/'episodes'/f'{case}_{m["animation_seed"]}_historical_replay.json.gz'
        if not p.exists():meta.append(dict(case=case,unavailable=True));continue
        r=load(p);cfg=Config(**r['config']);end=max(a['finish_time'] for a in r['agents']);times=np.unique(np.r_[np.linspace(0,end,72),[a['time'] for a in r['audits']],end]);frames=[]
        for now in times:
            fig,axs=plt.subplots(2,2,figsize=(12,7));world_axes(axs[0,0],r);axs[0,0].set_title(f'{case}: t={now:.2f}s; orange = evaluator risk')
            for i,color in enumerate(('tab:blue','tab:green')):
                ts=[t for t in r['trajectories'][i] if t['time']<=now]
                samples=[s for s in r['samples'][i] if s['time']<=now]
                if samples:
                    q=np.asarray([s['x'][:2] for s in samples]);axs[0,0].plot(*q.T,color=color);axs[0,0].plot(*q[-1],'o',color=color)
                ax=axs[0,1] if i==0 else axs[1,0];ax.set_title(f'A{i}: logged shaping Hamiltonian')
                if not ts:ax.text(.2,.5,'Not started',transform=ax.transAxes);continue
                t=ts[-1];model=t['control_model'];q=np.asarray(t['x'][:2]);v=np.asarray(t['x'][2:]);xx,yy=np.meshgrid(np.linspace(q[0]-1.3,q[0]+1.3,32),np.linspace(q[1]-1.2,q[1]+1.2,30));points=np.c_[xx.ravel(),yy.ravel()]
                h,f,mask=landscape(model,points,v,cfg);mask &= distance_to_network(points,r['spec']['network_nodes'],r['spec']['network_edges'])<=.68
                # H is kinetic + shaping; subtract constant kinetic for displayed shaping.
                values=np.ma.masked_where(~mask,h-.5*(v@v)).reshape(xx.shape);ax.pcolormesh(xx,yy,values,shading='auto',cmap='viridis');ax.plot(*q,'wo');ax.set_aspect('equal')
                route=t['navigation'].get('route',[])
                if route:axs[0,0].plot(*np.asarray(route).T,'--',color=color,alpha=.5)
                for name,c in [('attraction_force','cyan'),('shaping_force','orange'),('applied','red')]:
                    force=np.asarray(t['navigation'].get(name,[0.,0.]));ax.arrow(*q,*(.4*force),color=c,width=.018,length_includes_head=True)
                outcome='GOAL' if r['agents'][i]['success'] else 'COLLISION' if r['agents'][i]['collision'] else 'TIMEOUT'
                margin=t['navigation'].get('decision_route',{}).get('margin')
                ax.set_xlabel(f'A{i}: '+(outcome if now>=r['agents'][i]['finish_time'] else 'active')+f'; margin={margin:.2f}' if margin is not None else f'A{i}: no second route')
                available=[u for u in r['updates'][i] if u['time']<=now]
                if available:
                    for stage in available[-1]['post']['stages']:
                        for cf in stage['counterfactuals']:
                            path=np.asarray(cf['states'])
                            if len(path):axs[0,0].plot(path[:,0],path[:,1],':',color=color,alpha=.45)
                ax.text(.01,.97,'cyan attraction; orange shaping; red applied',transform=ax.transAxes,va='top',fontsize=7)
            ax=axs[1,1]
            for e in r['events']:
                if e['kind']=='delivered' and e['time']<=now:ax.scatter(e['time'],e['receiver'],marker='|',color='purple' if e['reason']=='eligible' else 'gray')
            for a in r['audits']:
                if a['time']<=now:ax.scatter(a['time'],a['receiver']+.15,color='red' if a['control_delta']>1e-10 else 'black',s=12)
            ax.set(xlim=(0,end),ylim=(-.3,1.4),yticks=[0,1],xlabel='Shared time (s)',title='Packets; red audit = nonzero applied-control change');ax.axvline(now,color='gray')
            fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=256));plt.close(fig)
        frames[0].save(figdir/f'{case}.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
        meta.append(dict(case=case,seed=m['animation_seed'],arm='historical_replay',times=times.tolist(),frames=len(frames),state_source='latest logged control-time model, never reconstructed from later evidence'))
    dump(figdir/'animation_frames.json',meta)
