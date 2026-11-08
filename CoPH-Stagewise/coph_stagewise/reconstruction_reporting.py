"""Reporting only: consumes frozen receiver diagnostic records."""
import json
from pathlib import Path
import numpy as np
from .evaluation import dump,load
from .receiver_reconstruction import METHODS,file_hash


def branch_entries(r,i):
    nodes=np.asarray(r['spec']['network_nodes']);edges=r['spec']['network_edges'];result=[];last=None
    for s in r['samples'][i]:
        q=np.asarray(s['x'][:2]);candidates=[]
        for k,(a,b) in enumerate(edges):
            d=nodes[b]-nodes[a];t=(q-nodes[a])@d/(d@d)
            if 0<t<1 and min(t,1-t)*np.linalg.norm(d)>.25:
                distance=np.linalg.norm(q-nodes[a]-t*d)
                if distance<.25:candidates.append((distance,k))
        if candidates:
            k=min(candidates)[1]
            if k!=last:result.append(dict(edge=k,time=s['time'],q=q.tolist()));last=k
    return result


def report(root):
    root=Path(root);m=load(root/'manifest.json');rows=[];audits=[];lineage=[];failures=[];incomplete=[];regressions=[];checks=[]
    for scene in m['settings']:
        for method in ['regression',*METHODS]:
            key=f'{scene}_{method}';p=root/'status'/f'{key}.json'
            if not p.exists():incomplete.append(dict(id=key,reason='not executed'));continue
            status=load(p)
            if not status['complete']:incomplete.append(status);continue
            path=root/'episodes'/status['path'];assert file_hash(path)==status['sha256'];r=load(path)
            if method=='regression':regressions.append(dict(scene=scene,**r['regression_comparison']))
            traversals=[branch_entries(r,i) for i in (0,1)]
            for i in (0,1):
                controls=r['trajectories'][i];trace=[u['diagnostic']['rr_trace'] for u in r['updates'][i]]
                first=next((s['time'] for s in r['samples'][i] if np.linalg.norm(np.array(s['x'][:2])-r['spec']['initials'][i][:2])>.25),None)
                receipts=[]
                for u in r['updates'][i]:
                    for e in u['diagnostic']['rr_trace']['events']:
                        row=dict(scene=scene,method=method,agent=i,**e,update_time=u['time'],accepted=u['diagnostic']['accepted'],
                            version_before=u['pre']['version'],version_after=u['post']['version'],first_branch_entry=first,
                            used_in_update=e.get('packet') in u['diagnostic']['rr_trace']['messages'].values(),
                            fitting_enabled=u['diagnostic']['rr_trace']['fit_enabled'])
                        lineage.append(row)
                        if e['kind']=='receipt':receipts.append(row)
                ids=[e['packet'] for e in receipts]
                delivered=[e['packet'] for e in r['events'] if e['kind']=='delivered' and e['receiver']==i]
                assert ids==delivered,'operational assimilation must be exactly once'
                if method!='regression':assert all(u['pre']['policy']==u['post']['policy'] for u in r['updates'][i])
                assert all(u['diagnostic']['counterfactual_steps']<=1024 for u in r['updates'][i])
                if method in ('shared_anchors','direct_local'):
                    assert all(not t['fit_enabled'] for t in trace)
                for u in r['updates'][i]:
                    if not u['diagnostic']['accepted']:
                        assert u['pre']['version']==u['post']['version']
                        assert [s['weights'] for s in u['pre']['stages']]==[s['weights'] for s in u['post']['stages']]
                assert sum(r['agents'][i]['bytes'].values())<=r['communication']['byte_budget']
                assert abs(sum(t['cost'] for t in controls)+r['agents'][i]['components']['failure']-r['agents'][i]['cost'])<1e-8
                points=np.array([s['x'][:2] for s in r['samples'][i]])
                stalls=sum(np.linalg.norm(points[j]-points[j-16])<.05 and np.linalg.norm(points[j]-r['spec']['goals'][i])>.25 for j in range(16,len(points)))
                row=dict(scene=scene,method=method,**r['agents'][i],branch_entries=traversals[i],first_branch_entry=first,stall_windows=int(stalls),
                    deliveries=len(receipts),replacements=sum(e.get('replaced') is not None for e in receipts),
                    fit_eligible_deliveries=sum(any(p['rectangle'] for p in e['eligibility']) for e in receipts),
                    local_supported_deliveries=sum(any(p['local_support'] for p in e['eligibility']) for e in receipts),
                    pre_entry_deliveries=sum(first is None or e['time']<first for e in receipts),
                    accepted_updates=sum(u['diagnostic']['accepted'] for u in r['updates'][i]),updates=len(trace),
                    diagnostic_seconds=r['receiver_diagnostics'][i]['seconds'],diagnostic_queries=r['receiver_diagnostics'][i]['queries'],
                    computation=r['receiver_diagnostics'][i],
                    audit_seconds=r['audit_seconds'],wall_seconds=r['wall_seconds'],route_changes=sum(t['navigation'].get('route_changed',False) for t in controls),
                    path_length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))
                rows.append(row)
                if not row['success']:failures.append(dict(id=key,agent=i,outcome=row,path=status['path']))
            for a in r['audits']:
                assert a['operational_state_unchanged'] and a['measurements_unchanged']
                b,c=a['branches']['no_fit'],a['branches']['fit'];q=a['queries'];errors=q['errors']
                assert b['message_fit_work']==0
                assert b['rankings']['anchors']==c['rankings']['anchors'],'fixed anchor source must be identical'
                change=lambda x,y: x['best']['nodes']!=y['best']['nodes'] if x['best'] and y['best'] else x['best']!=y['best']
                audits.append(dict(scene=scene,method=method,packet=a['packet'],time=a['time'],stage=a['stage'],reason=a['event']['reason'],
                    rectangle_pairs=sum(p['rectangle'] for p in a['eligibility']),local_supported_pairs=sum(p['local_support'] for p in a['eligibility']),
                    eligibility_pairs=len(a['eligibility']),query_count=q['query_count'],supported_count=q['supported_count'],errors=errors,
                    acceptance=dict(no_fit=b['accepted'],fit=c['accepted']),message_fit_work=c['message_fit_work'],
                    field_delta=float(np.linalg.norm(np.asarray(c['fields'])-np.asarray(b['fields']))),
                    control_delta=float(np.linalg.norm(np.asarray(c['control'])-np.asarray(b['control']))),
                    target_changed=c['target']!=b['target'],direct_ranking_changed=change(c['rankings']['supported_field'],b['rankings']['supported_field']),
                    natural_direct_ranking_changed=change(c['natural_rankings']['supported_field'],b['natural_rankings']['supported_field']),
                    seconds=a['seconds'],field_evaluations=a['field_evaluations']))
            checks.append(dict(id=key,sha256=status['sha256'],agents=2,updates=sum(map(len,r['updates'])),audits=len(r['audits']),passed=True))
    aggregates=[]
    for method in ['regression',*METHODS]:
        aa=[a for a in audits if a['method']==method];supported=[a for a in aa if a['supported_count']]
        count=sum(a['supported_count'] for a in supported)
        def mean(branch,metric):return sum(a['errors'][branch][metric]*a['supported_count'] for a in supported)/count if count else None
        aggregates.append(dict(method=method,audits=len(aa),supported_audits=len(supported),supported_queries=count,queries=sum(a['query_count'] for a in aa),
            value_mae={b:mean(b,'value_mae') for b in ('initial','no_fit','fit')},gradient_mae={b:mean(b,'gradient_mae') for b in ('initial','no_fit','fit')},
            value_improved=sum(a['errors']['fit']['value_mae']<a['errors']['no_fit']['value_mae']-1e-12 for a in supported),
            value_worsened=sum(a['errors']['fit']['value_mae']>a['errors']['no_fit']['value_mae']+1e-12 for a in supported),
            applied_control_changed=sum(a['control_delta']>1e-10 for a in aa),field_changed=sum(a['field_delta']>1e-10 for a in aa),
            direct_ranking_changed=sum(a['direct_ranking_changed'] for a in aa),acceptance_disagreement=sum(a['acceptance']['fit']!=a['acceptance']['no_fit'] for a in aa),
            fitting_work_nonzero=sum(a['message_fit_work']>0 for a in aa)))
    pairs=[]
    for scene in m['settings']:
        for agent in (0,1):
            indexed={r['method']:r for r in rows if r['scene']==scene and r['agent']==agent}
            for a,b in [('direct_local','shared_anchors'),('direct_cooperative','direct_local')]:
                if a in indexed and b in indexed:
                    x,y=indexed[a],indexed[b];pairs.append(dict(scene=scene,agent=agent,comparison=f'{a} minus {b}',
                        seconds=x['active_seconds']-y['active_seconds'],risk=x['components']['risk']-y['components']['risk'],
                        cost=x['cost']-y['cost'],branch_edges_equal=[e['edge'] for e in x['branch_entries']]==[e['edge'] for e in y['branch_entries']]))
    for name,obj in [('rows',rows),('audit_rows',audits),('aggregates',aggregates),('regressions',regressions),('paired',pairs),('failures',failures),('incomplete',incomplete),('checks',checks)]:dump(root/f'{name}.json',obj)
    dump(root/'lineage.json.gz',lineage)
    lines=['# Receiver reconstruction in the hazard pilot','',
        'Development diagnosis: four existing scenes, seed 79000, no new scenarios or tuning. Primary comparisons disable policy replay. Regessions retain the original pilot learner.',
        '',f'Completed episodes: {len(checks)}/16. Comparison goals: {sum(r["success"] for r in rows if r["method"]!="regression")}/24. Failures: {len(failures)}; incomplete: {len(incomplete)}.',
        f'Tracing regressions passed: {sum(r["passed"] for r in regressions)}/4. Bitwise after additive diagnostic exclusions: {sum(r["bitwise_passed"] for r in regressions)}/4.',
        '', '| Scene | Arm | Agent | Goal | Collision | Time (s) | Weighted risk | Deliveries / rectangle / local support |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['method']!='regression':lines.append(f'| {r["scene"]} | {r["method"]} | {r["agent"]} | {r["success"]} | {r["collision"]} | {r["active_seconds"]:.2f} | {r["components"]["risk"]:.5f} | {r["deliveries"]} / {r["fit_eligible_deliveries"]} / {r["local_supported_deliveries"]} |')
    lines+=['','## Matched receiver audits','', '| Arm | Supported audits / all | Supported queries / all | MAE no fit → fit | Gradient error no fit → fit | Better / worse | Field / control changes | Direct ranking changes |', '|---|---|---|---|---|---|---|---|']
    for a in aggregates:
        lines.append(f'| {a["method"]} | {a["supported_audits"]}/{a["audits"]} | {a["supported_queries"]}/{a["queries"]} | {a["value_mae"]["no_fit"]} → {a["value_mae"]["fit"]} | {a["gradient_mae"]["no_fit"]} → {a["gradient_mae"]["fit"]} | {a["value_improved"]}/{a["value_worsened"]} | {a["field_changed"]}/{a["applied_control_changed"]} | {a["direct_ranking_changed"]} |')
    lines+=['','Queries repeat across edges, connectors and audits; these are correlated counts, not independent trials. Both audit clones receive the same packet. Only received-residual fitting differs. Initial retained measurements and validation remain factual; no future observations are synthesized.',
        'Fixed-anchor rankings hold all anchor inputs constant to isolate field reconstruction. Natural rankings additionally include locally regenerated anchors. Node-route changes can be connector changes and are not automatically different branch traversal.',
        'Rectangle eligibility does not certify direct-field support. Remote-only anchors can still change routing. Operational traces record latest-sender replacement and expiry/eviction; missing legacy store histories remain unavailable.',
        'One noise seed and designed geometry do not establish generalization. Failures, rejected updates, unsupported predictions and zero-effect audits are retained. Source archive and manifest define reproduction; original pilot results are untouched.',
        '', '[Packet/model lineage](lineage.json.gz) · [Paired outcomes](paired.json) · [Regression checks](regressions.json) · [Failure inventory](failures.json) · [Incomplete inventory](incomplete.json)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(root);rows=report(root);m=load(root/'manifest.json');figdir=root/'figures';figdir.mkdir(exist_ok=True);gallery=figdir/'failures';gallery.mkdir(exist_ok=True)
    audits=load(root/'audit_rows.json');lineage=load(root/'lineage.json.gz');frames_meta={}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in METHODS:
        a=[r for r in audits if r['method']==method and r['supported_count']]
        axes[0].scatter([r['errors']['no_fit']['value_mae'] for r in a],[r['errors']['fit']['value_mae'] for r in a],label=method,alpha=.65)
        axes[1].scatter([r['time'] for r in a],[r['supported_count']/r['query_count'] for r in a],label=method,alpha=.65)
    lim=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([0,lim],[0,lim],'k--');axes[0].set(xlabel='No-fit MAE',ylabel='Fit MAE',title='Matched supported field errors')
    axes[1].set(xlabel='Global delivery time (s)',ylabel='Common supported query fraction',title='Coverage at audited packets')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figdir/'error_and_coverage.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for j,scene in enumerate(m['settings']):
        events=[e for e in lineage if e['scene']==scene and e['method']=='direct_cooperative' and e['agent']==1 and e['kind']=='receipt']
        for e in events:
            local=any(p['local_support'] for p in e['eligibility']);rectangle=any(p['rectangle'] for p in e['eligibility'])
            ax.scatter(e['time'],j,c='green' if local else 'orange' if rectangle else 'gray',marker='x' if e['replaced'] else 'o')
        r=next((r for r in rows if r['scene']==scene and r['method']=='direct_cooperative' and r['agent']==1),None)
        if r and r['first_branch_entry'] is not None:ax.scatter(r['first_branch_entry'],j,marker='|',s=250,c='black')
    ax.set(yticks=range(len(m['settings'])),yticklabels=list(m['settings']),xlabel='Global time (s)',title='Carrier: green local support; orange rectangle only; gray neither; × replaces sender; | first entry')
    fig.tight_layout();fig.savefig(figdir/'support_timing.png',dpi=140);plt.close(fig)
    for scene in m['settings']:
        rs={a:load(root/'episodes'/f'{scene}_{a}.json.gz') for a in METHODS if (root/'episodes'/f'{scene}_{a}.json.gz').exists()}
        if not rs:continue
        end=max(z['finish_time'] for r in rs.values() for z in r['agents']);times=np.unique(np.r_[np.linspace(0,end,64),25.]);frames=[]
        for now in times:
            fig,axes=plt.subplots(2,3,figsize=(14,7));fig.suptitle(f'{scene} | global time {now:.2f} s | seed 79000 | replay disabled')
            for col,(method,r) in enumerate(rs.items()):
                ax=axes[0,col];nodes=np.array(r['spec']['network_nodes'])
                for a,b in r['spec']['network_edges']:ax.plot(nodes[[a,b],0],nodes[[a,b],1],color='.8',lw=10,zorder=0)
                for px,py,sx,sy,amp in r['spec']['patches']:
                    from matplotlib.patches import Ellipse
                    ax.add_patch(Ellipse((px,py),4*sx,4*sy,color='red',alpha=.18));ax.text(px,py+.3,f'risk {amp}',fontsize=7)
                for i,color in enumerate(('tab:blue','tab:orange')):
                    samples=[s for s in r['samples'][i] if s['time']<=now]
                    if samples:
                        q=np.array([s['x'][:2] for s in samples]);ax.plot(q[:,0],q[:,1],color=color);ax.scatter(*q[-1],color=color,s=20)
                        state='GOAL' if samples[-1]['done'] and r['agents'][i]['success'] else 'FAILED' if samples[-1]['done'] else 'moving'
                        ax.text(.02,.98-i*.09,f'A{i}: {state}',color=color,va='top',transform=ax.transAxes,fontsize=8)
                    else:ax.text(.02,.98-i*.09,f'A{i}: waiting',va='top',transform=ax.transAxes,fontsize=8)
                    ts=[t for t in r['trajectories'][i] if t['time']<=now]
                    if ts and ts[-1]['navigation'].get('route'):
                        q=np.array(ts[-1]['navigation']['route']);ax.plot(q[:,0],q[:,1],'--',color=color,alpha=.5)
                for e in r['events']:
                    if e['kind']=='delivered' and 0<=now-e['time']<1.5:
                        q=np.array(e['anchors']);ax.scatter(q[:,0],q[:,1],marker='+',c='purple',s=20)
                ax.set(title=method,xlim=(-1,13),ylim=(-1,8.5),aspect='equal')
                ax=axes[1,col];trace=[t for t in r['trajectories'][1] if t['time']<=now]
                ax.plot([t['time'] for t in trace],[t['navigation']['rr_route']['margin'] if t['navigation']['rr_route']['margin'] is not None else np.nan for t in trace],color='tab:orange',lw=1)
                for e in r['events']:
                    if e['kind']=='delivered' and e['receiver']==1 and e['time']<=now:ax.axvline(e['time'],color='purple',alpha=.2)
                ax.set(xlim=(0,end),xlabel='Global time (s)',ylabel='Best–second route margin',title='Carrier route margin / packet deliveries')
            fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        frames[0].save(figdir/f'{scene}.gif',save_all=True,append_images=frames[1:],duration=160,loop=0);frames[-1].save(figdir/f'{scene}_final.png')
        frames_meta[scene]=dict(times=times.tolist(),source_episodes={k:file_hash(root/'episodes'/f'{scene}_{k}.json.gz') for k in rs},semantics='last logged state at/before global time; dashed planned routes, solid actual paths; evaluator hazards labeled')
        for f in load(root/'failures.json'):
            if f['id'].startswith(scene+'_'):frames[-1].save(gallery/f'{f["id"]}_agent{f["agent"]}.png')
    dump(root/'animation_frames.json',frames_meta)
    (gallery/'index.html').write_text('<h1>All failed and incomplete episodes</h1><pre>'+json.dumps(load(root/'failures.json')+load(root/'incomplete.json'),indent=2)+'</pre>')
    with (root/'REPORT.md').open('a') as f:
        f.write('\n![Errors and coverage](figures/error_and_coverage.png)\n\n![Support and timing](figures/support_timing.png)\n')
        for scene in frames_meta:f.write(f'\n[{scene} synchronized comparison](figures/{scene}.gif)\n')
