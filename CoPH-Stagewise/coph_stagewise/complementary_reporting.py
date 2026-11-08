"""Reports and visualizations from frozen complementary-coverage diagnostic records."""
import json
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .receiver_reconstruction import file_hash
from .complementary_reconstruction import METHODS,CASES,branch_entry


def audit_row(a,**labels):
    b,c=a['branches']['no_fit'],a['branches']['fit']
    return dict(labels,packet=a['packet'],receiver=a['receiver'],time=a['time'],stage=a['stage'],
        eligible=a['event']['reason']=='eligible',message_fit_work=c['message_fit_work'],
        supported=a['queries']['supported_count'],queries=a['queries']['query_count'],errors=a['queries']['errors'],
        accepted=dict(no_fit=b['accepted'],fit=c['accepted']),
        field_change=float(np.linalg.norm(np.array(c['fields'])-b['fields'])),
        control_change=float(np.linalg.norm(np.array(c['control'])-b['control'])),
        target_change=float(np.linalg.norm(np.array(c['target'])-b['target'])),
        fixed_anchor_scores_equal=b['rankings']['anchors']==c['rankings']['anchors'],
        direct_scores_equal=b['rankings']['supported_field']==c['rankings']['supported_field'],
        best_routes={name:{mode:branch['rankings'][mode]['best'] for mode in ('anchors','supported_field')} for name,branch in [('no_fit',b),('fit',c)]},
        route_margins={name:{mode:branch['rankings'][mode]['margin'] for mode in ('anchors','supported_field')} for name,branch in [('no_fit',b),('fit',c)]},
        natural_best_routes={name:{mode:branch['natural_rankings'][mode]['best'] for mode in ('anchors','supported_field')} for name,branch in [('no_fit',b),('fit',c)]},
        errors_change={k:None if c0 is None else a['queries']['errors']['fit'][k]-c0 for k,c0 in a['queries']['errors']['no_fit'].items()},
        audit_seconds=a['seconds'],field_evaluations=a['field_evaluations'],
        source_ids=a.get('source_ids',[]),purposes=a.get('purposes',['acquisition_gate']))


def report(root):
    root=Path(root);m=load(root/'manifest.json');gate=load(root/'gate.json') if (root/'gate.json').exists() else dict(passed=False,incomplete=True)
    gates=[];audits=[];rows=[];lineage=[];failures=[];incomplete=[];checks=[];pairs=[]
    for key,spec in m['specs'].items():
        p=root/'gate'/f'{key}.json'
        if p.exists():
            r=load(p);gates.append(r)
            for a in load(root/'gate'/f'{key}_audits.json.gz'):audits.append(audit_row(a,case=spec['scene'],seed=spec['seed'],method='gate'))
        for method in METHODS:
            identity=f'{key}_{method}';p=root/'episodes'/f'{identity}.json.gz'
            if not p.exists():
                status=root/'status'/f'{identity}.json'
                incomplete.append(load(status) if status.exists() else dict(id=identity,complete=False,executed=False,reason='eligibility gate failed' if not gate['passed'] else 'not run'));continue
            r=load(p)
            checks.append(dict(id=identity,acquisition=r['acquisition_check'],max_flow=max(a['max_update_work'] for a in r['agents']),
                packet_count=len(r['packets']),audit_isolation=all(a['operational_state_unchanged'] and a['measurements_unchanged'] for a in r['audits']),sha256=file_hash(p)))
            for a in r['audits']:audits.append(audit_row(a,case=spec['scene'],seed=spec['seed'],method=method))
            for i,a in enumerate(r['agents']):
                traversal=[]
                for s in r['samples'][i]:
                    e=branch_entry(s['x'][:2])
                    if e is not None and (not traversal or traversal[-1]['edge']!=e):traversal.append(dict(time=s['time'],edge=e))
                row=dict(a,id=identity,case=spec['scene'],seed=spec['seed'],method=method,branch_traversal=traversal,
                    unweighted_risk=a['components']['risk']/(4*spec['risk_weights'][i]),wall_seconds=r['wall_seconds'],audit_seconds=r['audit_seconds'])
                rows.append(row)
                if not a['success']:failures.append(row)
                for u in r['updates'][i]:
                    t=u['diagnostic'].get('rr_trace',{})
                    for e in t.get('events',[]):lineage.append(dict(e,id=identity,agent=i,version=u['post']['version'],accepted=u['diagnostic']['accepted'],
                        committed_routing_versions=t.get('routing_versions'),residuals=t.get('residuals_after')))
    for spec in m['specs'].values():
        for i in (0,1):
            rr={r['method']:r for r in rows if r['case']==spec['scene'] and r['seed']==spec['seed'] and r['agent']==i}
            for a,b in (('direct_local','shared_anchors'),('direct_cooperative','direct_local')):
                if a in rr and b in rr:pairs.append(dict(case=spec['scene'],seed=spec['seed'],agent=i,comparison=f'{a} minus {b}',
                    time=rr[a]['active_seconds']-rr[b]['active_seconds'],risk=rr[a]['unweighted_risk']-rr[b]['unweighted_risk'],
                    traversals_equal=[x['edge'] for x in rr[a]['branch_traversal']]==[x['edge'] for x in rr[b]['branch_traversal']]))
    for name,value in [('eligibility',gates),('audit_rows',audits),('rows',rows),('paired',pairs),('failures',failures),('incomplete',incomplete),('checks',checks)]:dump(root/f'{name}.json',value)
    dump(root/'lineage.json.gz',lineage)
    lines=['# Receiver reconstruction under complementary coverage','',
        'Development-only diagnosis. Fixed fixture, seeds 79100–79101, unchanged learner/controller/support. Policy replay disabled. Acquisition includes ordinary sensing, movement and fitting for 0.96 s before reciprocal exchange.',
        '',f'**Eligibility gate: {"PASS" if gate["passed"] else "FAIL — navigation comparison stopped"}.**',
        f'Navigation episodes executed: {len(checks)}/18; goals: {sum(r["success"] for r in rows)}/{len(rows)} executed missions; collisions: {sum(r["collision"] for r in rows)}; timeouts: {sum(r["timeout"] for r in rows)}.',
        'A failed eligibility gate means the prescribed fixture does not establish the intended pre-decision reconstruction test. No fixture adjustment or seed search was performed.',
        '', '| Case | Seed | Direction | Intended | Rectangle anchors / exported | Outside receiver footprint | Supported outgoing queries | Validation / fitting work | Failed checks |', '|---|---|---|---|---|---|---|---|---|']
    for g in gates:
        for d in g['directions']:
            a=next((a for a in audits if a['method']=='gate' and a['case']==g['scene'] and a['seed']==g['seed'] and a['receiver']==d['receiver']),None)
            support=', '.join(f"{e['edge']}: {e['count']}/{e['total']}" for e in d['outgoing'])
            lines.append(f"| {g['scene']} | {g['seed']} | {d['sender']}→{d['receiver']} | {d['intended']} | {len(d['rectangle_anchor_ids'])}/{d['exported_anchor_count']} | {len(d['outside_footprint_ids'])} | {support} | {d['checks']['valid_packet']} / {a['message_fit_work'] if a else 'unavailable'} | {', '.join(k for k,v in d['checks'].items() if not v) or 'none'} |")
    lines+=['','## Matched acquisition audits','',
        'Both copies receive the identical packet and remote route anchors. Only received-field fitting differs. All queries below are common-supported route quadrature, including connectors; counts are correlated and are not independent trials.',
        '', '| Case | Seed | Receiver | Supported / all queries | Value MAE no-fit → fit | Gradient error no-fit → fit | Fit accepted | Field / control change |', '|---|---|---|---|---|---|---|---|']
    for a in audits:
        if a['method']!='gate':continue
        e=a['errors'];lines.append(f"| {a['case']} | {a['seed']} | {a['receiver']} | {a['supported']}/{a['queries']} | {e['no_fit']['value_mae']} → {e['fit']['value_mae']} | {e['no_fit']['gradient_mae']} → {e['fit']['gradient_mae']} | {a['accepted']['fit']} | {a['field_change']:.6g} / {a['control_change']:.6g} |")
    lines+=['','## Interpretation and limits','',
        'Rectangle eligibility allows message residuals; it does not establish local-measurement support for route scoring. Outside-footprint anchors establish geometric complementary coverage only, not accuracy or benefit. Proposed, rejected and committed field changes are preserved separately in the full audit records.',
        'For redundant coverage, initial sensing coordinates coincide and indexed noise streams are independent. Subsequent sensing coordinates can differ because the agents retain different risk weights and independent observations. Actual equality is recorded in each eligibility record.',
        'The preserved 25-second relay is a non-overlap reference: its remote information can help routing despite not reaching field reconstruction. It is not an irrelevant-information control.',
        'Navigation outcomes, risk exposure, bytes and computation are separate from immediate estimation/control audits. A gate failure makes no claim about navigation success or comparative mission benefit. No held-out or final-test namespace was used.',
        '', '[Eligibility](eligibility.json) · [Audit rows](audit_rows.json) · [Paired outcomes](paired.json) · [Lineage](lineage.json.gz) · [Failures](failures.json) · [Incomplete / not-run inventory](incomplete.json)',
        '', 'Sources and exact specifications were archived before acquisition. Complete evaluator snapshots are local trusted pickle files; they are never transmitted or supplied to runtime learners.']
    if not rows:lines+=['','No three-arm GIFs are generated: the gate blocked all navigation comparisons. Acquisition/support and error figures show the evidence available.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return rows


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    from PIL import Image
    root=Path(root);report(root);m=load(root/'manifest.json');figdir=root/'figures';figdir.mkdir(exist_ok=True)
    gates=load(root/'eligibility.json');aa=load(root/'audit_rows.json');meta={}
    for case in CASES:
        g=next((x for x in gates if x['scene']==case and x['seed']==79100),None)
        if g is None:continue
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,d in zip(axes,g['directions']):
            spec=m['specs'][f'{case}_79100'];nodes=np.asarray(spec['network_nodes'])
            for i,j in spec['network_edges']:ax.plot(nodes[[i,j],0],nodes[[i,j],1],color='.8',lw=6,zorder=0)
            for s in d['stages']:ax.add_patch(Rectangle((s['center'][0]-s['width']/2,s['center'][1]-s['height']/2),s['width'],s['height'],fill=False,ec='green'))
            q=d['receiver_state'][:2];ax.add_patch(Circle(q,d['sensor_radius'],fill=False,ls='--',ec='orange'));ax.scatter(*q,c='orange',label='receiver')
            ax.scatter(*d['sender_state'][:2],c='blue',label='sender')
            if d['packet']:
                p=np.array(d['packet']['anchors']);ax.scatter(p[:,0],p[:,1],c='purple',marker='+',s=70,label='exported anchors')
            for e in d['outgoing']:
                p=np.array(e['points']);mask=np.array(e['supported']);ax.scatter(p[:,0],p[:,1],c=np.where(mask,'green','gray'),s=12)
            ax.set(xlim=(-1.2,6),ylim=(-2.5,3),aspect='equal',title=f"{case}: {d['sender']}→{d['receiver']}\n"+', '.join(k for k,v in d['checks'].items() if not v))
            ax.legend(fontsize=7)
        fig.suptitle('Actual acquisition boundary 0.96 s | green queries: committed local field support');fig.tight_layout();fig.savefig(figdir/f'{case}_eligibility.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for metric,ax in zip(('value_mae','gradient_mae'),axes):
        for case in CASES:
            a=[x for x in aa if x['case']==case and x['method']=='gate' and x['supported']]
            ax.scatter([x['errors']['no_fit'][metric] for x in a],[x['errors']['fit'][metric] for x in a],label=case)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],'k--');ax.set(xlabel='Fitting disabled',ylabel='Fitting enabled',title=metric);ax.legend()
    fig.tight_layout();fig.savefig(figdir/'estimation_errors.png',dpi=140);plt.close(fig)
    for case in CASES:
        paths={a:root/'episodes'/f'{case}_79100_{a}.json.gz' for a in METHODS}
        if not all(p.exists() for p in paths.values()):continue
        rs={a:load(p) for a,p in paths.items()};end=max(z['finish_time'] for r in rs.values() for z in r['agents']);times=np.unique(np.r_[np.linspace(0,end,64),.96]);frames=[]
        for now in times:
            fig,axes=plt.subplots(2,3,figsize=(14,7));fig.suptitle(f'{case} | seed 79100 | global time {now:.2f} s')
            for col,(method,r) in enumerate(rs.items()):
                ax=axes[0,col];nodes=np.array(r['spec']['network_nodes'])
                for i,j in r['spec']['network_edges']:ax.plot(nodes[[i,j],0],nodes[[i,j],1],color='.8',lw=8)
                ax.scatter(1.25,.45,c='red',alpha=.4,label='true risk center (evaluator)')
                for i,color in enumerate(('tab:blue','tab:orange')):
                    ss=[s for s in r['samples'][i] if s['time']<=now];q=np.array([s['x'][:2] for s in ss]);ax.plot(q[:,0],q[:,1],c=color);ax.scatter(*q[-1],c=color)
                    ax.text(.02,.97-.08*i,f"A{i}: {'GOAL' if ss[-1]['done'] and r['agents'][i]['success'] else 'FAILED' if ss[-1]['done'] else 'moving'}",transform=ax.transAxes,color=color)
                    ts=[t for t in r['trajectories'][i] if t['time']<=now]
                    if ts and ts[-1]['navigation'].get('route'):
                        p=np.array(ts[-1]['navigation']['route']);ax.plot(p[:,0],p[:,1],'--',c=color)
                for e in r['events']:
                    if e['kind']=='delivered' and 0<=now-e['time']<.96:
                        q=np.asarray(e['anchors']);ax.scatter(q[:,0],q[:,1],marker='+',c='purple')
                ax.set(title=method,xlim=(-1,9),ylim=(-1.5,3),aspect='equal')
                ax=axes[1,col]
                for i,color in enumerate(('tab:blue','tab:orange')):
                    ts=[t for t in r['trajectories'][i] if t['time']<=now]
                    ax.plot([t['time'] for t in ts],[t['navigation']['rr_route']['margin'] if t['navigation']['rr_route']['margin'] is not None else np.nan for t in ts],c=color,label=f'A{i}')
                for e in r['events']:
                    if e['kind']=='delivered' and e['time']<=now:ax.axvline(e['time'],color='purple',alpha=.12)
                ax.set(xlim=(0,end),xlabel='Time (s)',ylabel='Route margin (gaps: no alternative)');ax.legend()
            fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        frames[0].save(figdir/f'{case}.gif',save_all=True,append_images=frames[1:],duration=160,loop=0);frames[-1].save(figdir/f'{case}_final.png')
        meta[case]=dict(times=times.tolist(),sources={a:file_hash(p) for a,p in paths.items()},semantics='last logged sample at/before time; dashed plans are not traversed branches')
    dump(root/'animation_frames.json',meta)
    gallery=figdir/'failures';gallery.mkdir(exist_ok=True)
    (gallery/'index.html').write_text('<h1>All failures and incomplete/not-run episodes</h1><pre>'+json.dumps(load(root/'failures.json')+load(root/'incomplete.json'),indent=2)+'</pre>')
    with (root/'REPORT.md').open('a') as f:
        f.write('\n![Estimation errors](figures/estimation_errors.png)\n')
        for c in CASES:f.write(f'\n![{c} eligibility](figures/{c}_eligibility.png)\n')
        for c in meta:f.write(f'\n[{c} synchronized three-arm animation](figures/{c}.gif)\n')
