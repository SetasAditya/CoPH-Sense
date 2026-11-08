"""Evaluator-only summaries; no selection or tuning of receiver settings."""
import json
from pathlib import Path
import numpy as np
from .evaluation import dump,load
from .receiver_corrections import ARMS,VARIANTS
from .receiver_reconstruction import file_hash
from .complementary_reconstruction import branch_entry


def flatten(a,scope):
    rows=[];base=a['results']['no_fit/legacy'];be=base['common_errors']['proposed']
    for key,b in a['results'].items():
        d=b['diagnostic'];rd=d.get('receiver_fit',{});pe=b['common_errors']['proposed'];ce=b['common_errors']['committed']
        dv=None if pe['value_mae'] is None or be['value_mae'] is None else pe['value_mae']-be['value_mae']
        dg=None if pe['gradient_mae'] is None or be['gradient_mae'] is None else pe['gradient_mae']-be['gradient_mae']
        gate=rd.get('relative_gate_passed');ranks=b['ranking'];baseline=base['ranking']
        rows.append(dict(scope=scope,case=a['case'],seed=a['seed'],method=a.get('method','gate'),group=a.get('group'),intended=a.get('intended',False),
            packet=a['packet'],receiver=a['receiver'],time=a['time'],key=key,variant=b['variant'],validator=b['validator'],accepted=b['accepted'],reason=b['reason'],
            proposed_errors=pe,committed_errors=ce,value_delta=dv,gradient_delta=dg,
            field_delta=float(np.linalg.norm(np.array(b['committed_fields'])-base['committed_fields'])),control_delta=float(np.linalg.norm(np.array(b['control'])-base['control'])),
            target_delta=float(np.linalg.norm(np.array(b['target'])-base['target'])),route_changed=(ranks['best'] or {}).get('nodes')!=(baseline['best'] or {}).get('nodes'),
            margin=ranks['margin'],relative_gate=gate,relative_gate_applied=rd.get('additional_gate_applied',False),
            solver_success=rd.get('success'),solver_status=[s['status'] for s in rd.get('stages',[])],
            correction=b['proposal_max_correction'],stationarity=[s.get('stationarity') for s in rd.get('stages',[])],
            iterations=sum(s.get('iterations',0) for s in rd.get('stages',[])),validation_local=rd.get('local_validation'),validation_message=rd.get('message_validation'),
            threshold=rd.get('threshold'),flow=d['counterfactual_steps'],whole_model_checks=d.get('whole_model_checks_passed',b['accepted']),
            supported=pe['supported'],queries=pe['queries']))
    return rows


def summarize(rows):
    return dict(audits=len(rows),accepted=sum(r['accepted'] for r in rows),rejected=sum(not r['accepted'] for r in rows),
        improved=sum(r['value_delta'] is not None and r['value_delta']<-1e-12 for r in rows),worsened=sum(r['value_delta'] is not None and r['value_delta']>1e-12 for r in rows),
        unchanged=sum(r['value_delta'] is not None and abs(r['value_delta'])<=1e-12 for r in rows),unsupported=sum(r['value_delta'] is None for r in rows),
        mean_proposed_delta=float(np.mean([r['value_delta'] for r in rows if r['value_delta'] is not None])) if any(r['value_delta'] is not None for r in rows) else None,
        supported=sum(r['supported'] for r in rows),queries=sum(r['queries'] for r in rows),
        solver_failed=sum(r['solver_success'] is False for r in rows),route_changes=sum(r['route_changed'] for r in rows),
        control_changes=sum(r['control_delta']>1e-10 for r in rows),target_changes=sum(r['target_delta']>1e-10 for r in rows))


def report(root):
    root=Path(root);m=load(root/'manifest.json');audits=[];rows=[];lineage=[];failures=[];incomplete=[];checks=[];paired=[]
    for p in sorted((root/'offline').glob('*.json.gz')):
        for a in load(p):audits+=flatten(a,'saved')
    for key,spec in m['specs'].items():
        for arm in ARMS:
            identity=f'{key}_{arm}';status=root/'status'/f'{identity}.json'
            if not status.exists():incomplete.append(dict(id=identity,reason='not run; see gate/default checks',executed=False));continue
            s=load(status)
            if not s['complete']:incomplete.append(s);continue
            r=load(root/'episodes'/s['path'])
            for a in r['receiver_audits']:audits+=flatten(a,'fresh')
            checks.append(dict(id=identity,sha256=s['sha256'],acquisition=r['acquisition_check'],audits=len(r['receiver_audits']),flow=max(a['max_update_work'] for a in r['agents']),
                physical_and_measurement_isolation=all(a['state_unchanged'] and a['measurements_unchanged'] for a in r['receiver_audits'])))
            for i,outcome in enumerate(r['agents']):
                u=r['updates'][i];controls=r['trajectories'][i];traversal=[]
                for sample in r['samples'][i]:
                    edge=branch_entry(sample['x'][:2])
                    if edge and (not traversal or traversal[-1]['edge']!=edge):traversal.append(dict(time=sample['time'],edge=edge))
                reasons={}
                for z in u:
                    d=z['diagnostic'];reason=d['reason'];reasons[reason]=reasons.get(reason,0)+1
                    trace=d.get('rr_trace',{})
                    for e in trace.get('events',[]):lineage.append(dict(e,id=identity,agent=i,accepted=d['accepted'],reason=reason,version=z['post']['version'],
                        receiver_fit=d.get('receiver_fit'),residuals=trace.get('residuals_after')))
                row=dict(outcome,id=identity,case=spec['scene'],seed=spec['seed'],method=arm,unweighted_risk=outcome['components']['risk']/(4*spec['risk_weights'][i]),
                    branch_entries=traversal,updates=len(u),accepted_updates=sum(z['diagnostic']['accepted'] for z in u),reasons=reasons,
                    receiver_diagnostics=r['receiver_diagnostics'][i],route_expansions=sum(t['navigation'].get('route_expansions',0) for t in controls),
                    projection_candidates=sum(t['navigation'].get('projection_candidates',0) for t in controls),
                    wall_seconds=r['wall_seconds'],audit_seconds=r['audit_seconds'],audit_flow=r['audit_flow'],snapshot_seconds=r['snapshot_seconds'])
                rows.append(row)
                if not outcome['success']:failures.append(row)
    for key,spec in m['specs'].items():
        for i in (0,1):
            rr={r['method']:r for r in rows if r['case']==spec['scene'] and r['seed']==spec['seed'] and r['agent']==i}
            for a,b in [('corrected_fit','legacy'),('validation_only','legacy'),('combined','corrected_fit'),('combined','local_only')]:
                if a in rr and b in rr:paired.append(dict(case=spec['scene'],seed=spec['seed'],agent=i,comparison=f'{a} minus {b}',
                    time=rr[a]['active_seconds']-rr[b]['active_seconds'],risk=rr[a]['unweighted_risk']-rr[b]['unweighted_risk'],
                    same_branch=[e['edge'] for e in rr[a]['branch_entries']]==[e['edge'] for e in rr[b]['branch_entries']]))
    summaries=[];confusion=[]
    for scope in ('saved','fresh'):
        for group in ('intended_acquisition','all_acquisition','navigation'):
            sub=[r for r in audits if r['scope']==scope and ((r['group']=='acquisition' and (group=='all_acquisition' or group=='intended_acquisition' and r['intended'])) or r['group']=='navigation' and group=='navigation')]
            for key in sorted({r['key'] for r in sub}):
                rs=[r for r in sub if r['key']==key];summaries.append(dict(scope=scope,group=group,key=key,**summarize(rs)))
                gated=[r for r in rs if r['relative_gate'] is not None and r['value_delta'] is not None]
                confusion.append(dict(scope=scope,group=group,key=key,evaluable=len(gated),
                    harmful=sum(r['value_delta']>1e-12 for r in gated),harmful_rejected=sum(r['value_delta']>1e-12 and not r['relative_gate'] for r in gated),
                    useful=sum(r['value_delta']<-1e-12 for r in gated),useful_rejected=sum(r['value_delta']<-1e-12 and not r['relative_gate'] for r in gated),
                    unchanged=sum(abs(r['value_delta'])<=1e-12 for r in gated),gate_not_evaluable=len(rs)-len(gated)))
    for name,value in [('rows',rows),('checks',checks),('paired',paired),('failures',failures),('incomplete',incomplete),('summaries',summaries),('validator_confusion',confusion)]:dump(root/f'{name}.json',value)
    dump(root/'audit_rows.json.gz',audits);dump(root/'lineage.json.gz',lineage)
    defaults=load(root/'default_checks.json') if (root/'default_checks.json').exists() else None;gate=load(root/'gate.json') if (root/'gate.json').exists() else None
    lines=['# Receiver fitting and validation corrections','',
        'Opt-in development experiment. Fixed local-only field proposal; precision-scaled message correction with a fixed ±0.5 box; relative-local validation is an additional transactional gate. No controller, support, packet or replay changes.',
        '',f'Default reproduction: {defaults["passed"] if defaults else "unavailable"}; acquisition gate: {gate["passed"] if gate else "not run"}.',
        f'Completed episodes: {len(checks)}/30. Agent goals: {sum(r["success"] for r in rows)}/{len(rows)}. Collisions: {sum(r["collision"] for r in rows)}; timeouts: {sum(r["timeout"] for r in rows)}. Incomplete: {len(incomplete)}.',
        '', '## Saved-snapshot proposals and commitments','',
        '| Group | Variant / validator | Audits | Proposal improves / worsens | Accepted | Solver failures | Mean value-MAE change | Supported / all queries |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['scope']=='saved':lines.append(f"| {s['group']} | {s['key']} | {s['audits']} | {s['improved']} / {s['worsened']} | {s['accepted']} | {s['solver_failed']} | {s['mean_proposed_delta']} | {s['supported']} / {s['queries']} |")
    lines+=['','Proposal errors are evaluated before validation and rollback. Committed errors are saved separately. Convergence requires normalized projected-gradient stationarity; reaching the iteration limit is a rejected solve, not success. The ±0.5 bound is relative to the fresh local-only proposal, not the old committed model.',
        '', '## Navigation outcomes','', '| Case | Seed | Arm | Agent | Goal | Time | Unweighted integrated risk | Accepted / updates |', '|---|---|---|---:|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['seed']} | {r['method']} | {r['agent']} | {r['success']} | {r['active_seconds']:.2f} | {r['unweighted_risk']:.6f} | {r['accepted_updates']}/{r['updates']} |")
    lines+=['','## Interpretation contract','',
        'The local information matrix is a frozen delta-method approximation, not a calibrated posterior. Original overlap rows, ridge and marginal-block construction are preserved; clipping and repeated stage measurements retain their previous treatment. No truth or synthetic observations enter runtime fitting or validation.',
        'Relative-local validation uses 5% of max(local-only held-out MSE, sensor variance), plus 1e-12. Existing whole-model checks remain. Rejection rolls back the entire model, so local progress is also withheld; no partial commitment is claimed.',
        'Confusion tables distinguish rejected harmful proposals from rejected useful proposals using evaluator-only route-query error. Solver failures and unavailable validation data are separate exclusions. Query occurrences are correlated, not independent trials.',
        'All primary arms use supported-field routing, identical packet protocols and transmission opportunities. Realized packets may differ after trajectories diverge. Fitting, routing, projection, identification, audit and snapshot work are recorded separately. Wall times include instrumentation.',
        'Success requires valid isolation and reproducibility, not estimation or navigation superiority. No decision-sensitive fixture, fusion tuning, richer basis, adaptive rates or held-out evaluation was added.',
        '', '[Validator confusion](validator_confusion.json) · [Paired outcomes](paired.json) · [Episode/computation rows](rows.json) · [Lineage](lineage.json.gz) · [Failures](failures.json) · [Incomplete](incomplete.json)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return rows


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(root);m=load(root/'manifest.json');figdir=root/'figures';figdir.mkdir(exist_ok=True)
    summaries=load(root/'summaries.json');conf=load(root/'validator_confusion.json');audits=load(root/'audit_rows.json.gz')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    rs=[s for s in summaries if s['scope']=='saved' and s['group']=='navigation' and s['key'].endswith('/legacy')]
    axes[0].barh([s['key'].split('/')[0] for s in rs],[s['mean_proposed_delta'] or 0 for s in rs]);axes[0].axvline(0,c='k',lw=.5);axes[0].set(title='Saved navigation proposals',xlabel='Mean value MAE change vs local-only')
    rr=[r for r in conf if r['scope']=='saved' and r['group']=='navigation' and r['evaluable']]
    x=np.arange(len(rr));axes[1].bar(x-.2,[r['harmful_rejected']/r['harmful'] if r['harmful'] else np.nan for r in rr],width=.4,label='Harmful rejected');axes[1].bar(x+.2,[r['useful_rejected']/r['useful'] if r['useful'] else np.nan for r in rr],width=.4,label='Useful rejected')
    axes[1].set(xticks=x,xticklabels=[r['key'].split('/')[0] for r in rr],ylim=(0,1),title='Relative gate (evaluable proposals)');axes[1].tick_params(axis='x',rotation=45);axes[1].legend(fontsize=7)
    rr=[a for a in audits if a['scope']=='saved' and a['key']=='precision_iterative/legacy'];axes[2].scatter([r['iterations'] for r in rr],[r['correction'] for r in rr],c=['green' if r['solver_success'] else 'red' for r in rr],s=14);axes[2].axhline(.5,c='k',ls='--');axes[2].set(xlabel='Iterations summed over stages',ylabel='Maximum message correction',title='Green: converged; red: solver failure')
    fig.tight_layout();fig.savefig(figdir/'fitting_validation.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key in ('original/legacy','precision_iterative/legacy','precision_iterative/relative_local'):
        rr=[a for a in audits if a['scope']=='saved' and a['key']==key and a['value_delta'] is not None]
        axes[0].scatter([a['time'] for a in rr],[a['supported']/a['queries'] for a in rr],s=12,label=key)
        axes[1].scatter([a['proposed_errors']['value_mae'] for a in rr],[a['committed_errors']['value_mae'] for a in rr],s=12,label=key)
    axes[0].set(xlabel='Packet time',ylabel='Common supported query fraction');axes[1].set(xlabel='Proposed value MAE',ylabel='Committed value MAE');axes[0].legend(fontsize=6);axes[1].legend(fontsize=6)
    fig.tight_layout();fig.savefig(figdir/'support_and_errors.png',dpi=140);plt.close(fig)
    meta={}
    for case in ('complementary','reverse','redundant'):
        paths={arm:root/'episodes'/f'{case}_79200_{arm}.json.gz' for arm in ARMS}
        if not all(p.exists() for p in paths.values()):continue
        results={a:load(p) for a,p in paths.items()};end=max(a['finish_time'] for r in results.values() for a in r['agents']);times=np.unique(np.r_[np.linspace(0,end,64),.96]);frames=[]
        for now in times:
            fig,axes=plt.subplots(2,5,figsize=(20,7));fig.suptitle(f'{case} | seed 79200 | time {now:.2f}s | replay disabled')
            for col,(arm,r) in enumerate(results.items()):
                ax=axes[0,col];nodes=np.asarray(r['spec']['network_nodes'])
                for i,j in r['spec']['network_edges']:ax.plot(nodes[[i,j],0],nodes[[i,j],1],color='.8',lw=7)
                ax.scatter(1.25,.45,c='red',alpha=.4,s=25)
                for i,color in enumerate(('tab:blue','tab:orange')):
                    ss=[s for s in r['samples'][i] if s['time']<=now];q=np.array([s['x'][:2] for s in ss]);ax.plot(q[:,0],q[:,1],color=color);ax.scatter(*q[-1],color=color,s=12)
                    ax.text(.02,.98-.1*i,f"A{i}: {'GOAL' if ss[-1]['done'] and r['agents'][i]['success'] else 'FAILED' if ss[-1]['done'] else 'moving'}",transform=ax.transAxes,color=color,va='top',fontsize=7)
                    ts=[t for t in r['trajectories'][i] if t['time']<=now]
                    if ts and ts[-1]['navigation'].get('route'):
                        p=np.asarray(ts[-1]['navigation']['route']);ax.plot(p[:,0],p[:,1],'--',color=color,alpha=.5)
                for e in r['events']:
                    if e['kind']=='delivered' and 0<=now-e['time']<.96:
                        p=np.asarray(e['anchors']);ax.scatter(p[:,0],p[:,1],marker='+',c='purple',s=18)
                ax.set(title=arm,xlim=(-1,9),ylim=(-1.5,3),aspect='equal')
                ax=axes[1,col]
                for i,color in enumerate(('tab:blue','tab:orange')):
                    ts=[t for t in r['trajectories'][i] if t['time']<=now];ax.plot([t['time'] for t in ts],[t['navigation']['rr_route']['margin'] if t['navigation']['rr_route']['margin'] is not None else np.nan for t in ts],color=color)
                    for u in r['updates'][i]:
                        if u['time']<=now and not u['diagnostic']['accepted']:ax.scatter(u['time'],0,c=color,marker='x',s=14)
                ax.set(xlim=(0,end),ylim=(-.5,20),xlabel='Time (s)',ylabel='Route margin; × rejected update')
            fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        frames[0].save(figdir/f'{case}.gif',save_all=True,append_images=frames[1:],duration=160,loop=0);frames[-1].save(figdir/f'{case}_final.png')
        meta[case]=dict(times=times.tolist(),sources={a:file_hash(p) for a,p in paths.items()},semantics='logged states at or before global time; red point: evaluator risk center; missing alternative route margins are gaps')
    dump(root/'animation_frames.json',meta)
    gallery=figdir/'failures';gallery.mkdir(exist_ok=True)
    failures=load(root/'failures.json');incomplete=load(root/'incomplete.json')
    for failure in failures:
        path=root/'episodes'/f"{failure['id']}.json.gz";r=load(path);fig,ax=plt.subplots(figsize=(9,4));nodes=np.asarray(r['spec']['network_nodes'])
        for i,j in r['spec']['network_edges']:ax.plot(nodes[[i,j],0],nodes[[i,j],1],c='.8',lw=7)
        q=np.asarray([s['x'][:2] for s in r['samples'][failure['agent']]]);ax.plot(q[:,0],q[:,1]);ax.scatter(*q[-1],c='red');ax.set(title=f"{failure['id']} agent {failure['agent']}: collision={failure['collision']}, timeout={failure['timeout']}",aspect='equal');fig.savefig(gallery/f"{failure['id']}_{failure['agent']}.png",dpi=130);plt.close(fig)
    (gallery/'index.html').write_text('<h1>All failures and incomplete runs</h1><pre>'+json.dumps(failures+incomplete,indent=2)+'</pre>')
    with (root/'REPORT.md').open('a') as f:
        f.write('\n![Fitting and validation](figures/fitting_validation.png)\n\n![Support and errors](figures/support_and_errors.png)\n')
        for case in meta:f.write(f'\n[{case}: synchronized five-arm GIF](figures/{case}.gif)\n')
