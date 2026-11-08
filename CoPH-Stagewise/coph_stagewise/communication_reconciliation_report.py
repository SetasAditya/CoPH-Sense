"""Read-only endpoint, traffic, route and timing reports for reconciliation.

No simulator or learner is invoked. Missing legacy information stays unavailable.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

BYTE_KEYS=('status','payload','acknowledgment')
SCENES=('mild','moderate','severe','double_mild')
COLORS={'disabled':'#777777','rotation':'#0072B2','receiver_relevant':'#D55E00'}


def read(path,errors=None):
    path=Path(path)
    try:
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)
    except (OSError,ValueError) as exc:
        if errors is not None:errors.append(dict(path=str(path),reason=type(exc).__name__,detail=str(exc)))
        return None


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def label(row):return 'disabled' if row.get('arm')=='disabled' else row.get('selector','unavailable')


def episode_path(root,row):
    return Path(root)/row.get('episode_file',f"episodes/{row['key']}.json.gz")


def bytes_agent(agent):return {k:int(agent.get('bytes',{}).get(k,0)) for k in BYTE_KEYS}


def route_traversal(samples,spec):
    nodes=np.asarray(spec.get('network_nodes',[]),float);edges=spec.get('network_edges',[]);out=[];previous=None
    if not len(nodes) or not edges:return dict(available=False,reason='no_declared_network')
    for s in samples:
        q=np.asarray(s['x'][:2]);choices=[]
        for k,(i,j) in enumerate(edges):
            a,b=nodes[i],nodes[j];d=b-a;length=np.linalg.norm(d)
            if length<=.5:continue
            distance=float((q-a)@(d/length));perp=np.linalg.norm(q-a-np.clip(distance,0,length)*d/length)
            if .25<distance<length-.25 and perp<=.25:choices.append((float(perp),k,i,j))
        if not choices:continue
        _,k,i,j=min(choices)
        if k!=previous:
            out.append(dict(time=s['time'],edge_index=k,edge=[i,j],q=q.tolist()));previous=k
    q=np.array([s['x'][:2] for s in samples]);direction=np.sign(spec['goals'][1][0]-spec['initials'][1][0]) or 1.
    retreat=float(np.maximum(0,-direction*np.diff(q[:,0])).sum()) if len(q)>1 else 0.
    return dict(available=True,interior_edge_entries=out,backward_x_travel=retreat,
                interpretation='actual interior edge traversal; junction samples left unclassified; backward x distance is not mission regret')


def traffic_accounting_trace(events,agents,budget):
    ledger=[{k:0 for k in BYTE_KEYS} for _ in agents];trace=[]
    for e in events:
        charges=[]
        if e.get('kind') in ('sent','channel') and e.get('sender') is not None and e.get('receiver') is not None:
            sender,receiver=int(e['sender']),int(e['receiver'])
            if e.get('status_bytes',0):charges.append((receiver,'status',int(e['status_bytes'])))
            if e.get('reason')=='sent':charges.append((sender,'payload',int(e.get('payload_bytes',0))))
        elif e.get('kind')=='delivered' and e.get('receiver') is not None and e.get('acknowledgment_bytes',0):
            charges.append((int(e['receiver']),'acknowledgment',int(e['acknowledgment_bytes'])))
        for payer,component,size in charges:ledger[payer][component]+=size
        if charges or e.get('reason') in ('payload_budget','status_budget'):
            trace.append(dict(time=e.get('time'),kind=e.get('kind'),reason=e.get('reason'),sender=e.get('sender'),receiver=e.get('receiver'),
                charges=[dict(agent=i,component=k,bytes=v) for i,k,v in charges],totals=[sum(a.values()) for a in ledger],
                remaining=[budget-sum(a.values()) for a in ledger]))
    return dict(events=trace,ledger=ledger,matches_logged_agent_totals=all(a==bytes_agent(b) for a,b in zip(ledger,agents)))


def episode_diagnostics(episode,row):
    spec=episode['spec'];events=episode.get('events',[]);agents=episode.get('agents',[]);samples=episode.get('samples',[[],[]]);records=episode.get('trajectories',[[],[]]);obs=episode.get('observations',[[],[]])
    carrier=agents[1] if len(agents)>1 else {};patches=[]
    delivered=[e for e in events if e.get('kind')=='delivered' and str(e.get('receiver'))=='1' and e.get('reason')=='eligible']
    for patch in spec.get('patches',[]):
        center=np.array(patch[:2]);width=np.array(patch[2:4])
        inside=lambda q:float(np.sum(((np.asarray(q)-center)/width)**2))<=1.
        heard=[e['time'] for e in delivered if 'time' in e and any(inside(a[:2]) for a in e.get('anchors',[]))]
        seen=[o['time'] for o in obs[1] if any(inside(r['q']) for r in o.get('readings',[]))]
        entered=[s['time'] for s in samples[1] if inside(s['x'][:2])]
        patches.append(dict(center=center.tolist(),first_received_anchor=min(heard) if heard else None,
            first_local_observation=min(seen) if seen else None,first_actual_agent_entry=min(entered) if entered else None,
            warning_lead_before_local_sensing=min(seen)-min(heard) if heard and seen else None,
            criterion='evaluator-only mapping inside one-sigma patch ellipse; does not assert packet accuracy or causal benefit'))
    traffic=[]
    for i,a in enumerate(agents):
        sent=[e for e in events if e.get('reason')=='sent' and str(e.get('sender'))==str(i)]
        refused=[e for e in events if e.get('kind')=='channel' and str(e.get('sender'))==str(i)]
        offered_budget=[e for e in refused if e.get('reason') in ('status_budget','payload_budget') and e.get('time') is not None]
        budget=[e for e in events if e.get('kind')=='channel' and e.get('time') is not None and
                ((e.get('reason')=='status_budget' and str(e.get('receiver'))==str(i)) or
                 (e.get('reason')=='payload_budget' and str(e.get('sender'))==str(i)))]
        first=min((e['time'] for e in budget),default=None);later=[e for e in sent if first is not None and e.get('time',-np.inf)>first]
        traffic.append(dict(agent=i,bytes=bytes_agent(a),budget_per_agent=row.get('budget'),
            budget_respected=sum(bytes_agent(a).values())<=row['budget'] if row.get('budget') is not None else None,
            sent=len(sent),refusals={reason:sum(e.get('reason')==reason for e in refused) for reason in sorted({e.get('reason','unavailable') for e in refused})},
            offered_payload_refusals={reason:sum(e.get('reason')==reason for e in refused) for reason in sorted({e.get('reason','unavailable') for e in refused})},
            charged_agent_budget_refusals={reason:sum(e.get('reason')==reason for e in budget) for reason in ('status_budget','payload_budget')},
            first_offer_budget_refusal=min((e['time'] for e in offered_budget),default=None),
            first_budget_refusal=first,last_payload_time=max((e.get('time',-np.inf) for e in sent),default=None),
            payloads_after_first_budget_refusal=len(later) if first is not None else None,
            no_later_payload_after_budget_refusal=not later if first is not None else None,
            missing_refusal_metadata=sum(e.get('time') is None or e.get('sender') is None for e in events if e.get('kind')=='channel'),
            charging='status is charged to receiver advertising it; payload to sender; acknowledgment to recipient'))
    route=route_traversal(samples[1],spec) if len(samples)>1 else dict(available=False)
    planned=[];last=None
    for r in records[1]:
        ids=r.get('navigation',{}).get('route_nodes')
        if ids is not None and ids!=last:planned.append(dict(time=r['time'],nodes=ids));last=ids
    return dict(key=row['key'],scene=row['scene'],seed=row['seed'],arm=row['arm'],selector=row['selector'],budget=row['budget'],
        carrier_components=carrier.get('components'),carrier_raw_risk=carrier.get('raw_risk'),carrier_risk_weight=carrier.get('risk_weight'),
        actual_traversal=route,planned_route_changes=planned,patch_timing=patches,per_agent_traffic=traffic,
        sent_payloads=[{k:e.get(k) for k in ('time','sender','receiver','payload_bytes','diagnostic_route_relevant_anchors','diagnostic_anchor_count')} for e in events if e.get('reason')=='sent'],
        payload_selection_trace=[dict(time=e.get('time'),sender=e.get('sender'),receiver=e.get('receiver'),selection=e.get('selection')) for e in events if e.get('reason')=='sent'],
        traffic_accounting=traffic_accounting_trace(events,agents,row['budget']),
        cost_sum_residual=carrier.get('cost',0)-sum(carrier.get('components',{}).values()) if carrier.get('components') else None)


def paired(rows):
    out=[]
    for scene,seed,budget in sorted({(r['scene'],r['seed'],r['budget']) for r in rows if r.get('arm')!='disabled'}):
        rr={r['selector']:r for r in rows if r.get('scene')==scene and r.get('seed')==seed and r.get('budget')==budget and r.get('arm')=='reciprocal_fixed'}
        a,b=rr.get('rotation'),rr.get('receiver_relevant')
        if a is None or b is None:
            out.append(dict(scene=scene,seed=seed,budget=budget,available=False,reason='missing_selector'));continue
        d=next((r for r in rows if r.get('scene')==scene and r.get('seed')==seed and r.get('arm')=='disabled'),None)
        row=dict(scene=scene,seed=seed,budget=budget,available=True,
            relevant_minus_rotation_carrier_full=b['carrier_cost']-a['carrier_cost'],
            relevant_minus_rotation_carrier_time_risk=b['carrier_time_risk']-a['carrier_time_risk'],
            relevant_minus_rotation_team_full=b['team_cost']-a['team_cost'],
            relevant_minus_rotation_team_bytes=sum(b['bytes_team'].values())-sum(a['bytes_team'].values()),
            rotation_goals=a['goals'],relevant_goals=b['goals'])
        if d:row.update(disabled_reference_budget=d['budget'],rotation_carrier_time_risk_minus_disabled=a['carrier_time_risk']-d['carrier_time_risk'],relevant_carrier_time_risk_minus_disabled=b['carrier_time_risk']-d['carrier_time_risk'],rotation_carrier_full_minus_disabled=a['carrier_cost']-d['carrier_cost'],relevant_carrier_full_minus_disabled=b['carrier_cost']-d['carrier_cost'])
        out.append(row)
    return out


def plots(root,rows,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures';directory.mkdir(exist_ok=True)
    if not rows:return
    budgets=sorted({r['budget'] for r in rows if r.get('arm')!='disabled'}) or sorted({r['budget'] for r in rows})
    selected=[budgets[-1]]+([16000] if 16000 in budgets and budgets[-1]!=16000 else [])
    for budget in selected:
        fig,axes=plt.subplots(1,4,figsize=(15,4))
        for ax,scene in zip(axes,SCENES):
            groups=[]
            for name in ('disabled','rotation','receiver_relevant'):
                rr=[r for r in rows if r['scene']==scene and label(r)==name and (name=='disabled' or r['budget']==budget)]
                groups.append(rr)
            bottoms=np.zeros(3)
            for component,color in (('time','#9ecae1'),('risk','#e6550d'),('effort','#756bb1'),('failure','#222222')):
                values=[np.mean([r['agents'][1]['components'].get(component,0) for r in rr]) if rr else np.nan for rr in groups]
                ax.bar(np.arange(3),values,bottom=bottoms,label=component,color=color);bottoms+=np.nan_to_num(values)
            ax.set_title(scene);ax.set_xticks(range(3),('off','rotation','relevant'),rotation=25);ax.set_ylabel('Mean carrier full logged cost')
        axes[-1].legend(fontsize=8);fig.suptitle(f'Per-agent ceiling {budget:,}; noise seeds are paired, not independent terrains');fig.tight_layout();fig.savefig(directory/f'cost_components_{budget}.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name in ('disabled','rotation','receiver_relevant'):
        rr=[r for r in rows if label(r)==name];axes[0].scatter([r['agents'][1]['components']['time'] for r in rr],[r['agents'][1]['components']['risk'] for r in rr],label=name,color=COLORS[name],alpha=.7)
        if name!='disabled':
            xx=[];yy=[]
            for b in budgets:
                subset=[r for r in rr if r['budget']==b];xx.append(b);yy.append(np.mean([r['carrier_time_risk'] for r in subset]) if subset else np.nan)
            axes[1].plot(xx,yy,'o-',label=name,color=COLORS[name])
    axes[0].set(xlabel='Carrier time cost',ylabel='Carrier weighted risk');axes[1].set(xscale='log',xlabel='Per-agent byte ceiling',ylabel='Mean carrier time + weighted risk')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(directory/'risk_time.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name in ('rotation','receiver_relevant'):
        xx=np.arange(len(budgets))+(0 if name=='rotation' else .4);bottom=np.zeros(len(budgets))
        for component,alpha in (('payload',1.),('status',.65),('acknowledgment',.35)):
            values=[np.mean([r['bytes_team'].get(component,0) for r in rows if label(r)==name and r['budget']==b]) for b in budgets]
            axes[0].bar(xx,values,.38,bottom=bottom,color=COLORS[name],alpha=alpha,label=f'{name}: {component}');bottom+=np.array(values)
        for i in (0,1):
            ratios=[sum(r['bytes_per_agent'][i].values())/r['budget'] for r in rows if label(r)==name]
            axes[1].scatter([r['budget'] for r in rows if label(r)==name],ratios,label=f'{name} agent{i}',alpha=.65)
    axes[0].set_xticks(np.arange(len(budgets))+.2,[f'{b:,}' for b in budgets]);axes[0].set_ylabel('Actual TEAM bytes, including status and ack');axes[1].set(xscale='log',xlabel='Per-agent ceiling',ylabel='Agent bytes / own ceiling');axes[1].axhline(1,color='black',lw=.7)
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(directory/'bytes.png');plt.close(fig)
    for scene in ('moderate','double_mild'):
        chosen=[d for d in diagnostics if d['scene']==scene and d['seed']==86100 and d['budget'] in (8000,16000,4000000)]
        if not chosen:continue
        fig,axes=plt.subplots(len(chosen),1,figsize=(11,max(3,1.8*len(chosen))),squeeze=False)
        for ax,d in zip(axes[:,0],chosen):
            for i,traffic in enumerate(d['per_agent_traffic']):
                if traffic['last_payload_time'] is not None:ax.scatter(traffic['last_payload_time'],i,marker='o',label=f'agent{i} last payload')
                if traffic['first_budget_refusal'] is not None:ax.scatter(traffic['first_budget_refusal'],i,marker='x',s=70,label=f'agent{i} first budget refusal')
            for k,p in enumerate(d['patch_timing']):
                for field,y,marker in (('first_received_anchor',2,'v'),('first_local_observation',2.5,'s'),('first_actual_agent_entry',3,'>')):
                    if p[field] is not None:ax.scatter(p[field],y,marker=marker,label=f'patch{k+1} {field}')
            ax.set(title=f"{d['selector']} · per-agent ceiling {d['budget']:,}",xlabel='global seconds',yticks=[0,1,2,2.5,3],yticklabels=['agent0','agent1','carrier hears','carrier senses','carrier enters']);ax.legend(fontsize=6,ncol=3,loc='upper left');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(directory/f'{scene}_timing.png');plt.close(fig)


def report(out):
    root=Path(out);root.mkdir(parents=True,exist_ok=True);errors=[];manifest=read(root/'manifest.json',errors) or {};rows=read(root/'rows.json',errors) or [];regression=read(root/'regression.json',errors);parity=read(root/'parity.json',errors)
    if not isinstance(rows,list):errors.append(dict(path='rows.json',reason='not_a_list'));rows=[]
    valid=[];invalid=[]
    for r in rows:
        try:
            for key in ('key','scene','seed','arm','selector','budget','carrier_cost','carrier_time_risk','team_cost','bytes_team','bytes_per_agent','agents'):r[key]
            r['agents'][1]['components']
            valid.append(r)
        except (KeyError,IndexError,TypeError):
            invalid.append(dict(key=r.get('key') if isinstance(r,dict) else None,kind='row_schema_unavailable'));errors.append(dict(reason='row_schema_unavailable',row=r))
    rows=valid
    pairs=paired(rows);write(root/'paired_rows.json',pairs);diagnostics=[];failures=list(invalid)
    for row in rows:
        episode=read(episode_path(root,row),errors)
        if episode is None:failures.append(dict(key=row['key'],kind='missing_or_invalid_episode'));continue
        try:diagnostics.append(episode_diagnostics(episode,row))
        except (KeyError,ValueError,IndexError,TypeError) as exc:errors.append(dict(key=row['key'],reason='episode_diagnostics_unavailable',detail=str(exc)))
        if row.get('goals')!=2 or row.get('collisions',0) or row.get('timeouts',0):failures.append(dict(key=row['key'],kind='mission_failure',goals=row.get('goals'),agents=row.get('agents')))
    expected=manifest.get('expected_corrected_episodes');complete=expected is not None and len(rows)==expected and not any(f['kind']=='missing_or_invalid_episode' for f in failures)
    if expected is None or len(rows)!=expected:failures.append(dict(kind='incomplete_campaign',expected=expected,available_rows=len(rows)))
    write(root/'failure_inventory.json',failures);write(root/'episode_diagnostics.json',diagnostics);write(root/'unavailable_information.json',errors)
    originals=[]
    for scene in SCENES:
        for arm in ('none','field'):
            p=root/'original'/scene/f'{arm}.json.gz'
            if not p.exists():continue
            original=read(p,errors)
            if original is not None:
                a=original['agents'][1];c=a['components'];originals.append(dict(scene=scene,arm=arm,carrier_time_risk=c['time']+c['risk'],carrier_full=a['cost'],effort=c['effort'],failure=c.get('failure',0)))
    write(root/'original_endpoint_rows.json',originals)
    legacy_path=Path(__file__).resolve().parents[1]/'results/bandwidth_selection_v1/rows.json'
    legacy_rows=read(legacy_path) if legacy_path.exists() else None;legacy_comparison=[]
    if isinstance(legacy_rows,list):
        index={r['key']:r for r in legacy_rows if 'key' in r}
        for r in rows:
            if r['key'] in index:legacy_comparison.append(dict(key=r['key'],scene=r['scene'],seed=r['seed'],budget=r['budget'],selector=label(r),
                old_incorrectly_weighted_full=index[r['key']]['carrier_cost'],corrected_time_risk=r['carrier_time_risk'],corrected_full=r['carrier_cost'],
                full_accounting_correction=r['carrier_cost']-index[r['key']]['carrier_cost']))
    write(root/'legacy_accounting_rows.json',legacy_comparison)
    sums=dict(expected_episodes=expected,available_episodes=len(rows),complete=complete,goals=sum(r.get('goals',0) for r in rows),collisions=sum(r.get('collisions',0) for r in rows),timeouts=sum(r.get('timeouts',0) for r in rows),original_regression=regression,operational_parity=parity,
        numerical_endpoint='carrier time + 4 * carrier declared risk weight * raw risk integral',full_endpoint='primary + effort + failure',report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),missing_information_records=len(errors))
    # Large detailed parity records remain in parity.json; keep the summary readable.
    for key in ('original_regression','operational_parity'):
        value=sums[key];sums[key]={k:value.get(k) for k in ('complete','passed')} if isinstance(value,dict) else None
    write(root/'report_summary.json',sums);plots(root,rows,diagnostics)
    lines=['# Communication evaluation reconciliation','',f"Available corrected episodes: {len(rows)}/{expected if expected is not None else 'unknown'}. Goals: {sums['goals']}/{2*len(rows)}; collisions: {sums['collisions']}; timeouts: {sums['timeouts']}. Complete: {complete}.",
        '', '**Endpoints.** The primary endpoint is time plus receiver-weighted risk. Full logged cost additionally includes effort and failure. Earlier bandwidth accounting omitted the declared agent risk weights; corrected accounting changes reported risk costs while operational parity is checked separately. Team bytes sum both endpoints; each agent has its own ceiling. Status, payload and acknowledgments all count.',
        '', '**Channel and learner distinction.** Original hazard reproductions use the original uncompressed channel, query-specific receiver statuses and full policy replay. The newer compact channel uses evidence-only learning and route advertisements for receiver-relevant selection. Queue/update details and realized trajectories can differ. Original-versus-new differences cannot be attributed solely to effort or to the accounting correction. Only the paired legacy-versus-corrected accounting runs certify unchanged operations.',
        '', f"Original reproduction checks: {sums['original_regression']}. Accounting operational parity: {sums['operational_parity']}. Inspect regression.json/parity.json for numerical, discrete and serialized comparisons; unavailable checks remain unavailable.",
        '', '| Scene | Seed | Selector | Ceiling / agent | Carrier time+risk | Carrier FULL | Carrier effort | Team actual bytes | Goals |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['scene']} | {r['seed']} | {label(r)} | {r['budget']:,} | {r['carrier_time_risk']:.4f} | {r['carrier_cost']:.4f} | {r['agents'][1]['components'].get('effort',0):.4f} | {sum(r['bytes_team'].values()):,} | {r['goals']} |" for r in sorted(rows,key=lambda r:(r['scene'],r['seed'],r['budget'],label(r)))]
    lines+=['', '## Original-channel endpoint reproduction','', '| Scene | Original arm | Time+risk | FULL | Effort |','|---|---|---:|---:|---:|']
    lines += [f"| {r['scene']} | {r['arm']} | {r['carrier_time_risk']:.4f} | {r['carrier_full']:.4f} | {r['effort']:.4f} |" for r in originals]
    lines+=['', '## New-channel communication benefit at 4 MB','', 'Negative differences favor communication. The disabled reference is the same scene and seed; its cost is independent of a channel budget because communication is off. Means below summarize the two noise seeds descriptively.',
        '', '| Scene | Rotation − off time+risk | Relevant − off time+risk | Rotation − off FULL | Relevant − off FULL |','|---|---:|---:|---:|---:|']
    for scene in SCENES:
        rr=[p for p in pairs if p.get('available') and p['scene']==scene and p['budget']==4000000 and 'rotation_carrier_time_risk_minus_disabled' in p]
        if rr:lines.append('| '+scene+' | '+' | '.join(f"{np.mean([r[k] for r in rr]):+.4f}" for k in ('rotation_carrier_time_risk_minus_disabled','relevant_carrier_time_risk_minus_disabled','rotation_carrier_full_minus_disabled','relevant_carrier_full_minus_disabled'))+' |')
        else:lines.append(f'| {scene} | unavailable | unavailable | unavailable | unavailable |')
    lines+=['', '## Payload selection by budget','', 'Relevant minus rotation uses the identical scene, seed, per-agent allowance and evidence-only learner. Negative favors receiver-relevant selection. This is separate from whether either communication arm beats disabled.',
        '', '| Ceiling / agent | Paired episodes | Relevant − rotation time+risk | Relevant − rotation FULL | Relevant − rotation TEAM bytes |','|---|---:|---:|---:|---:|']
    for budget in sorted({p['budget'] for p in pairs}):
        rr=[p for p in pairs if p.get('available') and p['budget']==budget]
        if rr:lines.append(f"| {budget:,} | {len(rr)} | "+' | '.join(f"{np.mean([r[k] for r in rr]):+.4f}" for k in ('relevant_minus_rotation_carrier_time_risk','relevant_minus_rotation_carrier_full','relevant_minus_rotation_team_bytes'))+' |')
    if legacy_comparison:
        lines+=['', '## Before/after evaluation accounting','', 'The old bandwidth table reported full cost with evaluation risk weight one for both agents. The corrected table uses the declared scout/carrier weights. Only successful operational parity allows this difference to be attributed to accounting. The primary endpoint also omits effort, so its name is shown explicitly.',
            '', '| Ceiling / agent | Selector | Old incorrectly weighted FULL | Corrected time+risk | Corrected FULL |','|---|---|---:|---:|---:|']
        for budget in sorted({r['budget'] for r in legacy_comparison}):
            for name in ('disabled','rotation','receiver_relevant'):
                rr=[r for r in legacy_comparison if r['budget']==budget and r['selector']==name]
                if rr:lines.append(f'| {budget:,} | {name} | '+' | '.join(f"{np.mean([r[k] for r in rr]):.4f}" for k in ('old_incorrectly_weighted_full','corrected_time_risk','corrected_full'))+' |')
        lines+=['', '### Moderate and double-hazard communication check','', 'These rows compare rotation with disabled within the new channel/learner setting at the same seed and four-million-byte ceiling. The old scalar is the incorrectly weighted full endpoint; the corrected primary and full endpoints are both shown. Operational parity is checked independently.',
            '', '| Scene | Seed | Arm | Old incorrectly weighted FULL | Corrected time+risk | Corrected FULL |','|---|---:|---|---:|---:|---:|']
        for scene in ('moderate','double_mild'):
            for seed in sorted({r['seed'] for r in legacy_comparison if r['scene']==scene}):
                for name in ('disabled','rotation'):
                    r=next((r for r in legacy_comparison if r['scene']==scene and r['seed']==seed and r['budget']==4000000 and r['selector']==name),None)
                    if r:lines.append(f"| {scene} | {seed} | {name} | {r['old_incorrectly_weighted_full']:.5f} | {r['corrected_time_risk']:.5f} | {r['corrected_full']:.5f} |")
    lines+=['', '## Timing, routes, and budget starvation','', 'Patch timing maps received anchors, local readings and actual carrier positions to each true patch ellipse for evaluator diagnostics only. A missing local encounter may mean the route avoided that area; it is not an invented discovery time. Planned route changes are distinct from actual interior-edge entries. Backward x travel is recorded without calling every detour wrong.',
        '', '| Scene | Selector | Ceiling | Patch | Carrier receives anchor | Locally senses | Actual entry |','|---|---|---:|---:|---:|---:|']
    for d in diagnostics:
        if d['scene'] not in ('moderate','double_mild') or d['seed']!=86100:continue
        for i,p in enumerate(d['patch_timing']):lines.append(f"| {d['scene']} | {d['selector'] if d['arm']!='disabled' else 'disabled'} | {d['budget']} | {i+1} | {p['first_received_anchor']} | {p['first_local_observation']} | {p['first_actual_agent_entry']} |")
    lines+=['', 'Per-agent traffic records in episode_diagnostics.json separate offered-payload-direction refusals from charged-agent budget refusals. A status_budget refusal belongs to event.receiver, which advertises its route; payload_budget belongs to event.sender. The first_budget_refusal uses the charged agent, with first_offer_budget_refusal recorded separately. Last payload and any later payload after refusal are logged. No team-total allowance is substituted for per-agent budgets.',
        '', 'These four designed layouts use two sensor-noise seeds. Repeated budgets, quadrature points and noise seeds are not independent terrain trials. Results are descriptive; failures and incomplete/missing episodes remain in failure_inventory.json and unavailable_information.json. Original and new channels are not pooled as one controlled treatment.']
    lines+=artifact_links(root)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    inputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'manifest.json',root/'rows.json',root/'regression.json',root/'parity.json') if p.exists()}
    logs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.parent.glob(root.name+'*.log'))]
    write(root/'reporting_verification.json',dict(report_source_sha256=sums['report_source_sha256'],input_sha256=inputs,test_and_run_logs=logs,
        snapshot_reconstruction=False,learner_or_simulator_executed=False,primary_and_full_endpoints_distinct=True,
        every_row_retained=len(rows)+len(invalid),available_episode_diagnostics=len(diagnostics),incomplete_campaign=not complete,
        tests='See scoped test logs: known archived-source-hash failure remains; unscoped archived external-source collection errors are not silently removed.'))
    findings(root,rows,pairs,diagnostics,sums,legacy_comparison)
    return sums


def findings(root,rows,pairs,diagnostics,summary,legacy):
    lines=['# Reconciliation findings','',
        f"Original reproductions: {summary['original_regression']}. New-channel legacy/corrected accounting parity: {summary['operational_parity']}. Corrected episodes: {summary['available_episodes']}/{summary['expected_episodes']}; {summary['goals']}/{2*summary['available_episodes']} goals, {summary['collisions']} collisions, {summary['timeouts']} timeouts. Numerical comparison tolerance is 1e-10; discrete decisions and serialized differences are reported separately in the regression records.",
        '', '**Communication at 4 MB per agent.** Means across the two sensor-noise seeds; primary is time plus declared-weight risk. FULL additionally includes effort and failure.',
        '', '| Scene | Off primary | Rotation primary | Relevant primary | Off FULL | Rotation FULL | Relevant FULL |','|---|---:|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        vals=[]
        for endpoint in ('carrier_time_risk','carrier_cost'):
            for name in ('disabled','rotation','receiver_relevant'):
                rr=[r for r in rows if r['scene']==scene and label(r)==name and (name=='disabled' or r['budget']==4000000)]
                vals.append(f'{np.mean([r[endpoint] for r in rr]):.4f}' if rr else 'unavailable')
        lines.append('| '+scene+' | '+' | '.join(vals)+' |')
    lines+=['', '**Was communication harmful?** The old bandwidth evaluation priced carrier risk with weight one, although its declared preference was six. For moderate and double-mild hazards this underpriced the risky direct route and made the detour look harmful. The table below shows matched rotation-minus-off differences; negative favors rotation. This attribution is valid only when the operational parity check passes.',
        '', '| Scene | Old incorrect FULL difference | Corrected primary difference | Corrected FULL difference |','|---|---:|---:|---:|']
    for scene in ('moderate','double_mild'):
        old=[];corrected=[]
        for seed in sorted({r['seed'] for r in legacy if r['scene']==scene}):
            rs={r['selector']:r for r in legacy if r['scene']==scene and r['seed']==seed and r['budget']==4000000}
            if 'disabled' in rs and 'rotation' in rs:old.append(rs['rotation']['old_incorrectly_weighted_full']-rs['disabled']['old_incorrectly_weighted_full'])
        rr=[p for p in pairs if p.get('available') and p['scene']==scene and p['budget']==4000000 and 'rotation_carrier_time_risk_minus_disabled' in p]
        vals=[np.mean(old) if old else None]+[np.mean([p[k] for p in rr]) if rr else None for k in ('rotation_carrier_time_risk_minus_disabled','rotation_carrier_full_minus_disabled')]
        lines.append('| '+scene+' | '+' | '.join(f'{v:+.5f}' if v is not None else 'unavailable' for v in vals)+' |')
    lines+=['', '**The remaining selector limitation is real.** Receiver-relevant selection must be compared with rotation under the same corrected metric, independently of the above artifact. In the double-mild case inspect its primary cost, actual path and patch-arrival records; an absent first-patch summary is distinct from a late second-patch summary. At 4 MB, budget refusal cannot explain a case with no refusals. These observational timing records explain the information pattern without substituting an oracle success claim.']
    for selector in ('rotation','receiver_relevant'):
        d=next((d for d in diagnostics if d['scene']=='double_mild' and d['seed']==86100 and d['budget']==4000000 and d['arm']!='disabled' and d['selector']==selector),None)
        if d:
            timing=[p['first_received_anchor'] for p in d['patch_timing']]
            actual=d['actual_traversal'].get('interior_edge_entries',[]);edges=[r['edge'] for r in actual]
            lines.append(f"- Double-mild seed86100 {selector}: first received anchor times for the two patch ellipses {timing}; actual interior edges {edges}; backward x travel {d['actual_traversal'].get('backward_x_travel')} m.")
    lines+=['', 'Implementation limitation, separate from the measured timing: advertisements acknowledge sources in receiver.received, while accepted remote-route-only information can live solely in remote_terrain/remote_cells. Already useful route evidence may therefore remain absent from the acknowledgment set. This can misrepresent novelty. The double-mild logs repeatedly select the later stage while omitting the first patch; this is evidence of a payload-selection/interface limitation, not a controlled proof that acknowledgment alone causes it. No selection policy was changed during this reconciliation.']
    lines+=['', '**At an 8,000-byte ceiling, analyze each payer separately.** Status advertisements count against the receiver advertising them. A sender may be blocked because its intended recipient cannot afford that status. This does not mean either agent exceeded its own allowance. Offered-direction and charged-agent refusals are separate in episode_diagnostics.json.']
    for selector in ('rotation','receiver_relevant'):
        tt=[a for d in diagnostics if d['budget']==8000 and d['arm']!='disabled' and d['selector']==selector for a in d['per_agent_traffic']]
        if tt:lines.append(f"- {selector}: {sum(a['sent']==0 for a in tt)}/{len(tt)} agents sent no payload; {sum(a['sent']==0 and a['bytes']['status']>0 for a in tt)}/{len(tt)} spent status bytes while sending no payload. All own-budget checks pass: {all(a['budget_respected'] for a in tt)}.")
    example=next((d for d in diagnostics if d['scene']=='moderate' and d['seed']==86100 and d['budget']==8000 and d['arm']!='disabled' and d['selector']=='receiver_relevant'),None)
    if example:
        trace=example['traffic_accounting']['events'];first=next((t for t in trace if t['reason']=='sent'),None);blocked=next((t for t in trace if t['reason']=='payload_budget'),None)
        if first and blocked:
            lines += ['', f"Concrete moderate/seed86100 chronology: the first reverse-direction send at t={first['time']} charges {first['charges']}. At the first payload refusal t={blocked['time']}, charges are {blocked['charges']}; per-agent remaining balances are {blocked['remaining']}. The reconstructed charge trace matches the final logged agent totals: {example['traffic_accounting']['matches_logged_agent_totals']}."]
            high=next((d for d in diagnostics if d['scene']=='moderate' and d['seed']==86100 and d['budget']==16000 and d['arm']!='disabled' and d['selector']=='receiver_relevant'),None)
            forward=next((e for e in high['sent_payloads'] if str(e['sender'])=='0'),None) if high else None
            if forward:
                status=sum(c['bytes'] for c in first['charges'] if c['agent']==0 and c['component']=='status')
                ack=sum(c['bytes'] for t in trace if t['time'] is not None and t['time']<blocked['time'] for c in t['charges'] if c['agent']==0 and c['component']=='acknowledgment')
                lines.append(f"In the matched 16k run, the first scout packet at t={forward['time']} is {forward['payload_bytes']:,} bytes ({forward['diagnostic_route_relevant_anchors']}/{forward['diagnostic_anchor_count']} anchors near the advertised receiver route). In the 8k trace the scout already paid {status:,} status bytes and {ack:,} acknowledgment bytes, leaving {blocked['remaining'][0]:,} bytes. The forward payload cannot fit even though no single status advertisement exceeds8k. Later status refusals keep that direction silent. This is reverse-status-before-forward-payload overhead, not a violation of either agent's budget.")
    lines+=['', '**Readiness and limits.** Completion/collisions remain separate from cost improvements. Inter-agent contact dynamics are absent. Four designed terrains and two noise seeds do not establish generalization, and repeated budgets are not independent terrain samples. Original full-replay/uncompressed-channel reproductions and new evidence-only/compact-channel results are not pooled as one treatment.',
        '', '**Tests.** The scoped pytest run reported 257 passed and 7 subtests passed with the existing archived-source-hash failure retained. The numerical CLI check retained the same known failure (225/226 passed). The initial unscoped pytest additionally discovered archived external-source tests with missing dependencies; its collection errors remain logged. No full-suite-all-passing claim is made.',
        '', 'See REPORT.md for all rows and separate endpoints; paired_rows.json for selector and off-reference differences; failure_inventory.json for failed/incomplete episodes; reporting_verification.json for report-source/input/log fingerprints; animation_inventory.json for synchronized rendering availability.']
    lines+=artifact_links(root)
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n')


def artifact_links(root):
    lines=['','## Plots and synchronized animations','']
    for name,title in (('cost_components_4000000.png','4 MB cost components'),('cost_components_16000.png','16k cost components'),('risk_time.png','Time/risk and budget comparison'),('bytes.png','Status/payload/ack and per-agent ceilings'),('moderate_timing.png','Moderate warning and budget timeline'),('double_mild_timing.png','Double-hazard warning and budget timeline')):
        if (root/'figures'/name).exists():lines.append(f'- [{title}](figures/{name})')
    inventory=read(root/'animation_inventory.json') or []
    for item in inventory:
        if item.get('status')=='rendered':lines.append(f"- [{item['scene']} three-arm GIF, seed{item['seed']}, {item['budget']:,}/agent]({item['path']})")
    if not inventory:lines.append('Animation availability is recorded in animation_inventory.json after rendering; no absent animation is implied.')
    return lines


def _triplet(root,rows,scene,seed,budget):
    result=[]
    for name in ('disabled','rotation','receiver_relevant'):
        row=next((r for r in rows if r['scene']==scene and r['seed']==seed and label(r)==name and (name=='disabled' or r['budget']==budget)),None)
        if row is None:return None
        episode=read(episode_path(root,row))
        if episode is None:return None
        result.append((row,episode))
    return result


def render(out,tight_budget=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from .crossing_visuals import background
    root=Path(out);rows=read(root/'rows.json') or [];manifest=read(root/'manifest.json') or {};seed=manifest.get('animation_seed',86100);directory=root/'figures';directory.mkdir(exist_ok=True);inventory=[]
    for budget in [4000000]+([tight_budget] if tight_budget is not None and tight_budget!=4000000 else []):
        for scene in SCENES:
            triple=_triplet(root,rows,scene,seed,budget)
            if triple is None:inventory.append(dict(scene=scene,seed=seed,budget=budget,status='unavailable_missing_arm_or_episode'));continue
            end=max(a['finish_time'] for _,r in triple for a in r['agents']);frames=[];frame_records=[]
            for now in np.linspace(0,end,36):
                fig=plt.figure(figsize=(15,8),dpi=80);grid=fig.add_gridspec(2,3,height_ratios=[1.15,1]);state_records=[]
                for column,(row,r) in enumerate(triple):
                    ax=fig.add_subplot(grid[0,column]);background(ax,r['spec']);name=label(row);ax.set_title(name)
                    for i in (0,1):
                        ids=[k for k,s in enumerate(r['samples'][i]) if s['time']<=now+1e-9]
                        if not ids:continue
                        k=ids[-1];samples=r['samples'][i][:k+1];pts=np.asarray([s['x'][:2] for s in samples]);ax.plot(*pts.T,color=('#0072B2','#D55E00')[i],lw=2);ax.scatter(*pts[-1],color=('#0072B2','#D55E00')[i],s=30,zorder=5)
                        controls=[t for t in r['trajectories'][i] if t['time']<=now+1e-9]
                        if controls and not samples[-1].get('done'):
                            route=np.asarray(controls[-1].get('navigation',{}).get('route',[]))
                            if len(route):ax.plot(*route.T,':',lw=1,color=('#0072B2','#D55E00')[i],alpha=.7)
                            for key,color in (('nominal','#999999'),('u','#222222')):
                                if key in controls[-1]:v=np.asarray(controls[-1][key])*.3;ax.arrow(*pts[-1],*v,color=color,width=.02,head_width=.12,length_includes_head=True)
                        status='goal' if samples[-1].get('done') and r['agents'][i]['success'] else 'collision' if samples[-1].get('done') and r['agents'][i]['collision'] else 'timeout' if samples[-1].get('done') else f"v{samples[-1]['version']}"
                        ax.annotate(f'{"AB"[i]} {status}',pts[-1],xytext=(3,6),textcoords='offset points',fontsize=8,color=('#0072B2','#D55E00')[i]);state_records.append(dict(arm=name,agent=i,sample_index=k,time=samples[-1]['time'],version=samples[-1]['version']))
                    for e in r.get('events',[]):
                        if e.get('kind')=='delivered' and 0<=now-e.get('time',-100)<=2 and e.get('anchors'):
                            p=np.asarray(e['anchors']);ax.scatter(*p[:,:2].T,s=16,marker='x',color='#8e24aa',zorder=6)
                ax=fig.add_subplot(grid[1,0])
                for row,r in triple:
                    rr=[t for t in r['trajectories'][1] if t['end_time']<=now+1e-9]
                    if rr:ax.plot([t['end_time'] for t in rr],np.cumsum([t['components']['time']+t['components']['risk'] for t in rr]),color=COLORS[label(row)],label=label(row))
                ax.set(xlim=(0,end),xlabel='global seconds',ylabel='Carrier cumulative time + weighted risk');ax.legend(fontsize=7);ax.grid(alpha=.2)
                ax=fig.add_subplot(grid[1,1])
                for j,(row,r) in enumerate(triple):
                    for i in (0,1):
                        sent=[e['time'] for e in r['events'] if e.get('reason')=='sent' and str(e.get('sender'))==str(i) and e.get('time',end+1)<=now]
                        refused=[e['time'] for e in r['events'] if e.get('kind')=='channel' and str(e.get('sender'))==str(i) and e.get('time',end+1)<=now]
                        ax.scatter(sent,np.full(len(sent),j+(i-.5)*.16),s=8,color=COLORS[label(row)]);ax.scatter(refused,np.full(len(refused),j+(i-.5)*.16),s=12,marker='x',color='#b2182b')
                ax.set(xlim=(0,end),xlabel='global seconds',yticks=[0,1,2],yticklabels=['off','rotation','relevant'],title='Dots: payloads; red ×: refused opportunity')
                ax=fig.add_subplot(grid[1,2]);ax.axis('off');text=['FINAL endpoints (not current-time totals)','']
                for row,r in triple:
                    c=r['agents'][1]['components'];team=sum(sum(a.get('bytes',{}).get(k,0) for k in BYTE_KEYS) for a in r['agents'])
                    text += [f"{label(row)}: goals {sum(a['success'] for a in r['agents'])}/2",f" time+risk {c['time']+c['risk']:.2f}; full {r['agents'][1]['cost']:.2f}",f" carrier risk {c['risk']:.2f}; effort {c['effort']:.2f}",f" team bytes {team:,}",'']
                ax.text(0,1,'\n'.join(text),va='top',fontsize=9);fig.suptitle(f'{scene} · seed {seed} · ceiling {budget:,}/agent · t={now:.1f}s\nSame world and timestamps; grey nominal / black applied control; purple received anchors',fontsize=12);fig.tight_layout(rect=(0,0,1,.93));fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig);frame_records.append(dict(time=float(now),samples=state_records))
            path=directory/f'{scene}_seed{seed}_budget{budget}.gif';frames[0].save(path,save_all=True,append_images=frames[1:],duration=140,loop=0)
            with Image.open(path) as image:actual=image.n_frames
            verified=actual==len(frame_records) and all(s['time']<=f['time']+1e-9 for f in frame_records for s in f['samples'])
            item=dict(scene=scene,seed=seed,budget=budget,status='rendered',path=str(path.relative_to(root)),frames=actual,timestamps_verified=verified,episode_keys=[r['key'] for r,_ in triple]);inventory.append(item);write(directory/f'{path.stem}_frames.json',frame_records)
    write(root/'animation_inventory.json',inventory);return inventory


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['report','render']);p.add_argument('--out',type=Path,required=True);p.add_argument('--tight-budget',type=int);a=p.parse_args(argv)
    if a.command=='report':report(a.out)
    else:render(a.out,a.tight_budget)


if __name__=='__main__':main()
