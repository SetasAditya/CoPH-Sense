"""Saved-record reporting; evaluator-only truths never feed runtime decisions."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .decision_reporting import episode_summary,world_axes,draw_episode
from .field_route_suite import METHODS

def episodes(root):
    root=Path(root)
    for p in sorted((root/'status').glob('*.json')):
        s=load(p)
        if s['complete']:
            path=root/'episodes'/s['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256'];yield s,load(path)

def branch(nodes):return None if nodes is None else 'bypass' if 4 in nodes or 5 in nodes else 'direct'

def report(root):
    import matplotlib.pyplot as plt
    root=Path(root);m=load(root/'manifest.json');fig=root/'figures';fig.mkdir(exist_ok=True);(fig/'failures').mkdir(exist_ok=True)
    rows=[];audits=[];failures=[];regressions=[];lineage=[];done=set()
    for s,r in episodes(root):
        done.add(s['id']);row=episode_summary(r);row['id']=s['id'];row['route_field_mode']=r['config']['route_field_mode']
        controls=[t['navigation'] for ts in r['trajectories'] for t in ts]+[n for us in r['updates'] for u in us for n in u['diagnostic'].get('diagnostic_controls',[])]
        row['routing_work']=dict(field_evaluations=sum(n.get('route_field_work',{}).get('field_evaluations',0) for n in controls),
            seconds=sum(n.get('route_field_work',{}).get('seconds',0) for n in controls),
            diagnostic_field_evaluations=sum(n.get('route_field_diagnostic_work',{}).get('field_evaluations',0) for n in controls))
        row['routing_work'].update(total_field_evaluations=r['route_field_total_evaluations'],audit_field_evaluations=r['route_field_audit_evaluations'])
        rows.append(row)
        if 'baseline_comparison' in r:regressions.append(dict(id=s['id'],**r['baseline_comparison']))
        if not all(a['success'] for a in r['agents']):failures.append(dict(id=s['id'],agents=r['agents']));draw_episode(r,fig/'failures'/f"{s['id']}.png")
        packet_sources={p['identity']:p['sources'] for p in r['packets']}
        lineage.extend(dict(episode=s['id'],source_ids=packet_sources[e['packet']],**e) for e in r['events'] if e['kind']=='delivered')
        for audit in r['audits']:
            for name,b in audit['branches'].items():
                comparison=b['decision_audit']['route_field_comparison'];points=com=field=0;ae=de=afe=dfe=0.;sources={'field':0,'anchor':0,'unknown':0}
                for e in comparison['errors']:
                    mask=np.asarray(e['common_support']);fm=mask&np.asarray(e['field_support']);av=np.abs(e['anchor_error']);dv=np.abs(e['field_error']);points+=len(mask);com+=int(mask.sum());field+=int(fm.sum());ae+=float(av[mask].sum());de+=float(dv[mask].sum());afe+=float(av[fm].sum());dfe+=float(dv[fm].sum())
                for e in comparison['rankings']['supported_field']['quadrature']:
                    for v in e['route_field']['source']:sources[v]+=1
                best=lambda k:comparison['rankings'][k]['best']['nodes'] if comparison['rankings'][k]['best'] else None
                audits.append(dict(episode=s['id'],case=r['case'],seed=r['seed'],arm=r['method'],packet=audit['packet'],receiver=audit['receiver'],stage=audit['stage'],time=audit['time'],branch=name,
                    accepted=b['accepted'],version=b['version'],points=points,common_points=com,field_common_points=field,
                    anchor_abs_error=ae,direct_abs_error=de,anchor_field_abs_error=afe,direct_field_abs_error=dfe,sources=sources,
                    anchor_nodes=best('anchors'),direct_nodes=best('supported_field'),branch_ranking_changed=branch(best('anchors'))!=branch(best('supported_field')),
                    comparison=comparison,control_delta=audit['control_delta'],field_delta=audit['field_delta']))
    expected={f'{c}_{seed}_{a}' for c in m['cases'] for seed in m['seeds'] for a in m['arms']}
    errors=[load(f) for f in (root/'status').glob('*.json') if not load(f)['complete']]
    dump(root/'rows.json',rows);dump(root/'audit_rows.json.gz',audits);dump(root/'packet_lineage.json.gz',lineage);dump(root/'regressions.json',regressions)
    dump(root/'failures.json',failures+errors);dump(root/'incomplete.json',sorted(expected-done))
    groups=[]
    for c in m['cases']:
        for arm in METHODS:
            rr=[r for r in rows if r['case']==c and r['arm']==arm];aa=[a for r in rr for a in r['agents']];au=[a for a in audits if a['case']==c and a['arm']==arm]
            n=sum(a['common_points'] for a in au);nf=sum(a['field_common_points'] for a in au);points=sum(a['points'] for a in au)
            groups.append(dict(case=c,arm=arm,episodes=len(rr),goals=sum(a['success'] for a in aa),collisions=sum(a['collision'] for a in aa),timeouts=sum(a['timeout'] for a in aa),
                carrier_risk=float(np.mean([r['agents'][1]['components']['risk'] for r in rr])) if rr else None,
                carrier_seconds=float(np.mean([r['agents'][1]['active_seconds'] for r in rr])) if rr else None,
                common_points=n,field_common_points=nf,points=points,audit_branches=len(au),branch_ranking_changes=sum(a['branch_ranking_changed'] for a in au),
                anchor_mae=sum(a['anchor_abs_error'] for a in au)/n if n else None,direct_mae=sum(a['direct_abs_error'] for a in au)/n if n else None,
                anchor_field_mae=sum(a['anchor_field_abs_error'] for a in au)/nf if nf else None,direct_field_mae=sum(a['direct_field_abs_error'] for a in au)/nf if nf else None,
                sources={k:sum(a['sources'][k] for a in au) for k in ('field','anchor','unknown')}))
    dump(root/'aggregates.json',groups)
    fig1,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for c in m['cases']:
        gg=[g for g in groups if g['case']==c];ax[0].plot(range(4),[g['carrier_risk'] for g in gg],'o-',label=c)
        valid=[a for a in audits if a['case']==c and a['field_common_points']];ax[1].scatter([a['anchor_field_abs_error']/a['field_common_points'] for a in valid],[a['direct_field_abs_error']/a['field_common_points'] for a in valid],s=9,alpha=.4,label=c)
        ax[2].plot(range(4),[g['branch_ranking_changes']/g['audit_branches'] if g['audit_branches'] else np.nan for g in gg],'o-',label=c)
    for a in (ax[0],ax[2]):a.set_xticks(range(4),['anchor local','anchor fused','direct local','direct fused'],rotation=20)
    ax[0].set(title='Carrier risk component',ylabel='Recorded cost component');ax[0].legend(fontsize=8)
    ax[1].set(title='Same-state common-field error',xlabel='Anchor MAE',ylabel='Direct-field MAE');limit=max(ax[1].get_xlim()[1],ax[1].get_ylim()[1]);ax[1].plot([0,limit],[0,limit],'k--',lw=.8)
    ax[2].set(title='Same-state branch-ranking changes',ylabel='Changed / audited branches')
    fig1.savefig(fig/'field_route_comparison.png',dpi=150);plt.close(fig1)
    lines=['# Direct reconstructed-field route scoring','',f'Completed {len(rows)}/24 episodes; anchor numerical regressions {sum(r["passed"] for r in regressions)}/12. Goals {sum(a["success"] for r in rows for a in r["agents"])}/48.',
        '', 'This tests direct committed-field evaluation against the existing sparse local/remote anchor representation. Reconstruction already affects local anchors; this is not a newly created causal connection. Two noise seeds per fixed map, development only; replay disabled.',
        '', '| Case | Arm | Goals | Collisions / timeouts | Carrier time | Carrier risk cost | Field-common MAE: anchor → direct | Ranking changes / audit branches |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        fmt=lambda x:'unavailable' if x is None else f'{x:.6g}'
        lines.append(f'| {g["case"]} | {g["arm"]} | {g["goals"]}/{2*g["episodes"]} | {g["collisions"]} / {g["timeouts"]} | {fmt(g["carrier_seconds"])} | {fmt(g["carrier_risk"])} | {fmt(g["anchor_field_mae"])} → {fmt(g["direct_field_mae"])} | {g["branch_ranking_changes"]}/{g["audit_branches"]} |')
    lines+=['','Errors compare representations at identical audited physical/model states. Supported intersections, fallback fractions and denominators are in aggregates.json; repeated quadrature points are correlated, not independent trials. Between-arm trajectories need not visit identical states. First-eligible-packet audits may miss the decisive warning. Direct/bypass ranking changes differ from connector/node changes and actual traversal.',
        '', 'Only committed fields on retained local-measurement support override anchors. Unknown risk remains 1; remote-only support is not certified. Failed updates retain prior fields, and evicted measurements/stages can only shrink routing support. Mission costs are secondary; no superiority gate or tuning is applied.',
        '', 'Regressions use atol 1e-12, rtol 1e-10 for floats and exact discrete decisions/work. Bitwise and actual byte differences remain in regressions.json. New routing metadata is excluded from operational comparison; no operational state arrays are excluded.',
        '', '![Comparison](figures/field_route_comparison.png)','', 'See rows.json, aggregates.json, audit_rows.json.gz, packet_lineage.json.gz, the full episodes with per-control quadrature traces, failures.json and incomplete.json.']
    for c in m['cases']:lines.append(f'\n[{c} synchronized comparison](figures/{c}.gif)')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def render(root):
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(root);m=load(root/'manifest.json');meta=[]
    for case in m['cases']:
        runs=[load(root/'episodes'/f'{case}_{m["animation_seed"]}_{arm}.json.gz') for arm in METHODS];end=max(a['finish_time'] for r in runs for a in r['agents']);times=np.linspace(0,end,72);frames=[]
        for now in times:
            fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
            for ax,r in zip(axes.flat,runs):
                world_axes(ax,r);ax.set_title(f'{r["method"]} · t={now:.2f}s')
                for i,color in enumerate(('tab:blue','tab:green')):
                    ss=[s for s in r['samples'][i] if s['time']<=now];ts=[t for t in r['trajectories'][i] if t['time']<=now]
                    if ss:
                        q=np.asarray([s['x'][:2] for s in ss]);ax.plot(*q.T,color=color,label=f'A{i}');ax.plot(*q[-1],'o',color=color)
                    if ts:
                        t=ts[-1];route=t['navigation'].get('route',[])
                        if route:ax.plot(*np.asarray(route).T,'--',color=color,alpha=.45)
                        q=np.asarray(t['x'][:2]);u=np.asarray(t['u']);ax.arrow(*q,*(.5*u),color=color,width=.015)
                        if any(e['kind']=='delivered' and e['receiver']==i and 0<=now-e['time']<.6 for e in r['events']):ax.scatter(*q,s=170,facecolors='none',edgecolors='purple',lw=2)
                states=['GOAL' if a['success'] else 'COLLISION' if a['collision'] else 'TIMEOUT' for a in r['agents']]
                ax.set_xlabel(' / '.join(states[i] if now>=r['agents'][i]['finish_time'] else ('waiting' if now<r['agents'][i]['start_time'] else 'active') for i in (0,1)))
                ax.legend(loc='upper right',fontsize=7)
            fig.suptitle(f'{case}: dashed planned routes; solid execution; purple packet receipt; arrows applied control')
            fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=256));plt.close(fig)
        frames[0].save(root/'figures'/f'{case}.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
        meta.append(dict(case=case,seed=m['animation_seed'],frames=len(frames),times=times.tolist(),source='logged states and controls, shared time, all four arms'))
    dump(root/'figures/animation_frames.json',meta)

def verify(root):
    from PIL import Image
    root=Path(root);m=load(root/'manifest.json');count=goals=collisions=timeouts=packets=audits=updates=0;maxwork=0;reg=0
    with zipfile.ZipFile(root/'sources.zip') as z:
        for name,h in m['source_hashes'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    for status,r in episodes(root):
        count+=1;assert r['manifest_fingerprint']==m['fingerprint'];assert r['spec']==m['specs'][f'{r["case"]}_{r["seed"]}'];assert r['config']==m['configs'][r['method']]
        if 'baseline_comparison' in r:assert r['baseline_comparison']['passed'];reg+=1
        for i,a in enumerate(r['agents']):
            goals+=a['success'];collisions+=a['collision'];timeouts+=a['timeout'];assert abs(a['cost']-sum(a['components'].values()))<1e-8
            assert sum(a['bytes'].values())<=4000000;maxwork=max(maxwork,a['max_update_work']);assert a['max_update_work']<=1024
            for u in r['updates'][i]:
                updates+=1;assert u['diagnostic']['counterfactual_steps']<=1024;assert not u['diagnostic']['policy_trace']['policy_changed'];assert not u['diagnostic']['policy_trace']['backtracking_accepted']
            for t in r['trajectories'][i]:
                np.testing.assert_allclose(t['u'],t['navigation']['applied'],atol=1e-12);assert t['version']==t['control_model']['version']
                rank=t['navigation']['route_field_trace']
                for e in rank['quadrature']:
                    assert abs(sum(e['contributions'])-e['cost'])<1e-8
                    for source,v,support in zip(e['route_field']['source'],e['values'],e['support']):
                        assert 0<=v<=20
                        if source=='unknown':assert v==1 and not support
                        if source=='field':assert support and r['config']['route_field_mode']=='supported_field'
        obs=[{f'{i}:{v["uid"]}' for o in r['observations'][i] for v in o['readings']} for i in (0,1)]
        sizes={}
        for p in r['packets']:
            packets+=1;assert set(p['sources'])<=obs[int(p['sender'])]
            sizes[p['identity']]=len(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        for e in r['events']:
            if e['kind']=='sent':assert e['bytes']==sizes[e['packet']]
        for i in (0,1):assert r['agents'][i]['bytes']['payload']==sum(sizes[p['identity']] for p in r['packets'] if int(p['sender'])==i)
        deliveries=[e for e in r['events'] if e['kind']=='delivered'];assert len(deliveries)==len({e['packet'] for e in deliveries})
        for a in r['audits']:
            audits+=1;assert a['operational_state_unchanged'];assert a['randomness']['shared_between_branches']
            assert a['decomposition_closure_residual']<1e-8
            for b in a['branches'].values():
                for e in b['decision_audit']['route_field_comparison']['errors']:
                    assert len(e['points'])==len(e['truth'])==len(e['common_support'])
                for h,c in b['checkpoints'].items():
                    if c:assert c['path'][0]==a['start'] and c['steps']<=int(h)
    assert count==24 and reg==12;assert not load(root/'incomplete.json')
    for a in load(root/'figures/animation_frames.json'):
        im=Image.open(root/'figures'/f'{a["case"]}.gif');duration=0
        for j in range(im.n_frames):im.seek(j);duration+=im.info['duration']
        assert duration==160*a['frames'];assert a['seed']==78000
    result=dict(episodes=count,goals=goals,collisions=collisions,timeouts=timeouts,anchor_regressions=reg,updates=updates,packets=packets,audits=audits,max_update_flows=maxwork,source_files=len(m['source_hashes']),animations=3)
    dump(root/'artifact_checks.json',result);print(json.dumps(result,indent=2))
