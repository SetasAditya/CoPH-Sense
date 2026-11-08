"""Evaluator-only reporting for the terrain-warning development demonstration."""
from pathlib import Path
import numpy as np
from .evaluation import load,dump


def metrics(r):
    detection=[];switch=[]
    for i in range(2):
        ds=[o['time'] for o in r['observations'][i] if any(z['value']>=1 for z in o['readings'])]
        bs=[t['time'] for t in r['trajectories'][i] if t['navigation'].get('branch')==1]
        detection.append(min(ds) if ds else None);switch.append(min(bs) if bs else None)
    ds=[e for e in r['events'] if e['kind']=='delivered' and e['sender']==0 and e['reason']=='eligible' and max(a[2] for a in e['anchors'])>=1]
    warning=min((e['time'] for e in ds),default=None)
    before=bool(switch[1] is not None and (detection[1] is None or switch[1]<detection[1]))
    return dict(seed=r['spec']['seed'],arm=r['arm'],goals=sum(a['success'] for a in r['agents']),collisions=sum(a['collision'] for a in r['agents']),
        timeouts=sum(a['timeout'] for a in r['agents']),local_hazard_detection=detection,first_bypass_choice=switch,first_leader_warning=warning,
        carrier_bypass_before_local_detection=before,carrier_risk_integral=r['agents'][1]['components']['risk']/4,
        carrier_finish=r['agents'][1]['finish_time'],bytes=sum(sum(a['bytes'].values()) for a in r['agents']),
        max_update_work=max(a['max_update_work'] for a in r['agents']),audits=len(r['audits']),
        changed_control_audits=sum(a['control_delta']>1e-10 for a in r['audits']),wall_seconds=r['wall_seconds'])


def frame(pair,now):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from PIL import Image
    from .terrain_warning import ROUTES
    fig,axes=plt.subplots(2,2,figsize=(14,8));cols=['#0072B2','#009E73']
    for ax,arm in zip(axes[0],['disabled','reciprocal_fixed']):
        r=pair[arm];gridx,gridy=np.meshgrid(np.linspace(-.5,8.5,200),np.linspace(-1,2.6,100));risk=.02+4*np.exp(-.5*(((gridx-3.6)/.5)**2+(gridy/.38)**2))
        ax.contourf(gridx,gridy,risk,levels=[0,1,2,3,5],colors=['#fffef4','#ffdab3','#ff9977','#d94b4b'],alpha=.8)
        for path in ROUTES:
            p=np.array(path);ax.plot(*p.T,color='#b8b8b8',lw=12,zorder=0);ax.plot(*p.T,'--',color='gray',lw=1,zorder=1)
        ax.text(3.5,-.95,'Unknown terrain hazard (evaluator truth)',ha='center',fontsize=8)
        ax.text(3.5,2.02,'Surveyed bypass; terrain initially unknown',ha='center',fontsize=8)
        for i in range(2):
            ss=[s for s in r['samples'][i] if s['time']<=now]
            if ss:
                p=np.array([s['x'][:2] for s in ss]);q=p[-1];ax.plot(*p.T,color=cols[i],lw=2.5);ax.scatter(*q,color=cols[i],s=50)
                ax.annotate(('Scout' if i==0 else 'Carrier')+(' GOAL' if ss[-1]['done'] and r['agents'][i]['success'] else ''),q,xytext=(4,8+12*i),textcoords='offset points',fontsize=9,color=cols[i])
            else:ax.scatter(0,0,color='gray');ax.text(0,-.4,'Carrier waits until 5 s',fontsize=8)
        ev=[e for e in r['events'] if e['kind']=='delivered' and 0<=now-e['time']<=1.]
        for e in ev:
            color='purple' if e['reason']=='eligible' else 'red'
            ax.add_patch(FancyArrowPatch(e['sender_q'],e['receiver_q'],arrowstyle='->',mutation_scale=14,color=color,connectionstyle='arc3,rad=.25'))
            pts=np.array(e['anchors']);ax.scatter(pts[:,0],pts[:,1],marker='+',color=color,s=20)
        ts=[t for t in r['trajectories'][1] if t['time']<=now];choice=('bypass' if ts[-1]['navigation']['branch'] else 'shortcut') if ts else 'waiting'
        ax.set_title(('Communication OFF' if arm=='disabled' else 'Reciprocal fixed-rate')+' | carrier: '+choice)
        ax.set(xlim=(-.6,8.6),ylim=(-1.1,2.5),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
    ax=axes[1,0];r=pair['reciprocal_fixed'];m=metrics(r)
    for e in r['events']:
        if e['kind']=='delivered' and e['time']<=now:
            ax.scatter(e['time'],1 if e['sender']==0 else 0,marker='|',color='purple' if e['reason']=='eligible' else 'red',s=80)
    for label,t,color in [('Scout detects',m['local_hazard_detection'][0],'#0072B2'),('Carrier receives warning',m['first_leader_warning'],'purple'),('Carrier selects bypass',m['first_bypass_choice'][1],'#009E73'),('Carrier senses hazard',m['local_hazard_detection'][1],'#cc6600')]:
        if t is not None and t<=now:ax.axvline(t,color=color,alpha=.6);ax.text(t,1.2+(len(label)%3)*.16,label+f' {t:.2f}s',rotation=35,fontsize=7,color=color)
    ax.axvline(now,color='black');ax.axvline(5,color='gray',ls='--');ax.set(xlim=(0,max(a['finish_time'] for rr in pair.values() for a in rr['agents'])),ylim=(-.3,2.3),yticks=[0,1],yticklabels=['Carrier → scout','Scout → carrier'],xlabel='Global time (s)',title='Delivery timeline and detection order')
    ax=axes[1,1];ax.axis('off');ts=[t for t in r['trajectories'][1] if t['time']<=now];lines=['What changed for the carrier?','']
    if ts:
        t=ts[-1];scores=t['navigation']['route_scores'];lines += [f'Predicted remaining shortcut score: {scores[0]:.2f}',f'Predicted remaining bypass score:   {scores[1]:.2f}']
    audits=[a for a in r['audits'] if a['receiver']==1 and a['time']<=now]
    if audits:
        a=audits[-1];lines+=['',f'Packet-only audit at {a["time"]:.2f}s:',f'  Next-control difference: {a["control_delta"]:.4g} N',f'  Fixed-state shaping H change: {a["H_delta"]:+.4g} J','  Same receiver history; deliver vs withhold.']
    lines+=['','Known routes ≠ known terrain.','Remote reports do not become sensor observations.','Route target enters the scalar navigation potential.','Risk threshold 1 is evaluator detection labeling.']
    ax.text(0,1,'\n'.join(lines),va='top',family='monospace',fontsize=10)
    fig.suptitle(f'Unsafe shortcut: scout explores, carrier follows 5 s later | t = {now:.2f} s\nDevelopment mechanism demo • common surveyed routes • unknown terrain • independent plants',fontsize=12)
    fig.tight_layout();fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return im


def report(root):
    root=Path(root);(root/'figures').mkdir(exist_ok=True);rows=[];pairs={};failures=[]
    for p in sorted((root/'episodes').glob('*.json.gz')):
        r=load(p);row=metrics(r);rows.append(row)
        if row['goals']<2:failures.append(dict(file=p.name,metrics=row))
        if r['spec']['seed']==74000:pairs[r['arm']]=r
    dump(root/'rows.json',rows);dump(root/'failures.json',failures)
    end=max(a['finish_time'] for r in pairs.values() for a in r['agents']);times=sorted(set([0.,4.92,5.,5.12]+np.linspace(0,end,80).tolist()+[e['time'] for e in pairs['reciprocal_fixed']['events'] if e['kind']=='delivered' and 5<=e['time']<=14]))
    frames=[frame(pairs,t) for t in times];frames[0].save(root/'figures/unsafe_shortcut.gif',save_all=True,append_images=frames[1:],duration=220,loop=0)
    frames[-1].save(root/'figures/final.png');dump(root/'figures/frames.json',dict(times=times,seed=74000,playback='Compressed time; global-time labels authoritative'))
    lines=['# Unsafe-shortcut communication demonstration','','Development only. Shared surveyed route topology is an explicit new prior; terrain is unknown. The first agent starts at 0 s, the second at 5 s. Both use the same learner and risk objective. No blocked-wall communication or inter-agent collision avoidance is claimed.','','| Seed | Communication | Goals | Carrier first bypass | Carrier first local hazard detection | Carrier risk integral |','|---|---|---|---|---|---|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["arm"]} | {r["goals"]}/2 | {r["first_bypass_choice"][1]} | {r["local_hazard_detection"][1]} | {r["carrier_risk_integral"]:.4f} |')
    lines+=['','![Animation](figures/unsafe_shortcut.gif)','','Remote support is limited to .65 m around supported anchors; it is used only for route risk scoring, never replay-domain or geometric certification. Overlapping predictions use a conservative maximum, not calibrated confidence. Unknown risk remains 1; path cost is length plus 4 times predicted risk integral. Reports expire after 60 s in this static development scene and replace prior sender/stage contributions. The operational local learner retains reconstruction, validation, replay and rollback. Remote-only route estimates have not passed receiver measured-data validation.','','Packet-only audits isolate immediate model/control changes. Paired complete runs additionally test whether warning changes the route before local sensing; neither constitutes broad superiority evidence. All episodes, failures, work and bytes remain in machine-readable records.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
