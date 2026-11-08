"""Logged-state cooperative figures, synchronized GIFs, and honest reports."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .evaluation import load,dump
from .cooperative_evaluation import CooperativeWorld
from .records import Config,Reading
from .learner import Learner
from .diagnostics import probe_metrics
from .records import new_stage
from .model import tensor,risk,command



def episode_paths(root):
    root=Path(root)
    return [p for p in sorted((root/'episodes').glob('*.json.gz'))
            if not p.with_name(p.name.removesuffix('.json.gz')+'.error.json').exists()]


def report(root):
    root=Path(root);fig=root/'figures';fig.mkdir(exist_ok=True)
    rows=load(root/'summary.json');manifest=load(root/'manifest.json');cal=load(root/'calibration.json');predictor=load(root/'predictor.json')
    if rows:
        with (root/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    index={(r['scene'],r['seed'],r['budget'],r['arm']):r for r in rows}
    paired=[];groups={}
    for r in rows:
        base=index.get((r['scene'],r['seed'],r['budget'],'disabled'))
        if base:paired.append(dict(scene=r['scene'],seed=r['seed'],arm=r['arm'],budget=r['budget'],difference=r['team_cost']-base['team_cost']))
        groups.setdefault((r['budget'],r['arm']),[]).append(r)
    dump(root/'paired.json',paired)
    import hashlib
    renderer=Path(__file__).read_bytes();(root/'renderer_source.txt').write_bytes(renderer)
    dump(root/'report_source.json',dict(sha256=hashlib.sha256(renderer).hexdigest(),source='renderer_source.txt'))
    import platform,sys,torch
    dump(root/'provenance.json',dict(manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_archive_sha256=hashlib.sha256((root/'sources.zip').read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256((root/'calibration.json').read_bytes()).hexdigest(),
        predictor_sha256=hashlib.sha256((root/'predictor.json').read_bytes()).hexdigest(),
        python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform()))
    audits=[];common=[];failures=[];reasons={};learning=[];message_effects=[]
    # Common supported probes are evaluated at the same final clock for completed episodes
    # only in static scenes. Dynamic final-time errors are explicitly not paired.
    probe_index={}
    for path in episode_paths(root):
        r=load(path);key=(r['spec']['name'],r['spec']['seed'],r['communication']['byte_budget'])
        probe_index.setdefault(key,{})[r['arm']]=r['agents']
        for a in r['audits']:
            audits.append(dict(scene=key[0],seed=key[1],arm=r['arm'],budget=key[2],recipient=a['recipient'],
                predicted=a['predicted_improvement'],realized=a['realized_improvement'],valid=a['valid'] and a['model_pair']['valid']))
        for events in r['packet_events']:
            for e in events:reasons[e['reason']]=reasons.get(e['reason'],0)+1
        if r['summary']['failures']:failures.append(str(path.relative_to(root)))
        for i,updates in enumerate(r['updates']):
            for u in updates:
                if u['phase']!='message':continue
                cfg=Config(**r['config']);w=CooperativeWorld(r['spec'],cfg);w.time=u['tick']*cfg.dt
                probe=np.array(r['probe_points']);before=metrics_from_snapshot(u['pre'],cfg,w,probe)
                after=u['probes'];mask=np.array(before['support']) & np.array(after['support'])
                message_effects.append(dict(scene=key[0],seed=key[1],arm=r['arm'],budget=key[2],agent=i,tick=u['tick'],
                    packets=u['diagnostic'].get('messages',[]),accepted=u['diagnostic']['accepted'],common_probes=int(mask.sum()),
                    value_error_reduction=float(np.mean(np.array(before['squared_error'],dtype=float)[mask]-np.array(after['squared_error'],dtype=float)[mask])) if mask.any() else None,
                    gradient_error_reduction=float(np.mean(np.array(before['gradient_squared_error'],dtype=float)[mask]-np.array(after['gradient_squared_error'],dtype=float)[mask])) if mask.any() else None,
                    force_change=u['diagnostic'].get('force_change'),shaping_jump=u['diagnostic']['shaping_jump'],
                    interpretation='joint local-and-message update; not an isolated packet attribution'))
            for u in updates:learning.append(dict(scene=key[0],seed=key[1],arm=r['arm'],budget=key[2],agent=i,tick=u['tick'],
                phase=u['phase'],damping=u['post']['damping'],policy=u['post']['policy'],accepted=u['diagnostic']['accepted'],
                field_mse=u['probes']['mse'],gradient_mse=u['probes']['gradient_mse'],coverage=u['probes']['coverage']))
    for key,arms in probe_index.items():
        if 'disabled' not in arms or key[0]=='changing_hazard':continue
        for arm,agents in arms.items():
            for i,agent in enumerate(agents):
                a=agent['field'];b=arms['disabled'][i]['field'];mask=np.array(a['support']) & np.array(b['support'])
                common.append(dict(scene=key[0],seed=key[1],budget=key[2],arm=arm,agent=i,count=int(mask.sum()),
                    own_coverage=a['coverage'],baseline_coverage=b['coverage'],
                    mse=float(np.mean(np.array(a['squared_error'],dtype=float)[mask])) if mask.any() else None,
                    baseline_mse=float(np.mean(np.array(b['squared_error'],dtype=float)[mask])) if mask.any() else None,
                    gradient_mse=float(np.mean(np.array(a['gradient_squared_error'],dtype=float)[mask])) if mask.any() else None))
    dump(root/'message_effects.json',message_effects)
    dump(root/'audits.json',audits);dump(root/'common_probes.json',common);dump(root/'learning_traces.json',learning)
    dump(root/'failures.json',failures)
    selected_figures(root)
    lines=['# Cooperative development results','',
        f"Completed {len(rows)} / {manifest['expected']} episodes. Failures are retained. No across-episode navigation learning or held-out performance claim.",'',
        f"Uncertainty calibration: marginal 95% coverage {cal['coverage_95']:.3f}; development threshold passed: {cal['passed']}. This is not proof of joint posterior calibration.",
        f"Recipient predictor: ready={predictor['ready']}, valid training pairs={predictor.get('valid_rows',0)}. Labels use eight-step hypothetical continuations, not full-mission oracle values.",'',
        '| Byte ceiling per agent | Arm | Episodes | Team successes | Mean team cost | Mean bytes |',
        '|---:|---|---:|---:|---:|---:|']
    for (budget,arm),rs in sorted(groups.items()):
        lines.append(f"| {budget} | {arm} | {len(rs)} | {sum(r['team_success'] for r in rs)} | {np.mean([r['team_cost'] for r in rs]):.4f} | {np.mean([r['bytes'] for r in rs]):.0f} |")
    valid=[a for a in audits if a['valid'] and a['predicted'] is not None]
    agreement=float(np.mean([np.sign(a['predicted'])==np.sign(a['realized']) for a in valid])) if valid else None
    lines+=['',f'Valid paired model/truth packet audits: {len(valid)} / {len(audits)}; sign agreement: {agreement}.',
        f'Packet outcomes: {json.dumps(reasons,sort_keys=True)}.','',
        'Communication, historical replay, and scheduling are separate hypotheses. Lower team cost is beneficial; every paired difference is exported, including negative results.',
        'Budget comparisons use equal ceilings, not equal bytes or useful computation. Status queries, replies, duplicate transmissions and acknowledgments are charged.',
        'Dynamic estimation uses a three-second sliding observation window and refitting. Peer packets expire after two seconds. No forecasting of unseen hazard changes is claimed.',
        'Agents have independent physical plants: this milestone evaluates information cooperation, not inter-agent collision avoidance or coupled mechanical dynamics.',
        'Audits intervene on one receiver, with future communication withheld equally in both branches. Short-horizon improvement is not complete-mission improvement.',
        'Final common-probe comparisons exclude dynamic scenes because endpoints can occur at different times; dynamic per-update traces retain timestamps.',
        'Bootstrap uncertainty uses whole time blocks. Sparse single-block estimates rely on a calibrated floor; information-gain results remain proxy-based.',
        'See failures.json and figures/failures for every failed episode; missing episodes remain in completion.json.']
    for (budget,arm),rs in sorted(groups.items()):
        ps=[p['difference'] for p in paired if p['arm']==arm and p['budget']==budget]
        if ps:lines.append(f'{arm}, ceiling {budget}: mean paired cost difference {np.mean(ps):+.6f}; worse in {sum(v>1e-9 for v in ps)}/{len(ps)} episodes.')
    value_decisions=[]
    for path in [p for p in episode_paths(root) if '_recipient_value_' in p.name]:
        value_decisions.extend(load(path)['decisions'])
    lines.append(f'Recipient-value payload decisions: {sum(d["send"] for d in value_decisions)} sends / {len(value_decisions)} opportunities. Zero sends is a negative scheduling result, not successful cooperation.')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    budgets=sorted({r['budget'] for r in rows})
    f,axes=plt.subplots(1,2,figsize=(14,5))
    labels=[];diffs=[];success=[]
    for (b,arm),rs in sorted(groups.items()):
        labels.append(f'{arm}\n{b//1000}k');diffs.append(np.mean([p['difference'] for p in paired if p['arm']==arm and p['budget']==b]));success.append(sum(r['team_success'] for r in rs)/len(rs))
    axes[0].bar(range(len(labels)),diffs);axes[0].axhline(0,color='k',lw=.5);axes[0].set_ylabel('Mean paired team cost − disabled')
    axes[1].bar(range(len(labels)),success);axes[1].set_ylabel('Team success fraction')
    for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=7)
    f.tight_layout();f.savefig(fig/'ablations.png');plt.close(f)
    f,ax=plt.subplots(figsize=(6,5))
    if valid:ax.scatter([a['predicted'] for a in valid],[a['realized'] for a in valid],s=9,alpha=.4)
    ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(xlabel='Predicted short-horizon benefit',ylabel='Realized short-horizon benefit',title=f'Packet audits: {len(valid)} valid pairs')
    f.tight_layout();f.savefig(fig/'value_calibration.png');plt.close(f)
    f,axes=plt.subplots(1,2,figsize=(12,4))
    for arm in manifest['arms']:
        rs=[r for r in rows if r['arm']==arm]
        axes[0].scatter([r['bytes'] for r in rs],[r['team_cost'] for r in rs],s=8,alpha=.5,label=arm)
        axes[1].scatter([r['replay_steps'] for r in rs],[r['wall_seconds'] for r in rs],s=8,alpha=.5)
    axes[0].set(xlabel='Actual bytes',ylabel='Team mission cost');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Replay transitions',ylabel='Wall seconds including audits')
    f.tight_layout();f.savefig(fig/'cost_compute.png');plt.close(f)




def metrics_from_snapshot(snap,cfg,world,points):
    agent=Learner(cfg,snap['stages'][-1]['entry'],snap['stages'][-1]['control_target'])
    agent.stages=[]
    for record in snap['stages']:
        stage=new_stage(record['id'],record['center'],record['entry'],record['control_target'],cfg)
        stage.weights=tensor(record['weights']);stage.readings=[Reading(**r) for r in record['readings']]
        agent.stages.append(stage)
    return probe_metrics(agent,world,points)


def selected_figures(root):
    root=Path(root);manifest=load(root/'manifest.json');fig=root/'figures'
    seed=manifest['animation_seed'];budget=max(manifest['budgets'])
    for scene in dict.fromkeys(s['name'] for s in manifest['scenarios']):
        selected=[]
        for arm in manifest['arms']:
            path=root/'episodes'/f'{scene}_{seed}_{arm}_b{budget}.json.gz'
            if path in episode_paths(root):selected.append(load(path))
        if not selected:continue
        tick=max(len(t) for r in selected for t in r['trajectories'])-1
        f,axes=plt.subplots(3,3,figsize=(12,9));spec=selected[0]['spec'];cfg=Config(**selected[0]['config'])
        world=CooperativeWorld(spec,cfg);world.time=(tick+1)*cfg.dt
        lo,hi,bot,top=spec['bounds'];xx,yy=np.meshgrid(np.linspace(lo,hi,80),np.linspace(bot,top,50))
        z=world.risk_truth(np.column_stack((xx.ravel(),yy.ravel()))).reshape(xx.shape)
        for ax,r in zip(axes.flat,selected):
            ax.imshow(z,origin='lower',extent=(lo,hi,bot,top),cmap='YlOrRd',vmin=0,vmax=1.6,alpha=.65)
            for i,color in enumerate(('tab:blue','tab:green')):
                tr=r['trajectories'][i];p=np.array([tr[0]['x']]+[t['after'] for t in tr if t['tick']<=tick])
                ax.plot(p[:,0],p[:,1],color=color);ax.scatter(*p[-1,:2],color=color,s=12)
            ax.set(xlim=(lo,hi),ylim=(bot,top),title=f"{r['arm']}: {r['summary']['team_cost']:.2f}")
        f.suptitle(f'{scene}, seed {seed}: matched clock {(tick+1)*cfg.dt:.2f}s; completed paths held')
        f.tight_layout();f.savefig(fig/f'{scene}_paired.png');plt.close(f)
        full=next((r for r in selected if r['arm']=='reciprocal_fixed'),None)
        if full:
            f,axes=plt.subplots(2,3,figsize=(13,7))
            for i,color in enumerate(('tab:blue','tab:green')):
                us=full['updates'][i];times=[u['tick']*cfg.dt for u in us]
                for ax,key in [(axes[0,0],'mse'),(axes[0,1],'gradient_mse'),(axes[0,2],'coverage')]:
                    ax.plot(times,[u['probes'][key] if u['probes'][key] is not None else np.nan for u in us],color=color,label=f'agent {i}')
                axes[1,0].plot(times,[u['post']['damping'] for u in us],color=color)
                axes[1,1].plot(times,[u['post']['policy'][0] for u in us],color=color)
                axes[1,1].plot(times,[u['post']['policy'][1] for u in us],color=color,ls='--')
                axes[1,2].scatter(times,[int(u['diagnostic']['accepted'])+i*.1 for u in us],color=color,s=9)
            for ax,title in zip(axes.flat,['Field value MSE','Gradient MSE','Support coverage','Damping','Risk / barrier coefficients','Accepted updates']):
                ax.set(title=title,xlabel='Seconds')
            f.tight_layout();f.savefig(fig/f'{scene}_learning.png');plt.close(f)
    # Actual time, risk, effort, and failure components remain separately visible.
    sums={}
    for path in episode_paths(root):
        r=load(path)
        if r['communication']['byte_budget']!=budget:continue
        sums.setdefault(r['arm'],[]).append([sum(a['components'][k] for a in r['agents']) for k in ('time','risk','effort','failure')])
    if sums:
        f,ax=plt.subplots(figsize=(10,5));bottom=np.zeros(len(sums));names=list(sums)
        values=np.array([np.mean(sums[k],axis=0) for k in names])
        for k,label in enumerate(('time','risk','effort','failure')):
            ax.bar(names,values[:,k],bottom=bottom,label=label);bottom+=values[:,k]
        ax.tick_params(axis='x',rotation=45);ax.legend();ax.set_ylabel('Mean team mission cost')
        f.tight_layout();f.savefig(fig/'cost_components.png');plt.close(f)


def frame(result,tick):
    spec=result['spec'];cfg=Config(**result['config']);world=CooperativeWorld(spec,cfg);world.time=(tick+1)*cfg.dt
    f,axes=plt.subplots(2,2,figsize=(11,8));lo,hi,bot,top=spec['bounds']
    xs=np.linspace(lo,hi,90);ys=np.linspace(bot,top,55);xx,yy=np.meshgrid(xs,ys);points=np.column_stack((xx.ravel(),yy.ravel()))
    axes[0,0].imshow(world.risk_truth(points).reshape(xx.shape),extent=(lo,hi,bot,top),origin='lower',cmap='YlOrRd',vmin=0,vmax=1.6,alpha=.7)
    for i,color in enumerate(('tab:blue','tab:green')):
        tr=[r for r in result['trajectories'][i] if r['tick']<=tick]
        if not tr:continue
        path=np.array([tr[0]['x']]+[r['after'] for r in tr]);axes[0,0].plot(path[:,0],path[:,1],color=color,label=f'agent {i}')
        axes[0,0].scatter(*path[-1,:2],color=color);axes[0,0].scatter(*spec['goals'][i],marker='*',color=color)
        # Snapshot at/before requested time only. No interpolation of model versions.
        updates=[u for u in result['updates'][i] if u['tick']<=tick]
        if updates:
            snap=updates[-1]['post'];s=snap['stages'][-1];sample=np.array([r['q'] for r in s['readings'] if int(r['uid'].split(':')[-1])%5])
            bounds=s['bounds'];mask=(xx>=bounds[0])&(xx<=bounds[1])&(yy>=bounds[2])&(yy<=bounds[3])
            if len(sample):mask &= (np.sum((points[:,None,:]-sample[None,:,:])**2,axis=2).min(1).reshape(xx.shape)<=.65**2)
            else:mask[:]=False
            v=risk(tensor(points),tensor(s['weights']),tensor(s['centers']),cfg.rbf_width).numpy().reshape(xx.shape)
            axes[0,1].contour(xx,yy,np.where(mask,v,np.nan),levels=5,colors=color,alpha=.7)
            axes[0,1].text(.02,.96-i*.07,f'agent {i}: v{snap["version"]}, stage {s["id"]}',transform=axes[0,1].transAxes,color=color)
            q=path[-1,:2];u,_=command(tensor(path[-1]),tensor(s['control_target']),tensor(s['weights']),tensor(s['centers']),cfg.rbf_width,tensor(snap['policy']),s['obstacles'],cfg.force_limit)
            axes[0,1].quiver(q[0],q[1],float(u[0]),float(u[1]),color=color,angles='xy',scale_units='xy',scale=4)
            for stage in snap['stages']:
                from matplotlib.patches import Rectangle
                b=stage['bounds'];axes[1,0].add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],fill=False,edgecolor=color,alpha=.2,lw=.7))
                for cf in stage['counterfactuals']:
                    p=np.array(cf['states']);axes[1,0].plot(p[:,0],p[:,1],color=color,alpha=.5)
            times=[u['tick']*cfg.dt for u in updates];axes[1,1].plot(times,[u['post']['damping'] for u in updates],color=color,label=f'damping {i}')
            for u in updates:
                if not u['diagnostic']['accepted']:axes[1,1].scatter(u['tick']*cfg.dt,u['post']['damping'],marker='x',color='red',s=15)
        if tr:
            cumulative=np.cumsum([r['cost'] for r in tr]);axes[1,1].plot([r['tick']*cfg.dt for r in tr],cumulative/10,color=color,ls='--',label=f'cost/10 {i}')
    recent=[p for p in result['packets'] if tick*cfg.dt-cfg.update_every*cfg.dt<=p['timestamp']<=tick*cfg.dt]
    for p in recent:
        a=np.array(p['anchors']);axes[0,0].scatter(a[:,0],a[:,1],marker='+',c='purple',s=25)
        receiver=int(p['recipient']);tr=[r for r in result['trajectories'][receiver] if r['tick']*cfg.dt<=p['timestamp']]
        if tr:axes[0,0].annotate('',xy=tr[-1]['x'][:2],xytext=p['q'],arrowprops=dict(arrowstyle='->',color='purple',alpha=.4,lw=1))
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set(xlim=(lo,hi),ylim=(bot,top),aspect='equal')
    axes[0,0].set_title('True risk, execution, transmitted anchors (+)');axes[0,0].legend(fontsize=7)
    axes[0,1].set_title('Logged estimated fields and control forces')
    axes[1,0].set_title('Retained stages and counterfactual trajectories')
    decisions=[d for d in result['decisions'] if d['tick']<=tick]
    axes[1,0].text(.02,.03,f'Payload sends: {sum(d["send"] for d in decisions)}/{len(decisions)} opportunities',transform=axes[1,0].transAxes,fontsize=8)
    axes[1,1].set_title('Damping / cumulative cost; rejected updates ×');axes[1,1].legend(fontsize=6);axes[1,1].set_xlabel('Seconds')
    f.suptitle(f"{spec['name']} seed {spec['seed']} | {result['arm']} | t={(tick+1)*cfg.dt:.2f}s")
    f.tight_layout();f.canvas.draw();im=Image.fromarray(np.asarray(f.canvas.buffer_rgba())[:,:,:3].copy());plt.close(f);return im


def render(root):
    root=Path(root);manifest=load(root/'manifest.json');fig=root/'figures';fig.mkdir(exist_ok=True)
    gallery=fig/'failures';gallery.mkdir(exist_ok=True);failure_links=[]
    paths=episode_paths(root)
    for path in paths:
        # Filename filter keeps rendering memory bounded.
        selected=f"_{manifest['animation_seed']}_{manifest['animation_arm']}_b{max(manifest['budgets'])}." in path.name
        r=load(path);failed=r['summary']['failures']>0
        if not selected and not failed:continue
        end=max(len(t) for t in r['trajectories'])-1
        if selected:
            ticks=sorted(set([0,end]+list(range(0,end+1,4))))
            frames=[frame(r,n) for n in ticks];dest=fig/(path.name.removesuffix('.json.gz')+'.gif')
            frames[0].save(dest,save_all=True,append_images=frames[1:],duration=160,loop=0)
            frames[-1].save(fig/(path.name.removesuffix('.json.gz')+'.png'))
        if failed:
            dest=gallery/(path.name.removesuffix('.json.gz')+'.png');frame(r,end).save(dest)
            failure_links.append(f'<li>{path.name}: {r["summary"]["failures"]} failures <a href="{dest.name}">figure</a></li>')
    (gallery/'index.html').write_text('<h1>All failed cooperative episodes</h1><ul>'+''.join(failure_links)+'</ul>' if failure_links else '<h1>No failed completed episodes</h1><p>Check completion.json for incomplete episodes.</p>')


def main(argv,render_only=False):
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args(argv)
    (render if render_only else report)(args.root)
