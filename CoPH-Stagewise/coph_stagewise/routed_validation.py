"""Selected-controller held-out validation, without modifying the development experiment."""
import argparse,hashlib,json,time,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .completion_evaluation import job as frozen_job
from .complex_environments import SCENES,world_axes,paths,COLORS
from .records import Config
from .evaluation import load,dump

ARMS=('disabled','reciprocal_fixed')
ADDITIONS=('coph_stagewise/routed_validation.py','tests/test_routed_validation.py')
PACKAGE=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.loads(json.dumps(x,sort_keys=True,allow_nan=False))
def inventory(root):return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}

def verify_sources(reference):
    manifest=load(reference/'manifest.json');expected=manifest['source_hashes'];errors=[]
    with zipfile.ZipFile(reference/'sources.zip') as archive:
        for name,sha in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:errors.append('archive:'+name)
            path=PACKAGE/name
            if not path.is_file() or digest(path)!=sha:errors.append('current:'+name)
    extras={str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*.py') if '__pycache__' not in str(p)}-set(expected)
    if extras-set(ADDITIONS):errors+=['unexpected_source:'+n for n in sorted(extras-set(ADDITIONS))]
    if errors:raise ValueError('frozen source mismatch: '+', '.join(errors))
    return dict(frozen_source_hashes=expected,archive_sha256=digest(reference/'sources.zip'),
                additions={n:digest(PACKAGE/n) for n in ADDITIONS},verified=True)

def expected_keys(specs):return {(s['name'],s['seed'],a) for s in specs for a in ARMS}

def readiness(completed,specs):
    expected=expected_keys(specs);selected=[c for c in completed if c.get('mode')=='routed'];keys=[(c['scene'],c['seed'],c['arm']) for c in selected]
    exact=len(keys)==len(expected) and set(keys)==expected;groups=[]
    for arm in ARMS:
        cells=[c for c in selected if c['arm']==arm];rows=[c['metrics'] for c in cells if c.get('complete')]
        goals=sum(r['goals'] for r in rows);hits=sum(r['collisions'] for r in rows);timeouts=sum(r['timeouts'] for r in rows)
        groups.append(dict(arm=arm,episodes=len(rows),required_episodes=len(specs),goals=goals,required_goals=2*len(specs),collisions=hits,timeouts=timeouts,
            passed=exact and len(rows)==len(specs) and goals==2*len(specs) and hits==0 and timeouts==0 and all(r['goals']==2 and r['collisions']==0 and r['timeouts']==0 for r in rows)))
    return dict(passed=all(g['passed'] for g in groups),exact_pairing=exact,groups=groups)

def check_specs(specs):
    assert len(specs)==32
    assert {(s['name'],s['seed']) for s in specs}=={(name,seed) for name in SCENES for seed in range(110000,110008)}
    assert all(not s['dynamic'] and s['channel']=='normal' for s in specs)

def verify_episode(r,spec,arm,controller,communication):
    expected=canonical(asdict(replace(Config(**controller),steps=spec['steps'])))
    assert r['spec']==spec and r['arm']==arm,'scene or arm mismatch'
    assert r['config']==expected,'controller configuration mismatch'
    assert r['communication']==communication,'communication configuration mismatch'
    assert len(r['agents'])==2
    assert all(a['bytes']<=communication['byte_budget'] for a in r['agents'])
    assert abs(sum(sum(a['components'].values()) for a in r['agents'])-r['summary']['team_cost'])<1e-8
    for seq in r['trajectories']:assert 0<len(seq)<=spec['steps']

def job(args):
    spec,arm,root,communication,controller=args
    c=frozen_job((spec,'routed',arm,root,communication))
    if c['complete']:
        try:verify_episode(load(Path(root)/c['path']),spec,arm,controller,communication)
        except Exception as exc:c.update(complete=False,error='episode contract: '+repr(exc))
    return c

def frame(results,tick):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    fig,axes=plt.subplots(2,1,figsize=(12,7));spec=results[ARMS[0]]['spec'];dt=results[ARMS[0]]['config']['dt']
    end=max(t['tick'] for r in results.values() for seq in r['trajectories'] for t in seq);tick=min(tick,end)
    for ax,arm in zip(axes,ARMS):
        r=results[arm];world_axes(ax,spec);paths(ax,r,tick)
        labels=[]
        for i,seq in enumerate(r['trajectories']):
            ts=[t for t in seq if t['tick']<=tick]
            if not ts:continue
            t=ts[-1];n=t['navigation'];q=np.array(t['x'][:2]);color=COLORS[i]
            if n.get('route'):
                p=np.array(n['route']);ax.plot(p[:,0],p[:,1],':',color=color);ax.scatter(*n['waypoint'],marker='x',color=color,s=16)
            for v,c in ((n['nominal'],'orange'),(t['u'],color)):ax.arrow(*q,*(np.array(v)*.18),width=.012,color=c,length_includes_head=True)
            updates=[u for u in r['updates'][i] if u['tick']<=tick]
            if updates:
                u=updates[-1];snap=u['post'];labels.append(f'A{i} v{snap["version"]}, snapshot {u["tick"]*dt:.2f}s')
                for ox,oy,rad in snap['geometry']:ax.add_patch(Circle((ox,oy),rad+.17,fill=False,color=color,alpha=.3))
                for s in snap['stages']:
                    for cf in s['counterfactuals']:
                        p=np.array(cf['states']);ax.plot(p[:,0],p[:,1],'--',color=color,alpha=.6)
        for p in r['packets']:
            if 0<=tick*dt-p['timestamp']<dt*r['config']['update_every']:
                z=np.array(p['anchors']);ax.scatter(z[:,0],z[:,1],marker='+',color='purple',s=20)
        ax.set_title(f'routed-progress | {arm} | '+ '; '.join(labels),fontsize=9)
    fig.suptitle(f'{spec["name"]} | held-out seed {spec["seed"]} | t={(tick+1)*dt:.2f}s\nDotted route, dashed replay, sensed inflated disks; orange nominal / blue-green applied; purple packets.',fontsize=10)
    fig.tight_layout();fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return im

def report(root):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    manifest=load(root/'manifest.json');completed=load(root/'completion.json');rows=[c['metrics'] for c in completed if c['complete']]
    dump(root/'rows.json',rows);gate=readiness(completed,manifest['specs']);dump(root/'gate.json',gate)
    figures=root/'figures';gallery=figures/'failures';gallery.mkdir(parents=True,exist_ok=True);failures=[];selected={}
    for c in completed:
        if not c['complete']:failures.append(c);continue
        if c['seed']==manifest['animation_seed'] or c['metrics']['goals']!=2:
            r=load(root/'episodes'/c['path'])
            if c['seed']==manifest['animation_seed']:selected.setdefault(c['scene'],{})[c['arm']]=r
            if c['metrics']['goals']!=2:
                name=c['path'].replace('.json.gz','.png');fig,ax=plt.subplots(figsize=(10,3));world_axes(ax,r['spec']);paths(ax,r,10**9)
                ax.set_title(f'{c["scene"]} {c["seed"]} {c["arm"]}: goals {c["metrics"]["goals"]}/2; collisions {c["metrics"]["collisions"]}; timeouts {c["metrics"]["timeouts"]}',fontsize=9)
                fig.tight_layout();fig.savefig(gallery/name,dpi=110);plt.close(fig);failures.append(dict(c,figure=name,agent_outcomes=r['agents'],terminal_controls=[seq[-1].get('navigation',{}) for seq in r['trajectories']],terminal_states=[seq[-1]['after'] for seq in r['trajectories']]))
    dump(root/'failures.json',failures)
    (gallery/'index.html').write_text('<h1>Every failed or incomplete validation episode</h1>'+('No failures or incomplete episodes.' if not failures else ''.join('<p>'+c['path']+(f'<br><img width="850" src="{c["figure"]}">' if 'figure' in c else '<br>'+c.get('error','incomplete'))+'</p>' for c in failures)))
    paired=[]
    for s in manifest['specs']:
        rs={r['arm']:r for r in rows if (r['scene'],r['seed'])==(s['name'],s['seed'])}
        if len(rs)!=2:continue
        off,on=rs['disabled'],rs['reciprocal_fixed'];paired.append(dict(scene=s['name'],seed=s['seed'],goal_difference=on['goals']-off['goals'],secondary_cost_difference=on['team_cost']-off['team_cost'],bytes=on['bytes'],seconds_difference=on['wall_seconds']-off['wall_seconds']))
    dump(root/'paired.json',paired)
    lines=['# Routed-progress held-out validation','',f"**Navigation readiness: {'PASS' if gate['passed'] else 'FAIL'}.** Completed {len(rows)}/64 episodes. Selected after development, before held-out execution; controller and learner match the frozen development archive.",'','| Communication | Goals / agents | Collisions | Timeouts | Stagnation replans | Mean stationary seconds | Max stage | Interventions | Mean bytes | Mean compute seconds |','|---|---|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm]
        if rs:lines.append(f'| {arm} | {sum(r["goals"] for r in rs)}/{2*len(rs)} | {sum(r["collisions"] for r in rs)} | {sum(r["timeouts"] for r in rs)} | {sum(r["stagnation_replans"] for r in rs)} | {np.mean([r["near_stationary_seconds"] for r in rs]):.3f} | {max(r["max_stage"] for r in rs)} | {sum(r["interventions"] for r in rs)} | {np.mean([r["bytes"] for r in rs]):.0f} | {np.mean([r["wall_seconds"] for r in rs]):.2f} |')
    lines+=['','This evaluates static, independent-agent navigation with local sensing and fixed-rate environmental communication. It does not establish inter-agent collision avoidance, temporal forecasting, or adaptive communication reliability. Any failure blocks readiness; no parameters were tuned against held-out results. Final-test namespaces are untouched.','','Navigation reliability and communication benefit are separate questions. Existing isolated reciprocal audits show that evidence changes receiver models and controls. Equal completion rates provide no demonstrated completion advantage from communication; paired costs remain secondary diagnostics. The protected-progress failure and original overall development gate are preserved in the reference experiment.','','[Every failure or incomplete run](figures/failures/index.html). `rows.json` contains goal gaps, stage traversal, interventions, infeasibility, bytes, replay/identification work, routing/projection work, energy residuals and secondary costs. `paired.json` retains all 32 comparisons.','']
    for name,rs in selected.items():
        if set(rs)!=set(ARMS):continue
        end=max(t['tick'] for r in rs.values() for seq in r['trajectories'] for t in seq);ticks=sorted(set(np.linspace(0,end,32).astype(int)))
        frames=[frame(rs,int(t)) for t in ticks];frames[-1].save(figures/f'{name}_paired.png');frames[0].save(figures/f'{name}.gif',save_all=True,append_images=frames[1:],duration=max(100,int((end+1)*120/len(frames))),loop=0)
        dump(figures/f'{name}_frames.json',dict(seed=manifest['animation_seed'],ticks=[int(t) for t in ticks],geometry_and_replay='latest logged post-update snapshot at or before frame tick',execution='logged states and controls',truth_background='evaluator only'))
        lines += [f'![{name}](figures/{name}.gif)','']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return gate

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reference',type=Path,default=Path('results/completion_first_v1'));ap.add_argument('--workers',type=int,default=32);ap.add_argument('--resume',action='store_true');ap.add_argument('--render-only',action='store_true');args=ap.parse_args(argv)
    ref=args.reference.resolve();sources=verify_sources(ref);parent=load(ref/'manifest.json');specs=parent['specs']['validation'];check_specs(specs)
    development=readiness(load(ref/'development/completion.json'),parent['specs']['development'])
    if not development['passed']:ap.error('selected controller development gate failed')
    if args.out.resolve()==ref or ref in args.out.resolve().parents:ap.error('output must be separate from the preserved reference')
    if args.render_only and not args.out.exists():ap.error('render-only requires existing results')
    if not args.out.exists():
        args.out.mkdir(parents=True)
        manifest=dict(schema=1,selection_time_utc=datetime.now(timezone.utc).isoformat(),selection='routed-progress chosen after development, before held-out execution; protected retained as failed ablation',reference=str(ref),reference_inventory=inventory(ref),source_verification=sources,selected_development_gate=development,specs=specs,arms=ARMS,controller=parent['controllers']['routed'],communication=parent['communication'],animation_seed=110000,reserved_final_test=parent['reserved_final_test'],acceptance='64 episodes; 128 goals; zero collisions and timeouts; no required communication superiority',workers=args.workers)
        dump(args.out/'manifest.json',manifest)
        with zipfile.ZipFile(args.out/'evaluation_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for n in ADDITIONS:archive.write(PACKAGE/n,n)
    elif not(args.resume or args.render_only):ap.error('use a fresh output directory or --resume')
    manifest=load(args.out/'manifest.json')
    if manifest['source_verification']!=sources or manifest['reference_inventory']!=inventory(ref):ap.error('frozen reference or evaluation source changed')
    if manifest['specs']!=specs or manifest['controller']!=parent['controllers']['routed'] or manifest['communication']!=parent['communication']:ap.error('configuration mismatch')
    started=time.perf_counter();(args.out/'episodes').mkdir(exist_ok=True)
    completed=load(args.out/'completion.json') if (args.out/'completion.json').exists() else []
    if not args.render_only:
        done={(c['scene'],c['seed'],c['arm']) for c in completed}
        jobs=[(s,a,str(args.out/'episodes'),manifest['communication'],manifest['controller']) for s in specs for a in ARMS if (s['name'],s['seed'],a) not in done]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(job,j) for j in jobs]):
                c=f.result();completed.append(c);dump(args.out/'completion.json',completed);print(len(completed),c['scene'],c['seed'],c['arm'],c.get('metrics',{}).get('goals'),c.get('error',''),flush=True)
    result=report(args.out)
    assert inventory(ref)==manifest['reference_inventory'],'reference artifacts changed'
    dump(args.out/'verification.json',dict(reference_unchanged=True,source_verification=verify_sources(ref),readiness=result,seconds_this_invocation=time.perf_counter()-started,final_test_executed=False))

if __name__=='__main__':main()
