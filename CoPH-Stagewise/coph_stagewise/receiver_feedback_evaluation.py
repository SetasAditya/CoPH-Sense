"""Fixed-rate factorial evaluation of receiver feedback and boundary replay."""
import argparse,json,time,traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .records import Config
from .cooperative import CommunicationConfig
from .cooperative_evaluation import specification,execute
from .evaluation import dump,load
from .cli import source_hashes,archive_sources

SCENES=('complementary','early_warning','reverse_warning','redundant','irrelevant')
VARIANTS=('compact_fixed','receiver_fixed','compact_boundary','receiver_boundary')
HORIZONS=(8,16,32)

def job(args):
    spec,variant,destination=args;root=Path(destination);torch.set_num_threads(1)
    mode,horizon=variant.split('_');path=root/f"{spec['name']}_{spec['seed']}_{variant}.json.gz"
    try:
        result=execute(spec,'reciprocal_fixed',audits=False,replay_domain='retained',feedback_mode=mode,
                       horizon_mode='supported_boundary' if horizon=='boundary' else 'fixed',replay_cap=1024,feedback_audits=True)
        dump(path,result)
        return dict(scene=spec['name'],seed=spec['seed'],variant=variant,complete=True,path=path.name)
    except Exception as exc:
        dump(path,dict(error=str(exc),traceback=traceback.format_exc()))
        return dict(scene=spec['name'],seed=spec['seed'],variant=variant,complete=False,path=path.name,error=str(exc))

def case_key(r):return (r['scene'],r['seed'],r['receiver'],r['stage'],r['packet'])

def extract(episode,variant):
    rows=[];spec=episode['spec']
    for f in episode['feedback_records']:
        assessment=f['assessment'];truth=f['truth_audit']
        for h in HORIZONS:
            t=next(r for r in truth['rows'] if r['horizon']==h)
            p=next(r for r in assessment['rows'] if r['horizon']==h) if assessment else dict(valid=False,valid_draws=0,benefit=None,dispersion=None,exclusions=[f['summary_reason']])
            agreement=[];timing=[];goal_timing=[]
            if assessment:
                for d in assessment['draws']:
                    for branch,b in d['branches'].items():
                        ref=truth['branches'][branch];agreement.append(b['initial_update']['accepted']==ref['initial_update']['accepted'])
                        cp=b['checkpoints'][str(h)];ct=ref['checkpoints'][str(h)]
                        if cp['valid'] and ct['valid']:
                            timing.append(cp['executed_steps']-ct['executed_steps'])
                            if np.linalg.norm(np.array(ct['state'][:2])-spec['goals'][f['receiver']])<.25:
                                goal_timing.append(cp['executed_steps']-ct['executed_steps'])
            valid=t['valid'] and p['valid']
            rows.append(dict(scene=spec['name'],seed=spec['seed'],variant=variant,receiver=f['receiver'],stage=f['stage'],packet=f['packet'],
                state=f['state'],horizon=h,valid=bool(valid),valid_draws=p['valid_draws'],truth=t['benefit'],prediction=p['benefit'],dispersion=p['dispersion'],
                error=p['benefit']-t['benefit'] if valid else None,exclusions=p['exclusions'],
                acceptance_agreement=float(np.mean(agreement)) if agreement else None,
                termination_timing_error=float(np.mean(np.abs(timing))) if timing else None,
                goal_timing_error=float(np.mean(np.abs(goal_timing))) if goal_timing else None,
                goal_timing_pairs=len(goal_timing),feedback_sent=f['feedback_sent']))
    updates=[u for us in episode['updates'] for u in us]
    diagnostics=[u['diagnostic'] for u in updates]
    plans=[p for d in diagnostics for p in d.get('replay_plan',[])]
    checks=[d['counterfactual_steps'] for d in diagnostics]
    for f in episode['feedback_records']:
        branches=list(f['truth_audit']['branches'].values())
        if f['assessment']:branches += [b for d in f['assessment']['draws'] for b in d['branches'].values()]
        checks += [u['diagnostic']['counterfactual_steps'] for b in branches for u in b['updates']]
    assert all(c<=1024 for c in checks),'per-update replay ceiling exceeded'
    assert all(a['bytes']<=episode['communication']['byte_budget'] for a in episode['agents'])
    assert all(abs(a['cost']-sum(a['components'].values()))<1e-9 for a in episode['agents'])
    summary=dict(episode['summary'],variant=variant,
        opportunities=len(episode['feedback_records']),predictions=sum(f['assessment'] is not None for f in episode['feedback_records']),
        handoffs=sum(len(c.get('stage_events',[])) for u in updates for c in u['replay_calls']),
        actual_stage_handoffs=sum(sum(t2['stage']!=t1['stage'] for t1,t2 in zip(ts,ts[1:])) for ts in episode['trajectories']),
        pilots=sum(d.get('replay_pilot_steps',0) for d in diagnostics),
        boundary_plans=sum(p['reason']=='supported_boundary' for p in plans),
        selected_horizons=[p['horizon'] for p in plans],pilot_reasons=[p['reason'] for p in plans],
        nonzero_gradients=sum(d['policy_gradient_norm']>1e-12 for d in diagnostics),
        exhausted_updates=sum(d.get('replay_budget_exhausted',False) for d in diagnostics),
        max_update_work=max(checks,default=0),
        feedback_transitions=sum(f['assessment']['work']['transitions'] for f in episode['feedback_records'] if f['assessment']),
        feedback_replay_steps=sum(f['assessment']['work']['replay_steps'] for f in episode['feedback_records'] if f['assessment']),
        truth_transitions=sum(b['steps'] for f in episode['feedback_records'] for b in f['truth_audit']['branches'].values()),
        truth_replay_steps=sum(b['replay_steps'] for f in episode['feedback_records'] for b in f['truth_audit']['branches'].values()),
        payload_bytes=sum(a['payload_bytes'] for a in episode['agents']),status_bytes=sum(a['status_bytes'] for a in episode['agents']),
        ack_bytes=sum(a['ack_bytes'] for a in episode['agents']))
    return rows,summary

def report(root,completion):
    rows=[];episodes=[];feedback=[]
    for c in completion:
        if not c['complete']:continue
        ep=load(root/c['path']);r,s=extract(ep,c['variant']);rows+=r;episodes.append(s)
        feedback += [dict(scene=c['scene'],seed=c['seed'],variant=c['variant'],**f) for f in ep['feedback_records']]
    dump(root/'rows.json',rows);dump(root/'episodes.json',episodes);dump(root/'feedback_audits.json.gz',feedback);dump(root/'completion.json',completion)
    metrics=[]
    for h in HORIZONS:
        by_variant={v:{case_key(r):r for r in rows if r['variant']==v and r['horizon']==h} for v in VARIANTS}
        common=set.intersection(*[{k for k,r in rs.items() if r['valid']} for rs in by_variant.values()])
        # Physical state matching is essential: feedback traffic can alter later episodes.
        common={k for k in common if all(np.allclose(by_variant[v][k]['state'],by_variant[VARIANTS[0]][k]['state'],atol=1e-12,rtol=0) for v in VARIANTS)}
        for v,rs in by_variant.items():
            values=list(rs.values());valid=[r for r in values if r['valid']];matched=[rs[k] for k in common]
            signs=[r for r in valid if abs(r['truth'])>1e-6 and abs(r['prediction'])>1e-6]
            avg=lambda name,items:float(np.mean([r[name] for r in items if r[name] is not None])) if any(r[name] is not None for r in items) else None
            metrics.append(dict(variant=v,horizon=h,total=len(values),valid=len(valid),full_draws=sum(r['valid_draws']==4 and r['valid'] for r in values),
                mae=float(np.mean([abs(r['error']) for r in valid])) if valid else None,
                common=len(matched),common_mae=float(np.mean([abs(r['error']) for r in matched])) if matched else None,
                sign_agreement=float(np.mean([np.sign(r['truth'])==np.sign(r['prediction']) for r in signs])) if signs else None,sign_n=len(signs),
                acceptance_agreement=avg('acceptance_agreement',values),goal_timing_mae=avg('goal_timing_error',values),goal_timing_n=sum(r['goal_timing_pairs'] for r in values)))
    dump(root/'metrics.json',metrics)
    fmt=lambda x:'missing' if x is None else f'{x:.6g}'
    lines=['# Receiver-side evaluation and boundary-reaching replay','',
        'Development evidence: fixed-rate exchange; no predictor retraining or rate changes. Each estimate is audited from its own receiver snapshot. Across-arm common cases additionally require matching physical states, packet identities, and stages; models can differ.',
        'Four indexed reconstructed-environment draws share sensing randomness between deliver/withhold branches. Initial exact-receiver updates use actual measurements. Future observations and compact-summary validation are hypothetical. Dispersion is not calibrated confidence. Partial-draw estimates are conditional on support and must be read with coverage.',
        f"\nEpisodes: {len(completion)}; incomplete: {sum(not c['complete'] for c in completion)}. Failed agents: {sum(e['failures'] for e in episodes)}. Zero-opportunity episodes: {sum(e['opportunities']==0 for e in episodes)}.",
        '\n| Horizon | Variant | Valid / total (4 draws) | Common n / MAE | Own-case MAE | Acceptance agreement | Sign agreement (n) | Goal timing MAE (n) |',
        '|---|---|---|---|---|---|---|---|']
    for m in metrics:
        lines.append(f"| {m['horizon']} | {m['variant']} | {m['valid']}/{m['total']} ({m['full_draws']}) | {m['common']} / {fmt(m['common_mae'])} | {fmt(m['mae'])} | {fmt(m['acceptance_agreement'])} | {fmt(m['sign_agreement'])} ({m['sign_n']}) | {fmt(m['goal_timing_mae'])} ({m['goal_timing_n']}) |")
    lines += ['','| Variant | Team cost mean | Failed agents | Execution / replay handoffs | Boundary plans | Replay transitions | Feedback replay | Bytes / feedback bytes | Max update transitions |', '|---|---|---|---|---|---|---|---|---|']
    for v in VARIANTS:
        es=[e for e in episodes if e['variant']==v]
        total=lambda key:sum(e[key] for e in es)
        lines.append(f"| {v} | {fmt(float(np.mean([e['team_cost'] for e in es])) if es else None)} | {total('failures')} | {total('actual_stage_handoffs')} / {total('handoffs')} | {total('boundary_plans')} | {total('replay_steps')} | {total('feedback_replay_steps')} | {total('bytes')} / {total('feedback_bytes')} | {max([e['max_update_work'] for e in es],default=0)} |")
    lines += ['','Feedback replaces a basic acknowledgment when it fits the existing byte ceiling. Compact summaries and requested current uncertainty status are charged. Feedback cannot recover bytes already delivered; budget depletion can change later transmissions even at the same fixed opportunity rate. Payload, status, acknowledgments (including feedback), processing and evaluator audit work are separate in episodes.json. Feedback computation remains charged when the return message cannot fit.',
        'Historical pilot horizons are frozen per update. Unsupported continuations are missing, not zero benefit. Genuine goal/collision termination retains the normal cost definition. Sign agreement excludes effects within 1e-6 of zero. Goal timing errors compare executed steps at the checkpoint when the true-world branch reached its goal, including predictions that have not yet reached it.',
        'All failures and zero-opportunity episodes are retained. Raw updates, pilot fallback reasons, shaping jumps, trajectories and packet lineage are in the compressed episode files. Complete feedback and truth branches are in feedback_audits.json.gz. No held-out validation/test namespace was executed.']
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,h in zip(axs,HORIZONS):
        for v in VARIANTS:
            rs=[r for r in rows if r['variant']==v and r['horizon']==h and r['valid']]
            ax.scatter([r['truth'] for r in rs],[r['prediction'] for r in rs],s=9,alpha=.4,label=v)
        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],'k:');ax.set(title=f'{h} steps',xlabel='True-world benefit',ylabel='Predicted benefit')
    axs[0].legend(fontsize=6);fig.tight_layout();fig.savefig(root/'prediction.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for v in VARIANTS:
        ms=[m for m in metrics if m['variant']==v];axs[0].plot(HORIZONS,[m['valid']/max(1,m['total']) for m in ms],marker='o',label=v)
    axs[0].set(xlabel='Steps',ylabel='Valid estimate fraction');axs[0].legend(fontsize=6)
    for ax,key,label in [(axs[1],'replay_steps','Operational replay transitions'),(axs[2],'bytes','Actual bytes')]:
        ax.bar(range(4),[sum(e[key] for e in episodes if e['variant']==v) for v in VARIANTS]);ax.set_xticks(range(4),VARIANTS,rotation=25,fontsize=7);ax.set_ylabel(label)
    fig.tight_layout();fig.savefig(root/'coverage_work.png',dpi=150);plt.close(fig)
    baseline={(e['scene'],e['seed']):e for e in episodes if e['variant']=='compact_fixed'}
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for i,v in enumerate(VARIANTS):
        es=[e for e in episodes if e['variant']==v and (e['scene'],e['seed']) in baseline]
        axs[0].scatter([i]*len(es),[e['team_cost']-baseline[e['scene'],e['seed']]['team_cost'] for e in es],s=12,alpha=.5)
        axs[1].scatter([i]*len(es),[e['wall_seconds']-e['feedback_audit_seconds'] for e in es],s=12,alpha=.5)
    for ax in axs:ax.set_xticks(range(4),VARIANTS,rotation=20,fontsize=7)
    axs[0].set_ylabel('Paired cost difference vs compact fixed');axs[1].set_ylabel('Episode seconds excluding truth audits')
    fig.tight_layout();fig.savefig(root/'cost_compute.png',dpi=150);plt.close(fig)
    lines += ['','![Prediction](prediction.png)','','![Coverage and work](coverage_work.png)','','![Cost and computation](cost_compute.png)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(root/'failures.json',[dict(scene=e['scene'],seed=e['seed'],variant=e['variant'],failures=e['failures']) for e in episodes if e['failures']])
    return metrics

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args(argv)
    if args.out.exists():ap.error('use a fresh output directory')
    if args.workers<1:ap.error('workers must be positive')
    args.out.mkdir(parents=True)
    specs={profile:[specification(scene,60000+k,'development') for scene in SCENES for k in range(count)] for profile,count in [('smoke',2),('development',8)]}
    manifest=dict(specs=specs,variants=VARIANTS,communication=asdict(CommunicationConfig()),horizons=HORIZONS,draws=4,
        replay_domain='retained',replay_cap=1024,boundary_max_steps=32,boundary_follow_steps=4,
        resolved_configs={v:asdict(Config(steps=180,replay_objective='mission',replay_domain='retained',replay_update_cap=1024,
            replay_horizon_mode='supported_boundary' if v.endswith('boundary') else 'fixed')) for v in VARIANTS},
        costs=dict(time=1.,risk=4.,effort=.03,failure=40.),
        source_hashes=source_hashes(),numpy=np.__version__,torch=torch.__version__,
        draw_index='SeedSequence([scene_seed + tick*17 + receiver_index, draw_index, 2718])',
        reserved_validation=[100000,199999],reserved_test=[200000,299999],
        status='development only; fixed-rate; no retraining; smoke repeats first two development seeds')
    dump(args.out/'manifest.json',manifest);archive_sources(args.out)
    started=time.perf_counter();all_completion={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for profile in ('smoke','development'):
            root=args.out/profile;root.mkdir();completion=[]
            for item in pool.map(job,[(s,v,str(root)) for s in specs[profile] for v in VARIANTS]):
                completion.append(item);print(profile,item['scene'],item['seed'],item['variant'],item['complete'],item.get('error',''),flush=True)
            all_completion[profile]=completion;report(root,completion)
            if not all(c['complete'] for c in completion):break
    dump(args.out/'completion.json',dict(profiles=all_completion,seconds=time.perf_counter()-started))
    (args.out/'REPORT.md').write_text('# Receiver feedback factorial evaluation\n\n'+ '\n'.join(f'- [{p}]({p}/REPORT.md)' for p in all_completion)+'\n\nDefaults and sender predictor remain unchanged. See manifest.json for frozen settings and source fingerprints.\n')
    if any(not c['complete'] for cs in all_completion.values() for c in cs):raise SystemExit(1)

if __name__=='__main__':main()
