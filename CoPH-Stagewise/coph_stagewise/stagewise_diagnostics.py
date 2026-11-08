"""Frozen-manifest evaluations and retraining gate for supported stage prediction."""
import argparse,copy,json,time,traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .cooperative import CommunicationConfig,advertise
from .cooperative_evaluation import specification,execute
from .receiver_summary import summarize_receiver,SummaryCache
from .model_prediction import variant_prediction
from .continuations import continuation
from .evaluation import dump,load
from .cli import source_hashes,archive_sources

VARIANTS=('baseline','compact','cross_stage','both')
SCENES=('complementary','early_warning','reverse_warning','redundant','irrelevant')
HORIZONS=(8,16,32)


def diagnostic_case(agent,state,world,packet,tick,caches,spent):
    started=time.perf_counter();status=advertise(copy.deepcopy(agent),state,packet.sender,[z[:2] for z in packet.anchors])
    offered=summarize_receiver(agent,state,packet.sender)
    reference=continuation(agent,state,world,packet,tick,HORIZONS)
    variants={};traffic={};serialization_seconds=time.perf_counter()-started-reference['seconds']
    for variant in VARIANTS:
        summary=None;size=0;reason='legacy_status_only'
        # This is a separate hypothetical budget ledger per variant, not live traffic.
        key=(variant,agent.agent_id)
        status_size=status.bytes() if status else 32
        left=agent.communication.byte_budget-spent.get(key,0)
        if status_size>left:
            variants[variant]=dict(rows=[dict(horizon=h,valid=False,benefit=None,reason='status_budget') for h in HORIZONS],branches={},seconds=0.)
            traffic[variant]=dict(bytes=0,reason='status_budget');continue
        spent[key]=spent.get(key,0)+status_size
        if variant!='baseline':
            summary,size,reason=caches[variant].transfer(offered,packet.sender,agent.now,agent.communication.byte_budget-spent[key])
            spent[key]+=size
        variants[variant]=variant_prediction(packet,status,summary,agent.cfg,agent.communication,variant,
                                             seed=world.seed,horizons=HORIZONS)
        traffic[variant]=dict(bytes=status_size+size,status_bytes=status_size,summary_bytes=size,reason=reason,
                              cumulative=spent[key],summary_digest=summary.model_digest if summary else None)
    return dict(packet=packet.identity,recipient=agent.agent_id,stage=agent.stages[-1].id,tick=tick,start=state.tolist(),
        reference=reference,variants=variants,traffic=traffic,summary=asdict(offered),status=asdict(status) if status else None,
        packet_payload=asdict(packet),summary_generation_seconds=serialization_seconds,total_seconds=time.perf_counter()-started)


def job(args):
    spec,destination=args;root=Path(destination);torch.set_num_threads(1)
    cases=[];caches={v:SummaryCache() for v in VARIANTS};spent={};started=time.perf_counter()
    def capture(a,x,w,m,t):
        c=diagnostic_case(a,x,w,m,t,caches,spent);c.update(scene=spec['name'],seed=spec['seed']);cases.append(c)
    try:
        episode=execute(spec,'reciprocal_fixed',audits=True,diagnostic_callback=capture,replay_domain='retained')
        path=root/f"{spec['name']}_{spec['seed']}.json.gz"
        dump(path,dict(spec=spec,episode=episode,cases=cases,seconds=time.perf_counter()-started))
        return dict(scene=spec['name'],seed=spec['seed'],complete=True,cases=len(cases),path=path.name)
    except Exception as exc:
        path=root/f"{spec['name']}_{spec['seed']}.error.json"
        dump(path,dict(spec=spec,cases=cases,error=str(exc),traceback=traceback.format_exc()))
        return dict(scene=spec['name'],seed=spec['seed'],complete=False,cases=len(cases),path=path.name,error=str(exc))


def flatten(cases):
    rows=[]
    for c in cases:
        for variant in VARIANTS:
            prediction=c['variants'][variant]
            for h in HORIZONS:
                target=next(r for r in c['reference']['rows'] if r['horizon']==h)
                pred=next(r for r in prediction['rows'] if r['horizon']==h)
                valid=target['valid'] and pred['valid']
                disagreement=[]
                for branch in ('deliver','withhold'):
                    if branch in prediction['branches']:
                        disagreement.append(prediction['branches'][branch]['initial_update']['accepted']!=c['reference']['branches'][branch]['initial_update']['accepted'])
                rows.append(dict(scene=c['scene'],seed=c['seed'],packet=c['packet'],recipient=c['recipient'],stage=c['stage'],
                    variant=variant,horizon=h,valid=valid,reason=pred['reason'],truth=target['benefit'],prediction=pred['benefit'],
                    error=pred['benefit']-target['benefit'] if valid else None,
                    acceptance_disagreement=float(np.mean(disagreement)) if disagreement else None,
                    bytes=c['traffic'][variant]['bytes'],seconds=prediction['seconds']))
    return rows


def identity(r):return (r['scene'],r['seed'],r['packet'])


def metrics(rows):
    output=[]
    for h in HORIZONS:
        cohort=[r for r in rows if r['horizon']==h]
        common={identity(r) for r in cohort}
        for r in cohort:
            if not r['valid']:common.discard(identity(r))
        for v in VARIANTS:
            subset=[r for r in cohort if r['variant']==v];matched=[r for r in subset if identity(r) in common]
            nonzero=[r for r in matched if abs(r['truth'])>1e-6 and abs(r['prediction'])>1e-6]
            output.append(dict(horizon=h,variant=v,total=len(subset),valid=sum(r['valid'] for r in subset),common_valid=len(matched),
                common_mae=float(np.mean([abs(r['error']) for r in matched])) if matched else None,
                sign_agreement=float(np.mean([np.sign(r['truth'])==np.sign(r['prediction']) for r in nonzero])) if nonzero else None,
                sign_pairs=len(nonzero),acceptance_disagreement=float(np.mean([r['acceptance_disagreement'] for r in matched])) if matched else None,
                diagnostic_bytes=sum(r['bytes'] for r in subset),prediction_seconds=sum(r['seconds'] for r in subset)))
    return output


def gate(rows,complete=True):
    """Predeclared 16-step gate, independent of development outcomes."""
    base={identity(r):r for r in rows if r['horizon']==16 and r['variant']=='baseline'}
    both={identity(r):r for r in rows if r['horizon']==16 and r['variant']=='both'}
    matched=[k for k in base if base[k]['valid'] and both[k]['valid']]
    b=float(np.mean([abs(base[k]['error']) for k in matched])) if matched else None
    n=float(np.mean([abs(both[k]['error']) for k in matched])) if matched else None
    coverage_ok=all(not r['valid'] or both[k]['valid'] for k,r in base.items())
    passed=bool(complete and len(matched)>=10 and coverage_ok and n<b-1e-12)
    return dict(passed=passed,horizon=16,matched=len(matched),baseline_mae=b,combined_mae=n,
        baseline_valid=sum(r['valid'] for r in base.values()),combined_valid=sum(r['valid'] for r in both.values()),
        no_lost_baseline_cases=coverage_ok,episodes_complete=complete,
        criterion='at least 10 common cases; lower paired MAE; retain every baseline-valid prediction; all validation episodes complete',
        action='retrain_on_separate_training_manifest' if passed else 'retain_existing_predictor')


def report(root,completion):
    cases=[]
    for item in completion:
        data=load(root/item['path']);cases.extend(data['cases'])
    rows=flatten(cases);summary=metrics(rows)
    dump(root/'rows.json',rows);dump(root/'summary.json',summary);dump(root/'completion.json',completion)
    text=['# Supported stage prediction evaluation','',
        'Development mechanism evidence. All variants predict the same opt-in retained-replay receiver. Exact snapshots and true-world branches are evaluator-only.',
        'All prediction branches share an indexed common-anchor Gaussian proxy draw and sensor randomness. The atlas-wide posterior is not claimed calibrated.',
        'Baseline reconstructs the previous sparse status. Compact uses learned state and all retained summaries with single-stage prediction. Cross-stage uses the retained summaries but legacy initial policy/damping. Both uses learned state and retained-stage prediction. Stage geometry necessarily requires summary traffic even in the cross-stage ablation.',
        'Summary-derived readings and validation are hypothetical. Hypothetical traffic is charged on a separate per-variant ledger at audited opportunities; it is not the live episode traffic and does not claim exact resource equality.',
        '',f"Episodes: {len(completion)}; incomplete: {sum(not c['complete'] for c in completion)}; eligible packet cases: {len(cases)}.",'',
        '| Steps | Variant | Valid / total | Common | Common MAE | Sign agreement (n) | Acceptance disagreement |',
        '|---|---|---|---|---|---|---|']
    def fmt(v):return 'unavailable' if v is None else f'{v:.6f}'
    for r in summary:
        text.append(f"| {r['horizon']} | {r['variant']} | {r['valid']}/{r['total']} | {r['common_valid']} | {fmt(r['common_mae'])} | {fmt(r['sign_agreement'])} ({r['sign_pairs']}) | {fmt(r['acceptance_disagreement'])} |")
    text+=['','Metrics use common-valid cases at each horizon. Sign agreement excludes effects within 1e-6 of zero. Unsupported predictions remain missing. Empty scenes and incomplete runs remain in completion.json.',
           'No temporal forecasting, remote-only support, or private receiver snapshots are used by predictors. A retained atlas cannot certify unseen forward stages.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,h in zip(axes,HORIZONS):
        for v in VARIANTS:
            rs=[r for r in rows if r['horizon']==h and r['variant']==v and r['valid']]
            ax.scatter([r['truth'] for r in rs],[r['prediction'] for r in rs],label=v,s=12,alpha=.5)
        low,high=ax.get_xlim();ax.plot([low,high],[low,high],'k:');ax.set_title(f'{h} steps')
        ax.set_xlabel('True-world receiver benefit');ax.set_ylabel('Predicted benefit')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'prediction.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for v in VARIANTS:
        rs=[r for r in summary if r['variant']==v]
        axes[0].plot(HORIZONS,[r['valid']/max(1,r['total']) for r in rs],marker='o',label=v)
    axes[0].set(xlabel='Horizon steps',ylabel='Valid prediction fraction');axes[0].legend(fontsize=7)
    rs=[r for r in summary if r['horizon']==16]
    axes[1].bar([r['variant'] for r in rs],[r['diagnostic_bytes']/1000 for r in rs]);axes[1].set_ylabel('Audited status + summary kB')
    fig.tight_layout();fig.savefig(root/'coverage_bytes.png',dpi=150);plt.close(fig)
    text+=['','![Predictions](prediction.png)','','![Coverage and hypothetical bytes](coverage_bytes.png)']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    return rows


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args(argv)
    if args.out.exists():ap.error('use a fresh output directory')
    if args.workers<1:ap.error('workers must be positive')
    args.out.mkdir(parents=True)
    specs={profile:[specification(scene,base+k,'development') for scene in SCENES for k in range(count)]
        for profile,base,count in [('smoke',60000,2),('development',60000,8),('predictor_validation',45000,2),('training',30000,8)]}
    manifest=dict(specs=specs,variants=VARIANTS,horizons=HORIZONS,replay_domain='retained',
        communication=asdict(CommunicationConfig()),source_hashes=source_hashes(),numpy=np.__version__,torch=torch.__version__,
        gate='16 steps; >=10 common; lower MAE; no lost baseline-valid cases; complete validation',
        reserved_validation=[100000,199999],reserved_test=[200000,299999],
        training_policy='execute training only after predictor-validation gate passes',
        legacy_predictor='results/cooperative_development_final/predictor.json',
        environment_draw='shared indexed Gaussian common-anchor proxy; bounded value/gradient transport into supported bases; not calibrated atlas posterior',
        selection='first eligible packet per visited receiver stage',status='development evidence')
    dump(args.out/'manifest.json',manifest);archive_sources(args.out)
    all_completion={};rows_by_profile={};started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for profile in ('smoke','development','predictor_validation'):
            destination=args.out/profile;destination.mkdir()
            completion=[]
            for result in pool.map(job,[(s,str(destination)) for s in specs[profile]]):
                completion.append(result);print(profile,result['scene'],result['seed'],result['complete'],result['cases'],flush=True)
            all_completion[profile]=completion;rows_by_profile[profile]=report(destination,completion)
        outcome=gate(rows_by_profile['predictor_validation'],all(r['complete'] for r in all_completion['predictor_validation']))
        dump(args.out/'gate.json',outcome)
        if outcome['passed']:
            destination=args.out/'training';destination.mkdir();completion=list(pool.map(job,[(s,str(destination)) for s in specs['training']]))
            rows=report(destination,completion);all_completion['training']=completion
            # Train only on legal model continuations, never reference/audit outcomes.
            from .scheduling import value_features,fit_predictor
            from .cooperative import Packet
            training=[]
            for item in completion:
                data=load(destination/item['path'])
                for c in data['cases']:
                    if c['status'] is None:continue
                    packet=Packet(**c['packet_payload']);status=Packet(**c['status'])
                    label=next(r for r in c['variants']['both']['rows'] if r['horizon']==16)
                    training.append(dict(features=value_features(packet,status,.96).tolist(),label=label))
            predictor=fit_predictor(training);predictor['label']='16-step legal compact/retained model continuation; not true-world labels'
            predictor['promoted']=False
            dump(args.out/'candidate_predictor.json',predictor)
        else:dump(args.out/'training_status.json',dict(executed=False,reason='validation_gate_failed',retained_predictor=manifest['legacy_predictor']))
    dump(args.out/'completion.json',dict(profiles=all_completion,seconds=time.perf_counter()-started))
    (args.out/'REPORT.md').write_text('# Supported stagewise prediction\n\n'+
        f"Gate passed: **{outcome['passed']}**. Existing predictor remains unmodified.\n\n"+
        '\n'.join(f'- [{p}]({p}/REPORT.md)' for p in all_completion)+'\n\nSee [gate.json](gate.json) for predeclared criteria and results.\n')
    if any(not r['complete'] for group in all_completion.values() for r in group):raise SystemExit(1)

if __name__=='__main__':main()
