"""Frozen saved-snapshot comparison of global and compact received-field fits."""
import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import platform
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import torch

from .evaluation import dump,load
from .receiver_reconstruction import file_hash,fingerprint
from .receiver_corrections import load_saved_snapshot,errors
from .reconstruction_quality import _snapshot_jobs
from .reconstruction_diagnostics import field_queries,ranking
from .attribution_audits import digest
from .route_field import hypothetical_context,sync
from .model import tensor,risk
from .cooperative import field_vector

ROOT=Path(__file__).resolve().parents[1]
DEPENDENCIES=('model.py','learner.py','records.py','receiver_fitting.py','localized_receiver.py',
    'cooperative.py','receiver_summary.py','route_field.py','reconstruction_diagnostics.py',
    'diagnostics.py','stage_prediction.py','navigation.py','maze_learning.py','terrain_warning.py',
    'route_decision.py','localized_receiver_evaluation.py')
VARIANTS=('local_only','global_corrected','compact_corrected')


def input_jobs():
    jobs=[]
    current=ROOT/'results/current_campaign_v1/reconstruction'
    for p in sorted((current/'snapshots').glob('*.pkl.gz')):
        jobs.append(dict(path=str(p.resolve()),group='current',key=p.name[:-7],sha256=file_hash(p)))
    for path,meta in _snapshot_jobs(ROOT/'results/complementary_reconstruction_v1')[0]:
        p=Path(path)
        jobs.append(dict(path=str(p.resolve()),group='legacy',key=p.name[:-7],sha256=file_hash(p),meta=meta))
    return jobs


def manifest():
    m=dict(schema=1,development_only=True,inputs=input_jobs(),variants=VARIANTS,
        source_hashes={name:file_hash(ROOT/'coph_stagewise'/name) for name in DEPENDENCIES},
        convention='unchanged local-only 4x4 RBF proposal + compact raw/logit residual; radius .65; at most anchor_cap centers',
        objective='same delta-method local-precision regularization in compact coefficient coordinates; packet value/gradient covariance and age unchanged',
        receiver_options=dict(fit='precision_bounded',solver='active_set',search='diagnostic',validation='relative_local'),
        bounds=dict(total_coefficient_correction=.5,iterations=20,stationarity=1e-6),
        point_selection='frozen pre-delivery production route quadrature; common supported points; truth only in errors',
        outputs='proposed and committed errors, zero-effect outside compact union, solver/validator outcomes, matched controls',
        no_navigation_episodes=True,no_policy_replay=True,held_out_executed=False,
        numerical=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=1,
            import_order='numpy_then_torch',environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    return dict(m,fingerprint=fingerprint(m))


def freeze(root):
    root=Path(root);m=manifest()
    if (root/'manifest.json').exists():
        old=load(root/'manifest.json')
        if old['fingerprint']!=m['fingerprint']:raise ValueError('Frozen source/input/environment mismatch')
        return old
    root.mkdir(parents=True,exist_ok=True)
    for sub in ('audits','sources','figures'):(root/sub).mkdir(exist_ok=True)
    for name in DEPENDENCIES:(root/'sources'/name).write_bytes((ROOT/'coph_stagewise'/name).read_bytes())
    dump(root/'manifest.json',dict(m,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return m


def decode(entry):
    value=load_saved_snapshot(entry['path'])
    if entry['group']=='current':
        ctx,packet=value
        return [(ctx['agents'][1],ctx['states'][1],packet,ctx['worlds'][1],8)]
    if entry['meta']['group']=='navigation':return [value]
    ctx=value;snapshots=[]
    for sender in (0,1):
        a=copy.deepcopy(ctx['agents'][sender])
        packet=a.export(ctx['states'][sender].copy(),str(1-sender))
        if packet:snapshots.append((ctx['agents'][1-sender],ctx['states'][1-sender],packet,ctx['worlds'][1-sender],8))
    return snapshots


def evaluate(snapshot,key,group,index):
    agent,x,packet,world,tick=snapshot
    original=digest((agent,x,world));base=copy.deepcopy(agent);event=base.assimilate(packet)
    anchors=copy.deepcopy(base.navigation.get('terrain_anchors',[]))
    measured=digest([(s.readings,s.transitions) for s in base.stages])
    remote=digest(base.remote_terrain)
    quad=ranking(base,x,details=True)['quadrature']
    points=np.asarray([q for e in quad for q in e['points']])
    truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
    initial,initial_meta=field_queries(base,points)
    results={};predictions={};local_weights=None
    for variant in VARIANTS:
        a=copy.deepcopy(base);a.mode='evidence';a.fit_received=variant!='local_only'
        a.receiver_fit_mode='precision_bounded';a.receiver_solver_mode='active_set';a.receiver_search_mode='diagnostic'
        a.receiver_validation_mode='relative_local';a.receiver_localization_mode='compact' if variant=='compact_corrected' else 'global'
        before=a.counterfactual_steps;began=time.perf_counter();d=a.update(x.copy())
        proposed=[w.detach().clone() for w in a.reference]
        if variant=='local_only':local_weights=proposed
        if variant=='compact_corrected':assert all(torch.equal(w,z) for w,z in zip(proposed,local_weights))
        probe=copy.deepcopy(base)
        for k,(stage,w) in enumerate(zip(probe.stages,proposed)):
            stage.weights=w
            stage.local_residual=copy.deepcopy(a.receiver_localized_proposal[k]) if variant=='compact_corrected' and a.receiver_localized_proposal is not None else None
        sync(probe,publish=True)
        pp,pm=field_queries(probe,points);cc,cm=field_queries(a,points)
        predictions[variant]=(pp,cc)
        context=hypothetical_context(a);context['terrain_anchors']=copy.deepcopy(anchors)
        scored=ranking(a,x,context)
        controlled=copy.deepcopy(a);u,nominal=controlled.action(x.copy())
        locality=[]
        if variant=='compact_corrected':
            for stage in probe.stages:
                residual=getattr(stage,'local_residual',None)
                if residual is None:continue
                grid=np.array([(stage.center[0]+dx,stage.center[1]+dy) for dx in np.linspace(-1.5,1.5,25) for dy in np.linspace(-2,2,25)])
                distance=np.linalg.norm(grid[:,None,:]-np.asarray(residual['centers'])[None,:,:],axis=2).min(1)
                outside=distance>=residual['radius']
                before_values=field_vector(stage.weights,stage.centers,grid[outside],a.cfg.rbf_width)
                after_values=field_vector(stage.weights,stage.centers,grid[outside],a.cfg.rbf_width,residual)
                assert np.array_equal(before_values,after_values)
                locality.append(dict(stage=stage.id,outside_queries=int(outside.sum()),exact_value_gradient_equality=True,
                    max_coefficient=max(abs(c) for c in residual['coefficients']),centers=len(residual['centers'])))
        assert digest([(s.readings,s.transitions) for s in a.stages])==measured and digest(a.remote_terrain)==remote
        assert a.counterfactual_steps-before<=1024
        results[variant]=dict(accepted=d['accepted'],reason=d['reason'],diagnostic=d,seconds=time.perf_counter()-began,
            version=a.version,fields=[s.weights.tolist() for s in a.stages],proposed_fields=[w.tolist() for w in proposed],
            proposed_residuals=getattr(a,'receiver_localized_proposal',None),
            committed_residuals=[copy.deepcopy(getattr(s,'local_residual',None)) for s in a.stages],
            proposed_support=pm,committed_support=cm,control=u.tolist(),nominal=nominal.tolist(),
            shaping_energy=float(controlled.shaping_energy(x)),ranking_fixed_anchors=scored,
            flow=a.counterfactual_steps-before,fitting_work=a.message_fit_work-base.message_fit_work,
            locality=locality,measurements_immutable=True,remote_anchors_held_fixed=True)
    mask=np.isfinite(initial).all(1)
    for proposal,committed in predictions.values():mask&=np.isfinite(proposal).all(1)&np.isfinite(committed).all(1)
    for variant,(proposal,committed) in predictions.items():
        results[variant]['proposed_errors']=errors(proposal,truth,mask)
        results[variant]['committed_errors']=errors(committed,truth,mask)
    assert digest((agent,x,world))==original
    return dict(key=key,group=group,index=index,packet=packet.identity,receiver=agent.agent_id,event=event,
        query_points=points.tolist(),truth=truth.tolist(),common_support=mask.tolist(),initial_support=initial_meta,
        initial_errors=errors(initial,truth,mask),results=results,operational_state_unchanged=True,
        interpretation='matched saved-state update audit; no navigation treatment or generalization claim')


def job(task):
    root,entry=task;root=Path(root);torch.set_num_threads(1);began=time.perf_counter()
    if file_hash(entry['path'])!=entry['sha256']:raise ValueError('Input snapshot hash changed')
    rows=[evaluate(s,entry['key'],entry['group'],i) for i,s in enumerate(decode(entry))]
    dump(root/'audits'/f"{entry['group']}_{entry['key']}.json.gz",rows)
    return dict(key=entry['key'],group=entry['group'],audits=len(rows),seconds=time.perf_counter()-began)


def suite(root,workers):
    root=Path(root);m=freeze(root)
    tasks=[(str(root),e) for e in m['inputs'] if not (root/'audits'/f"{e['group']}_{e['key']}.json.gz").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    freeze(root)
    report(root)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);m=load(root/'manifest.json');rows=[r for p in sorted((root/'audits').glob('*.json.gz')) for r in load(p)]
    flat=[]
    for row in rows:
        baseline=row['results']['local_only']
        for variant,b in row['results'].items():
            flat.append(dict(key=row['key'],group=row['group'],index=row['index'],variant=variant,accepted=b['accepted'],reason=b['reason'],
                proposed_value=b['proposed_errors']['value_mae'],committed_value=b['committed_errors']['value_mae'],
                proposed_gradient=b['proposed_errors']['gradient_mae'],committed_gradient=b['committed_errors']['gradient_mae'],
                supported=b['committed_errors']['supported'],queries=b['committed_errors']['queries'],
                value_delta=b['committed_errors']['value_mae']-baseline['committed_errors']['value_mae'] if b['committed_errors']['value_mae'] is not None else None,
                control_delta=float(np.linalg.norm(np.asarray(b['control'])-baseline['control'])),
                flow=b['flow'],fitting_work=b['fitting_work'],solver=b['diagnostic'].get('receiver_fit'),locality=b['locality']))
    summary=[]
    for group in ('current','legacy'):
        for variant in VARIANTS:
            rr=[r for r in flat if r['group']==group and r['variant']==variant];valid=[r for r in rr if r['value_delta'] is not None]
            summary.append(dict(group=group,variant=variant,audits=len(rr),valid_errors=len(valid),accepted=sum(r['accepted'] for r in rr),
                improved=sum(r['value_delta']< -1e-12 for r in valid),worsened=sum(r['value_delta']>1e-12 for r in valid),
                mean_value_delta=float(np.mean([r['value_delta'] for r in valid])) if valid else None,
                changed_control=sum(r['control_delta']>1e-10 for r in rr),max_update_flow=max((r['flow'] for r in rr),default=0),
                reasons={reason:sum(r['reason']==reason for r in rr) for reason in sorted({r['reason'] for r in rr})}))
    dump(root/'rows.json',flat);dump(root/'summary.json',summary)
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,group in zip(axs,('current','legacy')):
        for i,variant in enumerate(VARIANTS):
            rr=[r for r in flat if r['group']==group and r['variant']==variant and r['value_delta'] is not None]
            ax.scatter([i]*len(rr),[r['value_delta'] for r in rr],alpha=.4,s=12)
        ax.axhline(0,color='k',lw=.7);ax.set(xticks=range(3),xticklabels=VARIANTS,ylabel='Committed value MAE − local-only',title=group)
        ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(root/'figures/committed_errors.png',dpi=160);plt.close(fig)
    text='# Localized receiver reconstruction\n\n'
    text+=f"Completed {len(rows)} matched saved-state audits ({sum(r['group']=='current' for r in rows)} current-campaign and {sum(r['group']=='legacy' for r in rows)} preserved complementary audits). No new mission episodes ran.\n\n"
    text+='| Group | Variant | Accepted | Error improves / worsens | Mean committed MAE change | Changed controls |\n|---|---|---:|---:|---:|---:|\n'
    for r in summary:text+=f"| {r['group']} | {r['variant']} | {r['accepted']}/{r['audits']} | {r['improved']}/{r['worsened']} of {r['valid_errors']} | {r['mean_value_delta']} | {r['changed_control']}/{r['audits']} |\n"
    text+='\nThe compact variant freezes the same-update local-only 4×4 field proposal and fits bounded Wendland C2 kernels in raw/logit space. It changes representation and the associated local information coordinates, not the sensor noise, packet residual weights, solver budgets, validators, or support rules. The same active-set/diagnostic solver and relative-local gate remain active. Whole-model rejection retains previous fields, compact residuals and routing snapshot.\n\n'
    text+='Compact corrections are exactly zero outside radius-.65 anchor neighborhoods; this does not certify those neighborhoods as measured support. Reserved measured records alone govern validation. Proposal and commitment errors remain separate, including solver and validator rejections. Controls are immediate fixed-state effects, not evidence of improved navigation. The old complementary cases are reused diagnostics and the current campaign contains three designed layouts with repeated sensor-noise seeds. No statistical generalization claim is made.\n\n'
    text+='All branch copies preserve operational state, measured histories, physical state and received route stores. Exported environmental packets continue to be reconstructed from local measurements only; operational residuals are not re-exported as independent evidence. Field, replay, route snapshots, summaries and force evaluation carry the same compact correction. Existing defaults remain global RBF fitting.\n\n![Committed errors](figures/committed_errors.png)\n'
    (root/'REPORT.md').write_text(text);(root/'FINDINGS.md').write_text(text)
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('freeze','suite','report'))
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args(argv)
    if args.command=='freeze':freeze(args.out)
    elif args.command=='suite':suite(args.out,args.workers)
    else:report(args.out)


if __name__=='__main__':main()
