"""Engineering regressions only on preserved contexts; no new navigation runs.

The replay_hold_v1 stop rule remains in force for its three fork fixtures.
Existing retained policy replay is the primary parity baseline. The flawed
historical discrete route voter is reported separately, never relabeled as that
policy learner.
"""
import argparse
import copy
import gzip
import json
import os
import pickle
import platform
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .historical_policy_replay import SharedPolicyReplayLearner
from .route_decision import RouteDecisionLearner,RouteDecisionOptions
from .evaluation import dump,load
from .receiver_reconstruction import file_hash,fingerprint
from .attribution_audits import digest
from .model import tensor

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/current_campaign_v1/reconstruction/snapshots'
VARIANTS=('original_policy','shared_policy_graph','shared_policy_current_route','old_historical_route_vote')
DEPENDENCIES=('historical_policy_replay.py','historical_policy_replay_suite.py','route_decision.py',
    'stage_prediction.py','replay_budget.py','learner.py','model.py','cooperative.py','receiver_fitting.py',
    'localized_receiver.py','maze_learning.py','navigation.py','route_field.py','terrain_warning.py',
    'records.py','diagnostics.py','maze_family.py','shapes.py','maze_geometry.py')


def manifest():
    inputs=[dict(path=str(p.resolve()),name=p.name,sha256=file_hash(p)) for p in sorted(SOURCE.glob('*.pkl.gz'))]
    if len(inputs)!=24:raise ValueError('Expected preserved24 engineering snapshots')
    m=dict(schema=1,scope='saved-context engineering regression; no new episodes',inputs=inputs,
        variants=VARIANTS,source_hashes={p:file_hash(ROOT/'coph_stagewise'/p) for p in DEPENDENCIES},
        evidence='same delivered packet, retained measured histories, learning objective, selected global receiver fit and support',
        receiver=dict(fit='precision_bounded',solver='active_set',search='diagnostic',validation='relative_local',localization='global'),
        actual_route_options=dict(mode='prefix',starts='current',terminal='graph',horizon=8),
        transitions='as captured; no strict/hold changes',flow_ceiling=1024,
        finite_difference=dict(epsilon=1e-5,atol=1e-6,rtol=1e-4,environment='committed reconstructed model; frozen update horizon plan'),
        baseline='original differentiable retained policy replay already uses own entry state and entry_navigation',
        bug_scope='historical discrete route voter alone shares current candidate IDs and averages their costs over past states',
        stop_rule=dict(source=str(ROOT/'results/replay_hold_v1/summary.json'),sha256=file_hash(ROOT/'results/replay_hold_v1/summary.json'),
            respected=True,no_new_navigation=True,no_fixture_search=True),
        numerical=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,dtype='float64',threads=1,
            import_order='numpy_then_torch',environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        held_out_executed=False)
    return dict(m,fingerprint=fingerprint(m))


def freeze(root):
    root=Path(root);m=manifest()
    if (root/'manifest.json').exists():
        prior=load(root/'manifest.json')
        if prior['fingerprint']!=m['fingerprint']:raise ValueError('Frozen input/runtime/environment mismatch')
        return prior
    root.mkdir(parents=True,exist_ok=True)
    for name in ('audits','sources','figures'):(root/name).mkdir(exist_ok=True)
    for name in DEPENDENCIES:(root/'sources'/name).write_bytes((ROOT/'coph_stagewise'/name).read_bytes())
    dump(root/'manifest.json',dict(m,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return m


def prepare(agent,variant):
    a=copy.deepcopy(agent)
    cls=SharedPolicyReplayLearner if variant.startswith('shared') else RouteDecisionLearner
    a.__class__=cls
    a.mode='evidence' if variant=='old_historical_route_vote' else 'full'
    a.route_options=RouteDecisionOptions(mode='prefix' if variant in ('old_historical_route_vote','shared_policy_current_route') else 'graph',
        starts='historical' if variant=='old_historical_route_vote' else 'current',terminal='graph',horizon=8)
    a.route_decisions=[];a.replay_start_mode='historical';a.decision_tracing=True
    a.receiver_fit_mode='precision_bounded';a.receiver_solver_mode='active_set';a.receiver_search_mode='diagnostic'
    a.receiver_validation_mode='relative_local';a.receiver_localization_mode='global'
    a.shared_policy_calls=[];a._shared_active=False;a._shared_signature=None
    return a


def model(agent,x):
    controlled=copy.deepcopy(agent);u,_=controlled.action(x.copy())
    return dict(version=agent.version,damping=agent.damping,policy=agent.policy.tolist(),
        fields=[s.weights.tolist() for s in agent.stages],control=u.tolist(),
        entries=[list(s.entry) for s in agent.stages])


def gradient_check(agent):
    """Independent diagnostic work; preserve a fixed committed model and plan."""
    def evaluate(policy,derivatives):
        a=copy.deepcopy(agent);a.shared_policy_calls=[];a._shared_active=False
        if getattr(a,'replay_plan',None):
            a.replay_plan_active=True;a.replay_budget_start=a.counterfactual_steps;a.replay_external_reserve=0
        before=a.counterfactual_steps
        p=tensor(policy).requires_grad_(derivatives)
        cost,paths=a.replay(p,[s.weights for s in a.stages],a.damping)
        if cost is None:return dict(available=False,reason='unsupported',flow=a.counterfactual_steps-before)
        grad=torch.autograd.grad(cost,p,allow_unused=True)[0] if derivatives and cost.requires_grad else None
        return dict(available=True,cost=float(cost.detach()),gradient=grad.detach().tolist() if grad is not None else [0.,0.],
            flow=a.counterfactual_steps-before,steps=[len(p)-1 for _,p,_ in paths],policy_independent=not cost.requires_grad)
    initial=evaluate(agent.policy.tolist(),True);flow=initial['flow']
    if not initial['available']:return dict(initial,diagnostic_flow=flow)
    differences=[]
    for k in range(2):
        p=agent.policy.detach().numpy().copy();m=p.copy();p[k]+=1e-5;m[k]-=1e-5
        plus=evaluate(p,False);minus=evaluate(m,False);flow+=plus['flow']+minus['flow']
        if not plus['available'] or not minus['available']:return dict(available=False,reason='perturbation_unsupported',diagnostic_flow=flow)
        differences.append((plus['cost']-minus['cost'])/2e-5)
    exact=np.asarray(initial['gradient']);fd=np.asarray(differences);error=np.abs(exact-fd)
    return dict(available=True,gradient=exact.tolist(),finite_difference=fd.tolist(),absolute_error=error.tolist(),
        agrees=bool(np.all(error<=1e-6+1e-4*np.maximum(np.abs(exact),np.abs(fd)))),
        nonzero=bool(np.linalg.norm(exact)>1e-12),steps=initial['steps'],diagnostic_flow=flow,
        semantics='committed reconstructed model and frozen plan; independent of recorded update gradient')


def job(task):
    root,entry=task;root=Path(root);torch.set_num_threads(1);began=time.perf_counter()
    if file_hash(entry['path'])!=entry['sha256']:raise ValueError('Input snapshot changed')
    with gzip.open(entry['path'],'rb') as f:ctx,packet=pickle.load(f)
    original=digest(ctx);x=ctx['states'][1].copy();base=copy.deepcopy(ctx['agents'][1]);receipt=base.assimilate(packet)
    measured=digest([(s.readings,s.transitions) for s in base.stages]);results={};agents={}
    for variant in VARIANTS:
        a=prepare(base,variant);before=a.policy.clone();d=a.update(x.copy());state=model(a,x)
        assert digest([(s.readings,s.transitions) for s in a.stages])==measured
        assert d['counterfactual_steps']<=1024
        policy_changed=bool(torch.linalg.vector_norm(a.policy-before)>1e-10)
        results[variant]=dict(update=d,model=state,policy_changed=policy_changed,
            flow=d['counterfactual_steps'],accepted=d['accepted'],
            policy_gradient_norm=d.get('policy_gradient_norm',0.),
            replay_valid=d.get('replay_before') is not None,
            actual_route=d.get('route_decision'),
            shared_trace=copy.deepcopy(getattr(a,'shared_policy_calls',[])))
        agents[variant]=a
    parity=results['original_policy']['model']==results['shared_policy_graph']['model']
    parity=parity and results['original_policy']['flow']==results['shared_policy_graph']['flow']
    parity=parity and results['original_policy']['accepted']==results['shared_policy_graph']['accepted']
    assert parity,'Wrapper changed original retained policy behavior'
    fd=gradient_check(agents['shared_policy_graph'])
    # The old voter and current-only route selector are different mechanisms;
    # their validity counts are not policy-learning superiority measurements.
    assert digest(ctx)==original
    row=dict(snapshot=entry['name'],receipt=receipt,results=results,original_policy_parity=parity,
        gradient_check=fd,measurements_immutable=True,operational_state_unchanged=True,
        no_navigation_execution=True,seconds=time.perf_counter()-began)
    dump(root/'audits'/f"{entry['name'][:-7]}.json.gz",row)
    return dict(snapshot=entry['name'],parity=parity,gradient=fd.get('agrees'),seconds=row['seconds'])


def suite(root,workers):
    root=Path(root);m=freeze(root)
    tasks=[(str(root),e) for e in m['inputs'] if not (root/'audits'/f"{e['name'][:-7]}.json.gz").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    freeze(root);return report(root)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);rows=[load(p) for p in sorted((root/'audits').glob('*.json.gz'))]
    summary=[]
    for variant in VARIANTS:
        rr=[r['results'][variant] for r in rows]
        summary.append(dict(variant=variant,updates=len(rr),accepted=sum(r['accepted'] for r in rr),
            valid_policy_batches=sum(r['replay_valid'] for r in rr),nonzero_gradients=sum(r['policy_gradient_norm']>1e-12 for r in rr),
            committed_policy_changes=sum(r['policy_changed'] for r in rr),max_flow=max((r['flow'] for r in rr),default=0),
            valid_actual_route_choices=sum(r['actual_route'] is not None and r['actual_route']['reason']=='supported_route_choice' for r in rr)))
    checks=[r['gradient_check'] for r in rows];available=[g for g in checks if g['available']]
    verdict=dict(component='historical_policy_replay',complete=len(rows)==24,
        mechanism_correct=all(r['original_policy_parity'] and r['operational_state_unchanged'] and r['measurements_immutable'] for r in rows) and all(g['agrees'] for g in available),
        performance_demonstrated=False,
        metrics=dict(audits=len(rows),original_policy_exact_parity=sum(r['original_policy_parity'] for r in rows),
            gradient_checks_available=len(available),gradient_checks_agree=sum(g['agrees'] for g in available),
            gradient_checks_nonzero=sum(g.get('nonzero',False) for g in available),
            max_gradient_absolute_error=max((max(g['absolute_error']) for g in available),default=None),
            diagnostic_gradient_flow=sum(g.get('diagnostic_flow',0) for g in checks),variants=summary),
        limitations=['Original retained policy replay was already own-state/context conditioned; no numerical policy bug was repaired.',
            'The separate historical discrete voter shared current route identities and detached costs; it is explicitly excluded from the new wiring.',
            'Saved fork contexts are engineering regressions only; the prior navigation stop remains in force.',
            'No new episodes, fixture search, support expansion or navigation-benefit claim.',
            'Current-only prefix can remain unsupported; rejection retains incumbent routing.',
            'Finite differences are away from discrete route/active-set changes; unavailable perturbations remain excluded.'],
        artifacts=['REPORT.md','manifest.json','summary.json','audits/','figures/replay_engineering.png'])
    dump(root/'summary.json',summary);dump(root/'VERDICT.json',verdict)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    labels=[r['variant'].replace('_','\n') for r in summary]
    axs[0].bar(labels,[r['valid_policy_batches'] for r in summary]);axs[0].set_ylabel('Valid policy batches / 24')
    axs[1].bar(labels,[r['committed_policy_changes'] for r in summary]);axs[1].set_ylabel('Committed shared-policy changes / 24')
    axs[2].bar(labels,[r['max_flow'] for r in summary]);axs[2].axhline(1024,color='r',ls='--');axs[2].set_ylabel('Maximum local-update flow work')
    for ax in axs:ax.tick_params(axis='x',labelsize=7)
    fig.tight_layout();fig.savefig(root/'figures/replay_engineering.png',dpi=160);plt.close(fig)
    text='# Shared historical policy learning: engineering diagnosis\n\n'
    text+='**The original retained policy replay was already conditioned on each entry’s own state and copied context. The identified bug belongs to the separate discrete route-voting experiment.** This opt-in wiring preserves the original shared two-coefficient policy learner and requires the actual route selector to use the current physical state alone. It does not average route identities across different starts.\n\n'
    text+='| Variant | Accepted updates | Valid policy batches | Nonzero gradients | Committed policy changes | Maximum flow | Valid actual prefix choices |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"| {r['variant']} | {r['accepted']}/{r['updates']} | {r['valid_policy_batches']}/{r['updates']} | {r['nonzero_gradients']}/{r['updates']} | {r['committed_policy_changes']}/{r['updates']} | {r['max_flow']} | {r['valid_actual_route_choices']}/{r['updates']} |\n"
    text+=f"\nThe wrapper reproduces the original full-policy model, controls, acceptance and flow work exactly in {sum(r['original_policy_parity'] for r in rows)}/24 snapshots. Finite-difference checks agree in {sum(g['agrees'] for g in available)}/{len(available)} available cases; {len(checks)-len(available)} remain unavailable. These committed-model derivative checks are separate from each update’s proposal-gradient record. All measured histories and live operational state remain unchanged.\n\n"
    text+='The full policy loss retains existing time, risk and effort terms, bounded policy changes, reconstruction, physical reconciliation, held-out validation, rollback and 1,024-flow accounting. Entry/context/horizon signatures must remain fixed within an update. Candidate-support failures and fixed-braking/no-route cases retain the existing behavior and remain visible; no support is created. The current-only prefix stage consumes only remaining work after learning and retains the incumbent if unsupported or out of budget.\n\n'
    text+='The old historical route-vote row uses evidence-only fitting, as that experiment did. Its absent policy gradients are therefore not a failed policy-learning result. Comparisons with it isolate the distinction between discrete voting and differentiable shared-policy learning; they do not establish navigation superiority.\n\n'
    text+='The previous `replay_hold_v1` stop rule is preserved. These 24 snapshots are reused solely as engineering regressions. No new mission episodes, substitute scenes, treatment search, or held-out runs occurred. Performance improvement is not demonstrated.\n\n![Engineering diagnostics](figures/replay_engineering.png)\n'
    (root/'REPORT.md').write_text(text)
    return verdict


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','suite','report'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args(argv)
    if args.command=='freeze':freeze(args.out)
    elif args.command=='suite':suite(args.out,args.workers)
    else:report(args.out)


if __name__=='__main__':main()
