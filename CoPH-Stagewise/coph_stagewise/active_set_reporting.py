"""Reports numerical selection separately from estimation and navigation."""
from pathlib import Path
import collections
import numpy as np
from .evaluation import load,dump
from .receiver_reconstruction import file_hash


def report(root):
    root=Path(root);selection=load(root/'selection.json');summary=load(root/'solver_summary.json');tests=load(root/'verification/tests.json')
    lines=['# Active-set receiver fitting: isolation results','',f"Numerical gate selected: **{selection['selected']}**. Archived reproduction passed: **{selection['reproduction_passed']}**.",'',
        'Objective, precision weighting, total ±0.5 box, stationarity tolerance and outer iteration limit are unchanged. Diagnostic search additionally uses 40 trials and a rounding exception. Equal iteration limits do not imply equal computation.',
        '', '| Import order | Step/search | Converged / 697 | Near tolerance | Accepted: legacy / relative | Linear solves | Residual rows |', '|---|---|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['order']} | {s['config']} | {s['converged']} | {s['near_tolerance']} | {s['accepted']['legacy']} / {s['accepted']['relative_local']} | {s['linear_solves']} | {s['residual_rows']} |")
    lines+=['','## Selection and numerical limitations','']
    for c in selection['candidates']:
        lines.append(f"- {c['config']}: eligible={c['eligible']}.")
        for o in c['orders']:lines.append(f"  - {o['order']}: {o['failed']} failures versus {o['baseline_failed']} original; {len(o['stable_regressions'])} losses among original solves below half the tolerance; validity={o['valid']}.")
    quality=[]
    for path in sorted((root/'diagnostics').glob('*.json.gz')):
        data=load(path)
        for validator in ('legacy','relative_local'):
            c=collections.Counter();deltas=[]
            for a in data['audits']:
                b=a['branches'][validator];local=b['local_error']['value_mae'];prop=b['proposal_error']['value_mae'];committed=b['committed_error']['value_mae']
                if local is None or prop is None:c['unsupported']+=1;continue
                category='harmful' if prop-local>1e-12 else 'useful' if prop-local<-1e-12 else 'unchanged'
                c[category]+=1;c[category+('_accepted' if b['accepted'] else '_rejected')]+=1;deltas.append(committed-local)
            quality.append(dict(config=data['config'],order=data['environment']['order'],validator=validator,counts=dict(c),mean_committed_value_delta=float(np.mean(deltas)) if deltas else None))
    dump(root/'fit_quality.json',quality)
    lines+=['','[Proposal/commitment quality and acceptance confusion](fit_quality.json). Full value/gradient errors and unsupported-query denominators are retained in diagnostic branches; these values are excluded from solver selection.']
    disagreements=load(root/'import_disagreements.json');lines += ['',f'Import-order status disagreements: {len(disagreements)}. Near-tolerance results and all seven archived active-set exhaustion identities remain in the machine-readable records. Neither import order was chosen to improve outcomes.',
        'Multi-start reference agreement and Hessians evaluated along sampled trajectories are numerical evidence, not proofs of global uniqueness or convexity. No reference objective or route-query truth determines solver selection.',
        '',f"Tests: {tests['total']-len(tests['errors'])}/{tests['total']} passed. Preserved errors: {tests['errors']}."]
    nav=root/'navigation';failures=[];incomplete=[]
    if (nav/'rows.json').exists():
        rows=load(nav/'rows.json');failures=load(nav/'failures.json');incomplete=load(nav/'incomplete.json')
        lines+=['','## Navigation', '',f"{len(rows)} agent missions recorded: {sum(r['success'] for r in rows)} goals, {sum(r['collision'] for r in rows)} collisions, {sum(r['timeout'] for r in rows)} timeouts; {len(incomplete)} incomplete episodes.",
            '[Navigation report, audits, error tables and synchronized GIFs](navigation/REPORT.md). [Baseline reproduction](navigation/baseline_checks.json).',
            '', '| Arm | Goals | Accepted updates | Solver rejection | Relative rejection |', '|---|---:|---:|---:|---:|']
        for arm in sorted({r['method'] for r in rows}):
            rr=[r for r in rows if r['method']==arm];reasons=collections.Counter()
            for r in rr:reasons.update(r['reasons'])
            lines.append(f"| {arm} | {sum(r['success'] for r in rr)}/{len(rr)} | {reasons['accepted']} | {reasons['receiver_solver_failure']} | {reasons['receiver_relative_validation']} |")
    else:
        incomplete=[dict(id=f'{case}_{seed}_{arm}',executed=False,reason='Numerical gate did not select a solver') for case in ('complementary','reverse','redundant') for seed in (79200,79201) for arm in ('local_only','legacy','corrected_fit','validation_only','combined')]
        lines+=['','## Navigation not executed','','The numerical gate did not select a solver. All 30 planned episodes remain explicitly unexecuted; no navigation GIFs or success claims are manufactured.']
    dump(root/'failures.json',failures);dump(root/'incomplete.json',incomplete)
    lines+=['','All modes remain opt-in. This is development evidence for solver reliability, not a demonstration of route-choice benefit. Historical replay, decision-sensitive fixtures, controller tuning, adaptive communication and held-out evaluation remain deferred.',
        '', '[Selection](selection.json) · [Import disagreements](import_disagreements.json) · [Archived exhaustion cases](archived_exhaustions.json) · [Manifest](manifest.json) · [Failure inventory](failures.json) · [Incomplete inventory](incomplete.json)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for j,order in enumerate(('numpy_first','torch_first')):
        rr=[s for s in summary if s['order']==order];x=np.arange(len(rr))
        ax[j].bar(x,[s['converged'] for s in rr]);ax[j].set(xticks=x,xticklabels=[s['config'] for s in rr],ylim=(0,697),title=order,ylabel='Converged stage solves');ax[j].tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(root/'figures/solver_comparison.png',dpi=140);plt.close(fig)
    with (root/'REPORT.md').open('a') as f:f.write('\n![Solver comparison](figures/solver_comparison.png)\n')
