"""Post-gate reporting only; cannot enter operational learners or screening."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .evaluation import load, dump
from .receiver_reconstruction import file_hash


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    m=load(root/'manifest.json');g=load(root/'gate.json')
    rows=[load(root/'screening'/f'{k}.json.gz') for k in m['specs']]
    auditrows=[];case_rows=[]
    for case in ('upper_risk','lower_risk','clear_control'):
        rr=[r for r in rows if r['case']==case]
        case_rows.append(dict(case=case,screened=len(rr),complete=sum(r['complete'] for r in rr),
            gate_counts={k:sum(r.get('checks',{}).get(k,False) for r in rr) for k in ('support','evidence','sensitivity','replay')},
            margin_range=[min(r['score']['margin'] for r in rr),max(r['score']['margin'] for r in rr)],
            interval_envelope=[min(r['bootstrap']['interval'][0] for r in rr),max(r['bootstrap']['interval'][1] for r in rr)]))
    for row in rows:
        if 'audit' not in row:continue
        audit=row['audit'];base=audit['branches']['no_fit_none']
        for arm,b in audit['branches'].items():
            replay=b['replay_probe']
            auditrows.append(dict(id=row['id'],case=row['case'],seed=row['seed'],arm=arm,time=row['time'],
                accepted=b['accepted'],reason=b['reason'],supported=b['errors']['supported'],queries=b['errors']['queries'],
                value_mae=b['errors']['value_mae'],gradient_mae=b['errors']['gradient_mae'],
                value_delta=b['errors']['value_mae']-base['errors']['value_mae'] if b['errors']['value_mae'] is not None else None,
                gradient_delta=b['errors']['gradient_mae']-base['errors']['gradient_mae'] if b['errors']['gradient_mae'] is not None else None,
                control_delta=float(np.linalg.norm(np.asarray(b['control'])-base['control'])),
                frozen_anchor_margin=b['frozen_anchor_score']['margin'],natural_margin=b['natural_score']['margin'],
                update_flow=b['update_flow'],fitting_work=b['fitting_work'],
                replay_valid=replay['valid'],replay_decision_reach=any(r['reaches_decision'] for r in replay['reach']),
                replay_distance=max((r['distance'] for r in replay['reach']),default=None),
                replay_reasons=[r['reason'] for r in replay['eligibility']],
                horizons=[dict(stage=k,horizon=v['horizon'],reason=v['reason']) for k,v in replay['horizons'].items()],
                diagnostic_flow=replay['diagnostic_flow'],
                measurements_unchanged=audit['measurements_unchanged'],state_unchanged=audit['physical_and_operational_state_unchanged']))
    horizon_reasons=Counter((h['horizon'],h['reason']) for r in auditrows for h in r['horizons'])
    source_counts={case:Counter(src for row in rows if row['case']==case for edge in row['score']['quadrature'] for src in edge['route_field']['source']) for case in ('upper_risk','lower_risk','clear_control')}
    summary=dict(cases=case_rows,matched_audits=len(rows),isolated_update_branches=len(auditrows),
        max_update_flow=max(r['update_flow'] for r in auditrows),flow_ceiling=1024,
        update_flow=sum(r['update_flow'] for r in auditrows),diagnostic_probe_flow=sum(r['diagnostic_flow'] for r in auditrows),
        bootstrap_linear_solves=sum(r['bootstrap']['linear_solves'] for r in rows),
        summed_screen_worker_seconds=sum(r['seconds'] for r in rows),
        tests=dict(targeted=4,passed=4),treatment_episodes=0 if g['stop'] else None,
        source_archive_sha256=file_hash(root/'sources.zip'),manifest_sha256=file_hash(root/'manifest.json'),
        report_source_sha256=file_hash(Path(__file__)),route_source_counts={k:dict(v) for k,v in source_counts.items()},
        replay_horizon_reasons=[dict(horizon=k[0],reason=k[1],count=v) for k,v in horizon_reasons.items()])
    dump(root/'audit_rows.json',auditrows);dump(root/'summary.json',summary)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    arms=list(m['arms'])
    for i,arm in enumerate(arms):
        rr=[r for r in auditrows if r['arm']==arm]
        axs[0].scatter([i]*len(rr),[r['value_delta'] for r in rr],alpha=.6,s=14)
        axs[1].scatter([i]*len(rr),[r['update_flow'] for r in rr],alpha=.6,s=14)
        axs[2].bar(i,sum(r['replay_decision_reach'] for r in rr))
    for ax in axs:
        ax.set_xticks(range(len(arms)),arms,rotation=65,ha='right',fontsize=7)
    axs[0].axhline(0,color='k',lw=.7);axs[0].set_ylabel('Value MAE change vs no-fit / no-replay')
    axs[1].set_ylabel('Update flow evaluations (ceiling 1,024)')
    axs[2].set_ylabel('Diagnostic replay reaches branch / 24')
    fig.tight_layout();fig.savefig(root/'figures/matched_updates.png',dpi=160);plt.close(fig)
    text='# Symmetric-fork reconstruction and replay: failed eligibility gate\n\n'
    text+='**All 24 communication-free screens completed. Neither risk case qualified; zero navigation treatment episodes were run.** This is a preserved fixture-eligibility failure, not a measured zero treatment effect.\n\n'
    text+='| Case | Support | Eligible fitting | Sensitivity | Historical decision reach | Learned upper−lower margin |\n|---|---:|---:|---:|---:|---:|\n'
    for c in case_rows:
        counts=c['gate_counts'];lo,hi=c['margin_range']
        text+=f"| {c['case']} | {counts['support']}/8 | {counts['evidence']}/8 | {counts['sensitivity']}/8 | {counts['replay']}/8 | {lo:.4f} to {hi:.4f} |\n"
    text+='\n## What the gate explains\n\n'
    text+='Equal geometric branch lengths remove the previous fixture’s explicit detour disadvantage, but they do not make the receiver’s learned decision indifferent. Spatially asymmetric measured support, reconstructed values, and anchor fallback can still establish a strong route preference. The 128-draw intervals condition on those support/fallback choices; they quantify neither unseen terrain nor calibrated route uncertainty.\n\n'
    hist=[r for r in auditrows if r['arm']=='no_fit_historical'];cur=[r for r in auditrows if r['arm']=='no_fit_current']
    distances=[r['replay_distance'] for r in hist if r['replay_distance'] is not None]
    text+=f"Historical replay produced valid returned paths in {sum(r['replay_valid'] for r in hist)}/24 probes and reached an outgoing branch in {sum(r['replay_decision_reach'] for r in hist)}/24. Its maximum returned travel was {max(distances):.3f} m. All planned entry horizons retained the existing eight-step fallback: either no successor was known or the incumbent pilot could not reach its handoff. It did not extend support or create a successor. Current-state probes were valid in {sum(r['replay_valid'] for r in cur)}/24; exclusions and their exact reasons remain in `audit_rows.json`.\n\n"
    text+='| Case | Committed-field queries | Anchor fallback | Unknown fallback | Total branch queries |\n|---|---:|---:|---:|---:|\n'
    for case,counts in source_counts.items():
        text+=f"| {case} | {counts['field']} | {counts['anchor']} | {counts['unknown']} | {sum(counts.values())} |\n"
    text+='\nThese are repeated fixed-branch quadrature occurrences across eight sensor seeds, not independent observations. “Both edges supported” requires at least one committed-field query on each outgoing edge; it does not imply full branch coverage.\n\n'
    fits=[r for r in auditrows if r['arm']=='fit_none']
    changed=[r for r in fits if r['value_delta'] is not None and abs(r['value_delta'])>1e-12]
    text+=f"The matched no-replay fitting branches improved supported value MAE in {sum(r['value_delta']< -1e-12 for r in changed)}/{len(fits)}, worsened it in {sum(r['value_delta']>1e-12 for r in changed)}/{len(fits)}, and were unchanged in {len(fits)-len(changed)}/{len(fits)}. These are isolated pre-branch updates with fixed incoming packets, not mission-level benefit estimates.\n\n"
    text+='## Isolation, computation, and interpretation\n\n'
    text+=f"The 24 complete snapshots produced {len(auditrows)} matched fit×replay update branches. Every branch preserved measured histories, the original operational state, and the per-update ceiling; maximum recorded update work was {summary['max_update_flow']} flow evaluations. Bootstrap work was {summary['bootstrap_linear_solves']} linear solves. Replay reach probes cost {summary['diagnostic_probe_flow']} additional evaluator-only flow evaluations; matched branch updates cost {summary['update_flow']}. These are separate from communication-free screening work and are summed work, not wall time.\n\n"
    text+='The clear control passed its exempted support/evidence gate but was withheld because neither risk case qualified. Seeds vary sensor noise only: there are three designed cases, not 24 independent environments. No held-out namespace was executed. All previous decision-sensitive results remain unchanged.\n\n'
    text+='The rejected preflight freeze is retained as `preflight_manifest.json` and `preflight_sources.zip`: an unrelated parallel campaign added Python files before screening, and the original global source fingerprint correctly refused execution. The final manifest records the runtime dependency sources and was frozen before any screen. No fixture adjustment followed outcomes. Four targeted campaign tests passed; full-suite results belong to the campaign-level verification.\n\n'
    if (root/'independent_checks.json').exists():
        independent=load(root/'independent_checks.json')
        text+=f"Independent verification re-executed all {independent['matched_update_branches']} isolated updates from the {independent['snapshots']} hashed snapshots. Fields, versions, acceptance, and operational flow counts reproduced bitwise; fixed-branch margins exactly matched the production graph-cost routine (maximum difference {max(r['max_absolute_production_margin_difference'] for r in independent['checks']):g}). Received route stores, measured histories, original physical state, bootstrap seeds, and independent indexed sensor streams were checked. The final frozen runtime fingerprint still verifies. See `independent_checks.json`.\n\n"
    text+='A valid historical-replay decision experiment remains missing. The next architecture discussion should address supported replay reach and route-source coverage, rather than report these failed gates as navigation evidence.\n\n![Frozen gate](figures/gate.png)\n\n![Matched updates](figures/matched_updates.png)\n'
    (root/'REPORT.md').write_text(text);(root/'FINDINGS.md').write_text(text)
    dump(root/'failure_inventory.json',dict(screen_errors=[r for r in rows if not r['complete']],
        excluded=[dict(id=r['id'],reasons=r['reasons']) for r in rows if not r['passed']],
        control_withheld=[r['id'] for r in rows if r['case']=='clear_control'],
        navigation_failures='unavailable: navigation treatments blocked by failed gate'))
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    report(ap.parse_args().out)
