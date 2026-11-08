"""Additional read-only diagnostics and verdict for the simple-fix campaign."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .simple_communication_suite import ARMS,BUDGETS
from .communication_reconciliation_report import episode_diagnostics


def report(out):
    root=Path(out);rows=load(root/'rows.json');good=[r for r in rows if 'error' not in r];diagnostics=[];baseline_checks=[];link_checks=[]
    archive=Path(__file__).resolve().parents[1]/'results/communication_reconciliation_v1/episodes'
    for row in good:
        r=load(root/row['episode_file']);diagnostics.append(episode_diagnostics(r,row))
        sent=[e for e in r['events'] if e.get('reason')=='sent'];modes={};prepared={};fallbacks={}
        for e in sent:
            selection=e.get('selection',{});mode=selection.get('advertisement_codec',{}).get('mode')
            if mode:
                prepared[mode]=prepared.get(mode,0)+1
                if e.get('status_bytes',0)>0:modes[mode]=modes.get(mode,0)+1
            fallback=selection.get('fallback')
            if fallback:fallbacks[fallback]=fallbacks.get(fallback,0)+1
        link_checks.append(dict(key=row['key'],condition=row['selector'],budget=row['budget'],codec_modes=modes,prepared_codec_modes=prepared,fallbacks=fallbacks,
            fixed_ticks=all(e['tick']%r['config']['update_every']==0 for e in sent),
            byte_caps=all(sum(a['bytes'][k] for k in ('status','payload','acknowledgment'))<=row['budget'] for a in r['agents']),
            work_cap=all(a['max_update_work']<=1024 for a in r['agents'])))
        if row['selector'] in ('disabled','rotation','existing_selector'):
            old_selector='rotation' if row['selector'] in ('disabled','rotation') else 'receiver_relevant'
            old_key=f"{row['scene']}_{row['seed']}_{row['arm']}_{old_selector}_{row['budget']}";path=archive/f'{old_key}.json.gz'
            if path.exists():
                old=load(path)
                controls_equal=all(
                    [(t['x'],t['u'],t['after'],t['version'],t['stage']) for t in aa]==[(t['x'],t['u'],t['after'],t['version'],t['stage']) for t in bb] for aa,bb in zip(r['trajectories'],old['trajectories']))
                baseline_checks.append(dict(key=row['key'],reference=old_key,controls_bitwise_equal=controls_equal,samples_equal=r['samples']==old['samples'],observations_equal=r['observations']==old['observations'],
                    carrier_primary_error=abs(row['carrier_time_risk']-old['agents'][1]['time_risk_cost']),carrier_full_error=abs(row['carrier_cost']-old['agents'][1]['cost'])))
    dump(root/'episode_diagnostics.json',diagnostics);dump(root/'baseline_checks.json',baseline_checks);dump(root/'link_checks.json',link_checks)
    paired=load(root/'paired_rows.json');baseline_ok=len(baseline_checks)==72 and all(c['controls_bitwise_equal'] and c['samples_equal'] and c['observations_equal'] and c['carrier_primary_error']<1e-10 and c['carrier_full_error']<1e-10 for c in baseline_checks)
    mechanism_ok=all(c['fixed_ticks'] and c['byte_caps'] and c['work_cap'] for c in link_checks) and all(a['traffic_accounting']['matches_logged_agent_totals'] for a in diagnostics)
    comparisons=[]
    for budget in BUDGETS:
        for arm in ARMS:
            for reference in ('rotation','existing_selector'):
                rr=[p for p in paired if p['budget']==budget and p['arm']==arm and p['reference']==reference]
                if rr:comparisons.append(dict(budget=budget,arm=arm,reference=reference,pairs=len(rr),mean_primary_difference=float(np.mean([r['primary_difference'] for r in rr])),mean_full_difference=float(np.mean([r['full_difference'] for r in rr])),mean_team_bytes_difference=float(np.mean([r['team_bytes_difference'] for r in rr])),better=sum(r['primary_difference']< -1e-8 for r in rr),worse=sum(r['primary_difference']>1e-8 for r in rr)))
    verdict=load(root/'VERDICT.json');verdict.update(mechanism_correct=mechanism_ok and baseline_ok,baseline_reproduction_passed=baseline_ok,comparisons=comparisons,
        performance_scope='Observed paired development differences only; no calibrated/generalization superiority claim',
        performance_demonstrated=any(c['arm']=='both' and c['reference']=='rotation' and c['mean_primary_difference']< -1e-8 for c in comparisons))
    verdict['artifacts']+=['FINDINGS.md','episode_diagnostics.json','baseline_checks.json','link_checks.json','animation_inventory.json']
    dump(root/'VERDICT.json',verdict)
    summaries=[]
    for name in ('rotation','existing_selector','ack_only','compact_status_only','both'):
        rr=[d for d in diagnostics if d['budget']==8000 and d['selector']==name]
        agents=[a for r in rr for a in r['per_agent_traffic']]
        summaries.append(dict(arm=name,episodes=len(rr),agents=len(agents),zero_payload=sum(a['sent']==0 for a in agents),status_no_payload=sum(a['sent']==0 and a['bytes']['status']>0 for a in agents)))
    dump(root/'starvation_8000.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4));names=[s['arm'] for s in summaries]
    axes[0].bar(names,[s['status_no_payload'] for s in summaries]);axes[0].set_ylabel('Agents spending status but sending no payload (8k)');axes[0].tick_params(axis='x',rotation=30)
    for reference,offset in (('rotation',-.17),('existing_selector',.17)):
        rr=[c for c in comparisons if c['arm']=='both' and c['reference']==reference];axes[1].bar(np.arange(len(rr))+offset,[c['mean_primary_difference'] for c in rr],.32,label='both − '+reference)
    axes[1].set_xticks(range(4),[str(b) for b in BUDGETS]);axes[1].set_ylabel('Mean paired carrier primary difference');axes[1].axhline(0,color='black',lw=.7);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'figures/starvation_and_paired.png');plt.close(fig)
    lines=['# Simple communication repair findings','',f"Completed {len(good)}/168 episodes; {verdict['goals']}/{2*len(good)} goals, {verdict['collisions']} collisions, {verdict['timeouts']} timeouts. Baseline reruns reproduce all72 archived control/observation/cost records: {baseline_ok}. Fixed-cadence, own-byte, flow and reconstructed charge-trace checks: {mechanism_ok}.",
        '', 'The ack/interface option and overhead option are separate. Ack-only keeps legacy status-before-payload charging. The overhead option combines lossless cached/delta coding and reserve admission/fallback; it does not isolate coding from admission. All arms retain packet contents/schema, learner/controller, sensing, horizons and fixed opportunities. Actual payloads differ after selection and behavior diverge.',
        '', '| Budget | Arm | Reference | Paired cases | Mean primary difference | Mean FULL difference | Mean TEAM byte difference | Better / worse |','|---|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {c['budget']:,} | {c['arm']} | {c['reference']} | {c['pairs']} | {c['mean_primary_difference']:+.5f} | {c['mean_full_difference']:+.5f} | {c['mean_team_bytes_difference']:+.1f} | {c['better']}/{c['worse']} |" for c in comparisons if c['arm'] not in ('rotation','existing_selector')]
    lines+=['', 'At 8k, status-only starvation counts (no own payload) are: '+', '.join(f"{s['arm']} {s['status_no_payload']}/{s['agents']}" for s in summaries)+'. These are per-agent balances; no team ceiling replaces them.',
        '', '## Double-hazard timing at 4 MB, seed86100','', '| Arm | First-patch anchor | Second-patch anchor | Actual edge sequence |','|---|---:|---:|---|']
    for name in ('rotation','existing_selector','ack_only','compact_status_only','both'):
        d=next((d for d in diagnostics if d['scene']=='double_mild' and d['seed']==86100 and d['budget']==4000000 and d['selector']==name),None)
        if d:lines.append(f"| {name} | {d['patch_timing'][0]['first_received_anchor']} | {d['patch_timing'][1]['first_received_anchor']} | {[e['edge'] for e in d['actual_traversal']['interior_edge_entries']]} |")
    lines+=['', 'All local preview fits, failed admission/codec/scoring/fallback handler time are counted for the new variants; receiver-local work is attributed to the receiver. Baseline failed-offer counters remain unavailable, so complete episode wall time is the comparable total. Unit preflight work is preserved in test logs separately. No GPU is used.',
        '', 'These are four designed static scenes with two sensor-noise seeds, not sixteen independent terrains. Pair counts and budgets are not independent trials. No adaptive-rate advantage, calibrated information gain, inter-agent collision avoidance or broad superiority is claimed. Any favorable result is limited to the tested paired conditions. Full failures and errors remain in failure_inventory.json and errors/.',
        '', '[Cost/byte curves](figures/cost_bytes.png), [starvation and paired differences](figures/starvation_and_paired.png).']
    if (root/'animation_inventory.json').exists():
        for a in load(root/'animation_inventory.json'):lines.append(f"- [Double-hazard six-arm synchronized GIF, {a['budget']:,}/agent]({a['path']})")
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    dump(root/'reporting_verification.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),row_sha256=hashlib.sha256((root/'rows.json').read_bytes()).hexdigest(),baseline_checks=baseline_ok,link_checks=mechanism_ok))
    return verdict


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.out)
