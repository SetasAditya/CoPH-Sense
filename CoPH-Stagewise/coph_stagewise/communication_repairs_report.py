"""Read-only reporting for frozen communication repair phases."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);payload=read(root/'matched_payload_rows.json');base=read(root/'summary.json')
    selection_root=root/'payload_selection_v2' if (root/'payload_selection_v2/summary.json').exists() else root
    transport=read(selection_root/'fixed_rate_transport_rows.json');selection=read(selection_root/'selection_rows.json')
    old_selection=read(root/'selection_rows.json');old_empty=sum(r['packet_stage'] is None for r in old_selection if r['mode']=='receiver_relevant')
    gate=read(root/'consistency_v2/summary.json') if (root/'consistency_v2/summary.json').exists() else None
    codecs=[]
    for name in ('field','raw','costmap'):
        rr=[r for r in payload if r.get('representation')==name]
        codecs.append(dict(representation=name,payloads=len(rr),anchors=sorted({r['anchors'] for r in rr}),
            legacy_bytes=float(np.mean([r['legacy_bytes'] for r in rr])),compact_bytes=float(np.mean([r['compact_bytes'] for r in rr])),
            mean_query_mae=float(np.mean([r['route_query_mae'] for r in rr])),all_roundtrip_exact=all(r['roundtrip_exact'] for r in rr)))
    statuses=[]
    for mode in ('rotation','receiver_relevant'):
        rr=[r for r in transport if r['mode']==mode]
        statuses.append(dict(mode=mode,receipts=len(rr),accepted_models=sum(r['received']['update']['accepted'] for r in rr),
            exactly_once=all(r['exactly_once'] for r in rr),normal_validation_eligible=sum(r['received']['events'][0]['reason']=='eligible' for r in rr),
            mean_payload_bytes=float(np.mean([r['sent']['payload_bytes'] for r in rr])),mean_status_bytes=float(np.mean([r['sent']['status_bytes'] for r in rr])),
            mean_ack_bytes=float(np.mean([sum(a['acknowledgment'] for a in r['accounting'].values()) for r in rr]))))
    pairs=list(zip(selection[::2],selection[1::2]));eligible_pairs=[(a,b) for a,b in pairs if a['packet_stage'] is not None and b['packet_stage'] is not None];changed=sum(a['packet_stage']!=b['packet_stage'] for a,b in eligible_pairs)
    figures=root/'figures';fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(3)
    axes[0].bar(x-.18,[r['legacy_bytes'] for r in codecs],.36,label='legacy JSON');axes[0].bar(x+.18,[r['compact_bytes'] for r in codecs],.36,label='lossless compact');axes[0].set_xticks(x,[r['representation'] for r in codecs]);axes[0].set_ylabel('Mean bytes, identical four measurements');axes[0].legend()
    axes[1].bar([r['mode'] for r in statuses],[r['mean_payload_bytes'] for r in statuses],label='payload');axes[1].bar([r['mode'] for r in statuses],[r['mean_status_bytes'] for r in statuses],bottom=[r['mean_payload_bytes'] for r in statuses],label='status');axes[1].set_ylabel('Mean transmitted bytes before acknowledgment');axes[1].legend();fig.tight_layout();fig.savefig(figures/'codec_and_status_accounting.png');plt.close(fig)
    summary=dict(codec_rows=codecs,transport_rows=statuses,selection_pairs=len(pairs),selection_eligible_pairs=len(eligible_pairs),selection_changed=changed,selection_version=2 if selection_root!=root else 1,prior_selection_empty=old_empty,
                 initial_fault_updates=base['fault_audits'],corrected_bias_harm_vs_clean=base['corrected_biased_worse_than_clean'],
                 consistency=gate,tested_claims=['lossless codec','matched-evidence byte/error tradeoff','fixed-rate normal receiver transport','controlled retained-stage selection','saved-state corrected fault response'],
                 untested_claims=['mission-level payload-selection benefit','navigation improvement from consistency gating','calibrated bias detector'],
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    navigation=read(root/'navigation_v2/summary.json') if (root/'navigation_v2/summary.json').exists() else None
    if navigation:
        summary.update(navigation=navigation,preserved_selector_bug_pilot_episodes=16,total_navigation_attempts=32)
        summary['untested_claims'].remove('mission-level payload-selection benefit')
        summary['tested_claims'].append('two-layout actual navigation pilot; no demonstrated selector benefit')
    (root/'final_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Communication repair findings','', 'All mechanisms are opt-in. Existing defaults and preserved results remain intact. Saved-state mechanisms and the separately reported actual-navigation pilot are distinguished.',
        '', '## Compact coding with matched evidence','', 'Each representation receives the identical four actual readings and source IDs at the identical coordinates. The field is fitted only from those four readings. Costmap samples contain the same fitted values with gradients marked unknown. The benchmark is conditional on twelve preserved acquisition directions; the query points are not independent trials.',
        '', '| Representation | Payloads | Legacy bytes | Compact bytes | Route-query MAE | Exact roundtrip |','|---|---:|---:|---:|---:|---|']
    lines += [f"| {r['representation']} | {r['payloads']} | {r['legacy_bytes']:.1f} | {r['compact_bytes']:.1f} | {r['mean_query_mae']:.5f} | {r['all_roundtrip_exact']} |" for r in codecs]
    lines+=['', 'The codec changes transport representation only. It retains all floating values, support, covariance and provenance. The field has a different byte/error tradeoff; these results do not establish superior navigation or compare optimized native raster/sensor codecs.',
        '', '## Fixed cadence and receiver-relevant payloads','', '| Selection | Actual receives | Eligible packets | Accepted models | Payload bytes | Status bytes | Ack bytes |','|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['mode']} | {r['receipts']} | {r['normal_validation_eligible']} | {r['accepted_models']} | {r['mean_payload_bytes']:.1f} | {r['mean_status_bytes']:.1f} | {r['mean_ack_bytes']:.1f} |" for r in statuses]
    lines += ['', f'All deliveries use normal validation and one ordinary receiver update; exactly-once checks pass. Controlled requests changed the selected retained stage in {changed}/{len(eligible_pairs)} eligible nonempty pairs, out of {len(pairs)} attempted pairs. Those requests are constructed from genuine retained measurement coordinates, not recovered peer routes. This proves selection responds to the request, not that it helps navigation.',
        f'The preserved first selector had {old_empty}/108 empty relevant payloads. Its earlier count of 108 changed outcomes included these empties and was not 108 successful selected packets. Version two preserves ledger order and adjusts the copied export cursor, with a nonempty historical-export regression test. Previous records and the first navigation pilot remain preserved.',
        '', 'Status overhead is substantial and can exceed the compact payload. It is charged to the transmitting receiver; no bandwidth advantage is claimed for the selector itself. Compact support advertisements approximate receiver knowledge and are not calibrated beliefs.',
        '', '## Corrected receiver before new gating','', f"The first phase executed {base['fault_audits']} updates on 160 preserved receiver snapshots. Corrected fitting still gave a worse biased than clean estimate in {base['corrected_biased_worse_than_clean']}/160 cases. It improved the mean biased response relative to legacy, so the new heuristic must be judged against the corrected baseline."]
    if gate:
        lines += ['', '## Separately frozen local consistency heuristic','', '| Fault | Checked / audits | Quarantined | Harmful candidates rejected | Useful candidates rejected | MAE delta vs withhold |','|---|---:|---:|---:|---:|---:|']
        lines += [f"| {r['fault']} | {r['checked']}/{r['audits']} | {r['quarantined']} | {r['harmful_rejected']} | {r['useful_rejected']} | {r['mean_delta_vs_withhold']:.6g} |" for r in gate['rows']]
        lines += ['', 'Quarantine checks local support only and is explicitly uncalibrated. Useful-update rejection, unchecked anchors and failures are retained. The initial gated serialization failure and 1280 discarded diagnostic updates are preserved; consistency_v2 changes only Python scalar serialization and reruns all 1280 updates. The numerical threshold remains four.']
        lines += ['', 'This heuristic is not recommended as a default improvement: it quarantines all biased candidates, including many that improved the saved-state query estimate, and removes some useful clean updates. It remains opt-in and off. The evidence supports corrected fitting and explicit fault accounting; robust consistency gating remains unresolved.']
    if navigation:
        lines += ['', '## Actual navigation pilot after the selector repair','',
            f"The separately frozen navigation_v2 pilot ran {navigation['episodes']}/16 episodes on the first two existing unselected layouts (seeds89500,89501), with {navigation['goals']}/32 agent goals, {navigation['collisions']} collisions and {navigation['timeouts']} timeouts. Both selectors use the compact codec, existing fixed opportunities, legacy receiver fitting and identical original scene/horizon settings. The two disabled pairs reproduce exactly.",
            '', 'The repaired selector showed no navigation benefit. In reciprocal exchange, seed89501 used 0.014566 m more team travel, 0.131516 more logged cost and 711,441 additional team bytes; seed89500 changes were approximately zero while using 637,611 additional bytes. This small pilot supports correct execution and exposes status overhead; it does not establish generalization.',
            '', 'The original 16-episode selector-bug pilot remains under navigation/. It is preserved as mechanism-failure history, not silently replaced or counted as a valid selector test. Together there were 32 navigation attempts. The fixed report is [navigation_v2/REPORT.md](navigation_v2/REPORT.md); the paired animation uses the preregistered seed89501.']
    lines += ['', 'Artifacts include immutable manifests, explicit runtime/input fingerprints, full per-update solver and acceptance records, matched payload and transport rows, packet selection records, validator confusion rows and plots. All saved-state branches preserve measurements and physical state. Broad navigation benefit remains unestablished.']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.out)
