"""Read-only artifact verification and pairwise matched-state feedback analysis."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .receiver_feedback_evaluation import HORIZONS,case_key

PAIRS=(('compact_fixed','receiver_fixed'),('compact_boundary','receiver_boundary'),
       ('compact_fixed','compact_boundary'),('receiver_fixed','receiver_boundary'))

def analyze(root):
    rows=load(root/'rows.json');paired=[];checks=0
    for h in HORIZONS:
        for left,right in PAIRS:
            a={case_key(r):r for r in rows if r['variant']==left and r['horizon']==h}
            b={case_key(r):r for r in rows if r['variant']==right and r['horizon']==h}
            matched=[k for k in a.keys() & b.keys() if a[k]['valid'] and b[k]['valid'] and np.allclose(a[k]['state'],b[k]['state'],atol=1e-12,rtol=0)]
            paired.append(dict(horizon=h,left=left,right=right,common=len(matched),
                left_mae=float(np.mean([abs(a[k]['error']) for k in matched])) if matched else None,
                right_mae=float(np.mean([abs(b[k]['error']) for k in matched])) if matched else None,
                keys=[list(k) for k in sorted(matched)]))
    completion=load(root/'completion.json')
    counts=dict(episodes=0,feedback_opportunities=0,draws=0,initial_exact_acceptance_checks=0,updates=0,max_update_transitions=0)
    for item in completion:
        if not item['complete']:continue
        ep=load(root/item['path']);counts['episodes']+=1
        seen=set()
        for f in ep['feedback_records']:
            counts['feedback_opportunities']+=1
            key=(f['receiver'],f['stage']);assert key not in seen;seen.add(key)
            events=[e for e in ep['packet_events'][f['receiver']] if e['packet']==f['packet']]
            assert len(events)==1 and events[0]['reason']=='eligible'
            truth=f['truth_audit'];assert truth['start']==f['state']
            branch_updates=[]
            for b in truth['branches'].values():
                assert b['path'][0]==f['state'];branch_updates+=b['updates']
                for cp in b['checkpoints'].values():
                    if cp['valid']:assert abs(cp['cost']-sum(cp['components'].values()))<1e-9
            assessment=f['assessment']
            if assessment:
                if f['mode']=='receiver':
                    assert f['feedback_bytes']==(assessment['wire_bytes'] if f['feedback_sent'] else 0)
                assert len(assessment['draws'])==4
                for d in assessment['draws']:
                    counts['draws']+=1
                    if not d['branches']:continue
                    assert d['branches']['deliver']['environment_seed']==d['branches']['withhold']['environment_seed']==d['seed']
                    for name,b in d['branches'].items():
                        assert b['path'][0]==f['state'];branch_updates+=b['updates']
                        assert b['observation_provenance']=='hypothetical'
                        if f['mode']=='receiver':
                            t=truth['branches'][name]
                            for key in ('accepted','reason','field_validation_after','plant_validation_after','policy'):
                                assert b['initial_update'][key]==t['initial_update'][key],(item['path'],f['packet'],key)
                            np.testing.assert_array_equal(b['initial_control'],t['initial_control'])
                            counts['initial_exact_acceptance_checks']+=1
                    if f['mode']=='receiver':
                        b=d['branches']['deliver']['initial_update']
                        assert b['accepted']==f['actual_accepted'] and b['version_after']==f['committed_version']
                for row in assessment['rows']:
                    assert 0<=row['valid_draws']<=4
                    assert (row['benefit'] is not None)==bool(row['valid_draws'])
            for u in branch_updates:
                work=u['diagnostic']['counterfactual_steps'];assert work<=1024
                counts['updates']+=1;counts['max_update_transitions']=max(counts['max_update_transitions'],work)
        for us in ep['updates']:
            for u in us:
                d=u['diagnostic'];assert d['counterfactual_steps']<=1024
                assert d['counterfactual_steps']==d.get('replay_pilot_steps',0)+sum(c['steps'] for c in u['replay_calls'])
                for p in d.get('replay_plan',[]):assert 8<=p['horizon']<=32
                counts['updates']+=1;counts['max_update_transitions']=max(counts['max_update_transitions'],d['counterfactual_steps'])
        for a in ep['agents']:
            assert a['bytes']==a['status_bytes']+a['payload_bytes']+a['ack_bytes']
            assert a['bytes']<=ep['communication']['byte_budget']
    dump(root/'pairwise_common.json',paired);dump(root/'contract_verification.json',dict(passed=True,**counts))
    fmt=lambda v:'missing' if v is None else f'{v:.6g}'
    text=['# Matched-state pairwise comparisons','',
          'Each pair requires the same scenario, seed, receiver, stage, packet identity and physical state (absolute tolerance 1e-12). Each prediction is compared against its own true-world receiver-learning audit; retained models/history may differ between replay arms. These are development comparisons. A zero denominator remains missing.',
          '', '| Horizon | Left | Right | Common n | Left MAE | Right MAE |','|---|---|---|---|---|---|']
    for r in paired:text.append(f"| {r['horizon']} | {r['left']} | {r['right']} | {r['common']} | {fmt(r['left_mae'])} | {fmt(r['right_mae'])} |")
    text+=['',f"Artifact contracts passed for {counts['episodes']} episodes, {counts['feedback_opportunities']} opportunities and {counts['updates']} operational/audit/prediction updates. Maximum replay work per update: {counts['max_update_transitions']} transitions.",
           'Receiver initial model, force, validation residuals and decisions equal the true-world branch initial updates; shared environment seeds, exactly-once packet events, operational commitment, byte sums, frozen horizon bounds, and pilot/replay work accounting are checked.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for variant in ('receiver_fixed','receiver_boundary'):
        rs=[r for r in rows if r['variant']==variant and r['horizon']==32 and r['valid']]
        xs=[r for r in rs if r['dispersion'] is not None]
        axes[0].scatter([r['dispersion'] for r in xs],[abs(r['error']) for r in xs],s=14,alpha=.5,label=variant)
        xs=[r for r in rs if r['goal_timing_error'] is not None]
        axes[1].scatter([r['goal_timing_error'] for r in xs],[abs(r['error']) for r in xs],s=14,alpha=.5,label=variant)
    axes[0].set(xlabel='Draw dispersion (not confidence)',ylabel='Absolute benefit error',title='Supported 32-step predictions')
    axes[1].set(xlabel='Mean absolute goal-timing error (steps)',ylabel='Absolute benefit error')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'error_dispersion.png',dpi=150);plt.close(fig)
    episodes=load(root/'episodes.json');fig,axes=plt.subplots(1,2,figsize=(12,4))
    for variant in ('receiver_fixed','receiver_boundary'):
        horizons=[h for e in episodes if e['variant']==variant for h in e['selected_horizons']]
        axes[0].hist(horizons,bins=np.arange(7.5,33.5,1),alpha=.5,label=variant)
    reasons=Counter(reason for e in episodes if e['variant']=='receiver_boundary' for reason in e['pilot_reasons'])
    axes[1].bar(range(len(reasons)),list(reasons.values()));axes[1].set_xticks(range(len(reasons)),list(reasons),rotation=25,fontsize=7)
    axes[0].set(xlabel='Frozen replay horizon',ylabel='Entry plans');axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Boundary pilot outcomes / fallbacks');fig.tight_layout();fig.savefig(root/'replay_horizons.png',dpi=150);plt.close(fig)
    text+=['','![Prediction error and timing](error_dispersion.png)','','![Frozen horizons and pilot outcomes](replay_horizons.png)']
    (root/'PAIRWISE.md').write_text('\n'.join(text)+'\n')
    return counts

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True,type=Path);args=ap.parse_args(argv)
    outcome={}
    for p in ('smoke','development'):outcome[p]=analyze(args.root/p)
    manifest=load(args.root/'manifest.json');package=Path(__file__).resolve().parents[1]
    differences=[name for name,digest in manifest['source_hashes'].items() if hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest]
    assert not differences, differences
    # This postprocessing module is separate from the frozen simulator source archive.
    source=Path(__file__);(args.root/'receiver_feedback_audit.py').write_bytes(source.read_bytes())
    dump(args.root/'artifact_verification.json',dict(profiles=outcome,frozen_source_hashes_match=True,
        analysis_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
