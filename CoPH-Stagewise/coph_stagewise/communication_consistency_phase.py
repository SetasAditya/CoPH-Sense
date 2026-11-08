"""Separately frozen consistency phase after corrected-receiver fault audits.

The initial phase found per-case bias harm. This runner preserves its source
archive and records the diagnostic scalar-serialization correction separately.
"""
import argparse,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .communication_repair_suite import fault_job


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(source,out,workers=8):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);m=load(source/'manifest.json')
    baseline={p.name:sha(p) for p in sorted((source/'fault_rows').glob('*.json.gz'))};rows=[r for p in sorted((source/'fault_rows').glob('*.json.gz')) for r in load(p)]
    grouped={}
    def key(r):return (r['case'],r['seed'],r['group'],r.get('method'),r.get('index'),r['packet'],r['receiver'])
    for r in rows:grouped.setdefault(key(r),{})[(r['mode'],r['fault'])]=r
    assert len(rows)==2560 and len(grouped)==160,'initial phase incomplete'
    harm=sum(g[('corrected','biased')]['value_mae']>g[('corrected','clean')]['value_mae']+1e-12 for g in grouped.values() if g[('corrected','biased')]['value_mae'] is not None)
    if not harm:raise ValueError('no remaining corrected bias harm; phase not justified')
    import coph_stagewise.communication_repairs as repairs
    source_path=Path(__file__);runtime_path=Path(repairs.__file__)
    for name,expected in m['sources'].items():
        if name!='communication_repairs.py' and sha(runtime_path.parent/name)!=expected:raise ValueError('initial runtime dependency changed: '+name)
    protocol=dict(schema=2,initial_manifest=sha(source/'manifest.json'),baseline_rows=baseline,trigger_corrected_bias_harm=harm,
        threshold=4.,scale='sqrt(local reserved MSE + .003^2)',confidence='uncalibrated local residual heuristic',
        action='quarantine whole candidate before insertion; ordinary local update still runs',
        amendment='Python diagnostic scalar serialization only; all arithmetic, threshold and gates unchanged',
        initial_sources=m['sources'],phase_sources={source_path.name:sha(source_path),runtime_path.name:sha(runtime_path)},
        expected_updates=1280,expected_receiver_snapshots=160,input_hashes={e['path']:e['sha256'] for e in m['inputs']})
    p=out/'manifest.json'
    if p.exists():
        if load(p)!=protocol:raise ValueError('consistency phase source/input changed')
    else:
        with p.open('x') as f:json.dump(protocol,f,indent=2)
        (out/'source_archive').mkdir();(out/'source_archive'/source_path.name).write_bytes(source_path.read_bytes());(out/'source_archive'/runtime_path.name).write_bytes(runtime_path.read_bytes());(out/'fault_rows').mkdir()
    completion=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(fault_job,(str(out),e,True)) for e in m['inputs']]):
            completion.append(f.result());dump(out/'completion.json',completion);print(completion[-1],flush=True)
    gated=[r for p in sorted((out/'fault_rows').glob('*.json.gz')) for r in load(p)];assert len(gated)==1280
    confusion=[];summary=[]
    for fault in ('no_packet','clean','duplicate','echo','stale','reordered','biased','correlated'):
        rr=[r for r in gated if r['fault']==fault];delta=[];checked=quarantined=harmful_rejected=useful_rejected=unchanged_rejected=0
        for r in rr:
            base=grouped[key(r)];candidate=base[('corrected',fault)];withheld=base[('corrected','no_packet')]
            available=any(c['available'] for c in r['consistency']);rejected=any(e['reason']=='local_consistency_quarantine' for e in r['events']);checked+=available;quarantined+=rejected
            d=None if candidate['value_mae'] is None or withheld['value_mae'] is None else candidate['value_mae']-withheld['value_mae']
            label='unsupported' if d is None else 'harmful' if d>1e-12 else 'useful' if d< -1e-12 else 'unchanged'
            if rejected:
                harmful_rejected+=label=='harmful';useful_rejected+=label=='useful';unchanged_rejected+=label=='unchanged'
            if r['value_mae'] is not None and withheld['value_mae'] is not None:delta.append(r['value_mae']-withheld['value_mae'])
            confusion.append(dict(case=r['case'],seed=r['seed'],group=r['group'],packet=r['packet'],receiver=r['receiver'],fault=fault,
                checked=available,quarantined=rejected,committed_candidate_label=label,ungated_delta=d,
                gated_value_mae=r['value_mae'],ungated_value_mae=candidate['value_mae'],withhold_value_mae=withheld['value_mae']))
        summary.append(dict(fault=fault,audits=len(rr),checked=checked,quarantined=quarantined,accepted=sum(r['accepted'] for r in rr),
            harmful_rejected=harmful_rejected,useful_rejected=useful_rejected,unchanged_rejected=unchanged_rejected,
            mean_delta_vs_withhold=float(np.mean(delta)) if delta else None))
    s=dict(updates=len(gated),receiver_snapshots=len(grouped),all_isolated=all(r['isolated'] for r in gated),rows=summary,complete=True,
           prior_aborted_diagnostic_updates=1280,claim='saved-state robustness heuristic only; no new navigation trials')
    dump(out/'summary.json',s);dump(out/'validator_confusion_rows.json',confusion)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4));names=[r['fault'] for r in summary];x=np.arange(len(names))
    axes[0].bar(x,[r['mean_delta_vs_withhold'] for r in summary]);axes[0].axhline(0,color='black',lw=.8);axes[0].set_ylabel('Gated committed MAE − withhold MAE')
    axes[1].bar(x,[r['harmful_rejected'] for r in summary],label='harmful candidate rejected');axes[1].bar(x,[r['useful_rejected'] for r in summary],bottom=[r['harmful_rejected'] for r in summary],label='useful candidate rejected')
    for ax in axes:ax.set_xticks(x,names,rotation=25)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'consistency_tradeoff.png');plt.close(fig)
    lines=['# Local consistency phase','', 'This phase follows the initial 2560-update receiver comparison; it did not replace those results. The fixed threshold was not tuned against evaluator truth. The first serialization attempt is preserved under ../aborted_serialization_v1; its 1280 computed diagnostic updates were discarded because JSON output was incomplete.',
        '', '| Fault | Audits | Checked | Quarantined | Accepted models | Mean MAE delta vs withhold | Harmful rejected | Useful rejected |','|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['fault']} | {r['audits']} | {r['checked']} | {r['quarantined']} | {r['accepted']} | {r['mean_delta_vs_withhold']} | {r['harmful_rejected']} | {r['useful_rejected']} |" for r in summary]
    lines+=['', 'The heuristic checks only locally supported anchors; unchecked remote-only information remains unchecked. False quarantines of useful clean updates are retained. These are paired saved-state estimation outcomes, not mission-level navigation or calibrated fault-detection guarantees.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return s


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();run(a.source,a.out,a.workers)
