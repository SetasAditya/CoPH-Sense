"""Evaluator-only fault quality and isolated ordering-guard ablations.

No operational module is modified. The disabled-order branch clears both the
field sequence watermark and route-summary timestamp guard before each receipt.
"""
import copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from . import campaign_communication as c
from .evaluation import dump,load
from .model import risk,tensor
from .attribution_audits import digest


def evaluate(root):
    root=Path(root);manifest=load(root/'manifest.json');source=Path(__file__).read_bytes()
    protocol=dict(input_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source).hexdigest(),seed_cases=manifest['protocol']['seeds'],
        comparison='same payload sequence, complete receiver snapshot, normal versus clear sender watermark and route-summary timestamp guard before each receipt',
        metrics='same locally supported .1m query lattice; truth evaluator only; query counts not independent trials')
    p=root/'ordering_ablation_protocol.json'
    if p.exists():
        if load(p)!=protocol:raise ValueError('Ordering audit protocol changed')
    else:
        with p.open('x') as f:json.dump(protocol,f,indent=2)
    (root/'ordering_ablation_source.py').write_bytes(source)
    originals=load(root/'audit_rows.json');quality=[];ablations=[];torch.set_num_threads(1)
    for case in manifest['protocol']['cases']:
        for seed in manifest['protocol']['seeds']:
            spec=c.complementary.specification(case,seed)
            ctx=c.complementary.acquire(spec,c.configuration(500),c.CommunicationConfig(byte_budget=4000000,ttl=300.))
            for sender in (0,1):
                receiver=1-sender;a=ctx['agents'][receiver];x=ctx['states'][receiver];world=ctx['worlds'][receiver]
                exporter=copy.deepcopy(ctx['agents'][sender]);packet=exporter.export(ctx['states'][sender],str(receiver))
                before=digest(a);s=a.stages[-1]
                pts=np.array([(xx,yy) for xx in np.arange(s.center[0]-1.5,s.center[0]+1.5001,.1) for yy in np.arange(s.center[1]-2,s.center[1]+2.0001,.1) if s.supported((xx,yy),a.cfg)])
                truth=world.risk_truth(pts);gradtruth=world.risk_gradient(pts)
                def error(weights):
                    q=tensor(pts).requires_grad_(True);v=risk(q,tensor(weights[-1]),s.centers,a.cfg.rbf_width);g=torch.autograd.grad(v.sum(),q)[0]
                    return dict(value_mae=float(np.abs(v.detach().numpy()-truth).mean()),gradient_mae=float(np.linalg.norm(g.detach().numpy()-gradtruth,axis=1).mean()))
                local=error([st.weights.tolist() for st in a.stages])
                for row in originals:
                    if row.get('case')==case and row.get('seed')==seed and row.get('sender')==sender and 'fault' in row:
                        quality.append(dict(case=case,seed=seed,sender=sender,fault=row['fault'],queries=len(pts),
                            before=local,proposal=error(row['proposed_weights']),committed=error(row['weights']),accepted=row['accepted'],reason=row['reason'],control=row['control']))
                for fault in ('duplicate','reordered'):
                    packets=c.fault_variants(packet,a)[fault]
                    for disabled in (False,True):
                        branch=copy.deepcopy(a);events=[];started=time.perf_counter()
                        for m in packets:
                            if disabled:
                                branch.last_sequences.pop(m.sender,None)
                                branch.remote_terrain.pop((m.sender,m.stage),None)
                            events.append(branch.assimilate(m))
                        d=branch.update(x.copy());u,_=branch.action(x.copy())
                        ablations.append(dict(case=case,seed=seed,sender=sender,fault=fault,ordering_disabled=disabled,
                            events=events,accepted=d['accepted'],reason=d['reason'],version=branch.version,
                            field_store_count=len(branch.received),route_store_count=len(branch.remote_terrain),
                            sources={k:list(m.sources) for k,m in branch.received.items()},
                            error=error([st.weights.tolist() for st in branch.stages]),control=u.tolist(),
                            operational_unchanged=digest(a)==before,physical_state_unchanged=np.array_equal(x,ctx['states'][receiver]),
                            measured_unchanged=digest([(st.readings,st.transitions) for st in branch.stages])==digest([(st.readings,st.transitions) for st in a.stages]),
                            work=d['counterfactual_steps'],seconds=time.perf_counter()-started))
                        assert digest(a)==before
    dump(root/'quality_rows.json',quality);dump(root/'ordering_ablation_rows.json',ablations)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=manifest['protocol']['faults'];fig,ax=plt.subplots(figsize=(10,4))
    ax.boxplot([[r['committed']['value_mae']-r['before']['value_mae'] for r in quality if r['fault']==name] for name in names],tick_labels=names)
    ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Committed − pre-receipt value MAE');ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(root/'figures'/'fault_estimation_changes.png');plt.close(fig)
    summaries=[]
    for name in names:
        rows=[r for r in quality if r['fault']==name]
        summaries.append(dict(fault=name,audits=len(rows),mean_committed_mae_change=float(np.mean([r['committed']['value_mae']-r['before']['value_mae'] for r in rows])),accepted=sum(r['accepted'] for r in rows),queries=sum(r['queries'] for r in rows)))
    text=['','## Estimation and isolated ordering ablation','',
        'Field errors use identical locally supported points in each complete receiver snapshot. Truth is used only here. Query counts are not independent trials.',
        '', '| Fault | Audits | Accepted models | Mean committed MAE change |','|---|---:|---:|---:|']
    text += [f"| {r['fault']} | {r['audits']} | {r['accepted']} | {r['mean_committed_mae_change']:+.6g} |" for r in summaries]
    text+=['','Ordering guard ablation: 48 isolated branch updates. Disabling order protection clears both sender watermarks and route-summary timestamps before each receipt; operational code is unchanged. Newer correlated summaries and duplicates replace a sender contribution rather than adding independent field observations. No age/provenance-disabled or mission-level fault arms were executed.']
    report=root/'REPORT.md';current=report.read_text().split('\n## Estimation and isolated ordering ablation')[0];report.write_text(current+'\n'.join(text)+'\n')
    summary=load(root/'summary.json');summary.update(quality=summaries,ordering_ablation_audits=len(ablations),
        ordering_ablation_isolation=all(r['operational_unchanged'] and r['physical_state_unchanged'] and r['measured_unchanged'] for r in ablations))
    dump(root/'summary.json',summary)
    return summary


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();print(evaluate(args.out))
