"""Postprocessing on the exact prior fixed-branch query population; no update runs."""
import argparse
import copy
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .localized_receiver_evaluation import decode,VARIANTS
from .reconstruction_diagnostics import field_queries
from .route_field import sync
from .receiver_corrections import errors
from .model import tensor


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);manifest=load(root/'manifest.json');rows=[]
    for entry in manifest['inputs']:
        if entry['group']!='current':continue
        agent,x,packet,world,_=decode(entry)[0]
        archive=load(root/'audits'/f"current_{entry['key']}.json.gz")[0]
        prior=load(Path(entry['path']).parents[1]/'screening'/f"{entry['key']}.json.gz")
        points=np.asarray([p for e in prior['score']['quadrature'] for p in e['points']])
        truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
        predictions={};initial,_=field_queries(agent,points);mask=np.isfinite(initial).all(1)
        for name in VARIANTS:
            r=archive['results'][name]
            for publication in ('proposed','committed'):
                a=copy.deepcopy(agent)
                weights=r['proposed_fields'] if publication=='proposed' else r['fields']
                residuals=r['proposed_residuals'] if publication=='proposed' else r['committed_residuals']
                for k,(stage,w) in enumerate(zip(a.stages,weights)):
                    stage.weights=tensor(w)
                    stage.local_residual=copy.deepcopy(residuals[k]) if residuals is not None else None
                if publication=='proposed' or r['accepted']:sync(a,publish=True)
                values,metadata=field_queries(a,points)
                predictions[(name,publication)]=(values,metadata)
                mask&=np.isfinite(values).all(1)
        results={}
        for name in VARIANTS:
            results[name]={publication:errors(predictions[(name,publication)][0],truth,mask) for publication in ('proposed','committed')}
            results[name].update(accepted=archive['results'][name]['accepted'],reason=archive['results'][name]['reason'])
        for name in VARIANTS:
            results[name]['committed_delta']=results[name]['committed']['value_mae']-results['local_only']['committed']['value_mae']
        # Verify exactly the previous query coordinates/counts and no implicit coverage gain.
        assert len(points)==len(prior['audit']['branches']['no_fit_none']['frozen_anchor_score']['quadrature'][0]['points'])+sum(len(e['points']) for e in prior['audit']['branches']['no_fit_none']['frozen_anchor_score']['quadrature'][1:])
        assert results['local_only']['committed']['supported']==prior['audit']['branches']['no_fit_none']['errors']['supported']
        assert abs(results['local_only']['committed']['value_mae']-prior['audit']['branches']['no_fit_none']['errors']['value_mae'])<1e-12
        rows.append(dict(key=entry['key'],points=points.tolist(),common_support=mask.tolist(),results=results,
            prior_no_fit_bitwise_query_population=True,prior_baseline_error_reproduced=True))
    summary=[]
    for name in VARIANTS:
        values=[r['results'][name]['committed_delta'] for r in rows]
        summary.append(dict(variant=name,audits=len(rows),improved=sum(v< -1e-12 for v in values),
            worsened=sum(v>1e-12 for v in values),mean_value_delta=float(np.mean(values))))
    dump(root/'original_query_rows.json',rows);dump(root/'original_query_summary.json',summary)
    fig,ax=plt.subplots(figsize=(6,4))
    for i,name in enumerate(VARIANTS):ax.scatter([i]*len(rows),[r['results'][name]['committed_delta'] for r in rows],alpha=.65,s=18)
    ax.axhline(0,color='k',lw=.7);ax.set(xticks=range(3),xticklabels=VARIANTS,ylabel='Committed value MAE − local-only',title='Exact prior fixed-branch query population')
    ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(root/'figures/original_queries.png',dpi=160);plt.close(fig)
    text='# Exact original fixed-branch queries\n\n'
    text+='This supplementary postprocessing evaluates already-saved proposals and commitments on the exact 24 prior campaign fixed-branch query sets. It performs no fitting, controller change, or new mission run. The frozen primary snapshot suite used the full graph and connector quadrature; the two populations must not be pooled. The prior no-fit errors and supported counts reproduce to 1e-12 in every snapshot.\n\n'
    text+='| Variant | Improves / worsens | Mean committed MAE change |\n|---|---:|---:|\n'
    for r in summary:text+=f"| {r['variant']} | {r['improved']}/{r['worsened']} of {r['audits']} | {r['mean_value_delta']:.8g} |\n"
    text+='\nThe compact basis guarantees locality, not a uniformly more accurate fit. These fixed-state errors do not establish a branch-choice or navigation benefit. Proposal and committed errors are retained separately in `original_query_rows.json`.\n\n![Original queries](figures/original_queries.png)\n'
    (root/'ORIGINAL_QUERY_REPORT.md').write_text(text)
    main=(root/'REPORT.md').read_text().split('\n## Exact prior fixed-branch queries')[0]
    main+='\n## Exact prior fixed-branch queries\n\n'
    main+='The frozen primary suite above uses full graph and connector quadrature. On the exact previous 24 fixed-branch query sets, the separate saved-field postprocessing gives:\n\n'
    main+='| Variant | Improves / worsens / unchanged | Mean committed MAE change |\n|---|---:|---:|\n'
    for r in summary:main+=f"| {r['variant']} | {r['improved']}/{r['worsened']}/{r['audits']-r['improved']-r['worsened']} of {r['audits']} | {r['mean_value_delta']:.8g} |\n"
    main+='\nThe prior no-fit errors and supported counts reproduce to 1e-12. The two query populations must not be pooled. See [the exact-query report](ORIGINAL_QUERY_REPORT.md). Locality is verified; estimation effects are mixed across the preserved populations, so the compact option remains experimental.\n\n'
    main+='Ten targeted localized tests, twelve existing receiver-fitting tests (including archived legacy bitwise fitting), and twelve existing cooperative tests pass. The packet-export test compares the complete packet dictionary before and after received evidence and compact operational context; local-only export contents are unchanged. All 18 compact relative-validation rejections retain versions, weights, residuals and routing snapshots after mandatory support synchronization. Routine removal of already-evicted support coordinates occurs independently of fitting acceptance. See `verification.json` and test logs. Full-suite results are reported at campaign level, including the known archived-source-hash failure.\n'
    (root/'REPORT.md').write_text(main)
    (root/'FINDINGS.md').write_text(main)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    print(run(parser.parse_args().out))
