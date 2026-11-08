"""Tables and figures for solver_diagnosis (evaluator-only)."""
from collections import Counter
from pathlib import Path
import numpy as np
from .evaluation import load,dump

GAP=1e-8


def summarize(root):
    root=Path(root);audits=load(root/'audits.json.gz');probs=[(a,p) for a in audits for p in a['problems']]
    fail=[(a,p) for a,p in probs if p['status'] not in ('converged',)]
    checks=dict(audits=len(audits),problems=len(probs),replay_exact=sum(p['replay_exact'] for _,p in probs),trace_exact=sum(p['trace_exact'] for _,p in probs),
                audits_by_scope=dict(Counter(a['scope'] for a in audits)))
    taxonomy=dict(status=dict(Counter(p['status'] for _,p in fail)),category=dict(Counter(p['category'] for _,p in fail)),
        by_status_category={f"{s}/{c}":n for (s,c),n in Counter((p['status'],p['category']) for _,p in fail).items()},
        fallback_or_clip_any=sum(any(s['fallback'] or s['clipped_components'] for s in p['steps']) for _,p in fail),
        median_ratio_failed=float(np.median([s['ratio'] for _,p in fail for s in p['steps'] if s['ratio'] is not None])) if fail else None,
        median_ratio_converged=float(np.median([s['ratio'] for _,p in probs if p['status']=='converged' for s in p['steps'] if s['ratio'] is not None])),
        indefinite_exact_hessian=sum(any(s['min_hessian_eig']<0 for s in p['steps']) for _,p in fail),
        reference_active_bounds_failed=float(np.mean([p['reference']['active']>0 for _,p in fail])) if fail else None,
        reference_active_bounds_converged=float(np.mean([p['reference']['active']>0 for _,p in probs if p['status']=='converged'])))
    ref=dict(reference_meets_tolerance=sum(p['reference']['stationarity']<=1e-6 for _,p in probs),distinct_local_minima_problems=sum(p['reference']['distinct_local_minima']>0 for _,p in probs),
        trust_constr_agrees=sum(abs(p['reference']['trust_constr_objective']-p['reference']['objective'])<=1e-8*max(1,abs(p['reference']['objective'])) or p['reference']['trust_constr_objective']>=p['reference']['objective'] for _,p in probs),
        converged_original_at_reference=sum(p['gap_to_reference']<=GAP for _,p in probs if p['status']=='converged'),converged_original=sum(p['status']=='converged' for _,p in probs),
        failed_gap_median=float(np.median([p['gap_to_reference'] for _,p in fail])) if fail else None,failed_at_reference=sum(p['gap_to_reference']<=GAP for _,p in fail))
    # Route-query error: does a reliable solve help, and what did rollbacks do?
    failed_audits=[a for a in audits if any(p['status']!='converged' for p in a['problems'])]
    def cmp(rows,key):
        d=np.array([r[key]-r['no_fit_error'] for r in rows if r[key] is not None and r['no_fit_error'] is not None]);return dict(n=len(d),improves=int(np.sum(d<-1e-12)),worsens=int(np.sum(d>1e-12)),mean=float(d.mean()) if len(d) else None)
    quality=dict(all_audits={k:cmp(audits,k) for k in ('proposal_error','reference_error','committed_error')},
                 failed_audits={k:cmp(failed_audits,k) for k in ('proposal_error','reference_error','committed_error')},n_failed_audits=len(failed_audits))
    cands={}
    for name in audits[0]['candidate_runs']:
        row=dict(gated_accepted=sum(a['candidate_runs'][name]['accepted'] for a in audits),gated_accepted_failed_audits=sum(a['candidate_runs'][name]['accepted'] for a in failed_audits))
        errs=[dict(no_fit_error=a['no_fit_error'],e=a['candidate_runs'][name]['committed_error']) for a in audits];row['committed_vs_nofit']=cmp(errs,'e')
        if name!='reference':
            c=[p['candidates'][name] for _,p in probs];cf=[p['candidates'][name] for _,p in fail]
            row.update(converged=sum(x['status']=='converged' for x in c),converged_on_original_failures=sum(x['status']=='converged' for x in cf),
                       at_reference=sum(x['gap']<=GAP for x in c),median_iterations=float(np.median([x['iterations'] for x in c])),status=dict(Counter(x['status'] for x in c)),
                       near_tolerance=sum(5e-7<x['stationarity']<2e-6 for x in c))
        cands[name]=row
    original=dict(gated_accepted=sum(a['accepted'] for a in audits),committed_vs_nofit=cmp(audits,'committed_error'),converged=sum(p['status']=='converged' for _,p in probs),
                  near_tolerance=sum(5e-7<p['stationarity']<2e-6 for _,p in probs))
    out=dict(checks=checks,taxonomy=taxonomy,reference=ref,quality=quality,original=original,candidates=cands)
    for k,v in out.items():dump(root/f'{k}.json',v)
    figures(root,probs,fail)
    return out


def figures(root,probs,fail):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    (root/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    for (a,p) in [x for x in probs if x[1]['status']=='converged'][:30]:
        st=[s['stationarity'] for s in p['steps']];ax[0].semilogy(range(len(st)),st,color='#0072B2',alpha=.4,lw=.8)
    for (a,p) in fail[:60]:
        st=[s['stationarity'] for s in p['steps']];ax[0].semilogy(range(len(st)),st,color='#C62828',alpha=.4,lw=.8)
    ax[0].axhline(1e-6,color='k',ls='--',lw=.7);ax[0].set(xlabel='iteration',ylabel='stationarity',title='original solver: converged (blue) vs failed (red)')
    rf=[s['ratio'] for _,p in fail for s in p['steps'] if s['ratio'] is not None];rc=[s['ratio'] for _,p in probs if p['status']=='converged' for s in p['steps'] if s['ratio'] is not None]
    bins=np.linspace(-1,1.5,41);ax[1].hist(np.clip(rc,-1,1.5),bins,alpha=.6,color='#0072B2',label='converged solves');ax[1].hist(np.clip(rf,-1,1.5),bins,alpha=.6,color='#C62828',label='failed solves')
    ax[1].set(xlabel='actual / Gauss–Newton-predicted decrease (unit step)',ylabel='steps');ax[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'figures/traces_and_ratios.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.2));bins=np.logspace(-16,2,37)
    ax.hist(np.clip([p['gap_to_reference'] for _,p in fail],1e-16,1e2),bins,alpha=.6,color='#C62828',label='original, failed solves')
    names=list(fail[0][1]['candidates']) if fail else []
    for name,color in zip(names,('#E69F00','#009E73','#8E24AA')):
        ax.hist(np.clip([max(p['candidates'][name]['gap'],1e-16) for _,p in fail],1e-16,1e2),bins,histtype='step',lw=1.5,color=color,label=name.replace('_',' '))
    ax.set_xscale('log');ax.set(xlabel='objective gap to certified reference',ylabel='problems (originally failed)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'figures/gap_to_reference.png',dpi=150);plt.close(fig)
