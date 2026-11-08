"""Tables, figures and DIAGNOSIS inputs for reconstruction_quality (evaluator-only)."""
import json
from pathlib import Path
import numpy as np
from .evaluation import load,dump


def groups(audits):
    acq=[a for a in audits if a['group']=='acquisition']
    return {'intended acquisition':[a for a in acq if a.get('intended')],'all acquisition':acq,'navigation':[a for a in audits if a['group']=='navigation']}


def table(audits):
    out={}
    for g,rows in groups(audits).items():
        if not rows:continue
        base=np.array([a['variants']['no_fit']['value_mae'] for a in rows]);t={}
        for v in rows[0]['variants']:
            val=np.array([a['variants'][v]['value_mae'] for a in rows]);grad=np.array([a['variants'][v]['gradient_mae'] for a in rows])
            gb=np.array([a['variants']['no_fit']['gradient_mae'] for a in rows]);held=np.array([a['variants'][v]['held_out']-a['variants']['no_fit']['held_out'] for a in rows])
            t[v]=dict(value_mae_mean=float(val.mean()),value_vs_nofit_mean=float((val-base).mean()),value_improves=int(np.sum(val<base-1e-12)),value_worsens=int(np.sum(val>base+1e-12)),
                      gradient_vs_nofit_mean=float((grad-gb).mean()),held_out_vs_nofit_mean=float(held.mean()),accepted=int(sum(a['variants'][v]['accepted'] for a in rows)),
                      route_changes=int(sum((a['variants'][v]['best'] or {}).get('nodes')!=(a['variants']['no_fit']['best'] or {}).get('nodes') for a in rows)),n=len(rows))
        out[g]=t
    return out


def summarize(root):
    root=Path(root);audits=load(root/'audits.json.gz');t=table(audits);dump(root/'variant_table.json',t)
    checks=dict(audits=len(audits),default_matches_operational=all(a['default_matches_operational'] for a in audits),
                archived_match=sum(a['archived_match'] is True for a in audits),archived_unmatched=sum(a['archived_match'] is False for a in audits),archived_missing=sum(a['archived_match'] is None for a in audits))
    dump(root/'checks.json',checks)
    # H0: anchor accuracy against truth.
    av=np.concatenate([a['anchor_error']['value'] for a in audits if a['anchor_error']]);dump(root/'anchor_quality.json',dict(value_mae=float(av.mean()),value_p90=float(np.percentile(av,90))))
    # H2/H4: per-query localization.
    dq=np.concatenate([a['queries']['delta_value_error'] for a in audits]);da=np.concatenate([a['queries']['distance_to_anchor'] for a in audits]);dh=np.concatenate([a['queries']['distance_to_heldout'] for a in audits])
    bins=[0,.3,.65,1.,1.5,2.5,np.inf];loc=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(da>=lo)&(da<hi)
        if m.any():loc.append(dict(anchor_distance=[lo,hi if np.isfinite(hi) else None],n=int(m.sum()),mean_delta=float(dq[m].mean()),worsens=float(np.mean(dq[m]>1e-12))))
    dump(root/'localization.json',loc)
    # H4: does local held-out validation (or leave-one-anchor-out) predict route-query change?
    nav=[a for a in audits];droute=np.array([a['variants']['default']['value_mae']-a['variants']['no_fit']['value_mae'] for a in nav])
    dheld=np.array([a['variants']['default']['held_out']-a['variants']['no_fit']['held_out'] for a in nav])
    dloao=np.array([np.mean(a['loao_delta']) if a['loao_delta'] else np.nan for a in nav])
    def agree(x):
        ok=np.isfinite(x)&(np.abs(droute)>1e-12)&(np.abs(x)>1e-12);return dict(n=int(ok.sum()),sign_agreement=float(np.mean(np.sign(x[ok])==np.sign(droute[ok]))) if ok.any() else None,
            spearman=float(np.corrcoef(np.argsort(np.argsort(x[ok])),np.argsort(np.argsort(droute[ok])))[0,1]) if ok.sum()>2 else None)
    dump(root/'validator.json',dict(held_out=agree(dheld),leave_one_anchor_out=agree(dloao)))
    margins=np.array([a['variants']['no_fit']['margin'] for a in audits if a['variants']['no_fit']['margin'] is not None])
    changes=np.array([max(abs((a['variants'][v]['margin'] or 0)-(a['variants']['no_fit']['margin'] or 0)) for v in a['variants']) for a in audits])
    dump(root/'decision_sensitivity.json',dict(margin_median=float(np.median(margins)),margin_min=float(margins.min()),max_margin_change_median=float(np.median(changes)),max_margin_change_max=float(changes.max())))
    figures(root,audits,dq,da,dh,droute,dheld,dloao)
    return t,checks


def figures(root,audits,dq,da,dh,droute,dheld,dloao):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    (root/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(9,3.2))
    for a_,d,label in ((ax[0],da,'distance to nearest received anchor (m)'),(ax[1],dh,'distance to nearest held-out local reading (m)')):
        a_.scatter(d,dq,s=3,alpha=.25,color='#555');a_.axhline(0,color='k',lw=.6);a_.set(xlabel=label,ylabel='Δ |value error| (fit − no fit)');a_.set_xlim(0,3)
    fig.tight_layout();fig.savefig(root/'figures/error_vs_distance.png',dpi=150);plt.close(fig)
    t=table(audits);fig,ax=plt.subplots(figsize=(6.5,3.2))
    for g,color in (('intended acquisition','#C62828'),('all acquisition','#E69F00'),('navigation','#0072B2')):
        if g not in t:continue
        lams=[0,.03,.1,.3,1,3,10];keys=['no_fit','lam_0.03','lam_0.1','lam_0.3','default','lam_3','lam_10']
        ax.plot(lams[1:],[t[g][k]['value_vs_nofit_mean'] for k in keys[1:]],'-o',color=color,label=g,ms=3)
        for k,m in (('calibrated','s'),('truth_anchors','*'),('value_only','v'),('converged','D')):
            ax.scatter(14 if k=='calibrated' else 18 if k=='truth_anchors' else 22 if k=='value_only' else 26,t[g][k]['value_vs_nofit_mean'],marker=m,color=color,s=25)
    ax.set_xscale('symlog',linthresh=.03);ax.axhline(0,color='k',lw=.6);ax.set(xlabel='message information scale λ  |  right: calibrated ■, truth anchors ★, value-only ▼, converged ◆',ylabel='mean Δ route value MAE vs no fit');ax.xaxis.label.set_size(7)
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/'figures/lambda_sweep.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(8,3.2))
    ax[0].scatter(dheld,droute,s=8,color='#555');ax[0].set(xlabel='Δ held-out local error (fit − no fit)',ylabel='Δ route value MAE')
    ax[1].scatter(dloao,droute,s=8,color='#555');ax[1].set(xlabel='Δ leave-one-anchor-out error',ylabel='Δ route value MAE')
    for a_ in ax:a_.axhline(0,color='k',lw=.6);a_.axvline(0,color='k',lw=.6)
    fig.tight_layout();fig.savefig(root/'figures/validator_agreement.png',dpi=150);plt.close(fig)
