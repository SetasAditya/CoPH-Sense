"""Offline diagnosis of receiver reconstruction on the preserved complementary snapshots.

Evaluator-only. Snapshots are trusted local pickles written by complementary_reconstruction;
truth is read from the snapshot's private world copy and never reaches a learner that acts.
Every variant runs on deep copies: the operational learner, packet schedule, validator and
controller are unchanged. Variants are diagnoses, not proposed methods.
"""
import argparse,copy,gzip,json,pickle,types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from .model import tensor,features
from .cooperative import CooperativeLearner
from .route_field import hypothetical_context
from .reconstruction_diagnostics import ranking,field_queries
from .attribution_audits import digest
from .evaluation import dump
from .cli import source_hashes

SOURCE=Path('results/complementary_reconstruction_v1')
SENSOR_SIGMA=.003  # declared sensor noise standard deviation (World.sense)
BASE=dict(lam=1.,calibrated=False,rows='all',steps=1,clamp=True,truth_anchors=False)
VARIANTS={'no_fit':None,'default':{},**{f'lam_{l:g}':dict(lam=l) for l in (.03,.1,.3,3.,10.)},
          'calibrated':dict(calibrated=True),'value_only':dict(rows='value'),'gradient_only':dict(rows='gradient'),
          'converged':dict(steps=20),'no_clamp':dict(clamp=False),'truth_anchors':dict(truth_anchors=True),
          'truth_value_only':dict(truth_anchors=True,rows='value'),'calibrated_converged':dict(calibrated=True,steps=20)}


def _local(agent):
    """The receiver's local-evidence fit, exactly as CooperativeLearner calls it."""
    return super(CooperativeLearner,agent)._field_fit()


def _calibrated_information(agent,k,w):
    s=agent.stages[k];rs=[r for r in s.readings if int(r.uid.split(':')[-1])%5]
    if not rs:return None
    phi=features(tensor([r.q for r in rs]),s.centers,agent.cfg.rbf_width);slope=torch.sigmoid(phi@w)/4
    c=(slope/SENSOR_SIGMA)**2
    return (phi*c[:,None]).T@phi+agent.cfg.fit_ridge*torch.eye(len(w),dtype=torch.float64)


def message_step(agent,weights,cov,p,world=None,drop=None):
    """CooperativeLearner's received-field step with explicit, diagnosable knobs.

    With BASE settings this is the operational computation, line for line.
    `drop=(packet_identity, anchor_index)` omits one anchor (leave-one-anchor-out).
    """
    p={**BASE,**(p or {})};adjusted=[]
    for k,(s,w) in enumerate(zip(agent.stages,weights)):
        blocks=[]
        for m in agent.received.values():
            ids=[i for i,a in enumerate(m.anchors) if s.contains(a[:2],agent.cfg) and (m.identity,i)!=drop]
            if not ids:continue
            points=tensor([m.anchors[i][:2] for i in ids]);z=tensor([m.anchors[i][2:] for i in ids])
            if p['truth_anchors']:
                q=np.asarray([m.anchors[i][:2] for i in ids]);z=tensor(np.column_stack((world.risk_truth(q),world.risk_gradient(q))))
            indices=[3*i+j for i in ids for j in range(3)]
            C=tensor(np.asarray(m.covariance)[np.ix_(indices,indices)])+torch.eye(len(indices),dtype=torch.float64)*(.01+.005*(agent.now-m.timestamp))
            keep=[n for n in range(len(indices)) if p['rows']=='all' or (n%3==0)==(p['rows']=='value')]
            blocks.append((points,z.reshape(-1),C,keep))
        if not blocks or p['lam']==0:adjusted.append(w);continue
        def vector(v,points):
            f=features(points,s.centers,agent.cfg.rbf_width);raw=f@v;sigmoid=torch.sigmoid(raw)
            df=-(points[:,None,:]-s.centers)/agent.cfg.rbf_width**2*f[:,:,None]
            val=torch.nn.functional.softplus(raw)/4;grad=sigmoid[:,None]/4*torch.einsum('nmd,m->nd',df,v)
            return torch.cat((val[:,None],grad),1).reshape(-1)
        if p['calibrated']:local=_calibrated_information(agent,k,w)
        else:local=None
        if local is None:local=torch.linalg.inv(cov[k]) if cov[k] is not None else torch.eye(len(w),dtype=torch.float64)
        w0=w
        for step in range(p['steps']):
            rows=[];targets=[]
            for points,z,C,keep in blocks:
                J=torch.autograd.functional.jacobian(lambda v:vector(v,points),w);r=(z-vector(w,points))[:,None]
                if len(keep)<len(z):
                    kt=torch.tensor(keep);J=J[kt];r=r[kt];C=C[kt][:,kt]
                L=torch.linalg.cholesky(C)
                rows.append(torch.linalg.solve_triangular(L,J,upper=False));targets.append(torch.linalg.solve_triangular(L,r,upper=False).flatten())
            A=torch.cat(rows);y=torch.cat(targets)
            if p['lam']!=1:A=A*np.sqrt(p['lam']);y=y*np.sqrt(p['lam'])
            rhs=A.T@y if step==0 else A.T@y+local@(w0-w)
            delta=torch.linalg.solve(local+A.T@A,rhs)
            w=w+(delta.clamp(-.5,.5) if p['clamp'] else delta)
            if step and float(torch.linalg.vector_norm(delta))<1e-10:break
        adjusted.append(w)
    return adjusted


def _patched_fit(p,world):
    def fit(self):
        weights,cov=_local(self)
        if not self.received:return weights,cov
        adjusted=message_step(self,weights,cov,p,world);self.reference=[w.clone() for w in adjusted];return adjusted,cov
    return fit


def _distances(points,others):
    if not len(others):return np.full(len(points),np.inf)
    return np.min(np.linalg.norm(np.asarray(points)[:,None,:]-np.asarray(others)[None,:,:],axis=2),axis=1)


def audit(snapshot,meta):
    agent,x,packet,world,tick=snapshot;original=digest(agent)
    base=copy.deepcopy(agent);event=base.assimilate(packet)
    quad=ranking(base,x,details=True)['quadrature'];points=np.array([q for e in quad for q in e['points']])
    truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
    initial,_=field_queries(base,points)
    eligible=[(m.identity,i,a) for m in base.received.values() for i,a in enumerate(m.anchors) if any(s.contains(a[:2],base.cfg) for s in base.stages)]
    anchor_q=np.array([a[:2] for _,_,a in eligible]).reshape(-1,2)
    anchor_err=[]
    if len(anchor_q):
        t=np.column_stack((world.risk_truth(anchor_q),world.risk_gradient(anchor_q)));z=np.array([a[2:] for _,_,a in eligible])
        anchor_err=dict(value=np.abs(z[:,0]-t[:,0]).tolist(),gradient=np.linalg.norm(z[:,1:]-t[:,1:],axis=1).tolist(),truth_value=t[:,0].tolist(),anchor_value=z[:,0].tolist())
    heldout=[r.q for s in base.stages for r in s.readings if int(r.uid.split(':')[-1])%5==0]
    out=dict(meta,packet=packet.identity,receiver=int(agent.agent_id),time=float(agent.now),event=event['reason'],query_count=len(points),
             eligible_anchors=len(eligible),anchor_error=anchor_err,variants={})
    preds={}
    for name,p in VARIANTS.items():
        c=copy.deepcopy(base);c.fit_received=p is not None
        if p is not None:c._field_fit=types.MethodType(_patched_fit(p,world),c)
        d=c.update(x.copy());pred,_=field_queries(c,points);preds[name]=pred
        ctx=hypothetical_context(c);ctx['route_field_mode']='supported_field';rk=ranking(c,x,ctx,True)
        out['variants'][name]=dict(accepted=bool(d['accepted']),held_out=float(d['field_validation_after']),best=rk['best'],margin=rk['margin'])
    # The patched default must equal the operational fit exactly.
    c=copy.deepcopy(base);c.fit_received=True;c.update(x.copy());op,_=field_queries(c,points)
    out['default_matches_operational']=bool(np.array_equal(np.nan_to_num(op,nan=-9),np.nan_to_num(preds['default'],nan=-9)))
    common=np.isfinite(initial).all(1)&np.isfinite(preds['no_fit']).all(1)&np.isfinite(preds['default']).all(1)
    for name,pred in preds.items():
        ok=common&np.isfinite(pred).all(1);delta=pred[ok]-truth[ok]
        out['variants'][name].update(value_mae=float(np.abs(delta[:,0]).mean()) if ok.any() else None,
                                     gradient_mae=float(np.linalg.norm(delta[:,1:],axis=1).mean()) if ok.any() else None,common=int(ok.sum()))
    e_fit=np.abs(preds['default'][common,0]-truth[common,0]);e_no=np.abs(preds['no_fit'][common,0]-truth[common,0])
    out['queries']=dict(delta_value_error=(e_fit-e_no).tolist(),distance_to_anchor=_distances(points[common],anchor_q).tolist(),
                        distance_to_heldout=_distances(points[common],heldout).tolist(),truth=truth[common,0].tolist())
    # Runtime-available check: leave one received anchor out, predict it with and without the rest.
    weights,cov=_local(copy.deepcopy(base));loao=[]
    for ident,i,a in eligible:
        k=next(j for j,s in enumerate(base.stages) if s.contains(a[:2],base.cfg));s=base.stages[k]
        fitted=message_step(base,weights,cov,{},drop=(ident,i))
        from .cooperative import field_vector
        with_rest=field_vector(fitted[k],s.centers,[a[:2]],base.cfg.rbf_width)[0];without=field_vector(weights[k],s.centers,[a[:2]],base.cfg.rbf_width)[0]
        loao.append(float(abs(with_rest-a[2])-abs(without-a[2])))
    out['loao_delta']=loao
    assert digest(agent)==original
    return out


def _snapshot_jobs(source):
    rows=json.loads((source/'audit_rows.json').read_text());jobs=[]
    for path in sorted((source/'snapshots').glob('*.pkl.gz')):
        parts=path.name[:-7].split('_');case,seed=parts[0],int(parts[1])
        if parts[2]=='acquisition':jobs.append((str(path),dict(case=case,seed=seed,group='acquisition')))
        else:jobs.append((str(path),dict(case=case,seed=seed,method='_'.join(parts[2:-1]),group='navigation',index=int(parts[-1]))))
    return jobs,rows


def _run(job):
    torch.set_num_threads(1);path,meta=job
    with gzip.open(path,'rb') as f:value=pickle.load(f)  # trusted local evaluator snapshot
    if meta['group']=='navigation':return [audit(value,meta)]
    out=[];ctx=value;intended=dict(complementary=0,reverse=1,redundant=None)[meta['case']]
    for sender in (0,1):
        receiver=1-sender;exported=copy.deepcopy(ctx['agents'][sender]);packet=exported.export(ctx['states'][sender].copy(),str(receiver))
        if packet is None:continue
        snap=(ctx['agents'][receiver],ctx['states'][receiver],packet,ctx['worlds'][receiver],8)
        out.append(audit(snap,dict(meta,sender=sender,intended=sender==intended)))
    return out


def run(out,source=SOURCE,workers=40):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);jobs,archived=_snapshot_jobs(source)
    dump(out/'manifest.json',dict(source=str(source),snapshots=[j[0] for j in jobs],variants={k:({**BASE,**v} if v is not None else None) for k,v in VARIANTS.items()},
         sensor_sigma=SENSOR_SIGMA,source_hashes=source_hashes(),semantics='evaluator-only counterfactual fits on deep copies; no operational change'))
    with ProcessPoolExecutor(max_workers=workers) as pool:results=[a for batch in pool.map(_run,jobs) for a in batch]
    # Archived fit / no-fit errors must be reproduced before anything else is read.
    index={(r['case'],r['seed'],r['method'],r['packet'],r['receiver']):r for r in archived}
    for a in results:
        key=(a['case'],a['seed'],a.get('method','gate'),a['packet'],a['receiver']);r=index.get(key)
        a['archived_match']=None if r is None else bool(abs(r['errors']['fit']['value_mae']-a['variants']['default']['value_mae'])<1e-12 and abs(r['errors']['no_fit']['value_mae']-a['variants']['no_fit']['value_mae'])<1e-12)
    dump(out/'audits.json.gz',results);return results


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    ap.add_argument('--summarize',action='store_true');args=ap.parse_args(argv)
    if not args.summarize:run(args.out,workers=args.workers)
    from .reconstruction_quality_report import summarize
    summarize(args.out)


if __name__=='__main__':main()
