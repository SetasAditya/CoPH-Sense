"""Certified two-route choices from the packet's own covariance (companion: field error -> decision regret).

Same matched evidence and path pairs as reconstruction_decision_suite (same RNG seeds). Each packet's
declared covariance (the field's bootstrap ensemble plus floor, exactly as matched_payloads builds it;
raw/costmap keep the value block and mark gradients unknown, which their scorer does not use) is
sampled 400 times; anchors are perturbed and both path costs re-priced with the production scorer.
A choice is certified when the same route wins in >=95% of samples.

Pre-declared endpoints: (a) wrong-choice rate among certified decisions (should be <=5%), (b) share of
decisions certified, (c) calibration: share of pairs whose true cost difference lies in the sampled 90%
interval. Saved-state, 12 groups; not navigation trials.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from .communication_repair_suite import inputs
from .receiver_corrections import load_saved_snapshot
from .terrain_warning import terrain_values
from .cooperative import bootstrap_field,field_vector
from .reconstruction_decision_suite import path,cost,PAIRS
from .evaluation import dump

SAMPLES=400


def packets(entry):
    obj=load_saved_snapshot(entry['path']);out=[]
    for sender in (0,1):
        s=copy.deepcopy(obj['agents'][sender]);x=obj['states'][sender]
        packet=s.export(x,str(1-sender));readings={r.q:r for st in s.stages for r in st.readings};points=[tuple(v[:2]) for v in packet.anchors]
        if not all(q in readings for q in points):continue
        batch=[readings[q] for q in points];stage=copy.deepcopy(next(st for st in s.stages if st.id==packet.stage));stage.readings=batch
        mean,ensemble,_=bootstrap_field(stage,s.cfg,s.communication,s.now,seed=int(entry['meta']['seed']*2+sender))
        vector=field_vector(mean,stage.centers,points,s.cfg.rbf_width).reshape(-1,3)
        vectors=np.array([field_vector(w,stage.centers,points,s.cfg.rbf_width) for w in ensemble])
        cov=np.cov(vectors,rowvar=False)+np.eye(3*len(points))*s.communication.covariance_floor
        field=np.array([tuple(q)+tuple(v) for q,v in zip(points,vector)])
        raw=field.copy();raw[:,2]=[readings[q].value for q in points];raw[:,3:]=0
        costmap=field.copy();costmap[:,3:]=0
        out.append(dict(group=f"{entry['meta']['case']}_{entry['meta']['seed']}_{sender}",world=obj['worlds'][sender],cov=cov,
                        reps=dict(field=(field,True),raw=(raw,False),costmap=(costmap,False))))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);torch.set_num_threads(1)
    groups=[g for e in inputs() if e['meta']['group']=='acquisition' for g in packets(e)];rows=[]
    for gi,g in enumerate(groups):
        rng=np.random.RandomState(187000+gi);noise=np.random.RandomState(188000+gi);n=len(g['reps']['field'][0])
        for k in range(PAIRS):
            P=[path(rng,g['reps']['field'][0]) for _ in range(2)]
            true=[cost(p,lambda q:g['world'].risk_truth(q)) for p in P];dtrue=true[0]-true[1];row=dict(group=g['group'],pair=k,dtrue=dtrue)
            for name,(A,grad) in g['reps'].items():
                C=g['cov'] if grad else g['cov'][np.ix_(range(0,3*n,3),range(0,3*n,3))]
                z=noise.multivariate_normal(np.zeros(len(C)),C,size=SAMPLES)
                d=[]
                for zz in z:
                    B=A.copy()
                    if grad:B[:,2:]+=zz.reshape(n,3)
                    else:B[:,2]+=zz
                    c=[cost(p,lambda q:terrain_values(q,list(map(tuple,B)),unknown=.02)) for p in P];d.append(c[0]-c[1])
                d=np.array(d);est=[cost(p,lambda q:terrain_values(q,list(map(tuple,A)),unknown=.02)) for p in P];pick=int(np.argmin(est))
                share=float((np.sign(d)==np.sign(est[0]-est[1])).mean())
                row[name]=dict(certified=bool(share>=.95),wrong=bool(true[pick]>min(true)+1e-12),regret=float(true[pick]-min(true)),
                    inside90=bool(np.percentile(d,5)<=dtrue<=np.percentile(d,95)))
            rows.append(row)
    dump(out/'rows.json',rows);res={}
    for name in ('field','raw','costmap'):
        cert=[r[name] for r in rows if r[name]['certified']];unc=[r[name] for r in rows if not r[name]['certified']]
        res[name]=dict(certified_share=len(cert)/len(rows),wrong_when_certified=float(np.mean([c['wrong'] for c in cert])) if cert else None,
            wrong_when_uncertain=float(np.mean([c['wrong'] for c in unc])) if unc else None,
            regret_certified=float(np.mean([c['regret'] for c in cert])) if cert else None,regret_uncertain=float(np.mean([c['regret'] for c in unc])) if unc else None,
            calibration90=float(np.mean([r[name]['inside90'] for r in rows])),overall_wrong=float(np.mean([r[name]['wrong'] for r in rows])))
    dump(out/'summary.json',res);print(json.dumps(res,indent=1))


if __name__=='__main__':main()
