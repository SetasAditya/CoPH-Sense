"""Decision-level test of field versus raw versus costmap packets built from identical evidence.

Reuses communication_repair_suite's matched-evidence construction exactly: the same actual
measurement batch, coordinates and source IDs; the field is refitted from that batch only.
For each packet we sample random pairs of straight candidate paths that lie entirely inside
the packet's .65 m support, price each with the production anchor scorer (terrain_values)
and the risk-averse carrier weight w=6, and choose the cheaper. Truth comes from the
simulator world only. Strata: paths through the anchors (offset 0) versus between/around them.

Pre-declared gate: field mean regret lower than both raw and costmap with a group-bootstrap
95% interval (over the 12 sender/snapshot groups) excluding zero. Saved-state decisions on
12 groups; not navigation trials.
"""
import argparse,copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .communication_repair_suite import inputs
from .receiver_corrections import load_saved_snapshot
from .terrain_warning import terrain_values
from .cooperative import bootstrap_field,field_vector
from .evaluation import dump

PAIRS=400;W=6.;DS=.05


def packets(entry):
    obj=load_saved_snapshot(entry['path']);out=[]
    for sender in (0,1):
        s=copy.deepcopy(obj['agents'][sender]);x=obj['states'][sender]
        packet=s.export(x,str(1-sender));readings={r.q:r for st in s.stages for r in st.readings};points=[tuple(v[:2]) for v in packet.anchors]
        if not all(q in readings for q in points):continue
        batch=[readings[q] for q in points];stage=copy.deepcopy(next(st for st in s.stages if st.id==packet.stage));stage.readings=batch
        mean,_,_=bootstrap_field(stage,s.cfg,s.communication,s.now,seed=int(entry['meta']['seed']*2+sender))
        vector=field_vector(mean,stage.centers,points,s.cfg.rbf_width).reshape(-1,3)
        field=[tuple(q)+tuple(v) for q,v in zip(points,vector)]
        out.append(dict(group=f"{entry['meta']['case']}_{entry['meta']['seed']}_{sender}",world=obj['worlds'][sender],
            reps=dict(field=field,costmap=[(*v[:3],0.,0.) for v in field],raw=[(*q,float(readings[q].value),0.,0.) for q in points])))
    return out


def path(rng,anchors):
    a=np.asarray(anchors)[:,:2]
    for _ in range(1000):
        c=a[rng.randint(len(a))]+rng.uniform(-.5,.5,2);th=rng.uniform(0,np.pi);L=rng.uniform(.4,1.)
        d=np.array([np.cos(th),np.sin(th)])*L/2;p=c-d+np.linspace(0,1,int(np.ceil(L/DS))+1)[:,None]*2*d
        if (np.linalg.norm(p[:,None]-a[None],axis=2).min(1)<=.65).all():return p
    raise RuntimeError('no supported path')


def cost(p,r):
    mid=(p[:-1]+p[1:])/2;ds=np.linalg.norm(np.diff(p,axis=0),axis=1);return float(np.sum(ds*(1+4*W*r(mid))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise SystemExit('Refuse existing output')
    out.mkdir(parents=True);torch.set_num_threads(1)
    groups=[g for e in inputs() if e['meta']['group']=='acquisition' for g in packets(e)]
    rows=[]
    for gi,g in enumerate(groups):
        rng=np.random.RandomState(187000+gi);anchors=g['reps']['field']
        for k in range(PAIRS):
            P=[path(rng,anchors) for _ in range(2)]
            true=[cost(p,lambda q:g['world'].risk_truth(q)) for p in P]
            # Distance of the paths from the nearest anchor: 'through' if within .15 everywhere.
            a=np.asarray(anchors)[:,:2];off=max(float(np.linalg.norm(p[:,None]-a[None],axis=2).min(1).mean()) for p in P)
            row=dict(group=g['group'],pair=k,true=true,mean_offset=off,gap=abs(true[0]-true[1]))
            for name,anc in g['reps'].items():
                est=[cost(p,lambda q:terrain_values(q,list(anc),unknown=.02)) for p in P]
                pick=int(np.argmin(est));row[name]=dict(est=est,regret=true[pick]-min(true),correct=bool(true[pick]==min(true)))
            rows.append(row)
    dump(out/'rows.json',rows)
    names=['field','raw','costmap'];G=sorted({r['group'] for r in rows});rng=np.random.RandomState(187999)
    def stats(rr):
        per={g:{n:np.mean([r[n]['regret'] for r in rr if r['group']==g]) for n in names} for g in G if any(r['group']==g for r in rr)}
        gs=list(per);res=dict(pairs=len(rr),groups=len(gs))
        for n in names:res[n]=dict(mean_regret=float(np.mean([r[n]['regret'] for r in rr])),accuracy=float(np.mean([r[n]['correct'] for r in rr])))
        for n in ('raw','costmap'):
            d=np.array([per[g][n]-per[g]['field'] for g in gs]);m=d[rng.randint(len(d),size=(10000,len(d)))].mean(1)
            res[f'{n}_minus_field']=dict(mean=float(d.mean()),interval95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],groups_field_better=int((d>0).sum()))
        return res
    q=np.quantile([r['mean_offset'] for r in rows],[1/3,2/3])
    res=dict(all=stats(rows),near_anchors=stats([r for r in rows if r['mean_offset']<=q[0]]),
             between=stats([r for r in rows if q[0]<r['mean_offset']<=q[1]]),far=stats([r for r in rows if r['mean_offset']>q[1]]),offset_terciles=q.tolist())
    a=res['all'];res['gate']=dict(passed=a['raw_minus_field']['interval95'][0]>0 and a['costmap_minus_field']['interval95'][0]>0)
    dump(out/'summary.json',res);print(json.dumps(res,indent=1))


if __name__=='__main__':main()
