"""Compact views of the graded-hazard runs (scene directories of one pilot root)."""
import gzip
import json
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load
from coph_stagewise.terrain_warning import terrain_values

CACHE_VERSION=2
DECISION_DELAY=3.  # seconds after activation: three packet opportunities
CARRIER=1
ROW=np.c_[np.linspace(.0,12.,241),np.full(241,4.)]


def _belief(r):
    """Carrier's route-store risk along the middle row at a declared moment: activation + DECISION_DELAY."""
    spec=r['spec'];start=spec['start_seconds'][CARRIER]
    updates=[u for u in r['updates'][CARRIER] if u['time']>=start]
    u=min(updates,key=lambda u:abs(u['time']-(start+DECISION_DELAY)));heard=any(z['packets'] for z in updates if z['time']<=u['time']);nav=u['post']['navigation']
    return dict(time=u['time'],heard=bool(heard),values=terrain_values(ROW,nav.get('terrain_anchors',[]),nav.get('unknown_risk',1.)).tolist())


def _compact(r):
    a=r['agents']
    return dict(scene=r['spec']['scene'],arm=r['arm'],spec=r['spec'],agents=[dict(cost=z['components']['time']+z['components']['risk'],time=z['components']['time'],
        risk=z['components']['risk'],success=z['success'],collision=z['collision'],finish=z['finish_time'],bytes=sum(z['bytes'].values())) for z in a],
        samples=[[s['x'][:2] for s in ss] for ss in r['samples']],belief=_belief(r))


def results(root):
    root=Path(root);sources=sorted(root.glob('*/episodes/*.json.gz'));cache=root/'hazard_cache.json.gz'
    stamp=[[str(p.relative_to(root)),p.stat().st_size,p.stat().st_mtime] for p in sources]
    if cache.exists():
        with gzip.open(cache,'rt') as h:data=json.load(h)
        if data.get('version')==CACHE_VERSION and data.get('stamp')==stamp:return data['episodes'],data['manifests'],sources
    episodes=[_compact(load(p)) for p in sources]
    manifests={d.name:json.loads((d/'manifest.json').read_text()) for d in sorted(root.iterdir()) if (d/'manifest.json').exists()}
    with gzip.open(cache,'wt') as h:json.dump(dict(version=CACHE_VERSION,stamp=stamp,episodes=episodes,manifests=manifests),h)
    return episodes,manifests,sources


def true_row(spec):
    from coph_stagewise.hazard_oracle import true_risk
    return np.array([true_risk(q,[tuple(p) for p in spec['patches']]) for q in ROW])
