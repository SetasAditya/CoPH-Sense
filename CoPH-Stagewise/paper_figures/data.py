"""Compact, cached views of logged episodes. Every number plotted comes from here."""
import gzip
import json
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load
from coph_stagewise.records import Config
from coph_stagewise.maze_visuals import landscape
from coph_stagewise.route_oracle import true_distance,predict
from .style import ARMS

CACHE_VERSION=2
WINDOW=4.  # seconds of model snapshots kept around each warning


def _doors(spec):
    return [tuple(o) for o in spec['obstacles'] if len(o)==5]


def _compact(r):
    spec=r['spec'];cfg=Config(**r['config']);out=dict(spec=spec,config=r['config'],arm=r['arm'],seed=spec['seed'],agents=r['agents'],
        geometry_events=r['geometry_events'],deliveries=[e for e in r['events'] if e['kind']=='delivered'],samples=[],trajectories=[],snapshots=[])
    for i in range(len(r['samples'])):
        goal=spec['goals'][i]
        out['samples'].append([dict(time=s['time'],x=s['x'],done=s['done'],remaining=true_distance(spec,s['x'],goal)) for s in r['samples'][i]])
        warnings=[e['time'] for e in r['geometry_events'][i] if e['kind']=='received_geometry']
        rows=[]
        for t in r['trajectories'][i]:
            m=t['control_model'];V,_,_=landscape(m,np.array([t['x'][:2]]),np.zeros(2),cfg)
            row=dict(time=t['time'],x=t['x'],u=t['u'],attraction=t['navigation'].get('attraction_force'),shaping=t['navigation'].get('shaping_force'),
                     waypoint=m['target'],V=float(V[0]),kinetic=.5*float(np.dot(t['x'][2:],t['x'][2:])),residual=t['energy']['residual'],version=t['version'])
            if any(abs(t['time']-w)<=WINDOW for w in warnings):row.update(control_model=m,route=t['navigation'].get('route'))
            rows.append(row)
        out['trajectories'].append(rows)
        out['snapshots'].append([dict(time=u['time'],stages=[dict(id=s['id'],bounds=s['bounds'],counterfactuals=[c['states'] for c in s['counterfactuals']]) for s in u['post']['stages']])
                                 for u in r['updates'][i] if any(abs(u['time']-w)<=WINDOW for w in warnings)])
    return out


def results(root):
    """{(seed, arm): compact episode} for a results directory, cached beside it."""
    root=Path(root);cache=root/'paper_cache.json.gz';sources=sorted((root/'episodes').glob('*.json.gz'))
    stamp=[[p.name,p.stat().st_size,p.stat().st_mtime] for p in sources]
    if cache.exists():
        with gzip.open(cache,'rt') as handle:data=json.load(handle)
        if data.get('version')==CACHE_VERSION and data.get('stamp')==stamp:
            return {(e['seed'],e['arm']):e for e in data['episodes']},sources
    episodes=[_compact(load(p)) for p in sources]
    with gzip.open(cache,'wt') as handle:json.dump(dict(version=CACHE_VERSION,stamp=stamp,episodes=episodes),handle)
    return {(e['seed'],e['arm']):e for e in episodes},sources


def path_length(samples):
    p=np.array([s['x'][:2] for s in samples]);return float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)))


def wasted(e):
    return [path_length(s)-s[0]['remaining'] for s in e['samples']]


def door_events(e):
    """First time each agent sensed / was told about each door: {(door, agent, kind): (time, q)}."""
    out={}
    for i,events in enumerate(e['geometry_events']):
        for g in events:
            if len(g['obstacle'])!=5:continue
            key=(tuple(g['obstacle']),i,'sensed' if g['kind']=='locally_sensed' else 'received')
            if key not in out or g['time']<out[key][0]:out[key]=(g['time'],g['q'],g.get('sender'))
    return out


def oracle(spec,arm,delay=0.,speed=1.):
    return predict(spec,arm,delay=delay,speed=speed)


def arms_present(episodes,seed):
    return [a for a in ARMS if (seed,a) in episodes]
