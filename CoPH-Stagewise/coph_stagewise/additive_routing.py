"""Opt-in nonnegative, additive integrals on surveyed graph edges.

Each edge has one fixed piecewise-constant density. Partial connectors use the
same cells, so splitting an edge cannot change its integral. No map/support is
certified by this route-cost representation.
"""
import copy
import json
from functools import lru_cache

import numpy as np

from .maze_geometry import corridor_clear, network, obstacle_clear
from .route_field import field_key, values
from .terrain_warning import dense

MODE = 'additive_v1'


def validate(context):
    if context.get('route_integral_mode', 'legacy') not in ('legacy', MODE):
        raise ValueError('incompatible_route_integral_mode')


def portion(profile, start, end):
    """Integral records for a subinterval in metres along a fixed edge."""
    a, b = sorted((float(start), float(end)))
    bounds = np.asarray(profile['bounds'])
    lengths = np.maximum(0., np.minimum(bounds[1:], b)-np.maximum(bounds[:-1], a))
    keep = lengths > 1e-14
    return dict(points=np.asarray(profile['points'])[keep].tolist(), lengths=lengths[keep].tolist(),
                values=np.asarray(profile['values'])[keep].tolist(), support=np.asarray(profile['support'])[keep].tolist())


def price(record, weight):
    return float(np.dot(record['lengths'], 1+4*weight*np.asarray(record['values'])))


@lru_cache(maxsize=64)
def _profiles(key):
    context=json.loads(key); nodes,edges=network(context); nodes=np.asarray(nodes)
    result=[]
    for index,(i,j) in enumerate(edges):
        if not obstacle_clear(nodes[i],nodes[j],context.get('reported_geometry',[])): continue
        path=dense([nodes[i],nodes[j]],.3); lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
        points=(path[:-1]+path[1:])/2; risk,meta=values(points,context,details=True)
        result.append(dict(edge=index,i=i,j=j,points=points.tolist(),bounds=np.r_[0.,np.cumsum(lengths)].tolist(),
            values=risk.tolist(),support=meta['support'],field_evaluations=meta['field_evaluations']))
    return tuple(result)


def profiles(context):
    validate(context)
    body=dict(network=network(context),terrain_anchors=context.get('terrain_anchors',[]),
              reported_geometry=context.get('reported_geometry',[]),unknown_risk=context.get('unknown_risk',1.))
    if field_key(context): body.update(route_field_mode='supported_field',route_field_snapshot=context['route_field_snapshot'])
    before=_profiles.cache_info().misses
    result=_profiles(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False))
    return result, sum(p['field_evaluations'] for p in result) if _profiles.cache_info().misses>before else 0


def records(q,context):
    """Canonical edge records plus projection-connector records and paths.

    A connector follows q -> orthogonal edge projection -> endpoint. Clearance
    is checked on that actual polyline. Duplicate endpoint connections choose
    minimum cost, then edge order, with deterministic float tolerance.
    """
    q=np.asarray(q); nodes,edges=network(context); nodes=np.asarray(nodes)
    weight=context.get('risk_weight',1.); obstacle=context.get('reported_geometry',[])
    tables,evaluations=profiles(context); output=[]; connections={}
    for profile in tables:
        i,j=profile['i'],profile['j']; a,b=nodes[i],nodes[j]; length=profile['bounds'][-1]
        output.append(dict(i=i,j=j,edge=profile['edge'],**portion(profile,0,length)))
        t=float(np.clip((q-a)@(b-a)/np.sum((b-a)**2),0,1)); projected=a+t*(b-a)
        if np.linalg.norm(q-projected)>.7: continue
        if not corridor_clear(q,projected,nodes=nodes,edges=edges) or not obstacle_clear(q,projected,obstacle): continue
        # This off-edge spur is separate from the canonical edge; it is paid
        # once, irrespective of which endpoint the connector chooses.
        spur=dense([q,projected],.15); mid=(spur[:-1]+spur[1:])/2
        risk,meta=values(mid,context,details=True); evaluations+=meta['field_evaluations']
        spur_lengths=np.linalg.norm(np.diff(spur,axis=0),axis=1)
        for k,position in ((i,0.),(j,length)):
            if not corridor_clear(projected,nodes[k],nodes=nodes,edges=edges) or not obstacle_clear(projected,nodes[k],obstacle): continue
            part=portion(profile,t*length,position)
            record=dict(i=-1,j=k,edge=profile['edge'],
                points=mid.tolist()+part['points'],lengths=spur_lengths.tolist()+part['lengths'],
                values=risk.tolist()+part['values'],support=meta['support']+part['support'],
                connector=[q.tolist(),projected.tolist(),nodes[k].tolist()],
                spur_queries=len(mid),edge_position=t*length,endpoint_position=position)
            cost=price(record,weight)
            if k not in connections or cost<price(connections[k],weight)-1e-12:
                connections[k]=record
    output.extend(connections.values())
    return output,dict(field_evaluations=evaluations,quadrature_queries=sum(len(r['points']) for r in output),
        integration_mode=MODE,connections={k:r['connector'] for k,r in connections.items()})


def graph_and_connections(q,context):
    rows,work=records(q,context); nodes,edges=network(context); graph=[[] for _ in nodes]; connections={}
    for r in rows:
        c=price(r,context.get('risk_weight',1.)); i,j=r['i'],r['j']
        if i==-1: connections[j]=c
        else: graph[i].append((j,c));graph[j].append((i,c))
    present={r['edge'] for r in rows if r['i']!=-1}
    return tuple(tuple(g) for g in graph),tuple(i for i in range(len(edges)) if i not in present),connections,work


def route_path(ids,q,context,work=None):
    if not ids: return np.asarray([q])
    if work is None: _,work=records(q,context)
    nodes=np.asarray(network(context)[0]); connector=work['connections'][ids[0]]
    return dense(connector+nodes[list(ids[1:])].tolist())
