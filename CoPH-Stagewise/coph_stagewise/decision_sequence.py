"""Nominal graph-level packet-sequence predictions from an explicit advertisement.

This is NOT physical pH replay. Virtual motion follows the surveyed centreline at
1 m/s, with frozen field evidence and no synthetic sensing or learner updates.
It measures a conditional route-cost proxy; true-world audits live elsewhere.
"""
import copy
import itertools
import json
import zlib
from dataclasses import asdict,dataclass

import numpy as np

from .additive_routing import MODE,portion,price,records
from .communication_repairs import canonical,encode_packet
from .decision_packet import DecisionAdvertisement,costs,packet_prediction,sender_reference,shortest
from .maze_geometry import network


@dataclass(frozen=True)
class SequenceAdvertisement:
    schema:int
    decision:dict
    nodes:tuple
    edges:tuple
    position:tuple
    convention:str='additive_v1; nominal_graph_speed=1m/s; frozen_evidence; no_future_sensing'

    @property
    def ad(self):
        body=copy.deepcopy(self.decision);body['records']=tuple(body['records'])
        return DecisionAdvertisement(**body)

    def wire(self):return zlib.compress(canonical(asdict(self)),9)

    def validate(self,sender,recipient,now):
        if self.schema!=1 or self.convention!='additive_v1; nominal_graph_speed=1m/s; frozen_evidence; no_future_sensing':raise ValueError('sequence_convention')
        self.ad.validate(sender,recipient,now)
        if not 1<=len(self.nodes)<=64 or len(self.edges)>128 or np.asarray(self.nodes).shape!=(len(self.nodes),2):raise ValueError('sequence_layout')
        if np.asarray(self.position).shape!=(2,) or not np.isfinite([*np.asarray(self.nodes).ravel(),*self.position]).all():raise ValueError('sequence_geometry')
        if any(len(e)!=2 or any(not 0<=int(k)<len(self.nodes) for k in e) for e in self.edges):raise ValueError('sequence_edges')

    @classmethod
    def decode(cls,wire):
        if len(wire)>1000000:raise ValueError('sequence_wire_limit')
        decoder=zlib.decompressobj()
        try:raw=decoder.decompress(wire,2000001)
        except zlib.error as exc:raise ValueError('sequence_wire_invalid') from exc
        if len(raw)>2000000 or not decoder.eof or decoder.unused_data:raise ValueError('sequence_decoded_limit')
        body=json.loads(raw);body.update(nodes=tuple(map(tuple,body['nodes'])),edges=tuple(map(tuple,body['edges'])),position=tuple(body['position']))
        body['decision']['records']=tuple(body['decision']['records'])
        return cls(**body)


def advertise(receiver,state,sender):
    if receiver.navigation.get('route_integral_mode')!=MODE:raise ValueError('sequence_requires_additive_mode')
    rows,_=records(state[:2],receiver.navigation);nodes,edges=network(receiver.navigation)
    ad=DecisionAdvertisement(1,receiver.agent_id,str(sender),receiver.now,
        receiver.now+receiver.cfg.dt*receiver.cfg.update_every,receiver.version,
        int(np.argmin(np.linalg.norm(np.asarray(nodes)-np.asarray(receiver.goal),axis=1))),
        receiver.navigation.get('risk_weight',1.),tuple(rows))
    return SequenceAdvertisement(1,asdict(ad),nodes,edges,tuple(map(float,state[:2])))


def profiles(ad):
    return {r['edge']:dict(r,bounds=np.r_[0.,np.cumsum(r['lengths'])].tolist()) for r in ad.records if r['i']!=-1}


def at_position(template,q,model):
    """Regenerate additive partial edges on the nominal centreline.

    The original advertised off-axis spurs are used only at the initial state.
    Future virtual states stay on the graph; they do not certify field support.
    """
    q=np.asarray(q);base=template.ad;nodes=np.asarray(template.nodes);tables=profiles(model)
    output=[copy.deepcopy(r) for r in model.records if r['i']!=-1]
    if np.linalg.norm(q-np.asarray(template.position))<1e-10:
        output.extend(copy.deepcopy(r) for r in model.records if r['i']==-1)
    else:
        connections={}
        for index,p in tables.items():
            i,j=p['i'],p['j'];a,b=nodes[i],nodes[j];length=p['bounds'][-1]
            t=float(np.clip((q-a)@(b-a)/np.sum((b-a)**2),0,1))
            if np.linalg.norm(q-a-t*(b-a))>1e-7:continue
            for k,where in ((i,0.),(j,length)):
                part=portion(p,t*length,where)
                r=dict(i=-1,j=k,edge=index,**part,connector=[q.tolist(),q.tolist(),nodes[k].tolist()],
                       spur_queries=0,edge_position=t*length,endpoint_position=where)
                if k not in connections or price(r,base.risk_weight)<price(connections[k],base.risk_weight)-1e-12:connections[k]=r
        # Before reaching the centreline, retain the untraversed portion of
        # the advertised initial spur. No new off-axis field query is invented.
        initial=np.asarray(template.position)
        for original in model.records:
            if original['i']!=-1:continue
            projected=np.asarray(original['connector'][1]);delta=projected-initial;norm=float(delta@delta)
            if norm<1e-20:continue
            progress=float((q-initial)@delta/norm)
            if not -1e-10<=progress<=1+1e-10 or np.linalg.norm(q-initial-progress*delta)>1e-7:continue
            n=original['spur_queries'];lengths=np.asarray(original['lengths'][:n]);bounds=np.r_[0.,np.cumsum(lengths)]
            rest=np.maximum(0.,bounds[1:]-np.maximum(bounds[:-1],max(0.,progress)*np.sqrt(norm)));keep=rest>1e-14
            indices=np.flatnonzero(keep).tolist();r=copy.deepcopy(original)
            for field in ('points','values','support'):
                r[field]=[original[field][k] for k in indices]+original[field][n:]
            r['lengths']=rest[keep].tolist()+original['lengths'][n:];r['spur_queries']=len(indices)
            r['connector'][0]=q.tolist();k=r['j']
            if k not in connections or price(r,base.risk_weight)<price(connections[k],base.risk_weight)-1e-12:connections[k]=r
        output.extend(connections.values())
    return DecisionAdvertisement(base.schema,base.receiver,base.recipient,base.timestamp,base.expires,
        base.version,base.goal_node,base.risk_weight,tuple(output))


def merge(model,packet):
    predictions=packet_prediction(model,packet); rows=[]; anchors=np.asarray(packet.anchors)
    for r,v in zip(model.records,predictions):
        row=copy.deepcopy(r);row['values']=v.tolist()
        points=np.asarray(r['points']).reshape(-1,2)
        covered=(np.linalg.norm(points[:,None]-anchors[None,:,:2],axis=-1)<=.65).any(1)
        row['support']=(np.asarray(r['support'])|covered).tolist();rows.append(row)
    return DecisionAdvertisement(model.schema,model.receiver,model.recipient,model.timestamp,model.expires,
        model.version,model.goal_node,model.risk_weight,tuple(rows))


def reference_lookup(ad,reference):
    return {tuple(q):float(value) for r,v in zip(ad.records,reference) for q,value in zip(r['points'],v)}


def referenced_cost(ad,route,lookup,unknown):
    return sum(price(dict(r,values=[lookup.get(tuple(q),unknown) for q in r['points']]),ad.risk_weight)
               for k in route['records'] for r in [ad.records[k]])


def first_branch(path,nodes,edges):
    """First observed exit from a degree>=3 junction, with explicit missingness."""
    nodes=np.asarray(nodes);degree=np.zeros(len(nodes),int)
    for i,j in edges:degree[i]+=1;degree[j]+=1
    entered={}
    for step,q in enumerate(np.asarray(path)[:,:2]):
        for junction in np.flatnonzero(degree>=3):
            if np.linalg.norm(q-nodes[junction])<=.25:entered.setdefault(int(junction),step)
        for junction,start in entered.items():
            if np.linalg.norm(q-nodes[junction])<=.25:continue
            for index,(i,j) in enumerate(edges):
                if junction not in (i,j):continue
                other=j if i==junction else i;d=nodes[other]-nodes[junction]
                t=float((q-nodes[junction])@d/(d@d))
                if t>.25/np.linalg.norm(d) and np.linalg.norm(q-nodes[junction]-t*d)<=.25:
                    return dict(valid=True,junction=junction,edge=index,step=step,observed_from=start)
    return dict(valid=False,reason='no_observed_junction_exit',junctions_seen=sorted(entered))


def advance(template,q,ad,route,distance,lookup,unknown):
    """Traverse a prefix and integrate reference risk on the same fixed cells."""
    tables=profiles(ad);nodes=np.asarray(template.nodes);q=np.asarray(q).copy();left=distance;cost=0.;travel=0.
    connector=ad.records[route['records'][0]];projection=np.asarray(connector['connector'][1])
    pieces=[(projection,None,connector), (nodes[route['nodes'][0]],connector['edge'],None)]
    pieces += [(nodes[j],ad.records[k]['edge'],None) for j,k in zip(route['nodes'][1:],route['records'][1:])]
    for target,edge,spur in pieces:
        delta=target-q;length=float(np.linalg.norm(delta))
        if length<1e-12:continue
        step=min(left,length);after=q+delta*(step/length)
        if edge is None:
            count=spur['spur_queries'];lengths=np.asarray(spur['lengths'][:count])
            risks=np.array([lookup.get(tuple(p),unknown) for p in spur['points'][:count]])
            bounds=np.r_[0.,np.cumsum(lengths)]
            used=np.maximum(0.,np.minimum(bounds[1:],step)-bounds[:-1])
            cost+=step+4*ad.risk_weight*float(np.dot(used,risks))
        else:
            p=tables[edge];a=nodes[p['i']];b=nodes[p['j']];unit=(b-a)/np.linalg.norm(b-a)
            part=portion(p,float((q-a)@unit),float((after-a)@unit))
            part['values']=[lookup.get(tuple(v),unknown) for v in part['points']]
            cost+=price(part,ad.risk_weight)
        q=after;travel+=step;left-=step
        if left<1e-12:break
    return q,float(cost),travel


def predict_sequence(template,packets,reference,*,dt=.12,interval=8,horizon=32,unknown=.02):
    """Fixed-rate frozen-content deliveries; no observations or physical dynamics."""
    model=template.ad;q=np.array(template.position);lookup=reference_lookup(model,reference)
    total=0.;distance=0.;path=[q.tolist()];checkpoints={};deliveries=[]
    for step in range(horizon):
        if step%interval==0 and step//interval<len(packets):
            packet=packets[step//interval]
            if model.timestamp+step*dt>packet.expires+1e-9:return dict(valid=False,reason='frozen_packet_expired',path=path)
            model=merge(model,packet);deliveries.append(dict(step=step,stage=packet.stage,packet=packet.identity))
        ad=at_position(template,q,model);route=shortest(ad,costs(ad))
        if route is None:return dict(valid=False,reason='nominal_no_connector',path=path)
        q,c,travel=advance(template,q,ad,route,dt,lookup,unknown);total+=c;distance+=travel;path.append(q.tolist())
        if step+1 in (8,16,32):
            end_ad=at_position(template,q,model);tail=shortest(end_ad,costs(end_ad))
            checkpoints[str(step+1)]=dict(prefix_reference_cost=total,reference_cost_with_tail=total+referenced_cost(end_ad,tail,lookup,unknown),
                nodes=tail['nodes'],q=q.tolist(),distance=distance)
    return dict(valid=True,path=path,checkpoints=checkpoints,deliveries=deliveries,
        first_branch=first_branch(path,template.nodes,template.edges),
        endpoint='nominal 1m/s spatial prefix plus frozen-reference route tail; not executed mission cost')


def rank_sequences(sender,state,recipient,advertisement,packets,rotation_stages,coverage_first):
    advertisement.validate(sender.agent_id,recipient,sender.now)
    reference,masks,work=sender_reference(sender,advertisement.ad)
    baseline=predict_sequence(advertisement,[],reference)
    by_stage={p.stage:p for p in packets}; orders=list(itertools.permutations(sorted(by_stage)))
    rows=[]
    for order in orders:
        seq=[by_stage[s] for s in order];result=predict_sequence(advertisement,seq,reference)
        benefit=baseline['checkpoints']['32']['reference_cost_with_tail']-result['checkpoints']['32']['reference_cost_with_tail'] if baseline['valid'] and result['valid'] else None
        rows.append(dict(stages=list(order),benefit=benefit,prediction=result,payload_bytes=sum(len(encode_packet(p)) for p in seq)))
    rotation=[s for s in rotation_stages if s in by_stage]
    rotation += [s for s in sorted(by_stage) if s not in rotation]
    coverage=([coverage_first] if coverage_first in by_stage else [])+[s for s in rotation if s!=coverage_first]
    positives=[r for r in rows if r['benefit'] is not None and r['benefit']>1e-10]
    chosen=max(positives,key=lambda r:(r['benefit'],-r['payload_bytes'],tuple(-x for x in r['stages'])))['stages'] if positives else rotation
    return dict(baseline=baseline,orders=rows,choices=dict(rotation=rotation,coverage_first=coverage,sequence=chosen),
        reference_work=work,sender_supported_queries=sum(int(m.sum()) for m in masks),queries=sum(len(m) for m in masks),
        status_bytes=len(advertisement.wire()),reason='positive_sequence_value' if positives else 'rotation_fallback',
        fixed_rate=True,adaptive_rate=False)
