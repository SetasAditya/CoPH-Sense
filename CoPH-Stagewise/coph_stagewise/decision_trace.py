"""Opt-in read-only decision diagnostics; no truth is passed to the learner."""
import copy
import heapq
import time
import numpy as np
from .attribution import AttributionLearner
from .maze_geometry import network, corridor_clear, obstacle_clear
from .terrain_warning import dense, terrain_values


def route_ranking(q, goal, context, truth=None, reference='estimated', trace_fields=False, respect_context=False):
    """Shortest and second distinct simple routes with production quadrature.

    Dijkstra uses the production (cost,node) ordering. The second route is the
    minimum shortest path obtained by deleting each edge of the first route.
    Immediate return over the virtual starting point is not a distinct route.
    Truth is an evaluator-only callback, never retained in context or a learner.
    """
    if reference not in ('estimated','supported_truth','full_truth'):raise ValueError(reference)
    nodes,edges=network(context);nodes=np.asarray(nodes);q=np.asarray(q)
    obstacles=context.get('reported_geometry',[]);anchors=context.get('terrain_anchors',[])
    weight=4*context.get('risk_weight',1.) if respect_context else 4.
    unknown=context.get('unknown_risk',1.) if respect_context else 1.
    target=int(np.argmin(np.linalg.norm(nodes-np.asarray(goal),axis=1)))
    graph={i:[] for i in range(len(nodes))};graph[-1]=[];quadrature=[];forbidden=set()
    def score(a,b,spacing,label):
        points=dense([a,b],spacing);mid=(points[:-1]+points[1:])/2;length=np.linalg.norm(np.diff(points,axis=0),axis=1)
        support=(np.linalg.norm(mid[:,None,:]-np.asarray(anchors)[None,:,:2],axis=2)<=.65).any(1) if anchors else np.zeros(len(mid),bool)
        from .route_field import values as route_values
        values,metadata=route_values(mid,context,details=True)
        support=np.asarray(metadata['support'])
        if reference!='estimated':
            if truth is None:raise ValueError('oracle requires evaluator callback')
            exact=np.asarray(truth(mid));values=np.where(support,exact,unknown) if reference=='supported_truth' else exact
        cost=float(np.sum(length*(1+weight*values)))
        quadrature.append(dict(edge=label,points=mid.tolist(),support=support.tolist(),values=values.tolist(),lengths=length.tolist(),cost=cost))
        if trace_fields:quadrature[-1].update(route_field=metadata,contributions=(length*(1+weight*values)).tolist())
        return cost
    for i,j in edges:
        if obstacle_clear(nodes[i],nodes[j],obstacles):
            c=score(nodes[i],nodes[j],.3,[i,j]);graph[i].append((j,c));graph[j].append((i,c))
    connections={}
    for i,j in edges:
        a,b=nodes[i],nodes[j];t=np.clip((q-a)@(b-a)/np.sum((b-a)**2),0,1)
        if np.linalg.norm(q-a-t*(b-a))>.7:continue
        for k in (i,j):
            if corridor_clear(q,nodes[k],nodes=nodes,edges=edges) and obstacle_clear(q,nodes[k],obstacles):
                connections[k]=score(q,nodes[k],.15,[-1,k])
        if 1e-9<t<1-1e-9:
            forbidden.update(((i,j),(j,i)))
    graph[-1]=list(connections.items())
    def shortest(banned=None):
        queue=[(0.,-1)];dist={-1:0.};parents={}
        while queue:
            cost,i=heapq.heappop(queue)
            if cost>dist[i]+1e-12:continue
            if i==target:
                path=[i]
                while path[-1]!=-1:path.append(parents[path[-1]])
                return dict(cost=cost,nodes=list(reversed(path))[1:])
            for j,c in graph[i]:
                if banned is not None and (i,j)==banned:continue
                proposed=cost+c
                if proposed<dist.get(j,float('inf'))-1e-12:
                    dist[j]=proposed;parents[j]=i;heapq.heappush(queue,(proposed,j))
        return None
    def canonical(path):
        ids=list(path)
        while ids and np.linalg.norm(nodes[ids[0]]-q)<1e-9:ids.pop(0)
        if len(ids)>1 and (ids[0],ids[1]) in forbidden:ids.pop(0)
        return ids
    first=shortest();second=None
    if first:
        path=[-1]+first['nodes'];alternatives=[]
        for edge in zip(path[:-1],path[1:]):
            other=shortest(edge)
            if other is not None and canonical(other['nodes'])!=canonical(first['nodes']):alternatives.append(other)
        if alternatives:second=min(alternatives,key=lambda v:(v['cost'],v['nodes']))
    for item in (first,second):
        if item is not None:item['nodes']=canonical(item['nodes'])
    return dict(best=first,second=second,margin=second['cost']-first['cost'] if second and first else None,
                quadrature=quadrature,reference=reference,unknown_risk=unknown,risk_weight=weight)


def path_reach(path,context):
    nodes,edges=network(context);nodes=np.asarray(nodes);points=np.asarray(path,dtype=float)
    if points.size==0:return dict(available=False,reason='no_returned_path')
    points=points[:,:2];degree=np.zeros(len(nodes),int)
    for i,j in edges:degree[i]+=1;degree[j]+=1
    junctions=np.flatnonzero(degree>=3);near=[];entered=[]
    for j in junctions:
        distances=np.linalg.norm(points-nodes[j],axis=1);visits=np.flatnonzero(distances<=.25)
        if len(visits):
            near.append(int(j))
            for k in range(int(visits[0])+1,len(points)):
                if np.linalg.norm(points[k]-nodes[j])<=.25:continue
                for ei,(a,b) in enumerate(edges):
                    if j not in (a,b):continue
                    other=b if a==j else a;d=nodes[other]-nodes[j];t=(points[k]-nodes[j])@d/(d@d)
                    if .25/np.linalg.norm(d)<t<1 and np.linalg.norm(points[k]-nodes[j]-t*d)<.25:
                        entered.append(dict(junction=int(j),edge=ei,step=k));break
                if entered and entered[-1]['junction']==j:break
    return dict(available=True,distance=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
        minimum_junction_distance=float(np.linalg.norm(points[:,None,:]-nodes[junctions][None,:,:],axis=2).min()) if len(junctions) else None,
        near_junctions=near,outgoing_entries=entered,reaches_decision=bool(entered))


def force_components(state,x):
    nav=state['navigation'];nom=np.asarray(state['nominal']);applied=np.asarray(state['control']);clipped=np.asarray(nav['nominal_clipped'])
    return dict(attraction=nav.get('attraction_force'),soft_shaping=nav.get('shaping_force'),
        damping=(-1.4*np.asarray(x)[2:]).tolist(),nominal=nom.tolist(),clipped=clipped.tolist(),
        clipping=(clipped-nom).tolist(),safety=(applied-clipped).tolist(),applied=applied.tolist(),
        reason=nav['reason'],constraint_residual=nav.get('constraint_residual'))


class DecisionLearner(AttributionLearner):
    def __init__(self,*args,layout=None,**kwargs):
        super().__init__(*args,**kwargs);self.decision_tracing=True;self.decision_seconds=0.;self._replay_trace=None
        if layout is not None:
            self.navigation['network']=copy.deepcopy(layout)
            self.stages[0].entry_navigation=copy.deepcopy(self.navigation)

    def action(self,x):
        u,raw=super().action(x)
        started=time.perf_counter()
        rank=route_ranking(x[:2],self.goal,self.navigation,trace_fields=self.cfg.route_field_mode=='supported_field')
        evaluations=sum(e.get('route_field',{}).get('field_evaluations',0) for e in rank['quadrature'])
        points=sum(len(e['points']) for e in rank.pop('quadrature'))
        self.last_control['decision_route']=rank
        # Diagnostic metadata never modifies route inputs or packet contents.
        elapsed=time.perf_counter()-started
        self.last_control['decision_work']=dict(seconds=elapsed,quadrature_points=points)
        if self.cfg.route_field_mode=='supported_field':self.last_control['route_field_diag_evals']=evaluations
        self.decision_seconds+=elapsed
        return u,raw

    def replay(self,policy,weights,damping):
        cost,paths=super().replay(policy,weights,damping)
        if self._replay_trace is not None:
            started=time.perf_counter()
            self._replay_trace.append(dict(policy=policy.detach().tolist(),valid=cost is not None,
                missing_paths_reason=None if cost is not None else 'rejected_batch_partial_paths_not_returned',
                eligibility=copy.deepcopy(getattr(self,'replay_eligibility',[])),
                handoffs=copy.deepcopy(getattr(self,'replay_stage_events',[])),
                paths=[dict(stage=s,path=copy.deepcopy(p),reach=path_reach(p,self.navigation)) for s,p,c in paths]))
            self.decision_seconds+=time.perf_counter()-started
        return cost,paths

    def update(self,x):
        before=self.decision_seconds;self._replay_trace=[]
        result=super().update(x)
        result['decision_replay']=self._replay_trace;self._replay_trace=None
        result['decision_seconds']=self.decision_seconds-before
        return result
