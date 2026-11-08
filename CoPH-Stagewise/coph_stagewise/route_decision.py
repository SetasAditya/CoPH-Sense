"""Opt-in route decisions from supported physical prefixes and graph continuation.

No evaluator imports. Graph continuation is an approximate cost-to-go at nominal
speed 1 m/s, with the existing route-source precedence and unknown-risk prior.
It is not a certified prediction of unseen terrain or a learned HJB value.
"""
import copy
from dataclasses import dataclass
import time
import numpy as np
from .model import tensor,flow
from .maze_geometry import network,corridor_clear,obstacle_clear
from .terrain_warning import dense
from .route_field import values,field_key
from .stage_prediction import FrozenStages,DomainExit


@dataclass(frozen=True)
class RouteDecisionOptions:
    mode: str = 'graph'
    starts: str = 'current'
    terminal: str = 'graph'
    horizon: int = 8
    candidates: int = 4
    expansion_cap: int = 4096
    path_cap: int = 64
    nominal_speed: float = 1.

    def __post_init__(self):
        if self.mode not in ('graph','prefix') or self.starts not in ('current','historical') or self.terminal not in ('none','graph'):raise ValueError('route decision mode')
        if min(self.horizon,self.candidates,self.expansion_cap,self.path_cap)<1 or self.horizon>32 or self.candidates>4:raise ValueError('route decision budget')
        if self.nominal_speed!=1.:raise ValueError('fixed declared continuation speed 1 m/s')


def connectors(q,context):
    """Same .7 corridor eligibility and .15 quadrature as production routing."""
    if context.get('route_integral_mode')=='additive_v1':
        from .additive_routing import graph_and_connections
        _,_,connections,work=graph_and_connections(q,context)
        return connections,work['quadrature_queries']
    nodes,edges=network(context);nodes=np.asarray(nodes);obstacles=context.get('reported_geometry',[])
    output={};work=0
    for i,j in edges:
        a,b=nodes[i],nodes[j];t=np.clip((q-a)@(b-a)/np.sum((b-a)**2),0,1)
        if np.linalg.norm(q-(a+t*(b-a)))>.7:continue
        for k in (i,j):
            if corridor_clear(q,nodes[k],nodes=nodes,edges=edges) and obstacle_clear(q,nodes[k],obstacles):
                p=dense([q,nodes[k]],.15);mid=(p[:-1]+p[1:])/2
                r,m=values(mid,context,details=True);work+=len(mid)
                output[k]=float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)*(1+4*context.get('risk_weight',1.)*r)))
    return output,work


def graph_data(context):
    if context.get('route_integral_mode')=='additive_v1':
        from .additive_routing import graph_and_connections
        # Graph profiles are independent of the virtual start.
        graph,blocked,_,_=graph_and_connections(np.asarray(network(context)[0][0]),context)
        return graph,blocked
    from .maze_learning import graph_costs
    layout=network(context)
    graph,blocked,_=graph_costs(tuple(tuple(a) for a in context.get('terrain_anchors',[])),
        tuple(tuple(o) for o in context.get('reported_geometry',[])),layout,field_key(context),
        context.get('risk_weight',1.),context.get('unknown_risk',1.))
    return graph,blocked


def conditioned_path(ids,q,context):
    """Join a frozen route through exactly the production-eligible connectors."""
    ids=tuple(map(int,ids));graph,_=graph_data(context);connections,work=connectors(np.asarray(q),context)
    if not ids or len(set(ids))!=len(ids):return None,None,work
    edge_cost={(i,j):c for i,neighbors in enumerate(graph) for j,c in neighbors}
    if any((i,j) not in edge_cost for i,j in zip(ids[:-1],ids[1:])):return None,None,work
    suffix=[0.]*len(ids)
    for k in range(len(ids)-2,-1,-1):suffix[k]=suffix[k+1]+edge_cost[ids[k],ids[k+1]]
    choices=[(connections[node]+suffix[k],k) for k,node in enumerate(ids) if node in connections]
    if not choices:return None,None,work
    cost,k=min(choices,key=lambda z:(z[0],z[1]));return list(ids[k:]),float(cost),work


def route_candidates(atlas,x,context,options):
    from .maze_learning import maze_route
    baseline_context=copy.deepcopy(context);baseline_context.pop('route_candidate_nodes',None)
    _,metadata=maze_route(atlas,tensor(x),baseline_context,0)
    incumbent=metadata.get('route_nodes',[])
    if not incumbent:return [],dict(reason='no_incumbent_route',expansions=0)
    graph,_=graph_data(context);start=incumbent[0];goal=incumbent[-1]
    paths=[];stack=[(start,(start,),0.)];expanded=0
    while stack and expanded<options.expansion_cap and len(paths)<options.path_cap:
        node,path,cost=stack.pop();expanded+=1
        if node==goal:paths.append((cost,path));continue
        for nxt,edge in sorted(graph[node],key=lambda v:v[0],reverse=True):
            if nxt not in path:stack.append((nxt,path+(nxt,),cost+edge))
    unique={tuple(incumbent)}|{p for c,p in paths}
    scored=[]
    for ids in unique:
        suffix,cost,work=conditioned_path(ids,np.asarray(x[:2]),context)
        if suffix is not None:scored.append((cost,ids))
    scored.sort(key=lambda v:(v[0],v[1]))
    # Always retain the original route, even if enumeration hits its search cap.
    selected=[tuple(incumbent)]+[p for c,p in scored if p!=tuple(incumbent)][:options.candidates-1]
    return selected,dict(reason='enumerated',expansions=expanded,enumeration_truncated=bool(stack),
        generated=len(unique),incumbent=list(incumbent),frozen_incumbent_connector=start,
        candidate_nodes=[list(p) for p in selected])


def prefix(atlas,state,index,policy,damping,context,nodes,horizon,terminal,charge):
    x=tensor(state);context=copy.deepcopy(context);context['route_candidate_nodes']=list(nodes)
    context['route_candidate_schema']=1;cost=0.;path=[list(state)];handoffs=[];controls=[]
    for step in range(horizon):
        # Historical starts use the existing handoff rule; no unseen stages created.
        new=atlas.advance(x[:2].detach().numpy(),index)
        if new!=index:handoffs.append(dict(step=step,old=atlas.stages[index].id,new=atlas.stages[new].id))
        index=new
        u,_=atlas.control(x,index,policy,damping=damping,context=context,step=step)
        meta=copy.deepcopy(context.get('last_control',{}));charge(meta.get('guard_steps',0))
        if meta.get('infeasible') or meta.get('no_route'):raise DomainExit('route_prefix_safety_or_route')
        y=flow(x,u,damping,atlas.cfg.dt);charge(1)
        if not atlas.stages[index].supported(y[:2].detach().numpy(),atlas.cfg):raise DomainExit('unsupported_route_prefix')
        cost+=atlas.cfg.dt*(1+4*context.get('risk_weight',1.)*float(atlas.value(y[:2],index).detach())+.03*float(u.square().sum().detach()))
        controls.append(meta);x=y;path.append(x.detach().tolist())
        if np.linalg.norm(x[:2].detach().numpy()-atlas.goal)<.25:break
    ended=np.linalg.norm(x[:2].detach().numpy()-atlas.goal)<.25
    tail=0.;queries=0
    if terminal=='graph' and not ended:
        _,tail,queries=conditioned_path(nodes,x[:2].detach().numpy(),context)
        if tail is None:raise DomainExit('unavailable_route_continuation')
    return dict(prefix_cost=float(cost),remaining_cost=float(tail),total=float(cost+tail),
        states=path,controls=controls,handoffs=handoffs,terminal_goal=bool(ended),
        route_queries=queries,horizon=horizon,executed_steps=len(path)-1,
        terminal_semantics='approximate remaining length+risk at 1 m/s; no terminal effort model; existing unknown prior')


def evaluate_routes(agent,x,options,flow_allowance=None):
    started=time.perf_counter();atlas=FrozenStages.freeze(agent);context=copy.deepcopy(agent.navigation)
    candidates,enumeration=route_candidates(atlas,x,context,options)
    result=dict(schema=1,mode=options.mode,starts_mode=options.starts,terminal_mode=options.terminal,
        model_version=agent.version,enumeration=enumeration,candidates=[],starts=[],excluded_starts=[],
        selected=None,flow_work=0,reason='no_supported_candidates',horizon=options.horizon)
    if options.mode=='graph':result.update(reason='legacy_graph',seconds=time.perf_counter()-started);return result
    try:current_index=atlas.index_at(np.asarray(x[:2]))
    except DomainExit:
        result.update(reason='unsupported_current_state',seconds=time.perf_counter()-started);return result
    starts=[('current',current_index,tuple(map(float,x)))]
    if options.starts=='historical':
        for index,s in enumerate(atlas.stages):
            if tuple(s.entry)==starts[0][2]:continue
            if not s.supported(s.entry[:2],agent.cfg):result['excluded_starts'].append(dict(stage=s.id,reason='unsupported_entry'));continue
            starts.append((s.id,index,s.entry))
    result['starts']=[dict(label=label,stage=atlas.stages[i].id,state=list(state),horizon=options.horizon) for label,i,state in starts]
    cap=agent.cfg.replay_update_cap or 1024
    allowance=min(cap,cap if flow_allowance is None else flow_allowance)
    multiplier=7 if agent.cfg.controller_mode in ('protected','routed') else 1
    reserved=len(candidates)*len(starts)*options.horizon*multiplier
    result.update(flow_allowance=allowance,reserved_flow=reserved)
    if not candidates or reserved>allowance:
        result.update(reason='route_batch_budget' if candidates else 'no_route_candidates',seconds=time.perf_counter()-started);return result
    def charge(count):
        if result['flow_work']+count>allowance:raise RuntimeError('route computation cap violated')
        result['flow_work']+=count;agent.counterfactual_steps+=count
    for nodes in candidates:
        row=dict(nodes=list(nodes),valid=True,starts=[])
        for label,index,state in starts:
            try:row['starts'].append(dict(label=label,**prefix(atlas,state,index,agent.policy,agent.damping,context,nodes,options.horizon,options.terminal,charge)))
            except DomainExit as error:
                row.update(valid=False,reason=str(error));break
        row['cost']=float(np.mean([s['total'] for s in row['starts']])) if row['valid'] and row['starts'] else None
        result['candidates'].append(row)
    # An incomplete candidate batch retains the incumbent. Excluding a costly
    # route after a support failure must not manufacture a better decision.
    if result['candidates'] and all(c['valid'] for c in result['candidates']):
        chosen=min(enumerate(result['candidates']),key=lambda v:(v[1]['cost'],v[0]))[1]
        result.update(selected=chosen['nodes'],reason='supported_route_choice')
    else:result['reason']='candidate_batch_unsupported'
    result['seconds']=time.perf_counter()-started
    return result


from .maze_family import NetworkLearner

class RouteDecisionLearner(NetworkLearner):
    """Opt-in decision stage after ordinary transactional local learning."""
    def __init__(self,*args,route_options=None,**kwargs):
        super().__init__(*args,**kwargs);self.route_options=route_options or RouteDecisionOptions()
        self.route_decisions=[]
    def update(self,x):
        report=super().update(x)
        if self.route_options.mode=='graph' or self.mode=='frozen':return report
        prior=report.get('counterfactual_steps',0);cap=self.cfg.replay_update_cap or 1024
        decision=evaluate_routes(self,x,self.route_options,max(0,cap-prior))
        self.route_decisions.append(decision);report['route_decision']=decision
        report['counterfactual_steps']=prior+decision['flow_work']
        if decision['selected'] is not None:
            self.navigation.update(route_candidate_schema=1,route_candidate_nodes=copy.deepcopy(decision['selected']))
        assert report['counterfactual_steps']<=cap
        return report
