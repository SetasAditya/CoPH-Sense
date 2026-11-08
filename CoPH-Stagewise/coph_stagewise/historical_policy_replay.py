"""Opt-in separation of historical policy learning from current route choice.

The original retained_replay already starts each trajectory at its own retained
entry with its own copied entry_navigation and optimizes one shared policy. This
module preserves that implementation. It prevents the separate discrete route
voter from reusing current-position candidate identities across historical starts.
"""
import copy
import hashlib
import json

from .route_decision import RouteDecisionLearner,RouteDecisionOptions


def signature(agent):
    """Read-only identity of the fixed entry/context/horizon batch, not its policy."""
    plan=getattr(agent,'replay_plan',{}) if getattr(agent,'replay_plan_active',False) else {}
    entries=[]
    for stage in agent.stages:
        context=copy.deepcopy(stage.entry_navigation or {})
        payload=json.dumps(context,sort_keys=True,separators=(',',':'),default=lambda v:v.tolist())
        entries.append(dict(stage=stage.id,state=list(stage.entry),
            context_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            horizon=plan.get(stage.id,{}).get('horizon',agent.cfg.replay_steps)))
    return dict(version=agent.version,starts_mode=getattr(agent,'replay_start_mode','historical'),entries=entries)


class SharedPolicyReplayLearner(RouteDecisionLearner):
    """Existing shared-policy replay followed by an explicitly current-only route.

    No replay cost, integrator, support rule, validation, solver, or policy bounds
    are replaced. Historical route labels remain attached to their own starts;
    they never vote for the route to execute at the current physical state.
    """
    def __init__(self,*args,route_options=None,**kwargs):
        options=route_options or RouteDecisionOptions(mode='prefix',starts='current',terminal='graph')
        if options.starts!='current':raise ValueError('actual route choice must use the current physical state')
        if kwargs.get('mode','full')!='full':raise ValueError('shared policy replay requires full learning mode')
        kwargs['mode']='full'
        super().__init__(*args,route_options=options,**kwargs)
        if self.cfg.replay_update_cap is None or not 0<self.cfg.replay_update_cap<=1024:
            raise ValueError('shared policy replay requires a positive flow ceiling at most 1024')
        self.replay_start_mode='historical'
        self.shared_policy_calls=[]
        self._shared_active=False
        self._shared_signature=None

    def replay(self,policy,weights,damping):
        batch=signature(self)
        if getattr(self,'_shared_active',False):
            if self._shared_signature is None:self._shared_signature=copy.deepcopy(batch)
            elif batch!=self._shared_signature:raise RuntimeError('replay entries, contexts, or horizons changed within one update')
        before=self.counterfactual_steps
        cost,paths=super().replay(policy,weights,damping)
        first={}
        for control in getattr(self,'replay_controls',[]):
            first.setdefault(control['entry'],dict(stage=control['entry'],step=control['step'],
                route_nodes=copy.deepcopy(control.get('route_nodes',[])),waypoint=copy.deepcopy(control.get('waypoint')),
                reason=control.get('route_reason'),no_route=control.get('no_route',False)))
        row=dict(batch=batch,policy=policy.detach().tolist(),valid=cost is not None,
            cost=float(cost.detach()) if cost is not None else None,flow=self.counterfactual_steps-before,
            eligibility=copy.deepcopy(getattr(self,'replay_eligibility',[])),
            handoffs=copy.deepcopy(getattr(self,'replay_stage_events',[])),first_controls=list(first.values()),
            trajectories=[dict(stage=s,states=copy.deepcopy(path),cost=float(value.detach())) for s,path,value in paths],
            semantics='per-entry own-state control; scalar losses train one shared policy; route identities are not aggregated')
        if not hasattr(self,'shared_policy_calls'):self.shared_policy_calls=[]
        self.shared_policy_calls.append(row)
        return cost,paths

    def update(self,x):
        if self.route_options.starts!='current':raise ValueError('historical route voting is disabled in this opt-in learner')
        self.replay_start_mode='historical'
        self.shared_policy_calls=[];self._shared_signature=None;self._shared_active=True
        try:
            report=super().update(x)
        finally:
            self._shared_active=False
        report['shared_policy_replay']=dict(calls=copy.deepcopy(self.shared_policy_calls),
            shared_coefficients=self.policy.tolist(),actual_route_starts='current',
            original_retained_policy_implementation=True,
            mechanism='historical trajectories optimize shared policy; current state alone selects discrete route')
        assert report['counterfactual_steps']<=self.cfg.replay_update_cap
        return report
