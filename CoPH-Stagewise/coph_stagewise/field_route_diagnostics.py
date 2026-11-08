"""Opt-in routing diagnostics and evaluator-only paired representation audits."""
import copy,time
import numpy as np
from .decision_trace import DecisionLearner,route_ranking
from .decision_audits import branch_observer
from .attribution_audits import packet_audit,digest
from .route_field import hypothetical_context,sync

class FieldRouteLearner(DecisionLearner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.track_route_field=True
        self.route_field_evaluations=0;self.route_field_diagnostic_evaluations=0;self.route_field_seconds=0.
        self.navigation['route_field_trace_enabled']=True
        self.stages[0].entry_navigation=copy.deepcopy(self.navigation)

    def action(self,x):
        u,raw=super().action(x);start=time.perf_counter()
        rank=route_ranking(x[:2],self.goal,self.navigation,trace_fields=True)
        self.last_control['route_field_trace']=rank
        self.last_control['route_field_diagnostic_work']=dict(seconds=time.perf_counter()-start,
            field_evaluations=sum(e['route_field']['field_evaluations'] for e in rank['quadrature'])+self.last_control.get('route_field_diag_evals',0))
        work=self.last_control.get('route_field_work',{})
        self.route_field_evaluations+=work.get('field_evaluations',0)
        self.route_field_seconds+=work.get('seconds',0.)
        self.route_field_diagnostic_evaluations+=self.last_control['route_field_diagnostic_work']['field_evaluations']
        return u,raw


def representation_audit(agent,x,world):
    before=digest(agent);truth_before=digest(world);start=time.perf_counter()
    ordinary=branch_observer(agent,x,world)
    contexts={'anchors':copy.deepcopy(agent.navigation),'supported_field':hypothetical_context(agent)}
    contexts['anchors']['route_field_mode']='anchors'
    ranks={k:route_ranking(x[:2],agent.goal,c,trace_fields=True) for k,c in contexts.items()}
    common=[];all_points=[]
    for a,d in zip(ranks['anchors']['quadrature'],ranks['supported_field']['quadrature']):
        assert a['points']==d['points'] and a['edge']==d['edge']
        truth=np.asarray(world.risk_truth(np.asarray(a['points'])))
        mask=np.asarray(a['support'])&np.asarray(d['support'])
        field=np.asarray(d['route_field']['field_support']);length=np.asarray(a['lengths'])
        common.append(dict(edge=a['edge'],points=a['points'],truth=truth.tolist(),common_support=mask.tolist(),field_support=field.tolist(),lengths=a['lengths'],
            anchor_error=(np.asarray(a['values'])-truth).tolist(),field_error=(np.asarray(d['values'])-truth).tolist()))
    ordinary['route_field_comparison']=dict(rankings=ranks,errors=common,seconds=time.perf_counter()-start,
        field_evaluations=sum(e['route_field']['field_evaluations'] for r in ranks.values() for e in r['quadrature']),
        semantics='same committed model and physical state; evaluator truth never enters runtime; branch-route changes are not traversal')
    assert digest(agent)==before and digest(world)==truth_before
    return ordinary


def field_packet_audit(agent,x,packet,world,tick):
    from . import route_field
    before=route_field.FIELD_EVALUATIONS
    a=packet_audit(agent,x,packet,world,tick,branch_observer=representation_audit)
    a['decision_intermediates']=[dict(label=s['label'],route=s['navigation'].get('decision_route')) for s in a['intermediate_states']]
    a['decision_guard_flows']=sum(b['decision_audit']['guard_flows'] for b in a['branches'].values())
    a['route_field_total_audit_evaluations']=route_field.FIELD_EVALUATIONS-before
    return a


def compatible_baseline(new,old):
    from .decision_suite import compare_operational
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if not k.startswith('route_field')}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    result=compare_operational(clean(new),clean(old))
    result['excluded_additive_metadata']='route_field* diagnostic/configuration keys only; operational arrays, controls, decisions, work and packet identity remain compared'
    return result
