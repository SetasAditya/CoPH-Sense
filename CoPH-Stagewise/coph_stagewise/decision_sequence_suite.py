"""Frozen additive-routing and packet-sequence diagnostic on preserved hazards."""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import os
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from pathlib import Path

import numpy as np
import torch

from . import communication_navigation as cn
from .additive_routing import MODE,records,route_path
from .attribution_audits import digest
from .communication_reconciliation import compare_operations,episode_row
from .communication_repairs import canonical,encode_packet
from .cooperative import CommunicationConfig
from .crossing_demo import configuration
from .decision_packet import candidates,costs,rank,shortest
from .decision_packet_suite import actual_route,true_route_cost
from .decision_sequence import SequenceAdvertisement,advertise,first_branch,rank_sequences
from .evaluation import dump,load
from .hazard_demo import SCENES,specification
from .maze_family import NetworkWorld
from .records import Transition
from .simple_communication import SimpleOptions,SimpleRepairLink,novel_payload,route_advertisement
from .staggered_demo import sense
from .terrain_warning import dense

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/decision_sequence_v1'
OLD=ROOT/'results/decision_packet_v1'
SEEDS=(86100,86101)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def enable(a):
    a.navigation['route_integral_mode']=MODE
    for stage in a.stages:
        if stage.entry_navigation is not None:stage.entry_navigation['route_integral_mode']=MODE
    return a


def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    entries=load(OLD/'snapshot_inventory.json'); first=[]
    for scene in SCENES:
        for seed in SEEDS:
            match=[e for e in entries if e['key']==f'{scene}_{seed}' and str(e['sender'])=='0' and str(e['receiver'])=='1']
            first.append(min(match,key=lambda e:e['time']))
    inputs={str((OLD/e['path']).relative_to(ROOT)):e['sha256'] for e in entries}
    for name in ('manifest.json','rows.json','snapshot_inventory.json'):
        inputs[str((OLD/name).relative_to(ROOT))]=sha(OLD/name)
    for scene in SCENES:
        for seed in SEEDS:
            for arm in ('rotation','both'):
                p=ROOT/f'results/simple_fixes_v1/communication/episodes/{scene}_{seed}_{arm}_4000000.json.gz';inputs[str(p.relative_to(ROOT))]=sha(p)
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'coph_stagewise').glob('*.py'))}
    manifest=dict(schema=1,sources=sources,inputs=inputs,snapshot_entries=entries,sequence_entries=first,
        specs=[specification(s,seed) for s in SCENES for seed in SEEDS],
        protocol=dict(integration='additive_v1 fixed .3 nominal edge grid, piecewise constant nonnegative densities; projection spur plus canonical edge partials; legacy default',
            legacy_parity='eight archived rotation episodes, operational events/states/controls/versions/work; stop if mismatched',
            instant='all 64 preserved states, same real receiver updates and packets; evaluator uses actual NetworkWorld risk including background .02',
            sequence='earliest pre-offer scout-to-carrier snapshot in each scene/seed; all permutations of up to three frozen retained-stage packets plus withhold',
            sender_prediction='nominal 1m/s centreline graph, no sensing/dynamics/learning; frozen supported sender reference with receiver fallback; spatial prefix plus route tail',
            actual_audits='isolated real receiver updates, true simulator, indexed sensor noise; fixed events at steps 0/8/16; checkpoints8/16/32; no other incoming packets',
            packet_semantics='frozen local evidence re-exported hypothetically at each scheduled time; q/state/header model held frozen; original absolute expiry; no future sender sensing; not a live future-packet prediction',
            choices='rotation order; repaired coverage first then remaining rotation order; maximum positive forecast sequence benefit then bytes/stage ordering, otherwise rotation',
            costs='incurred time+declared weighted risk primary; effort/failure separate; prefix+nominal1m/s true-risk route-tail secondary, never full mission cost',
            budgets='fresh 4MB per-agent diagnostic window, actual full status+payload+ack wire sizes; not remaining lifetime allowance of archived snapshots',
            branch='first observed degree>=3 junction exit beyond .25m along an outgoing edge; unavailable if not observed; planned routes separate',
            gate='all integrity checks; sequence mean secondary selection regret strictly below rotation AND coverage-first, with a beneficial selected outcome in at least two scenes; otherwise no live sequence-selector navigation',
            additive_navigation='16 full development episodes: additive rotation/coverage, four unchanged hazards and two old noise seeds; compare preserved legacy counterparts',
            truth_correction='old decision_packet_suite.truth_values omitted .02 background; keep v1 intact and publish corrected endpoints separately',
            visualization='seed86100 double_mild frozen before execution; sequence branches, model forecasts, timelines; no favorable replacement'),
        runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),dtype='float64',
            import_order='numpy then torch then runtime',threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    m=json.loads(json.dumps(manifest));p=out/'manifest.json'
    if p.exists():
        if load(p)!=m:raise ValueError('frozen source/input/config/runtime mismatch')
    else:
        with p.open('x') as f:json.dump(m,f,indent=2)
        for name in ('sources','legacy','instant','sequences','episodes','figures','errors'):(out/name).mkdir()
        for path in sources:(out/'sources'/Path(path).name).write_bytes((ROOT/path).read_bytes())
    return m


def run_episode(task):
    spec,arm,out=task;root=Path(out);torch.set_num_threads(1)
    key=f"{spec['scene']}_{spec['seed']}_{arm}";p=root/('legacy' if arm=='legacy' else 'episodes')/f'{key}.json.gz'
    if p.exists():return load(p.with_name(p.name+'.row.json'))
    old_factory,old_link=cn.factory,cn.FixedRateRepairLink
    def factory(specification):
        make=old_factory(specification)
        def instantiate(*args,**kwargs):
            a=make(*args,**kwargs)
            return a if arm=='legacy' else enable(a)
        return instantiate
    def link(*args,**kwargs):return SimpleRepairLink(*args,simple_options=SimpleOptions(arm=='coverage',arm=='coverage'),**kwargs)
    try:
        cn.factory=factory;cn.FixedRateRepairLink=link
        result=cn.execute(spec,'reciprocal_fixed','receiver_relevant' if arm=='coverage' else 'rotation',byte_budget=4000000)
    finally:cn.factory,cn.FixedRateRepairLink=old_factory,old_link
    dump(p,result);row=episode_row(result,key,4000000);row['condition']=arm
    baseline='both' if arm=='coverage' else 'rotation'
    archived=load(ROOT/f"results/simple_fixes_v1/communication/episodes/{spec['scene']}_{spec['seed']}_{baseline}_4000000.json.gz")
    if arm=='legacy':row['parity']=compare_operations(result,archived)
    row['carrier_primary_change']=result['agents'][1]['time_risk_cost']-archived['agents'][1]['time_risk_cost']
    dump(p.with_name(p.name+'.row.json'),row);return row


def load_snapshot(entry):
    path=OLD/entry['path'];assert sha(path)==entry['sha256']
    with gzip.open(path,'rb') as f:return pickle.load(f)


def world_for(spec,a):
    world=NetworkWorld(dict(spec,initial=list(a.current_state),goal=a.goal,risk_weight=a.navigation.get('risk_weight',1.)),a.cfg)
    world.seed=spec['seed']*2+int(a.agent_id);world.time=a.now;return world


def instant_job(task):
    entry,out=task;root=Path(out);p=root/'instant'/(Path(entry['path']).name+'.json')
    if p.exists():return load(p)
    torch.set_num_threads(1);s,x,r,rx,spec,tick=load_snapshot(entry)
    before=digest((s,r));physical=copy.deepcopy(r.current_state);measured=digest([(g.readings,g.transitions) for g in r.stages])
    a=enable(copy.deepcopy(r));sender=copy.deepcopy(s);sender.now=a.now=max(s.now,r.now)
    status=advertise(a,rx,sender.agent_id);ad=status.ad
    packets,work=candidates(sender,x,a.agent_id);selected,pred=rank(sender,x,a.agent_id,ad,packets)
    coverage,_=novel_payload(copy.deepcopy(sender),x,a.agent_id,route_advertisement(a,rx,sender.agent_id))
    rotation=copy.deepcopy(sender).export(x,a.agent_id)
    truth=world_for(spec,a);true=costs(ad,[truth.risk_truth(np.asarray(rec['points']).reshape(-1,2)) for rec in ad.records])
    branches=[]
    for packet in [None]+packets:
        b=copy.deepcopy(a);event=b.assimilate(packet) if packet else None;d=b.update(rx.copy());nodes=actual_route(b,rx)
        assert b.current_state==physical and digest([(g.readings,g.transitions) for g in b.stages])==measured
        branches.append(dict(stage=packet.stage if packet else None,nodes=nodes,cost=true_route_cost(ad,nodes,true),
            event=event,accepted=d['accepted'],reason=d['reason'],flow=d['counterfactual_steps']))
    assert before==digest((s,r))
    predicted={v['stage']:v for v in pred['candidates']}
    for b in branches[1:]:
        b['benefit']=branches[0]['cost']-b['cost'] if b['cost'] is not None and branches[0]['cost'] is not None else None
        b['predicted_benefit']=predicted[b['stage']]['benefit'];b['prediction_agrees']=b['nodes']==predicted[b['stage']]['route']['nodes']
    result=dict(entry,scene=spec['scene'],branches=branches,prediction=pred,
        selected=dict(rotation=rotation.stage if rotation else None,coverage=coverage.stage if coverage else rotation.stage,
                      decision=selected.stage if selected else rotation.stage),
        production_parity=actual_route(a,rx)==shortest(ad,costs(ad))['nodes'],state_isolation=True,export_work=work)
    dump(p,result);return result


def correction(out):
    """Reprice v1 planned routes only; do not rerun/overwrite v1 outcomes."""
    from .decision_packet import advertise as old_advertise
    rows=[]
    for row in load(OLD/'rows.json'):
        s,x,r,rx,spec,tick=load_snapshot(row);r.now=max(s.now,r.now);world=world_for(spec,r)
        ad=old_advertise(r,rx,s.agent_id)
        true=costs(ad,[world.risk_truth(np.asarray(v['points']).reshape(-1,2)) for v in ad.records])
        branches=[]
        base=true_route_cost(ad,row['branches'][0]['nodes'],true)
        for b in row['branches'][1:]:
            c=true_route_cost(ad,b['nodes'],true)
            branches.append(dict(stage=b['stage'],corrected_cost=c,corrected_benefit=base-c,
                previous_benefit=b['actual_benefit']))
        rows.append(dict(key=row['key'],time=row['time'],sender=row['sender'],branches=branches,
            selected=row['selection']))
    dump(Path(out)/'v1_endpoint_correction.json',dict(reason='missing background .02 in diagnostic helper; simulator and mission logging already correct',rows=rows))
    return rows


def true_tail(a,x,world):
    rows,work=records(x[:2],a.navigation)
    from .decision_packet import DecisionAdvertisement
    ad=DecisionAdvertisement(1,a.agent_id,str(1-int(a.agent_id)),a.now,a.now+1,a.version,
        int(np.argmin(np.linalg.norm(np.asarray(a.navigation['network'][0])-a.goal,axis=1))),
        a.navigation.get('risk_weight',1.),tuple(rows))
    route=shortest(ad,costs(ad))
    if route is None:return None,[]
    path=route_path(route['nodes'],x[:2],a.navigation,work)
    # Evaluator-only finer quadrature of the fixed selected polyline.
    line=dense(path,.03);mid=(line[:-1]+line[1:])/2
    return float(np.dot(np.linalg.norm(np.diff(line,axis=0),axis=1),1+4*a.navigation.get('risk_weight',1.)*world.risk_truth(mid))),route['nodes']


def continuation(receiver,x,spec,packets,order,status_bytes):
    a=copy.deepcopy(receiver);world=world_for(spec,a);state=np.array(x).copy();path=[state.tolist()]
    parts=dict(time=0.,risk=0.,effort=0.,failure=0.);checkpoints={};events=[];updates=[];controls=[]
    sender_bytes=0;receiver_bytes=status_bytes;by_stage={p.stage:p for p in packets};now=a.now;dt=a.cfg.dt
    initial_measured=digest([(s.readings,s.transitions) for s in a.stages]);physical=a.current_state
    tick0=int(round((now-spec['start_seconds'][int(a.agent_id)])/dt))
    for k in range(32):
        a.now=now+k*dt;world.time=a.now
        if k%a.cfg.update_every==0:
            index=k//a.cfg.update_every
            if index<len(order):
                original=by_stage[order[index]]
                packet=replace(original,timestamp=a.now,identity=f'{original.identity}:frozen-sequence:{k}')
                if a.now<=packet.expires:
                    wire=encode_packet(packet);event=a.assimilate(packet);ack=canonical(event)
                    sender_bytes+=len(wire);receiver_bytes+=len(ack)
                    events.append(dict(step=k,stage=packet.stage,packet=packet.identity,source_packet=original.identity,
                        event=event,payload_bytes=len(wire),ack_bytes=len(ack),source_ids=list(packet.sources),
                        interpretation='hypothetical re-export of frozen evidence; no new sender observations'))
                    if k==0:
                        assert a.current_state==physical and digest([(s.readings,s.transitions) for s in a.stages])==initial_measured
            d=a.update(state.copy());assert d['counterfactual_steps']<=1024
            updates.append(dict(step=k,version=a.version,accepted=d['accepted'],reason=d['reason'],flow=d['counterfactual_steps']))
        u,nom=a.action(state.copy());after,hit,cost,energy=world.step(state,u)
        exposure=dt*float(world.risk_truth(after[:2]));parts['time']+=dt;parts['risk']+=4*a.navigation.get('risk_weight',1.)*exposure
        parts['effort']+=dt*.03*float(u@u)
        assert abs(cost-(dt+4*a.navigation.get('risk_weight',1.)*exposure+dt*.03*float(u@u)))<1e-10
        controls.append(dict(step=k,nominal=nom.tolist(),applied=u.tolist(),energy=energy))
        time_now=now+(k+1)*dt;obs=sense(world,after,tick0+k+1,time_now)
        transition=Transition(tick0+k,time_now,tuple(state),tuple(u),tuple(after),a.version)
        a.observe(obs,transition);state=after;a.advance(state);path.append(state.tolist())
        if k+1 in (8,16,32):
            tail,nodes=true_tail(a,state,world)
            checkpoints[str(k+1)]=dict(components=copy.deepcopy(parts),incurred_primary=parts['time']+parts['risk'],
                primary_plus_nominal_true_tail=parts['time']+parts['risk']+tail if tail is not None else None,
                true_tail=tail,planned_nodes=nodes,state=state.tolist(),version=a.version,
                branch=first_branch(path,spec['network_nodes'],spec['network_edges']))
        if hit or np.linalg.norm(state[:2]-a.goal)<.25:
            return dict(valid=False,reason='collision' if hit else 'early_goal',steps=k+1,path=path,checkpoints=checkpoints,events=events,updates=updates,controls=controls)
    assert sender_bytes<=4000000 and receiver_bytes<=4000000
    return dict(valid=True,path=path,checkpoints=checkpoints,events=events,updates=updates,controls=controls,
        bytes=dict(sender_payload=sender_bytes,receiver_status=status_bytes,receiver_ack=receiver_bytes-status_bytes),
        observation_provenance='evaluator-only true-world indexed sensor samples confined to this branch',
        steps=32,first_branch=first_branch(path,spec['network_nodes'],spec['network_edges']))


def sequence_job(task):
    entry,out=task;root=Path(out);p=root/'sequences'/f"{entry['key']}.json.gz"
    if p.exists():return load(p)
    torch.set_num_threads(1);start=time.perf_counter();s,x,r,rx,spec,tick=load_snapshot(entry);before=digest((s,r))
    sender=copy.deepcopy(s);receiver=enable(copy.deepcopy(r));sender.now=receiver.now=max(s.now,r.now)
    status=advertise(receiver,rx,sender.agent_id);wire=status.wire();decoded=SequenceAdvertisement.decode(wire)
    decoded.validate(sender.agent_id,receiver.agent_id,sender.now)
    assert asdict(status)==asdict(decoded)
    packets,work=candidates(sender,x,receiver.agent_id)
    rotating=copy.deepcopy(sender);rotation=[]
    for _ in range(len(sender.stages)):
        packet=rotating.export(x,receiver.agent_id)
        if packet and packet.stage not in rotation:rotation.append(packet.stage)
    coverage,_=novel_payload(copy.deepcopy(sender),x,receiver.agent_id,route_advertisement(receiver,rx,sender.agent_id))
    prediction=rank_sequences(sender,x,receiver.agent_id,decoded,packets,rotation,coverage.stage if coverage else None)
    branches=[dict(stages=[],actual=continuation(receiver,rx,spec,packets,[],0))]
    for row in prediction['orders']:
        branches.append(dict(stages=row['stages'],actual=continuation(receiver,rx,spec,packets,row['stages'],len(wire))))
    assert digest((s,r))==before
    result=dict(entry,scene=spec['scene'],seed=spec['seed'],status_bytes=len(wire),prediction=prediction,branches=branches,
        state_isolation=True,export_work=work,seconds=time.perf_counter()-start,
        endpoint='matched 32-step receiver continuations, not navigation episodes or new independent trials')
    dump(p,result);return result


def suite(out,workers=4):
    m=freeze(out);root=Path(out)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        baseline=list(pool.map(run_episode,[(s,'legacy',str(root)) for s in m['specs']]))
    dump(root/'legacy_parity.json',baseline)
    if not all(r['parity']['passed'] and r['parity']['bitwise_serialized_equal'] for r in baseline):raise RuntimeError('legacy parity failed')
    correction(root)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        instant=list(pool.map(instant_job,[(e,str(root)) for e in m['snapshot_entries']]))
    dump(root/'instant_rows.json',instant)
    if not all(r['production_parity'] and r['state_isolation'] for r in instant):raise RuntimeError('additive routing/advertisement parity failed')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        seq=list(pool.map(sequence_job,[(e,str(root)) for e in m['sequence_entries']]))
    dump(root/'sequence_rows.json.gz',seq)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        episodes=list(pool.map(run_episode,[(s,a,str(root)) for s in m['specs'] for a in ('rotation','coverage')]))
    dump(root/'episode_rows.json',episodes)
    return report(root)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(out);instant=load(root/'instant_rows.json');sequences=load(root/'sequence_rows.json.gz');episodes=load(root/'episode_rows.json')
    candidates_all=[b for r in instant for b in r['branches'][1:]]
    false_positive=[dict(key=r['key'],time=r['time'],stage=b['stage'],predicted=b['predicted_benefit'],actual=b['benefit']) for r in instant for b in r['branches'][1:] if b['predicted_benefit'] is not None and b['predicted_benefit']>1e-10 and b['benefit']< -1e-10]
    paired=[]
    for row in sequences:
        by_order={tuple(b['stages']):b['actual'] for b in row['branches']};withhold=by_order[()]
        for h in (8,16,32):
            key=str(h);valid=[b for b in by_order.values() if key in b['checkpoints'] and b['checkpoints'][key]['primary_plus_nominal_true_tail'] is not None]
            best=min((b['checkpoints'][key]['primary_plus_nominal_true_tail'] for b in valid),default=None)
            for name,order in row['prediction']['choices'].items():
                b=by_order[tuple(order)];cp=b['checkpoints'].get(key);wp=withhold['checkpoints'].get(key)
                pred=next(v for v in row['prediction']['orders'] if v['stages']==order)['prediction']
                pp=pred.get('checkpoints',{}).get(key);pb=row['prediction']['baseline'].get('checkpoints',{}).get(key)
                available=cp is not None and wp is not None and cp['primary_plus_nominal_true_tail'] is not None
                actual=wp['primary_plus_nominal_true_tail']-cp['primary_plus_nominal_true_tail'] if available else None
                forecast=pb['reference_cost_with_tail']-pp['reference_cost_with_tail'] if pp and pb else None
                paired.append(dict(key=row['key'],scene=row['scene'],seed=row['seed'],horizon=h,arm=name,stages=order,
                    available=available,actual_hybrid_benefit=actual,forecast_spatial_benefit=forecast,
                    incurred_primary_difference=cp['incurred_primary']-wp['incurred_primary'] if available else None,
                    selection_regret=cp['primary_plus_nominal_true_tail']-best if available else None,
                    predicted_branch=pred.get('first_branch'),actual_branch=b.get('first_branch'),
                    status_bytes=row['status_bytes'],actual_bytes=b.get('bytes')))
    dump(root/'paired.json',paired);dump(root/'false_positive_packets.json',false_positive)
    metrics={}
    for name in ('rotation','coverage_first','sequence'):
        rr=[r for r in paired if r['horizon']==32 and r['arm']==name and r['available']]
        metrics[name]=dict(available=len(rr),mean_selection_regret=float(np.mean([r['selection_regret'] for r in rr])) if rr else None,
            mean_hybrid_benefit=float(np.mean([r['actual_hybrid_benefit'] for r in rr])) if rr else None,
            beneficial=sum(r['actual_hybrid_benefit']>1e-10 for r in rr),harmful=sum(r['actual_hybrid_benefit']< -1e-10 for r in rr))
    seq_cases=[r for r in paired if r['horizon']==32 and r['arm']=='sequence' and r['available']]
    gate=bool(len(seq_cases)==8 and len({r['scene'] for r in seq_cases if r['actual_hybrid_benefit']>1e-10})>=2 and all(metrics['sequence']['mean_selection_regret']<metrics[n]['mean_selection_regret']-1e-10 for n in ('rotation','coverage_first')))
    failures=[r for r in episodes if r['goals']!=2 or r['collisions'] or r['timeouts']]
    incomplete=[dict(key=r['key'],stages=b['stages'],reason=b['actual'].get('reason')) for r in sequences for b in r['branches'] if not b['actual']['valid']]
    summary=dict(legacy_bitwise_reproductions=len(load(root/'legacy_parity.json')),instant_snapshots=len(instant),instant_packet_candidates=len(candidates_all),
        instant_route_agreement=sum(b['prediction_agrees'] for b in candidates_all),instant_false_positive_packets=len(false_positive),
        sequence_snapshots=len(sequences),sequence_true_continuations=sum(len(r['branches']) for r in sequences),sequence_metrics=metrics,
        sequence_gate=gate,live_sequence_navigation_episodes=0,additive_navigation_episodes=len(episodes),
        goals=sum(r['goals'] for r in episodes),collisions=sum(r['collisions'] for r in episodes),timeouts=sum(r['timeouts'] for r in episodes),
        failures=len(failures),incomplete_continuations=len(incomplete),status_bytes=[r['status_bytes'] for r in sequences],
        max_continuation_update_flow=max(u['flow'] for r in sequences for b in r['branches'] for u in b['actual']['updates']),
        diagnostic_seconds=sum(r['seconds'] for r in sequences),
        caveats=['four development hazards and two noise-only seeds','states generated under legacy rotation','frozen packet contents; no future sender sensing',
            'nominal graph prediction omits receiver dynamics/future sensing/field fitting','hybrid tail metric is not full mission cost','fresh diagnostic-window bytes are not remaining archived mission budget'])
    dump(root/'summary.json',summary);dump(root/'failure_inventory.json',dict(episodes=failures,incomplete_continuations=incomplete,adverse_predictions=false_positive))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    vv=[r for r in paired if r['horizon']==32 and r['available'] and r['forecast_spatial_benefit'] is not None]
    for name in metrics:
        rr=[r for r in vv if r['arm']==name];axes[0].scatter([r['forecast_spatial_benefit'] for r in rr],[r['actual_hybrid_benefit'] for r in rr],label=name)
    axes[0].axhline(0,color='gray');axes[0].axvline(0,color='gray');axes[0].set(xlabel='Nominal sequence forecast',ylabel='True continuation + nominal true-tail benefit');axes[0].legend(fontsize=7)
    axes[1].bar(list(metrics),[v['mean_selection_regret'] or 0 for v in metrics.values()]);axes[1].set(ylabel='Mean 32-step hybrid selection regret')
    for name in ('rotation','coverage'):
        rr=[r for r in episodes if r['condition']==name];axes[2].plot(range(len(rr)),[r['carrier_primary_change'] for r in rr],'o',label=name)
    axes[2].axhline(0,color='gray');axes[2].set(xlabel='Scene/seed (see episode_rows.json)',ylabel='Additive − legacy full mission primary');axes[2].legend()
    fig.tight_layout();fig.savefig(root/'figures/results.png');plt.close(fig)
    lines=['# Additive route integration and decision-window packet sequences','',
        f"Legacy operational reproduction: {summary['legacy_bitwise_reproductions']}/8 bitwise. Additive snapshot routing agreement: {summary['instant_route_agreement']}/{len(candidates_all)} candidate packets. False-positive instantaneous packet benefits: {len(false_positive)}.",'',
        'The common mode uses one fixed nonnegative edge density and its exact partial-cell integrals. An off-axis connector explicitly follows a checked projection spur and then the canonical edge. Execution, retained-stage replay and graph continuation use this mode through the copied navigation context. Legacy remains the default.', '',
        f"Sequence audits: {len(sequences)} preserved scout-to-carrier states and {summary['sequence_true_continuations']} matched 32-step continuations. Sequence progression gate: **{'PASS' if gate else 'FAIL'}**. Live sequence-selector navigation episodes: 0.", '',
        '| Initial ordering | Available / 8 | Mean hybrid selection regret | Mean hybrid benefit vs withhold | Beneficial / harmful |','|---|---:|---:|---:|---:|']
    lines += [f"| {n} | {v['available']} | {v['mean_selection_regret']} | {v['mean_hybrid_benefit']} | {v['beneficial']} / {v['harmful']} |" for n,v in metrics.items()]
    lines += ['', 'The runtime forecast is a frozen-evidence graph model at 1 m/s. It has no future local sensing, no physical pH rollout, and no receiver fitting. Its objective includes nominal reference risk along the prefix and the remaining route. The evaluator separately runs the actual receiver learner and physical simulator with identical indexed noise, updates and frozen packet schedules. Incurred time+weighted-risk is reported separately from the hybrid prefix+nominal-tail endpoint; the latter cannot establish a full navigation win.', '',
        f"Additive navigation smoke/development reproductions: {len(episodes)} episodes; goals {summary['goals']}/{2*len(episodes)}, collisions {summary['collisions']}, timeouts {summary['timeouts']}. These test the common integration correction with rotation and repaired coverage at the original four-million-byte allowance. All original conditions and legacy counterparts are preserved.", '',
        'The old v1 diagnostic omitted background risk 0.02 in its evaluator helper. v1_endpoint_correction.json reprices its saved selected routes using NetworkWorld.risk_truth. Operational mission logging was already correct. Old v1 files remain untouched.', '',
        'Full standalone status, payload and acknowledgment bytes, failed/incomplete branches, model versions and controls are retained. Sequence budgets are fresh per-agent diagnostic-window allowances. No cached-advertisement savings, equal-computation claim, adaptive-rate claim, independent-seed generalization claim, or access to future sender evidence is assumed.', '',
        '[Results plot](figures/results.png). Full records: paired.json, sequence_rows.json.gz, instant_rows.json, episode_rows.json, failure_inventory.json and manifest.json.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return summary


def render(out):
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(out);row=load(root/'sequences/double_mild_86100.json.gz');choices=row['prediction']['choices'];by_order={tuple(b['stages']):b['actual'] for b in row['branches']};images=[];index=[]
    for step in range(33):
        fig,axes=plt.subplots(1,3,figsize=(13,4),dpi=90)
        for ax,(name,order) in zip(axes,choices.items()):
            b=by_order[tuple(order)];path=np.asarray(b['path']);prediction=next(r['prediction'] for r in row['prediction']['orders'] if r['stages']==order);virtual=np.asarray(prediction['path'])
            for i,j in SCENE_EDGES():
                from .crossing_demo import NODES
                nodes=np.asarray(NODES);ax.plot(*nodes[[i,j]].T,color='.8')
            k=min(step,len(path)-1);v=min(step,len(virtual)-1)
            ax.plot(*path[:k+1,:2].T,'-',color='#D55E00',label='actual receiver');ax.scatter(*path[k,:2],color='#D55E00')
            ax.plot(*virtual[:v+1].T,'--',color='#0072B2',label='nominal graph forecast')
            ax.set(title=f'{name}: stages {order}',xlim=(-.5,4.5),ylim=(2.8,7.9),aspect='equal');ax.legend(fontsize=7)
            for e in b['events']:
                if e['step']<=step:ax.text(.02,.96-.08*(e['step']//8),f"k={e['step']}: stage {e['stage']} ({e['event']['reason']})",transform=ax.transAxes,fontsize=7)
        fig.suptitle(f'Double hazard, seed86100 · {step} steps · actual states versus frozen nominal predictions');fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        index.append(dict(step=step,actual_indices={n:min(step,len(by_order[tuple(o)]['path'])-1) for n,o in choices.items()}))
    images[0].save(root/'figures/double_mild_86100.gif',save_all=True,append_images=images[1:],duration=180,loop=0)
    dump(root/'animation_frames.json',index)


def SCENE_EDGES():
    from .crossing_demo import EDGES
    return EDGES


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','suite','report','render']);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    print({'freeze':lambda:freeze(a.out),'suite':lambda:suite(a.out,a.workers),'report':lambda:report(a.out),'render':lambda:render(a.out)}[a.command]())
