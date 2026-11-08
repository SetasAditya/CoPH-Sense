"""Frozen saved-snapshot diagnosis for opt-in communication repairs.

No episodes are regenerated. All measurements and packets come from preserved
complementary artifacts. Temporal packet changes are evaluator-only faults.
"""
import argparse,copy,gzip,hashlib,json,pickle,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from .communication_repairs import *
from .evaluation import load,dump
from .receiver_corrections import load_saved_snapshot
from .reconstruction_diagnostics import ranking,field_queries
from .route_field import sync
from .attribution_audits import digest
from .terrain_warning import terrain_values
from .cooperative import bootstrap_field,field_vector
from .model import potential

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/complementary_reconstruction_v1'
FAULTS=('no_packet','clean','duplicate','echo','stale','reordered','biased','correlated')
MODES=('legacy','corrected')
DEPENDENCIES=('communication_repairs.py','communication_repair_suite.py','model.py','learner.py','cooperative.py','maze_learning.py',
 'navigation.py','route_field.py','stage_prediction.py','receiver_fitting.py','diagnostics.py','reconstruction_diagnostics.py',
 'records.py','shapes.py','terrain_warning.py','sharing.py','maze_family.py','maze_geometry.py','receiver_corrections.py',
 'replay_budget.py','decision_trace.py','attribution_audits.py','decision_audits.py','world.py','scenarios.py','cooperative_evaluation.py','evaluation.py','route_decision.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    result=[]
    for path in sorted((SOURCE/'snapshots').glob('*.pkl.gz')):
        parts=path.name[:-7].split('_');case,seed=parts[0],int(parts[1])
        meta=dict(case=case,seed=seed,group='acquisition' if parts[2]=='acquisition' else 'navigation')
        if meta['group']=='navigation':meta.update(method='_'.join(parts[2:-1]),index=int(parts[-1]))
        result.append(dict(path=str(path.resolve()),sha256=sha(path),meta=meta))
    return result


def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=dict(schema=1,source=str(SOURCE),inputs=inputs(),sources={n:sha(ROOT/'coph_stagewise'/n) for n in DEPENDENCIES},
        fault_modes=list(MODES),faults=list(FAULTS),fault_bias=.1,corrected=dict(fit='precision_bounded',solver='active_set',search='diagnostic',validation='relative_local'),
        consistency_phase='only after corrected fault audit shows residual harm; separately frozen if needed',
        fair_payloads=dict(locations='identical actual source-reading coordinates from saved acquisition packets',
            evidence='identical selected raw measurement batch and source IDs; field reconstructed afresh from only this batch using original local ridge model',
            field='batch-fitted values and gradients',raw='actual batch values at same coordinates; gradients unknown',costmap='same batch-fitted scalar samples, gradients unknown; no spatial-grid advantage'),
        payload_selection='controlled retained-entry route requests on actual retained snapshots; fixed opportunity, not adaptive rates',
        codec='lossless canonical JSON + zlib9 in versioned binary frame; exact float64 roundtrip',
        execution=dict(device='cpu',dtype='float64',threads=1,python=sys.version,numpy=np.__version__,torch=torch.__version__),
        no_new_episodes=True,limitations=['old current_campaign fault contexts were not pickled; preserved original complementary snapshots used instead',
        'snapshot fault, codec and selection mechanisms do not establish navigation improvement'])
    p=out/'manifest.json'
    if p.exists():
        if load(p)!=m:raise ValueError('frozen protocol/input/source changed')
        return m
    with p.open('x') as f:json.dump(m,f,indent=2)
    (out/'source_archive').mkdir()
    for name in DEPENDENCIES:(out/'source_archive'/name).write_bytes((ROOT/'coph_stagewise'/name).read_bytes())
    (out/'fault_rows').mkdir();return m


def snapshots(entry):
    obj=load_saved_snapshot(entry['path']);meta=entry['meta']
    if meta['group']=='navigation':return [(obj,meta)]
    result=[]
    for sender in (0,1):
        a=copy.deepcopy(obj['agents'][sender]);packet=a.export(obj['states'][sender],str(1-sender))
        result.append(((obj['agents'][1-sender],obj['states'][1-sender],packet,obj['worlds'][1-sender],8),dict(meta,sender=sender)))
    return result


def packets_for(packet,receiver,fault):
    if fault=='no_packet':return []
    if fault=='clean':return [packet]
    if fault=='duplicate':return [packet,packet]
    if fault=='echo':return [replace(packet,sources=(receiver.agent_id+':echo',))]
    if fault=='stale':return [replace(packet,expires=receiver.now-.01)]
    if fault=='reordered':return [packet,replace(packet,identity=packet.identity+':old',timestamp=packet.timestamp-.01)]
    if fault=='biased':return [replace(packet,identity=packet.identity+':biased',anchors=tuple((*a[:2],a[2]+.1,*a[3:]) for a in packet.anchors))]
    if fault=='correlated':return [packet,replace(packet,identity=packet.identity+':correlated',timestamp=packet.timestamp+1e-10)]
    raise ValueError(fault)


def configure(a,mode):
    a.mode='evidence';a.fit_received=True
    if hasattr(a,'receiver_diagnostic_variant'):del a.receiver_diagnostic_variant
    a.receiver_fit_mode='legacy' if mode=='legacy' else 'precision_bounded'
    a.receiver_validation_mode='legacy' if mode=='legacy' else 'relative_local'
    a.receiver_solver_mode='projected' if mode=='legacy' else 'active_set'
    a.receiver_search_mode='legacy' if mode=='legacy' else 'diagnostic'
    return a


def fault_snapshot(snapshot,meta,include_gate=False):
    agent,x,packet,world,tick=snapshot;original=digest((agent,x,world));measured=digest([(s.readings,s.transitions) for s in agent.stages]);physical=agent.current_state
    quad=ranking(agent,x,details=True)['quadrature'];points=np.array([q for e in quad for q in e['points']]);truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points)))
    initial,_=field_queries(agent,points);mask=np.isfinite(initial).all(1);rows=[]
    for mode in (('corrected_gated',) if include_gate else MODES):
        for fault in FAULTS:
            a=configure(copy.deepcopy(agent),'corrected' if mode=='corrected_gated' else mode);start=time.perf_counter();events=[];checks=[]
            for p in packets_for(packet,a,fault):
                if mode=='corrected_gated':
                    trial=copy.deepcopy(a);e=trial.assimilate(p)
                    if e['reason']=='eligible':
                        check=consistency_check(a,p);checks.append(check)
                        if not check['accepted']:
                            events.append(dict(reason='local_consistency_quarantine',packet=p.identity));continue
                events.append(a.assimilate(p))
            d=a.update(x.copy());pred,metadata=field_queries(a,points);valid=mask&np.isfinite(pred).all(1)
            delta=pred[valid]-truth[valid]
            proposal=copy.deepcopy(a)
            if getattr(a,'reference',None) is not None:
                for s,w in zip(proposal.stages,a.reference):s.weights=w.detach().clone()
                sync(proposal,publish=True)
            proposed,pm=field_queries(proposal,points);pv=mask&np.isfinite(proposed).all(1)
            u,_=a.action(x.copy());fit=getattr(a,'receiver_fit_diagnostic',None)
            isolated=digest((agent,x,world))==original and digest([(s.readings,s.transitions) for s in a.stages])==measured and a.current_state==physical
            assert isolated and d['counterfactual_steps']<=1024
            rows.append(dict(meta,packet=packet.identity,receiver=agent.agent_id,time=agent.now,mode=mode,fault=fault,events=events,consistency=checks,
                accepted=d['accepted'],reason=d['reason'],version=a.version,fit=fit,control=u.tolist(),
                value_mae=float(np.abs(delta[:,0]).mean()) if len(delta) else None,gradient_mae=float(np.linalg.norm(delta[:,1:],axis=1).mean()) if len(delta) else None,
                proposed_value_mae=float(np.abs(proposed[pv,0]-truth[pv,0]).mean()) if pv.any() else None,
                supported=int(valid.sum()),queries=len(points),received_count=len(a.received),message_fit_work=a.message_fit_work-agent.message_fit_work,
                operational_flow_work=d['counterfactual_steps'],isolated=isolated,seconds=time.perf_counter()-start))
    return rows


def fault_job(job):
    root,entry,gated=job;torch.set_num_threads(1)
    if sha(entry['path'])!=entry['sha256']:raise ValueError('snapshot changed')
    path=Path(root)/'fault_rows'/f"{Path(entry['path']).name[:-7]}{'_gated' if gated else ''}.json.gz"
    if path.exists():return dict(path=path.name,rows=len(load(path)),sha256=sha(path))
    rows=[r for snapshot,meta in snapshots(entry) for r in fault_snapshot(snapshot,meta,gated)];dump(path,rows)
    return dict(path=path.name,rows=len(rows),sha256=sha(path))


def fault_suite(root,workers=8,gated=False):
    root=Path(root);m=freeze(root)
    if gated:
        previous=summary(root)
        harmed=previous['corrected_biased_worse_than_clean']
        if not harmed:raise ValueError('No residual corrected bias harm; consistency phase not justified')
        p=root/'consistency_phase_protocol.json';v=dict(threshold=4.,scale='sqrt(local reserved MSE + .003^2)',input_manifest=sha(root/'manifest.json'),
            trigger=dict(corrected_biased_worse_than_clean=harmed,initial_results={p.name:sha(p) for p in sorted((root/'fault_rows').glob('*.json.gz')) if not p.name.endswith('_gated.json.gz')}),
            action='quarantine whole message before route/field insertion; local update still runs',confidence='uncalibrated local heuristic')
        if p.exists():
            if load(p)!=v:raise ValueError('Consistency phase input changed')
        else:
            with p.open('x') as f:json.dump(v,f,indent=2)
    completed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fault_job,(str(root),entry,gated)) for entry in m['inputs']]):
            completed.append(future.result());dump(root/('gated_completion.json' if gated else 'fault_completion.json'),completed)
            print(completed[-1],flush=True)
    return summary(root)


def matched_payloads(root):
    root=Path(root);m=freeze(root);rows=[];selection=[];transport=[];torch.set_num_threads(1)
    for entry in m['inputs']:
        if entry['meta']['group']!='acquisition':continue
        obj=load_saved_snapshot(entry['path'])
        for sender in (0,1):
            s=copy.deepcopy(obj['agents'][sender]);a=copy.deepcopy(obj['agents'][1-sender]);x=obj['states'][sender];rx=obj['states'][1-sender]
            packet=s.export(x,str(1-sender));readings={r.q:r for st in s.stages for r in st.readings};points=[tuple(v[:2]) for v in packet.anchors]
            if not all(q in readings for q in points):
                rows.append(dict(entry['meta'],sender=sender,unavailable='packet anchor not an exact raw reading location'));continue
            batch=[readings[q] for q in points];stage=copy.deepcopy(next(st for st in s.stages if st.id==packet.stage));stage.readings=batch
            mean,ensemble,_=bootstrap_field(stage,s.cfg,s.communication,s.now,seed=int(entry['meta']['seed']*2+sender))
            vector=field_vector(mean,stage.centers,points,s.cfg.rbf_width).reshape(-1,3)
            vectors=np.array([field_vector(w,stage.centers,points,s.cfg.rbf_width) for w in ensemble])
            covariance=np.cov(vectors,rowvar=False)+np.eye(3*len(points))*s.communication.covariance_floor
            H=.5*float(np.sum(np.asarray(x[2:])**2))+float(potential(tensor(x[:2]),tensor(s.control_target(stage)),mean,stage.centers,s.cfg.rbf_width,s.policy,s.control_obstacles(stage),s.cfg))
            packet=replace(packet,anchors=tuple(tuple(q)+tuple(v) for q,v in zip(points,vector)),covariance=tuple(map(tuple,covariance)),
                           sources=tuple(sorted(r.uid for r in batch)),support_points=tuple(points),H=H,
                           uncertainty='controlled identical measurement batch; original ridge reconstruction; not an operational model')
            cov=np.asarray(packet.covariance).copy()
            for k in range(len(points)):
                for j in (1,2):idx=3*k+j;cov[idx,:]=0;cov[:,idx]=0;cov[idx,idx]=1e4
            samples=dict(field=packet,costmap=replace(packet,anchors=tuple((*v[:3],0.,0.) for v in packet.anchors),covariance=tuple(map(tuple,cov))),
                raw=replace(packet,anchors=tuple((*v[:2],float(readings[tuple(v[:2])].value),0.,0.) for v in packet.anchors),covariance=tuple(map(tuple,cov))))
            query=np.array([np.asarray(q)+off for q in points for off in ((0.,0.),(.15,0.),(-.15,0.),(0.,.15),(0.,-.15))]);truth=obj['worlds'][sender].risk_truth(query)
            for name,p in samples.items():
                t=time.perf_counter();wire=encode_packet(p);encoding=time.perf_counter()-t;t=time.perf_counter();decoded=decode_packet(wire);decoding=time.perf_counter()-t
                assert asdict(decoded)==asdict(p)
                values=terrain_values(query,list(p.anchors),unknown=.02)
                rows.append(dict(entry['meta'],sender=sender,representation=name,anchors=len(p.anchors),points=points,source_ids=list(p.sources),
                    legacy_bytes=p.bytes(),compact_bytes=len(wire),encoding_seconds=encoding,decoding_seconds=decoding,roundtrip_exact=True,
                    route_query_mae=float(np.abs(values-truth).mean()),queries=len(query),
                    anchor_value_mae=float(np.abs(np.array([v[2] for v in p.anchors])-obj['worlds'][sender].risk_truth(np.array(points))).mean())))
            for mode in ('rotation','receiver_relevant'):
                opts=CommunicationRepairOptions('compact_v1',mode);link=FixedRateRepairLink(opts)
                ss=copy.deepcopy(s);aa=copy.deepcopy(a);state=rx.copy();version=aa.version
                sent=link.offer(ss,x,aa,state,8);received=link.deliver_due(aa,state);again=link.deliver_due(aa,state)
                transport.append(dict(entry['meta'],sender=sender,mode=mode,sent=sent,received=received,accounting=link.accounting,
                    exactly_once=again['update'] is None,version_before=version,version_after=aa.version))
    for entry in m['inputs']:
        if entry['meta']['group']!='navigation':continue
        agent,x,packet,world,tick=load_saved_snapshot(entry['path'])
        if len(agent.stages)<2:continue
        # Controlled request, not a recovered peer route: ask about the oldest
        # retained entry using only measured support coordinates, no truth.
        route=tuple(r.q for r in agent.stages[0].readings if int(r.uid.split(':')[-1])%5)[:16]
        ad=PayloadAdvertisement(1,packet.sender,agent.agent_id,agent.now,agent.now+.96,0,route,(),())
        for mode in ('rotation','receiver_relevant'):
            a=copy.deepcopy(agent);before=digest(a);p,meta=select_payload(a,x,packet.sender,ad,mode)
            selection.append(dict(entry['meta'],request_kind='controlled oldest-retained measured route request',packet_stage=p.stage if p else None,
                 payload_bytes=len(encode_packet(p)) if p else 0,**meta))
    dump(root/'matched_payload_rows.json',rows);dump(root/'fixed_rate_transport_rows.json',transport);dump(root/'selection_rows.json',selection)
    return summary(root)


def summary(root):
    root=Path(root);rows=[r for p in (root/'fault_rows').glob('*.json.gz') for r in load(p)];groups={}
    for row in rows:
        key=(row['case'],row['seed'],row['group'],row.get('method'),row.get('index'),row['packet'],row['receiver']);groups.setdefault(key,{})[(row['mode'],row['fault'])]=row
    report=[];harm=0
    for mode in ('legacy','corrected','corrected_gated'):
        for fault in FAULTS:
            rs=[r for r in rows if r['mode']==mode and r['fault']==fault]
            if not rs:continue
            deltas=[]
            for group in groups.values():
                r=group.get((mode,fault));b=group.get((mode,'no_packet'))
                if r and b and r['value_mae'] is not None and b['value_mae'] is not None:deltas.append(r['value_mae']-b['value_mae'])
            report.append(dict(mode=mode,fault=fault,audits=len(rs),accepted=sum(r['accepted'] for r in rs),mean_delta_vs_no_packet=float(np.mean(deltas)) if deltas else None,
                worsening_vs_no_packet=sum(d>1e-12 for d in deltas),eligible_updates=sum(bool(r.get('fit') and r['fit'].get('eligible_rows')) for r in rs),
                quarantined=sum(any(e['reason']=='local_consistency_quarantine' for e in r['events']) for r in rs)))
    for group in groups.values():
        b=group.get(('corrected','biased'));c=group.get(('corrected','clean'))
        if b and c and b['value_mae'] is not None and c['value_mae'] is not None and b['value_mae']>c['value_mae']+1e-12:harm+=1
    payload=load(root/'matched_payload_rows.json') if (root/'matched_payload_rows.json').exists() else []
    s=dict(fault_audits=len(rows),input_snapshot_audits=len(groups),corrected_biased_worse_than_clean=harm,fault_summary=report,
        all_isolated=all(r['isolated'] for r in rows),matched_payloads=len(payload),navigation_claim='none; saved-state mechanisms and estimation only')
    dump(root/'summary.json',s)
    if rows or payload:render_report(root,s,payload,rows)
    return s


def render_report(root,s,payload,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=root/'figures';figs.mkdir(exist_ok=True)
    if payload:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name in ('field','raw','costmap'):
            p=[r for r in payload if r.get('representation')==name]
            axes[0].scatter([r['legacy_bytes'] for r in p],[r['compact_bytes'] for r in p],label=name)
            axes[1].scatter([r['compact_bytes'] for r in p],[r['route_query_mae'] for r in p],label=name)
        axes[0].set(xlabel='Legacy full-envelope bytes',ylabel='Lossless compact bytes');axes[1].set(xlabel='Compact bytes, identical locations',ylabel='Evaluator-only route-query MAE')
        for ax in axes:ax.legend()
        fig.tight_layout();fig.savefig(figs/'matched_evidence_bytes_error.png');plt.close(fig)
    if rows:
        fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(FAULTS));width=.25
        for i,mode in enumerate(('legacy','corrected','corrected_gated')):
            values=[next((r['mean_delta_vs_no_packet'] for r in s['fault_summary'] if r['mode']==mode and r['fault']==fault),np.nan) for fault in FAULTS]
            ax.bar(x+(i-1)*width,values,width,label=mode)
        ax.set_xticks(x,FAULTS,rotation=25);ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Committed MAE − withhold-current-packet MAE');ax.legend();fig.tight_layout();fig.savefig(figs/'corrected_fault_comparison.png');plt.close(fig)
    lines=['# Communication mechanism repairs','',f"Saved-state fault audits: {s['fault_audits']}; distinct receiver snapshots: {s['input_snapshot_audits']}. All isolation checks: {s['all_isolated']}.",
        '', 'No old fault acquisition episode was regenerated: that campaign saved rows, not complete snapshots. This suite uses the original archived complementary acquisitions and navigation snapshots with frozen input hashes.',
        '', 'The no-packet branch withholds the current candidate while retaining the same prior received store and local measured history. It is not a received-fitting-disabled baseline.',
        '', '| Receiver | Fault | Audits | Accepted models | Mean MAE change vs withhold | Worse vs withhold | Quarantined |','|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['mode']} | {r['fault']} | {r['audits']} | {r['accepted']} | {r['mean_delta_vs_no_packet'] if r['mean_delta_vs_no_packet'] is not None else 'unavailable'} | {r['worsening_vs_no_packet']} | {r['quarantined']} |" for r in s['fault_summary']]
    lines+=['',f"Corrected biased fits worse than the identical clean branch: {s['corrected_biased_worse_than_clean']} snapshots. A consistency phase, if present, is a separately frozen uncalibrated local residual heuristic; it is not a Bayesian confidence guarantee.",
        '', 'Compact coding is exact full-message JSON+zlib in a versioned binary frame; covariance/support/provenance are preserved. Fair field/raw/costmap comparisons use the identical selected actual measurement batch, coordinates, support and evidence IDs. The controlled field is refitted using only that batch with the original ridge model. This is a matched-evidence codec/model benchmark, not the operational full-history field. Costmap means sampled graded values at those same coordinates, not a denser raster. Raw and costmap gradients are explicitly unknown.',
        '', 'Fixed-rate transport demonstration decodes compact packets through normal packet validation and the ordinary receiver update. Status, payload and acknowledgment bytes are charged independently; repeated delivery is empty. It does not establish a mission benefit.',
        '', 'Multi-stage payload selection uses controlled advertised requests derived from actual retained measured coordinates. It is a mechanism check, not a recovered peer route or a navigation outcome. No adaptive rate selection is implemented.',
        '', 'All options remain opt-in; legacy defaults and previous results are preserved. New navigation benefit remains untested.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','payloads','faults','gated','report']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args(argv)
    if args.command=='freeze':freeze(args.out)
    elif args.command=='payloads':matched_payloads(args.out)
    elif args.command in ('faults','gated'):fault_suite(args.out,args.workers,args.command=='gated')
    else:summary(args.out)


if __name__=='__main__':main()
