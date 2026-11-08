"""Small opt-in actual-navigation pilot for fixed-rate payload selection.

Shared-clock ordering follows staggered_demo; transport is FixedRateRepairLink.
This is a separate pilot, not a replacement for preserved navigation results.
"""
import argparse,copy,hashlib,heapq,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .communication_repairs import CommunicationRepairOptions,FixedRateRepairLink,decode_packet
from .cooperative import CommunicationConfig
from .sharing import LEDGER_MODES
from .crossing_demo import configuration
from .maze_family import NetworkWorld
from .records import Transition
from .staggered_demo import sense
from .evaluation import dump,load
from .communication_repair_suite import DEPENDENCIES

ROOT=Path(__file__).resolve().parents[1]
ARMS=('disabled','one_way_01','one_way_10','reciprocal_fixed')
SELECTORS=('rotation','receiver_relevant')


def factory(spec):
    network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges'])))
    def make(cfg,x,goal,agent_id,communication):
        options={}
        if 'risk_weights' in spec:options['risk_weight']=spec['risk_weights'][int(agent_id)]
        if 'unknown_risk' in spec:options['unknown_risk']=spec['unknown_risk']
        return LEDGER_MODES['field'](cfg,x,goal,mode='evidence',agent_id=agent_id,communication=communication,network=network,**options)
    return make


def execute(spec,arm,selector,*,byte_budget=4000000,accounting_mode='corrected'):
    # Legacy accounting exists only to reproduce the superseded evaluation.
    # Neither mode changes the learner's declared task objective.
    if accounting_mode not in ('corrected','legacy'):raise ValueError('accounting_mode')
    torch.set_num_threads(1);cfg=configuration(spec['steps']);comm=CommunicationConfig(byte_budget=byte_budget,ttl=300.)
    senders={'disabled':(),'one_way_01':(0,),'one_way_10':(1,),'reciprocal_fixed':(0,1)}[arm]
    link=FixedRateRepairLink(CommunicationRepairOptions('compact_v1',selector),byte_budget=comm.byte_budget)
    x=[np.array(z,float) for z in spec['initials']];agents=[];worlds=[];make=factory(spec)
    for i in (0,1):
        weight=spec.get('risk_weights',(1.,1.))[i] if accounting_mode=='corrected' else 1.
        w=NetworkWorld(dict(spec,initial=spec['initials'][i],goal=spec['goals'][i],risk_weight=weight),cfg);w.seed=spec['seed']*2+i;worlds.append(w)
        agents.append(make(cfg,x[i],spec['goals'][i],agent_id=str(i),communication=comm))
    starts=spec['start_seconds'];active=[False]*2;done=[False]*2;success=[False]*2;collision=[False]*2;pending=[None]*2
    records=[[],[]];samples=[[],[]];updates=[[],[]];observations=[[],[]];events=[];packet_details={}
    components=[dict(time=0.,risk=0.,effort=0.,failure=0.) for _ in (0,1)]
    raw_risk=[0.,0.]
    receive_slots=[{int(np.ceil((starts[1-i]+k*cfg.dt-starts[i])/cfg.dt-1e-9)) for k in range(0,cfg.steps,cfg.update_every) if starts[1-i]+k*cfg.dt>=starts[i]-1e-9} for i in (0,1)]
    schedule=[(round(starts[i]*1e6),i,0) for i in (0,1)];heapq.heapify(schedule);dt_us=round(cfg.dt*1e6);began=time.perf_counter();update_seconds=0.
    while schedule:
        stamp=schedule[0][0];now=stamp/1e6;batch=[]
        while schedule and schedule[0][0]==stamp:batch.append(heapq.heappop(schedule))
        for _,i,tick in batch:
            a,w=agents[i],worlds[i];w.time=now
            if pending[i] is None:
                active[i]=True;obs=sense(w,x[i],tick,now);a.observe(obs);observations[i].append(asdict(obs))
            else:
                p=pending[i];after,hit,cost,energy=w.step(x[i],np.array(p['u']))
                exposure=cfg.dt*float(w.risk_truth(after[:2]));raw_risk[i]+=exposure
                parts=dict(time=cfg.dt,risk=4*w.spec['risk_weight']*exposure,effort=cfg.dt*.03*float(np.dot(p['u'],p['u'])))
                for k,v in parts.items():components[i][k]+=v
                records[i].append(dict(p,after=after.tolist(),end_time=now,cost=cost,components=parts,raw_risk=exposure,energy=energy))
                tr=Transition(tick-1,now,tuple(x[i]),tuple(p['u']),tuple(after),p['version']);x[i]=after
                obs=sense(w,x[i],tick,now);a.observe(obs,tr);observations[i].append(asdict(obs))
                collision[i]=bool(hit);success[i]=bool(np.linalg.norm(x[i][:2]-a.goal)<.25);done[i]=bool(hit or success[i] or tick>=cfg.steps)
                if not done[i]:a.advance(x[i])
            samples[i].append(dict(time=now,x=x[i].tolist(),stage=a.stages[-1].id,version=a.version,done=done[i]))
        # Both sensing steps precede any advertisement or export at this time.
        for _,i,tick in batch:
            j=1-i
            if i not in senders or done[i] or done[j] or not active[j] or tick%cfg.update_every:continue
            event=link.offer(agents[i],x[i],agents[j],x[j],tick)
            for key,value in dict(time=now,sender=str(i),receiver=str(j)).items():event.setdefault(key,value)
            event['kind']='sent' if event['reason']=='sent' else 'channel'
            if event['reason']=='sent':
                wire=link.queue[-1][2];packet=decode_packet(wire,'compact_v1');path=agents[j].navigation.get('path',[])
                packet_details[packet.identity]=dict(sender_q=list(packet.q),anchors=packet.anchors)
                if path:
                    distances=np.linalg.norm(np.asarray([z[:2] for z in packet.anchors])[:,None]-np.asarray(path)[None],axis=-1).min(1)
                    event['diagnostic_route_relevant_anchors']=int((distances<=.65).sum());event['diagnostic_anchor_count']=len(packet.anchors)
                else:event.update(diagnostic_route_relevant_anchors=0,diagnostic_anchor_count=len(packet.anchors))
            events.append(event)
        for _,i,tick in batch:
            if done[i]:continue
            a=agents[i];before=a.version;before_route=copy.deepcopy(a.navigation.get('route_nodes',[]))
            receipt=link.deliver_due(a,x[i],commit=False);delivered=receipt['events']
            if tick%cfg.update_every==0 or tick in receive_slots[i] or delivered:
                t=time.perf_counter();d=a.update(x[i]);update_seconds+=time.perf_counter()-t
                assert d['counterfactual_steps']<=1024
                updates[i].append(dict(time=now,tick=tick,diagnostic=d,version_before=before,version_after=a.version,packets=[e['packet'] for e in delivered]))
                for event in delivered:events.append(dict(event,kind='delivered',receiver=i,sender=int(event['sender']),receiver_q=x[i][:2].tolist(),
                    **packet_details[event['packet']],version_before=before,version_after=a.version,model_accepted=d['accepted'],force_change=d['force_change']))
            u,nominal=a.action(x[i]);navigation=copy.deepcopy(a.last_control)
            if delivered:events.append(dict(kind='post_delivery_decision',time=now,receiver=i,route_before=before_route,route_after=a.navigation.get('route_nodes',[]),control=u.tolist()))
            pending[i]=dict(time=now,tick=tick,x=x[i].tolist(),u=u.tolist(),nominal=nominal.tolist(),version=a.version,stage=a.stages[-1].id,navigation=navigation)
            heapq.heappush(schedule,(stamp+dt_us,i,tick+1))
    outcomes=[]
    for i in (0,1):
        components[i]['failure']=0. if success[i] else 40.;pts=np.array([s['x'][:2] for s in samples[i]])
        budget=link.accounting.get(str(i),dict(status=0,payload=0,acknowledgment=0,processing_seconds=0.))
        assert sum(budget[k] for k in ('status','payload','acknowledgment'))<=comm.byte_budget
        assert abs(sum(r['cost'] for r in records[i])+components[i]['failure']-sum(components[i].values()))<1e-8
        outcomes.append(dict(agent=i,success=success[i],collision=collision[i],timeout=not success[i] and not collision[i],
            cost=sum(components[i].values()),components=components[i],raw_risk=raw_risk[i],risk_weight=worlds[i].spec['risk_weight'],
            time_risk_cost=components[i]['time']+components[i]['risk'],path_length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),
            finish_time=samples[i][-1]['time'],goal_gap=float(np.linalg.norm(x[i][:2]-agents[i].goal)),stage_traversal=len({s['stage'] for s in samples[i]})-1,
            bytes=budget,interventions=sum(r['navigation']['reason']!='nominal' for r in records[i]),identification_work=agents[i].identification_steps,
            max_update_work=max(r['diagnostic']['counterfactual_steps'] for r in updates[i])))
    return dict(spec=spec,config=asdict(cfg),communication=asdict(comm),arm=arm,selector=selector,agents=outcomes,trajectories=records,samples=samples,updates=updates,observations=observations,
        events=events,geometry_events=[a.geometry_events for a in agents],wall_seconds=time.perf_counter()-began,update_seconds=update_seconds,pending_packets=len(link.queue),
        independent_agents=True,accounting_mode=accounting_mode,packet_protocol='local-only LedgerExport, compact codec, fixed opportunities, legacy receiver fitting',
        receive_update_slots=[sorted(z) for z in receive_slots])


def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/current_campaign_v1/reciprocity/manifest.json';old=load(source)['protocol'];specs=sorted(old['specs'],key=lambda s:s['seed'])[:2]
    assert [s['seed'] for s in specs]==[89500,89501]
    names=tuple(dict.fromkeys(DEPENDENCIES+('communication_navigation.py','staggered_demo.py')))
    m=dict(specs=specs,source_manifest=str(source),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),arms=list(ARMS),selectors=list(SELECTORS),expected_episodes=16,
        selection='first two by seed, no outcome selection',animation_seed=89501,animation_arm='reciprocal_fixed',
        config={str(s['seed']):asdict(configuration(s['steps'])) for s in specs},communication=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.)),
        sources={n:hashlib.sha256((ROOT/'coph_stagewise'/n).read_bytes()).hexdigest() for n in names},
        assumptions='same surveyed graph, independent plants without contact dynamics; evidence-only local fitting, legacy receiver; compact codec both selectors',
        accounting='per-agent status/payload/ack ceilings; status selection overhead included; actual work need not match',cadence='exports each8 sender steps after sensing; deliveries receiver boundaries; exogenous receive-update slots preserved')
    m=json.loads(json.dumps(m))
    path=out/'manifest.json'
    if path.exists():
        if load(path)!=m:raise ValueError('navigation pilot source/input/config changed')
        return m
    with path.open('x') as f:json.dump(m,f,indent=2)
    (out/'episodes').mkdir();(out/'source_archive').mkdir()
    for name in names:(out/'source_archive'/name).write_bytes((ROOT/'coph_stagewise'/name).read_bytes())
    return m


def job(arg):
    root,spec,arm,selector=arg;root=Path(root);key=f"{spec['seed']}_{arm}_{selector}";path=root/'episodes'/f'{key}.json.gz'
    r=load(path) if path.exists() else execute(spec,arm,selector)
    if not path.exists():dump(path,r)
    sent=[e for e in r['events'] if e.get('reason')=='sent'];delivered=[e for e in r['events'] if e.get('kind')=='delivered']
    return dict(key=key,seed=spec['seed'],arm=arm,selector=selector,goals=sum(a['success'] for a in r['agents']),collisions=sum(a['collision'] for a in r['agents']),timeouts=sum(a['timeout'] for a in r['agents']),
        team_cost=sum(a['cost'] for a in r['agents']),team_path=sum(a['path_length'] for a in r['agents']),
        bytes={k:sum(a['bytes'][k] for a in r['agents']) for k in ('status','payload','acknowledgment')},wall_seconds=r['wall_seconds'],
        packets=len(sent),relevant_anchors=sum(e['diagnostic_route_relevant_anchors'] for e in sent),anchors=sum(e['diagnostic_anchor_count'] for e in sent),
        accepted_deliveries=sum(e['model_accepted'] for e in delivered),force_revisions=sum(abs(e['force_change'])>1e-10 for e in delivered),
        route_revisions=sum(e['route_before']!=e['route_after'] for e in r['events'] if e.get('kind')=='post_delivery_decision'))


def suite(out,workers=8):
    m=freeze(out);rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(job,(str(out),s,a,c)) for s in m['specs'] for a in ARMS for c in SELECTORS]):
            rows.append(f.result());dump(Path(out)/'completion.json',rows);print(rows[-1],flush=True)
    return report(out)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);rows=load(out/'completion.json');pairs=[]
    for seed in (89500,89501):
        for arm in ARMS:
            rr={r['selector']:r for r in rows if r['seed']==seed and r['arm']==arm}
            if len(rr)==2:pairs.append(dict(seed=seed,arm=arm,path_difference=rr['receiver_relevant']['team_path']-rr['rotation']['team_path'],cost_difference=rr['receiver_relevant']['team_cost']-rr['rotation']['team_cost'],byte_difference=sum(rr['receiver_relevant']['bytes'].values())-sum(rr['rotation']['bytes'].values())))
    summary=dict(episodes=len(rows),expected_episodes=16,goals=sum(r['goals'] for r in rows),collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeouts'] for r in rows),complete=len(rows)==16,independent_agent_contacts='not modeled',pairs=pairs)
    dump(out/'summary.json',summary);dump(out/'paired_rows.json',pairs);dump(out/'failure_inventory.json',[r for r in rows if r['goals']!=2])
    (out/'figures').mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(11,4));labels=[f"{p['seed']}\n{p['arm']}" for p in pairs]
    axes[0].bar(labels,[p['path_difference'] for p in pairs]);axes[0].set_ylabel('Relevant − rotation team path (m)');axes[1].bar(labels,[p['byte_difference'] for p in pairs]);axes[1].set_ylabel('Relevant − rotation actual team bytes')
    for ax in axes:ax.tick_params(axis='x',labelrotation=45);ax.axhline(0,color='black',lw=.7)
    fig.tight_layout();fig.savefig(out/'figures/selection_navigation.png');plt.close(fig)
    lines=['# Fixed-rate payload-selection navigation pilot','',f"{len(rows)}/16 episodes; {summary['goals']}/32 goals; {summary['collisions']} collisions; {summary['timeouts']} timeouts. Agent-agent contact dynamics are absent.",
        '', 'The first two existing unselected layouts by seed were frozen before execution. Both selectors use compact packets, local-only retained-stage export and legacy receiver fitting. World, sensing, horizons, safety, goals and per-agent four-million-byte ceilings match the existing scene specifications. Receiver advertisements and acknowledgments are charged. The new loop has controlled parity tests against existing disabled execution; this is a separate pilot, not a bitwise reproduction of the prior communication channel.',
        '', '| Seed | Direction | Selector | Goals | Team path | Team cost | Payload | Status | Ack | Relevant anchors / anchors | Force changes |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['arm']} | {r['selector']} | {r['goals']} | {r['team_path']:.4f} | {r['team_cost']:.4f} | {r['bytes']['payload']} | {r['bytes']['status']} | {r['bytes']['acknowledgment']} | {r['relevant_anchors']}/{r['anchors']} | {r['force_revisions']} |" for r in sorted(rows,key=lambda r:(r['seed'],r['arm'],r['selector']))]
    lines+=['', 'Payload relevance is evaluated against the actual receiver route at send time, with a 0.65 m distance criterion; it is a diagnostic, not an information-value guarantee. No rates or controller parameters were tuned. Two layouts cannot establish generalization; small, zero and negative changes remain present.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return summary


def render(out):
    import matplotlib.pyplot as plt
    from PIL import Image
    from .crossing_visuals import world_panel
    out=Path(out);episodes=[load(out/'episodes'/f'89501_reciprocal_fixed_{c}.json.gz') for c in SELECTORS];end=max(a['finish_time'] for r in episodes for a in r['agents']);frames=[];verification=[]
    for now in np.linspace(0,end,40):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,r in zip(axes,episodes):world_panel(ax,r,float(now));ax.set_title(r['selector'])
        fig.suptitle(f'Fixed cadence, compact packets · seed89501 · t={now:.1f}s');fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        verification.append(dict(time=float(now),samples=[[max((i for i,s in enumerate(ss) if s['time']<=now+1e-9),default=None) for ss in r['samples']] for r in episodes]))
    frames[0].save(out/'figures/selected_comparison.gif',save_all=True,append_images=frames[1:],duration=150,loop=0);dump(out/'animation_frames.json',verification)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','suite','report','render']);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.command=='freeze':freeze(a.out)
    elif a.command=='suite':suite(a.out,a.workers)
    elif a.command=='render':render(a.out)
    else:report(a.out)
