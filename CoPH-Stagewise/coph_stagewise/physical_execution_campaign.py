"""Protected execution supplement to the frozen controlled numerical campaign."""
import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from .records import Config,Transition
from .learner import Learner
from .cooperative_evaluation import CooperativeWorld
from .model import tensor,potential
from .stage_prediction import FrozenStages


def execute(scene,dt):
    cfg=Config(controller_mode='protected',shaping_mode='progress',replay_domain='retained',replay_objective='mission',dt=dt,steps=round(24/dt))
    spec=dict(name=scene,seed=83000,initial=[0.,0.,0.,0.],goal=[5.,0.],damping=.85,dynamic=False,
        obstacles=[(2.,.65,.3)] if scene=='obstacle' else [],patches=[(2.,.35,.6,.6,1.)] if scene=='risk' else [])
    world=CooperativeWorld(spec,cfg);a=Learner(cfg,spec['initial'],spec['goal'],mode='evidence');x=np.array(spec['initial']);a.observe(world.sense(x,0));rows=[];handoffs=[];hit=False;start=time.perf_counter()
    for tick in range(cfg.steps):
        if tick%cfg.update_every==0:a.update(x)
        u,nominal=a.action(x);control=dict(a.last_control);y,hit,cost,energy=world.step(x,u)
        active=a.stages[-1]
        old=float(potential(tensor(y[:2]),tensor(a.control_target()),active.weights,active.centers,cfg.rbf_width,a.policy,a.control_obstacles(active),cfg))
        a.observe(world.sense(y,tick+1),Transition(tick,(tick+1)*dt,tuple(x),tuple(u),tuple(y),a.version))
        new=float(potential(tensor(y[:2]),tensor(a.control_target()),active.weights,active.centers,cfg.rbf_width,a.policy,a.control_obstacles(active),cfg))
        rows.append(dict(tick=tick,time=(tick+1)*dt,x=x.tolist(),after=y.tolist(),u=u.tolist(),nominal=nominal.tolist(),energy=energy,
            observation_context_jump=new-old,safety_modification_work=float(np.dot(u-nominal,y[:2]-x[:2])),control=control,version=a.version,stage=active.id))
        x=y
        if hit or np.linalg.norm(x[:2]-a.goal)<.25:break
        previous=copy.deepcopy(a)
        if a.advance(x):
            # Same physical state and frozen measured context on isolated copies.
            left=FrozenStages.freeze(previous);right=FrozenStages.freeze(a);before_context=copy.deepcopy(previous.navigation);after_context=copy.deepcopy(a.navigation)
            before_u,before_p=left.control(tensor(x),len(left.stages)-1,a.policy,a.damping,before_context,tick)
            after_u,after_p=right.control(tensor(x),len(right.stages)-1,a.policy,a.damping,after_context,tick)
            handoffs.append(dict(time=(tick+1)*dt,x=x.tolist(),from_stage=previous.stages[-1].id,to_stage=a.stages[-1].id,
                nominal_force_jump=float(torch.linalg.vector_norm(after_p-before_p)),applied_force_jump=float(torch.linalg.vector_norm(after_u-before_u)),
                shaping_jump=a.context_jumps[-1]['shaping_jump'],physical_jump=0.,diagnostic_only=True))
    return dict(scene=scene,dt=dt,config=asdict(cfg),specification=spec,trajectory=rows,stage_jumps=a.context_jumps,fixed_state_handoffs=handoffs,updates=a.diagnostics,
        summary=dict(success=bool(np.linalg.norm(x[:2]-a.goal)<.25 and not hit),collision=bool(hit),timeout=bool(np.linalg.norm(x[:2]-a.goal)>=.25 and not hit),
            goal_gap=float(np.linalg.norm(x[:2]-a.goal)),max_energy_residual=max(abs(r['energy']['residual']) for r in rows),
            interventions=int(sum(np.linalg.norm(np.array(r['u'])-r['nominal'])>1e-8 for r in rows)),stage_handoffs=len(a.context_jumps),seconds=time.perf_counter()-start))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in ['navigation.py','stage_prediction.py','learner.py','model.py','cooperative_evaluation.py']]]
    manifest=dict(scenes=['clear','risk','obstacle'],dt=[.12,.06,.03],duration=24.,seed=83000,controller='protected-progress',learner='evidence',
        update_every_steps=8,warning='Timestep sweep changes update/sensing frequencies in physical time; do not attribute mission differences solely to integration.',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));archive=out/'sources';archive.mkdir()
    for p in files:(archive/p.name).write_bytes(p.read_bytes())
    results=[]
    for scene in manifest['scenes']:
        for dt in manifest['dt']:
            record=execute(scene,dt);(out/('%s_%.3f.json'%(scene,dt))).write_text(json.dumps(record,indent=2));results.append(dict(scene=scene,dt=dt,**record['summary']));print(results[-1],flush=True)
    (out/'outcomes.json').write_text(json.dumps(results,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for i,scene in enumerate(manifest['scenes']):
        for dt in manifest['dt']:
            r=json.loads((out/('%s_%.3f.json'%(scene,dt))).read_text());tr=r['trajectory'];t=[e['time'] for e in tr]
            axes[i,0].plot(t,[np.linalg.norm(e['u']) for e in tr],label='dt=%g'%dt)
            axes[i,1].semilogy(t,[max(abs(e['energy']['residual']),1e-18) for e in tr])
        axes[i,0].set(title=scene,ylabel='Applied force norm',xlabel='Time (s)');axes[i,0].legend()
        axes[i,1].set(ylabel='Physical energy residual',xlabel='Time (s)')
    fig.tight_layout();fig.savefig(out/'protected_accounting.png',dpi=150);plt.close(fig)
    text='# Protected execution supplement\n\nNine fixed-duration controlled episodes use the unchanged protected-progress controller and online measured-evidence learning. All outcomes, interventions, context jumps and stage handoffs are retained. These are diagnostic cases, not a safety guarantee. The timestep sweep also changes physical sensing/update cadence and is not a pure integration convergence experiment.\n\n'
    text+='| Scene | dt | Goal | Collision | Timeout | Handoffs | Max energy residual |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in results:text+='| %s | %g | %s | %s | %s | %d | %.3g |\n'%(r['scene'],r['dt'],r['success'],r['collision'],r['timeout'],r['stage_handoffs'],r['max_energy_residual'])
    (out/'REPORT.md').write_text(text)


if __name__=='__main__':main()
