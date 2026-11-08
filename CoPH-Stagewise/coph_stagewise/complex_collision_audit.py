"""Evaluator-only diagnosis of recorded obstacle collisions; no learner changes."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from .evaluation import load,dump
from .model import tensor,risk,command

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args(argv);rows=[]
    torch.set_num_threads(1)
    for item in load(args.root/'development/completion.json'):
        if not item['complete']:continue
        r=load(args.root/'development/episodes'/item['path'])
        for i,a in enumerate(r['agents']):
            if not a['collision']:continue
            rec=r['trajectories'][i][-1];x=np.array(rec['x']);after=np.array(rec['after']);delta=after[:2]-x[:2]
            def swept(o):
                t=np.clip(np.dot(np.array(o[:2])-x[:2],delta)/(delta@delta+1e-20),0,1)
                return np.linalg.norm(x[:2]+t*delta-o[:2])-o[2]-.12
            obstacle=min(r['spec']['obstacles'],key=swept)
            seen=[o['time'] for o in r['observations'][i] if obstacle in o['obstacles'] and o['time']<=rec['tick']*r['config']['dt']+1e-10]
            snap=r['final_models'][i];st=next(s for s in snap['stages'] if s['id']==rec['stage'])
            u,proposed=command(tensor(x),tensor(r['spec']['goals'][i]),tensor(st['weights']),tensor(st['centers']),r['config']['rbf_width'],tensor(snap['policy']),st['obstacles'],r['config']['force_limit'])
            match=bool(np.allclose(proposed.detach().numpy(),rec['proposed'],atol=1e-10,rtol=0))
            goal=np.array(r['spec']['goals'][i])-x[:2];drag=-1.4*x[2:]
            q=tensor(x[:2]).requires_grad_(True)
            field=-torch.autograd.grad(snap['policy'][0]*risk(q,tensor(st['weights']),tensor(st['centers']),r['config']['rbf_width']),q)[0].numpy()
            barrier=proposed.detach().numpy()-goal-drag-field
            rows.append(dict(scene=item['scene'],seed=item['seed'],arm=item['arm'],agent=i,tick=rec['tick'],
                obstacle=obstacle,first_seen=min(seen) if seen else None,sensed_before_impact=bool(seen),
                present_in_final_active_stage=obstacle in st['obstacles'],stage_bounds=st['bounds'],
                force_reconstruction_matches=match,goal_force=float(np.linalg.norm(goal)),
                risk_force=float(np.linalg.norm(field)) if match else None,
                obstacle_force=float(np.linalg.norm(barrier)) if match else None,
                velocity_damping_force=float(np.linalg.norm(drag)),executed_force=rec['u'],position=rec['x'],
                policy=snap['policy'],swept_clearance=float(swept(obstacle))))
    dump(args.root/'collision_audit.json',rows)
    summary=[]
    for name in sorted({r['scene'] for r in rows}):
        rs=[r for r in rows if r['scene']==name and r['arm']=='reciprocal_fixed']
        summary.append(dict(scene=name,collisions=len(rs),sensed=sum(r['sensed_before_impact'] for r in rs),
            omitted=sum(r['sensed_before_impact'] and not r['present_in_final_active_stage'] for r in rs),
            reconstructed=sum(r['force_reconstruction_matches'] for r in rs),
            mean_goal_force=float(np.mean([r['goal_force'] for r in rs if r['force_reconstruction_matches']])),
            mean_obstacle_force=float(np.mean([r['obstacle_force'] for r in rs if r['force_reconstruction_matches']]))))
    dump(args.root/'collision_summary.json',summary)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));xs=np.arange(len(summary))
    ax.bar(xs-.18,[r['mean_goal_force'] for r in summary],.36,label='Goal attraction')
    ax.bar(xs+.18,[r['mean_obstacle_force'] for r in summary],.36,label='Obstacle repulsion')
    ax.set_xticks(xs,[r['scene'] for r in summary],rotation=15,fontsize=8);ax.set_ylabel('Mean pre-clipping force norm at impact');ax.legend();fig.tight_layout()
    fig.savefig(args.root/'development/figures/collision_forces.png',dpi=150);plt.close(fig)
    source=Path(__file__);(args.root/'collision_audit_source.txt').write_bytes(source.read_bytes())
    dump(args.root/'collision_audit_provenance.json',dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        note='Read-only final-snapshot force reconstruction is interpreted only where it matches the recorded pre-clipping command. Geometry visibility uses recorded observations.'))
    print(summary)

if __name__=='__main__':main()
