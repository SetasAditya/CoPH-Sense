#!/usr/bin/env bash
# Supplementary figures from existing logs; never runs or changes a learner.
set -eu
cd "$(dirname "$0")/.."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/coph-mpl}" python - "$@" <<'PY'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from coph_stagewise.evaluation import load,dump
from coph_stagewise.complex_environments import SCENES,COLORS,world_axes,paths
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/navigation_fix_v3')
manifest=load(root/'manifest.json');out=root/'development/figures';out.mkdir(exist_ok=True)
def render(scene):
    rs={'legacy':load(Path(manifest['reference'])/'development/episodes'/f'{scene}_61000_reciprocal_fixed.json.gz')}
    for mode in ('geometry','protected','routed'):rs[mode]=load(root/'development/episodes'/f'{scene}_61000_{mode}_reciprocal_fixed.json.gz')
    end=max(t['tick'] for r in rs.values() for seq in r['trajectories'] for t in seq);frames=[]
    for tick in sorted(set(np.linspace(0,end,36).astype(int))):
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for ax,(mode,r) in zip(axes.flat,rs.items()):
            world_axes(ax,r['spec']);paths(ax,r,int(tick))
            for i,seq in enumerate(r['trajectories']):
                seq=[t for t in seq if t['tick']<=tick]
                if not seq:continue
                t=seq[-1];n=t.get('navigation',{});c=COLORS[i]
                if n.get('route'):
                    p=np.array(n['route']);ax.plot(p[:,0],p[:,1],':',color=c);ax.scatter(*n['waypoint'],marker='x',color=c,s=18)
                if 'proposed' in t:
                    for vector,color in ((t['proposed'],'orange'),(t['u'],c)):
                        v=np.array(vector)*.12;ax.arrow(*t['x'][:2],*v,width=.01,color=color,length_includes_head=True)
                us=[u for u in r['updates'][i] if u['tick']<=tick]
                if us:
                    u=us[-1];snap=u['post']
                    for ox,oy,rad in snap.get('geometry',[]):ax.add_patch(Circle((ox,oy),rad+.17,fill=False,color=c,lw=.7,alpha=.5))
                    for st in snap['stages']:
                        for cf in st['counterfactuals']:
                            p=np.array(cf['states']);ax.plot(p[:,0],p[:,1],'--',color=c,alpha=.4,lw=.8)
                    ax.text(.02,.97-i*.065,f'Agent {i}: snapshot {u["tick"]*.12:.2f}s, model {snap["version"]}',transform=ax.transAxes,color=c,fontsize=6,va='top')
            ax.set_title(mode+(' (legacy routing/guard logs unavailable)' if mode=='legacy' else ''),fontsize=9)
        fig.suptitle(f'{scene} | all four controllers, reciprocal fixed-rate communication | seed 61000 | t={(tick+1)*.12:.2f}s\nDotted: routes; dashed: retained replay; orange: nominal control; blue/green: applied. Terminal failures retained.',fontsize=10)
        fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    frames[-1].save(out/f'{scene}_controllers.png')
    frames[0].save(out/f'{scene}_controllers.gif',save_all=True,append_images=frames[1:],duration=max(100,int((end+1)*120/len(frames))),loop=0)
    return scene
with ProcessPoolExecutor(max_workers=4) as pool:
    for name in pool.map(render,SCENES):print('controller comparison GIF:',name,flush=True)
controlled=root/'controlled'
if (controlled/'completion.json').exists():
    destination=controlled/'figures';destination.mkdir(exist_ok=True)
    for item in load(controlled/'completion.json'):
        if not item['complete']:continue
        r=load(controlled/'episodes'/item['path']);end=max(t['tick'] for ts in r['trajectories'] for t in ts);frames=[]
        for tick in sorted(set(np.linspace(0,end,32).astype(int))):
            fig,ax=plt.subplots(figsize=(7,4));world_axes(ax,r['spec']);paths(ax,r,int(tick))
            for i,seq in enumerate(r['trajectories']):
                ts=[t for t in seq if t['tick']<=tick]
                if not ts:continue
                t=ts[-1];n=t['navigation'];p=np.asarray(n.get('route',[]))
                if len(p):ax.plot(p[:,0],p[:,1],':',color=COLORS[i]);ax.scatter(*n['waypoint'],marker='x',color=COLORS[i])
                for vector,color in ((n['nominal'],'orange'),(t['u'],COLORS[i])):
                    v=np.asarray(vector)*.15;ax.arrow(*t['x'][:2],*v,width=.01,color=color,length_includes_head=True)
            ax.set_title(f"{r['spec']['name']} | learning + reciprocal fixed-rate exchange | t={(tick+1)*.12:.2f}s\nIndependent agents; sensed-space routes dotted; nominal orange, applied blue/green",fontsize=9)
            fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        frames[0].save(destination/f"{item['scene']}.gif",save_all=True,append_images=frames[1:],duration=max(80,int((end+1)*120/len(frames))),loop=0)
script=Path('scripts/render_navigation_ablations.sh')
dump(root/'rendering_manifest.json',dict(renderer=str(script),renderer_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    simulation_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
    purpose='supplementary rendering only; does not change frozen simulation sources or records',animation_seed=61000))
PY
