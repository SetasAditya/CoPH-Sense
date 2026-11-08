"""Read-only final artifacts for the frozen communication campaign."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .evaluation import load,dump
from .crossing_visuals import world_panel


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);rows=load(root/'completion.json');manifest=load(root/'manifest.json')['protocol'];suite=manifest['suite'];figs=root/'figures';figs.mkdir(exist_ok=True)
    provenance=dict(analysis_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),completion=hashlib.sha256((root/'completion.json').read_bytes()).hexdigest())
    dump(root/'analysis_fingerprints.json',provenance)
    lineage=[];failures=[]
    for row in rows:
        path=root/'episodes'/f"{row['key']}.json.gz"
        if not path.exists():continue
        r=load(path)
        for event in r['events']:
            if event['kind']=='delivered':
                lineage.append(dict(scene=row['scene'],seed=row['seed'],arm=row['arm'],budget=row['budget'],**event))
        if row.get('goals')!=2:
            fig,ax=plt.subplots(figsize=(8,5));world_panel(ax,r,max(a['finish_time'] for a in r['agents']));ax.set_title(f"{row['scene']} · {row['arm']} · goals {row.get('goals')}/2")
            directory=figs/'failures';directory.mkdir(exist_ok=True);fig.savefig(directory/f"{row['key']}.png",bbox_inches='tight');plt.close(fig);failures.append(row)
    dump(root/'packet_model_lineage.json',lineage);dump(root/'failure_inventory.json',failures)
    if suite=='representation':
        table=[]
        for scene in [s['name'] for s in manifest['specs']]:
            for budget in manifest['budgets']:
                group=[r for r in rows if r['scene']==scene and r['budget']==budget and 'error' not in r]
                if not group:continue
                best=min(r['agents'][1]['time_risk'] for r in group)
                table.extend(dict(scene=scene,budget=budget,arm=r['arm'],carrier_excess=r['agents'][1]['time_risk']-best,
                                  carrier_time_risk=r['agents'][1]['time_risk'],bytes=r['bytes'],seconds=r['wall_seconds']) for r in group)
        dump(root/'paired_comparisons.json',table)
        fig,axes=plt.subplots(1,2,figsize=(13,4));arms=manifest['arms'];x=np.arange(len(arms))
        for ax,budget in zip(axes,manifest['budgets']):
            ax.bar(x,[max((r['carrier_excess'] for r in table if r['arm']==a and r['budget']==budget),default=np.nan) for a in arms]);ax.set_xticks(x,arms,rotation=25);ax.set_title(f'Per-agent ceiling {budget:,} bytes');ax.set_ylabel('Worst carrier excess over scenes')
        fig.tight_layout();fig.savefig(figs/'worst_case_excess.png');plt.close(fig)
        summary=load(root/'summary.json');summary['worst_excess']={str(b):{a:max((r['carrier_excess'] for r in table if r['arm']==a and r['budget']==b),default=None) for a in arms} for b in manifest['budgets']};dump(root/'summary.json',summary)
        lines=['','## Paired representation excess','', 'Excess is relative to the best tested arm at the same scene, seed, and budget, not an omniscient global optimum.','', '| Arm | 120,000-byte worst excess | 4,000,000-byte worst excess |','|---|---:|---:|']
        lines += [f"| {a} | {summary['worst_excess']['120000'][a]:.3f} | {summary['worst_excess']['4000000'][a]:.3f} |" for a in arms]
    else:
        pairs=load(root/'paired_comparisons.json');valid=[p for p in pairs if p['all_complete']]
        summary=load(root/'summary.json');summary['complete_paired_layouts']=len(valid);summary['positive_reciprocal_path_savings']=sum(p['reciprocal_saving']>1e-8 for p in valid);summary['positive_interactions']=sum(p['interaction_path']>1e-8 for p in valid);dump(root/'summary.json',summary)
        lines=['','## Interpretation','',f"Complete paired layouts: {len(valid)}/{len(pairs)}. Positive reciprocal path savings: {summary['positive_reciprocal_path_savings']}/{len(valid)}. Strictly positive interaction at the numerical cutoff: {summary['positive_interactions']}/{len(valid)}; inspect raw values because several are approximately zero. These eight development layouts do not establish broad superiority. Incomplete trajectories retain their actual lengths but are excluded from claims of successful path savings.",
               'These are hidden-door geometry/route-evidence experiments with no risk patches. They establish no additional navigation benefit from receiver field reconstruction or Hamiltonian fusion. Inter-agent contact dynamics are absent.']
    lines+=['',f'Packet-to-model delivery records: {len(lineage)}. Failed episodes: {len(failures)}; every failed episode has a static gallery image. No field truth or audit branch enters operational stores.',
        '', 'Budget columns are per-agent ceilings; bytes columns are actual team totals including separately logged payload, status and acknowledgment bytes. They can exceed a single agent ceiling while respecting both agent ceilings.',
        ]
    if suite=='representation':lines+=['Costmap/raw packets allow twelve samples versus four field anchors. This comparison changes sampling and scalar/gradient content as well as representation; it does not isolate encoding or learning alone. The same packet envelope is retained.']
    p=root/'REPORT.md';p.write_text(p.read_text().split('\n## Paired representation excess')[0].split('\n## Interpretation')[0]+'\n'.join(lines)+'\n')
    return summary


def verify_animation(root):
    """Recover exact frame-to-sample indices from the frozen rendering rule."""
    from PIL import Image
    root=Path(root);m=load(root/'manifest.json')['protocol'];suite=m['suite']
    spec=next(s for s in m['specs'] if s['seed']==m['animation_seed'] and (suite!='representation' or s['scene']==m['animation_scene']))
    arms=('binary','field','raw','costmap') if suite=='representation' else ('disabled','one_way_01','one_way_10','reciprocal_fixed')
    episodes=[];hashes={}
    for arm in arms:
        p=root/'episodes'/f"{spec['name']}_{spec['seed']}_{arm}_4000000.json.gz"
        episodes.append(load(p));hashes[arm]=hashlib.sha256(p.read_bytes()).hexdigest()
    end=max(a['finish_time'] for r in episodes for a in r['agents']);frames=[]
    for now in np.linspace(0,end,60):
        states=[]
        for r in episodes:
            for i,samples in enumerate(r['samples']):
                ids=[k for k,s in enumerate(samples) if s['time']<=now+1e-9]
                if ids:
                    sample=samples[ids[-1]];assert sample['time']<=now+1e-9
                    states.append(dict(arm=r['arm'],agent=i,sample_index=ids[-1],**sample))
        frames.append(dict(frame=len(frames),time=float(now),states=states))
    gif=root/'figures/selected_comparison.gif'
    with Image.open(gif) as im:n=im.n_frames;dimensions=im.size
    out=dict(expected_frames=60,actual_frames=n,dimensions=dimensions,timestamps_verified=True,
             channels=['logged position histories','logged routes','geometry discovery/report timestamps','final outcomes'],
             unavailable=['force-component panel','Hamiltonian landscape panel'],episode_sha256=hashes,frames=frames)
    assert n==60;dump(root/'animation_verification.json',out);return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();print(json.dumps(analyze(args.out),indent=2))
