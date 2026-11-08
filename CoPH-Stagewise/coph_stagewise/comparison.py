"""Paired before/after evaluation under the same frozen scene/cost contract."""
import argparse
from pathlib import Path
import numpy as np
from .evaluation import load,dump,ARMS


def compare_runs(baseline,run):
    old=load(baseline/'manifest.json');new=load(run/'manifest.json')
    for key in ('profile','seeds','scenarios','arms','mission_cost','replay_cost'):
        if old[key]!=new[key]:raise ValueError(f'Cannot pair differing {key}')
    configs=[]
    for m in (old,new):
        cfg=m['config'].copy();cfg.pop('navigation_target',None);configs.append(cfg)
    if configs[0]!=configs[1]:raise ValueError('Other learner settings changed')
    key=lambda r:(r['scene'],r['seed'],r['arm'])
    before={key(r):r for r in load(baseline/'summary.json')}
    after={key(r):r for r in load(run/'summary.json')}
    if before.keys()!=after.keys() or len(after)!=new['expected_episodes']:
        raise ValueError('Incomplete or unmatched runs; no complete paired comparison')
    rows=[]
    for k in sorted(after):
        a,b=before[k],after[k]
        rows.append(dict(scene=k[0],seed=k[1],arm=k[2],success_before=a['success'],success_after=b['success'],
            recovered=not a['success'] and b['success'],regressed=a['success'] and not b['success'],
            collision_before=a['collision'],collision_after=b['collision'],
            cost_before=a['mission_cost'],cost_after=b['mission_cost'],cost_delta=b['mission_cost']-a['mission_cost'],
            stationary_before=a['near_stationary_seconds'],stationary_after=b['near_stationary_seconds']))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();rows=compare_runs(args.baseline,args.run)
    dump(args.run/'before_after.json',rows)
    lines=['# Mission-goal attraction: paired evaluation','',
        f'Baseline: `{args.baseline}`. Corrected run: `{args.run}`. Matched episodes: {len(rows)}.',
        '', 'Only the executed attraction target changes. Replay controls/objectives retain stage-local targets; replay support handling is unchanged. Energy diagnostics use the executed potential. Scenes, seeds, horizons, sensing, costs and other learner settings match.',
        '', '| Arm | Success before → after | Recovered | Regressed | Collisions before → after | Mean cost before → after | Paired cost change |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        g=[r for r in rows if r['arm']==arm]
        lines.append(f"| {arm} | {sum(r['success_before'] for r in g)}/{len(g)} → {sum(r['success_after'] for r in g)}/{len(g)} | {sum(r['recovered'] for r in g)} | {sum(r['regressed'] for r in g)} | {sum(r['collision_before'] for r in g)} → {sum(r['collision_after'] for r in g)} | {np.mean([r['cost_before'] for r in g]):.3f} → {np.mean([r['cost_after'] for r in g]):.3f} | {np.mean([r['cost_delta'] for r in g]):+.3f} |")
    lines.extend(['','## Full-method scenario results','','| Scenario | Success before → after | Recovered | Regressed | Cost change |','|---|---:|---:|---:|---:|'])
    for scene in dict.fromkeys(r['scene'] for r in rows):
        g=[r for r in rows if r['arm']=='full' and r['scene']==scene]
        lines.append(f"| {scene} | {sum(r['success_before'] for r in g)} → {sum(r['success_after'] for r in g)} / {len(g)} | {sum(r['recovered'] for r in g)} | {sum(r['regressed'] for r in g)} | {np.mean([r['cost_delta'] for r in g]):+.3f} |")
    full=[r for r in rows if r['arm']=='full']
    lines.extend(['',f"Full-method mean near-stationary time: {np.mean([r['stationary_before'] for r in full]):.3f} → {np.mean([r['stationary_after'] for r in full]):.3f} seconds.",
        '', 'These are development results on previously examined seeds. A repaired controller is not evidence that historical replay, cooperation or across-episode learning improves performance. Compare against the corrected frozen and evidence-only arms. The current-state comparator still includes explicitly charged non-learning padding.',
        '', '[Within-run report](REPORT.md) · [All recorded failures](figures/failures/index.html) · [Paired numerical rows](before_after.json)'])
    (args.run/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n')
    from .visualize import plt
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,arm in enumerate(ARMS):
        g=[r for r in rows if r['arm']==arm]
        axs[0].scatter(np.full(len(g),i),[r['cost_delta'] for r in g],s=9,alpha=.5)
        axs[1].bar(i-.18,sum(r['success_before'] for r in g),width=.36,color='gray',label='before' if i==0 else None)
        axs[1].bar(i+.18,sum(r['success_after'] for r in g),width=.36,color='steelblue',label='mission attraction' if i==0 else None)
    for ax in axs:ax.set_xticks(range(len(ARMS)),ARMS,rotation=45,ha='right')
    axs[0].axhline(0,color='black',ls='--');axs[0].set_ylabel('Paired complete cost: after minus before')
    axs[1].set_ylabel('Successful episodes');axs[1].legend()
    (args.run/'figures').mkdir(exist_ok=True);fig.savefig(args.run/'figures'/'before_after.png',dpi=140);plt.close(fig)
    print(f'Compared {len(rows)} matched episodes; full recovered {sum(r["recovered"] for r in full)}, regressed {sum(r["regressed"] for r in full)}')


if __name__=='__main__':main()
