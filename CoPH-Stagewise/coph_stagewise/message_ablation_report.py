"""Summarize message_ablation runs: carrier excess cost, bytes per packet and per component."""
import argparse,json
from pathlib import Path
import numpy as np
from .evaluation import load
from .hazard_demo import SCENES
from .message_ablation import ARMS,SEEDS

# Packet fields grouped as in the manuscript's m_{i->j}.
GROUPS=dict(header=('schema','sender','recipient','identity','timestamp','expires','version','convention','stage','uncertainty'),
            sender_state=('q','p','v','H','V_task','goal'),support=('bounds','support_points'),anchors=('anchors',),
            covariance=('covariance',),provenance=('sources','acknowledged_sources'),geometry=('geometry_schema','geometry'))


def size(value):
    return len(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def summarize(root):
    root=Path(root);cost={};packets={}
    for scene in SCENES:
        for seed in SEEDS:
            for arm in ARMS:
                r=load(root/scene/'episodes'/f'{seed}_{arm}.json.gz');c=r['agents'][1]['components']
                assert all(a['success'] for a in r['agents']),(scene,seed,arm)
                cost[scene,seed,arm]=c['time']+c['risk'];packets.setdefault(arm,[]).extend(r['packets'])
    excess={(s,k,a):cost[s,k,a]-min(cost[s,k,b] for b in ARMS) for s in SCENES for k in SEEDS for a in ARMS}
    rows={}
    for a in ARMS:
        worst=[max(excess[s,k,a] for s in SCENES) for k in SEEDS]
        per_scene={s:float(np.mean([excess[s,k,a] for k in SEEDS])) for s in SCENES}
        p=packets[a];full=[size(m) for m in p]
        rows[a]=dict(worst_case_mean=float(np.mean(worst)),worst_case_per_seed=worst,worst_case_max=float(max(worst)),mean_excess_per_scene=per_scene,
                     paired_vs_field=float(np.mean([cost[s,k,a]-cost[s,k,'field'] for s in SCENES for k in SEEDS])),
                     packets=len(p),bytes_per_packet=float(np.mean(full)) if p else 0.,
                     group_bytes={g:float(np.mean([sum(size(m[f]) for f in fs if f in m) for m in p])) if p else 0. for g,fs in GROUPS.items()})
    return dict(rows=rows,cost={f'{s}|{k}|{a}':v for (s,k,a),v in cost.items()})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);args=ap.parse_args(argv)
    out=summarize(args.root);(args.root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f"{'arm':16s} {'worst(mean)':>11s} {'worst(max)':>10s} {'vs field':>9s} {'B/pkt':>7s}  "+'  '.join(f'{s:>11s}' for s in SCENES))
    for a,r in out['rows'].items():
        print(f"{a:16s} {r['worst_case_mean']:11.1f} {r['worst_case_max']:10.1f} {r['paired_vs_field']:9.1f} {r['bytes_per_packet']:7.0f}  "+'  '.join(f"{r['mean_excess_per_scene'][s]:11.1f}" for s in SCENES))
    print('bytes by group (field):',{g:round(v) for g,v in out['rows']['field']['group_bytes'].items()})


if __name__=='__main__':main()
