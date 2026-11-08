#!/usr/bin/env python3
"""Liveness with a braking-derived interaction radius, on fresh seeds.

The shared protection layer considers peers within protection.braking_radius() = 0.641 m instead of 0.45 m.
The radius is derived from native physics (top speed with |u_x|,|u_y|<=1, unit braking, one reaction step
for the unknown first-sighting velocity), not tuned. Priority rule and all other settings as in
liveness_campaign. Fresh evaluation seeds 200-229 (unused by v1 development 0-9 and evaluation 100-129).

Pre-declared gate, unchanged from v1 and applied to the '_priority_range' arms against their
protection-only (0.45 m) counterparts: >= 27/30 all goals in every family, zero new contacts, more successes
than the counterpart wherever it is below 27, and no fewer successes anywhere.
"""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import liveness_campaign as m

SEEDS=range(200,230);ARMS=('attraction','attraction_priority_range','coph_online','coph_online_priority_range')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise SystemExit('Refuse existing output')
    args.out.mkdir(parents=True);(args.out/'manifest.json').write_text(json.dumps(dict(seeds=list(SEEDS),arms=ARMS,radius=m.braking_radius(),gate=__doc__),indent=1))
    with ProcessPoolExecutor(44) as pool:rows=list(pool.map(m.run,[(f,s,a) for f in m.FAMILIES for s in SEEDS for a in ARMS]))
    (args.out/'rows.json').write_text(json.dumps(rows));summary={}
    for f in m.FAMILIES:
        for a in ARMS:
            rr=[r for r in rows if r['family']==f and r['arm']==a];t=sorted(r['time_to_goals'] for r in rr if r['time_to_goals'])
            summary[f'{f}/{a}']=dict(success=sum(r['success'] for r in rr),new_contacts=sum(r['new_contacts'] for r in rr),
                median_time=t[len(t)//2] if t else None,min_separation=min(r['min_separation'] for r in rr))
    checks={}
    for f in m.FAMILIES:
        for a in ('attraction','coph_online'):
            t,b=summary[f'{f}/{a}_priority_range'],summary[f'{f}/{a}']
            checks[f'{f}/{a}']=dict(success=t['success']>=27,contacts=t['new_contacts']==0,vs_base=t['success']>=b['success'] and (b['success']>=27 or t['success']>b['success']))
    gate=all(all(v.values()) for v in checks.values())
    res=dict(summary=summary,checks=checks,gate=gate);(args.out/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))


if __name__=='__main__':main()
