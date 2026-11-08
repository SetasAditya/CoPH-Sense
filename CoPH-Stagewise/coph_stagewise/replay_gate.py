"""Compare aligned/legacy replay and measured-evidence learning on a frozen manifest."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import traceback
import numpy as np
import torch
from .evaluation import execute,load,dump
from .cli import source_hashes,archive_sources


def job(args):
    spec,name,root=args;torch.set_num_threads(1)
    try:
        result=execute(spec,'development','evidence' if name=='evidence' else 'full',
                       replay_objective='mission' if name=='aligned' else 'legacy')
        result['summary']['arm']=name
        dump(Path(root)/'episodes'/f"{spec['name']}_{spec['seed']}_{name}.json.gz",result)
        return result['summary']
    except Exception:
        dump(Path(root)/'episodes'/f"{spec['name']}_{spec['seed']}_{name}.error.json",dict(traceback=traceback.format_exc()))
        return None


def main(argv):
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('results/mission_goal_development/manifest.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args(argv)
    if args.out.exists():p.error('use a fresh output directory')
    original=load(args.manifest);args.out.mkdir(parents=True);(args.out/'episodes').mkdir()
    manifest=dict(source_manifest=str(args.manifest),scenarios=original['scenarios'],arms=['legacy','evidence','aligned'],
                  source_hashes=source_hashes(),expected=3*len(original['scenarios']),status='paired development, unchanged scene/seed/horizon/cost')
    dump(args.out/'manifest.json',manifest);archive_sources(args.out);rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(job,(spec,arm,str(args.out))) for spec in manifest['scenarios'] for arm in manifest['arms']]
        for f in as_completed(jobs):
            r=f.result()
            if r is not None:rows.append(r)
            dump(args.out/'summary.json',rows);print(f'{len(rows)}/{len(jobs)} replay gate episodes',flush=True)
    dump(args.out/'completion.json',dict(completed=len(rows),expected=manifest['expected']))
    lines=['# Replay alignment development gate','','| Arm | Completed | Success | Mean cost |','|---|---:|---:|---:|']
    for name in manifest['arms']:
        rs=[r for r in rows if r['arm']==name]
        lines.append(f"| {name} | {len(rs)} | {sum(r['success'] for r in rs)} | {np.mean([r['mission_cost'] for r in rs]):.5f} |")
    lines+=['','Aligned replay uses execution attraction and time/risk/effort running cost. Its finite horizon is truncated; it has no artificial distance or timeout penalty.',
            'This gate tests consistency, not guaranteed replay superiority. Unsupported entry and domain-exit diagnostics remain visible. Existing results were not overwritten.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
