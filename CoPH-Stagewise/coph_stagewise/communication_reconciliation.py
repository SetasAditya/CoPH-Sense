"""Frozen accounting reconciliation; no controller or communication-policy changes.

Run from CoPH-Stagewise with single-thread numerical libraries:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m coph_stagewise.communication_reconciliation all --workers 16

Legacy accounting is rerun solely to certify operational parity and reproduce the
superseded bandwidth rows. It is never included as a corrected treatment arm.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch

from . import communication_navigation as navigation
from .bandwidth_selection_suite import BUDGETS, SEEDS
from .crossing_demo import configuration
from .evaluation import dump, load
from . import hazard_demo
from .maze_family import NetworkWorld
from .staggered_demo import execute as original_execute

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/communication_reconciliation_v1'
BYTE_KEYS = ('status', 'payload', 'acknowledgment')
TOLERANCE = 1e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items()
                if not k.endswith('_seconds') and k not in ('seconds', 'latency', 'elapsed')}
    if isinstance(value, (tuple, list)):
        return [strip_timing(v) for v in value]
    return value


def operational_view(episode):
    """Includes full control diagnostics and update decisions/work; excludes costs/timers."""
    trajectory_keys = ('time', 'tick', 'x', 'u', 'nominal', 'after', 'end_time', 'version', 'stage', 'navigation')
    updates = [[dict(time=u['time'], tick=u['tick'], diagnostic=strip_timing(u['diagnostic']))
                for u in us] for us in episode['updates']]
    return dict(samples=episode['samples'], observations=episode['observations'], updates=updates,
                trajectories=[[{k: r[k] for k in trajectory_keys if k in r} for r in rs]
                              for rs in episode['trajectories']],
                geometry_events=episode['geometry_events'])


def compare_values(a, b, path='', errors=None, stats=None):
    if errors is None: errors = []
    if stats is None: stats = dict(max_abs_error=0., discrete_mismatches=0, numeric_mismatches=0)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            stats['discrete_mismatches'] += 1
            if len(errors) < 12: errors.append(path + ': keys differ')
        for k in sorted(set(a) & set(b)): compare_values(a[k], b[k], path + '/' + str(k), errors, stats)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            stats['discrete_mismatches'] += 1
            if len(errors) < 12: errors.append(path + ': lengths differ')
        for i, (p, q) in enumerate(zip(a, b)): compare_values(p, q, path + '/' + str(i), errors, stats)
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)) and not isinstance(a, bool) and not isinstance(b, bool):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                stats['discrete_mismatches'] += 1
                if len(errors) < 12: errors.append(path + ': integers differ')
        else:
            error = abs(a-b)
            stats['max_abs_error'] = max(stats['max_abs_error'], error)
            if not np.isfinite(error) or error > TOLERANCE:
                stats['numeric_mismatches'] += 1
                if len(errors) < 12: errors.append(path + ': numeric difference ' + str(error))
    elif a != b:
        stats['discrete_mismatches'] += 1
        if len(errors) < 12: errors.append(path + ': values differ')
    return dict(**stats, examples=errors, passed=not errors,
                bitwise_serialized_equal=canonical(a) == canonical(b))


def compare_operations(a, b, include_events=True):
    left, right = operational_view(a), operational_view(b)
    if include_events:
        left['events'], right['events'] = strip_timing(a['events']), strip_timing(b['events'])
    result = compare_values(left, right)
    result['operational_sha256'] = [hashlib.sha256(canonical(z)).hexdigest() for z in (left, right)]
    return result


def frozen_manifest():
    source = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'coph_stagewise').glob('*.py'))
              if p.name != 'communication_reconciliation_report.py'}
    inputs = [ROOT/'results/bandwidth_selection_v1/manifest.json', ROOT/'results/bandwidth_selection_v1/rows.json']
    for scene in hazard_demo.SCENES:
        folder = ROOT/'results/hazard_pilot_v2'/scene
        inputs.append(folder/'manifest.json')
        inputs.extend(folder/'episodes'/f'79000_{arm}.json.gz' for arm in ('none', 'field'))
    specs = [hazard_demo.specification(scene, seed=seed) for scene in hazard_demo.SCENES for seed in SEEDS]
    return dict(schema=1, sources=source, inputs={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                specs=specs, original_specs=[hazard_demo.specification(s, seed=79000) for s in hazard_demo.SCENES],
                seeds=list(SEEDS), budgets=list(BUDGETS), animation_seed=86100,
                animations='all four scenes, disabled vs rotation vs receiver_relevant at 4 MB; tight-budget timelines also reported',
                expected_corrected_episodes=72, expected_legacy_parity_episodes=72, expected_original_reproductions=8,
                config=asdict(configuration(1800)), tolerance=TOLERANCE,
                runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
                             import_order='numpy then torch, followed by runtime modules', torch_threads=1,
                             thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}),
                endpoints=dict(primary='time + 4 * declared agent risk weight * raw risk integral',
                               full='primary + effort + failure', raw_risk='dt * truth risk at each executed step endpoint'),
                corrections=['evaluation omitted per-agent risk weight; learner objective was correct',
                             'team-total advertisement bytes were compared with a per-agent ceiling'],
                reporting_provenance='report-only module fingerprint recorded separately; excluded from operational source seal',
                boundaries=['development reproduction; seeds vary sensor noise only',
                            'original hazard channel and full-replay learner differ from compact evidence-only bandwidth runner',
                            'original audit branches disabled during reproduction; operational parity is checked against saved episodes',
                            'no controller, schedule, payload, receiver fitting, support or budget tuning'])


def freeze(out):
    out = Path(out)
    manifest = frozen_manifest()
    if (out/'manifest.json').exists():
        if load(out/'manifest.json') != json.loads(json.dumps(manifest)):
            raise ValueError('Refuse resume: source, configuration, numerical environment or input fingerprint changed')
        return manifest
    if out.exists() and any(out.iterdir()): raise ValueError('Refuse nonempty output without frozen manifest')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('episodes','legacy','original','regressions','figures','errors'): (out/name).mkdir()
    dump(out/'manifest.json', manifest)
    with zipfile.ZipFile(out/'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for relative in manifest['sources']: archive.write(ROOT/relative, relative)
    return manifest


def original_job(task):
    scene, out = task
    out = Path(out); spec = hazard_demo.specification(scene, seed=79000)
    torch.set_num_threads(1); result = []
    for arm in ('none','field'):
        path=out/'original'/scene/f'{arm}.json.gz';path.parent.mkdir(exist_ok=True)
        if path.exists(): episode=load(path)
        else:
            factories=hazard_demo.learner(spec,arm)
            def make(cfg,x,goal,agent_id,communication):
                return factories[int(agent_id)](cfg,x,goal,agent_id=agent_id,communication=communication)
            episode=original_execute(spec,hazard_demo.ARMS[arm][2],hazard_demo.communication(arm),audit=False,
                                     learner_factory=make,world_factory=NetworkWorld,config=configuration(spec['steps']))
            episode['arm']=arm;episode['channel']=hazard_demo.ARMS[arm][2];dump(path,episode)
        reference_path=ROOT/'results/hazard_pilot_v2'/scene/'episodes'/f'79000_{arm}.json.gz'
        archived=load(reference_path)
        comparison=compare_operations(archived,episode,include_events=False)
        packets=compare_values(archived.get('packets'),episode.get('packets'))
        outcomes=compare_values([a['components'] for a in archived['agents']], [a['components'] for a in episode['agents']])
        result.append(dict(scene=scene,arm=arm,operations=comparison,packets=packets,cost_components=outcomes,
                           passed=comparison['passed'] and packets['passed'] and outcomes['passed'],
                           archived_time_risk=[a['components']['time']+a['components']['risk'] for a in archived['agents']],
                           reproduced_time_risk=[a['components']['time']+a['components']['risk'] for a in episode['agents']],
                           episode_file=str(path.relative_to(out)),serialized_gzip_equal=digest(path)==digest(reference_path),
                           serialization_note='timers, omitted evaluator-only audit branches and gzip headers differ; operational projections and packets checked separately'))
        dump(out/'regressions'/f'original_{scene}_{arm}.json',result[-1])
    return result


def regression(out,workers=4):
    freeze(out);out=Path(out);rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(original_job,(s,str(out))) for s in hazard_demo.SCENES]):
            rows.extend(f.result());dump(out/'regression.json',dict(original=rows,passed=all(r['passed'] for r in rows),complete=len(rows)==8))
            print('original regression',len(rows),'/8',flush=True)
    if not all(r['passed'] for r in rows): raise RuntimeError('Original reproduction gate failed; preserved diagnostics, stop before bandwidth comparisons')
    return rows


def episode_row(r,key,budget):
    sent=[e for e in r['events'] if e.get('reason')=='sent']
    refused=[e for e in r['events'] if e.get('kind')=='channel']
    return dict(key=key,scene=r['spec']['scene'],seed=r['spec']['seed'],arm=r['arm'],selector=r['selector'],budget=budget,
                agents=r['agents'],goals=sum(a['success'] for a in r['agents']),collisions=sum(a['collision'] for a in r['agents']),
                timeouts=sum(a['timeout'] for a in r['agents']),carrier_cost=r['agents'][1]['cost'],
                carrier_time_risk=r['agents'][1]['time_risk_cost'],team_cost=sum(a['cost'] for a in r['agents']),
                team_time_risk=sum(a['time_risk_cost'] for a in r['agents']),
                bytes_team={k:sum(a['bytes'][k] for a in r['agents']) for k in BYTE_KEYS},
                bytes_per_agent=[{k:a['bytes'][k] for k in BYTE_KEYS} for a in r['agents']],
                packets=len(sent),refusals={reason:sum(e['reason']==reason for e in refused) for reason in sorted({e['reason'] for e in refused})},
                wall_seconds=r['wall_seconds'],update_seconds=r['update_seconds'],
                episode_file='episodes/'+key+'.json.gz')


def paired_job(task):
    scene,seed,arm,selector,budget,out=task;out=Path(out);spec=hazard_demo.specification(scene,seed=seed)
    key=f'{scene}_{seed}_{arm}_{selector}_{budget}'
    old_path=out/'legacy'/f'{key}.json.gz';new_path=out/'episodes'/f'{key}.json.gz'
    torch.set_num_threads(1)
    old=load(old_path) if old_path.exists() else navigation.execute(spec,arm,selector,byte_budget=budget,accounting_mode='legacy')
    if not old_path.exists(): dump(old_path,old)
    new=load(new_path) if new_path.exists() else navigation.execute(spec,arm,selector,byte_budget=budget,accounting_mode='corrected')
    if not new_path.exists(): dump(new_path,new)
    parity=compare_operations(old,new)
    archived_path=ROOT/'results/bandwidth_selection_v1'/f'{key}.json'
    archived=load(archived_path)
    legacy_row=episode_row(old,key,budget)
    archived_metrics={k:archived[k] for k in ('carrier_cost','team_cost','goals','collisions','packets','refusals')}
    reproduced_metrics={k:legacy_row[k] for k in archived_metrics}
    # Legacy row refusal counts included sent and delivery reasons as well.
    refused=[e.get('reason') for e in old['events'] if e.get('reason') not in ('sent',None)]
    reproduced_metrics['refusals']={k:refused.count(k) for k in set(refused)}
    archived_metrics['bytes']=archived['bytes'];reproduced_metrics['bytes']=legacy_row['bytes_team']
    archived_parity=compare_values(archived_metrics,reproduced_metrics)
    check=dict(key=key,operations=parity,archived_metrics=archived_parity,passed=parity['passed'] and archived_parity['passed'],
               serialized_gzip_equal=digest(old_path)==digest(new_path),
               serialized_difference_expected='corrected cost/raw exposure fields and runtime timers; operational digest reported separately')
    dump(out/'regressions'/f'{key}.json',check)
    row=episode_row(new,key,budget);row['regression_passed']=check['passed'];return row,check


def suite(out,workers=16):
    freeze(out);out=Path(out)
    gate=load(out/'regression.json')
    if not gate['complete'] or not gate['passed']:raise RuntimeError('Require complete passing original reproduction first')
    tasks=[(s,k,'disabled','rotation',4000000,str(out)) for s in hazard_demo.SCENES for k in SEEDS]
    tasks += [(s,k,'reciprocal_fixed',c,b,str(out)) for s in hazard_demo.SCENES for k in SEEDS for c in navigation.SELECTORS for b in BUDGETS]
    rows=[];checks=[];errors=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(paired_job,t):t for t in tasks}
        for f in as_completed(futures):
            try:
                row,check=f.result();rows.append(row);checks.append(check)
                print(len(rows),'/72',row['key'],'parity',check['passed'],flush=True)
            except Exception as error:
                import traceback
                failure=dict(task=futures[f],error=str(error),traceback=traceback.format_exc());errors.append(failure)
                dump(out/'errors'/f'{len(errors)}.json',failure)
            dump(out/'rows.json',sorted(rows,key=lambda r:r['key']))
            dump(out/'parity.json',dict(checks=checks,errors=errors,complete=len(rows)==72,
                                      passed=len(rows)==72 and not errors and all(r['passed'] for r in checks),wall_seconds=time.perf_counter()-started))
    if errors or not all(r['passed'] for r in checks):raise RuntimeError('Accounting parity failed; inspect preserved parity.json before interpretation')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','regression','suite','report','render','all'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.out)
    elif args.command=='regression':regression(args.out,min(args.workers,4))
    elif args.command=='suite':suite(args.out,args.workers)
    elif args.command in ('report','render'):
        from .communication_reconciliation_report import report,render
        (report if args.command=='report' else render)(args.out)
    else:
        regression(args.out,min(args.workers,4));suite(args.out,args.workers)
        from .communication_reconciliation_report import report,render
        report(args.out);render(args.out)


if __name__=='__main__':main()
