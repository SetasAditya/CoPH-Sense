"""Frozen saved-state experiment; no operational decision selector is enabled."""
import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import communication_navigation as cn
from .attribution_audits import digest
from .communication_reconciliation import compare_operations
from .decision_packet import advertise, candidates, costs, rank, shortest
from .evaluation import dump, load
from .hazard_demo import SCENES, specification
from .maze_learning import maze_route
from .model import tensor
from .simple_communication import SimpleRepairLink, novel_payload, route_advertisement

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'results/decision_packet_v1'
SEEDS = (86100, 86101)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'coph_stagewise').glob('*.py'))}
    inputs = {f'results/simple_fixes_v1/communication/episodes/{s}_{seed}_rotation_4000000.json.gz':
        sha(ROOT/f'results/simple_fixes_v1/communication/episodes/{s}_{seed}_rotation_4000000.json.gz')
        for s in SCENES for seed in SEEDS}
    m = dict(schema=1, sources=sources, inputs=inputs, specs=[specification(s, seed) for s in SCENES for seed in SEEDS],
        capture='first send opportunity for each (sender, receiver retained-stage ID, sender retained-stage ID) pair; all directions; no outcome selection',
        packet_candidates='one ordinary local-only bootstrap export per retained stage; incumbent rotation cursor preserved',
        endpoint='true spatial production-quadrature route score of actual isolated post-update route; no execution or mission-benefit claim',
        estimate='sender locally measured bootstrap means on certified support; receiver advertised estimates elsewhere; nominal 1 m/s route score',
        selection='largest positive predicted route improvement, then fewer payload bytes, then newer stage; otherwise ordinary rotation',
        readiness_gate='baseline operational parity, state isolation, production-route parity; strict lower mean actual selection regret than BOTH rotation and coverage, and positive selected benefit in at least two scenes; no navigation treatments in this diagnostic',
        timing='pre-offer frozen physical states; receiver clock advanced on isolated copy to max(sender,receiver) for legal immediate receipt; no simulated new sensing; actual next-boundary delivery is not predicted',
        resources='full standalone zlib advertisement measured, no cache savings assumed; diagnostic exports, bootstrap work, updates, flow, timing counted separately',
        limitations=['development scenes and noise-only seeds', 'true complete-route quadrature is evaluator-only and omits future learning/motion',
            'anchor prediction omits bounded cell eviction/replacement and subsequent local-model fitting', 'no calibrated uncertainty or rate adaptation',
            'no arbitrary conversion between bytes and route cost; benefit ranking and actual bytes reported separately'],
        runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}, import_order='numpy,torch,runtime'))
    p = out/'manifest.json'
    if p.exists():
        if load(p) != json.loads(json.dumps(m)): raise ValueError('source/config/input/runtime fingerprint changed')
    else:
        with p.open('x') as f: json.dump(m, f, indent=2)
        for d in ('sources','episodes','snapshots','audits','figures'): (out/d).mkdir()
        for name in sources: (out/'sources'/Path(name).name).write_bytes((ROOT/name).read_bytes())
    return m


def capture_job(task):
    spec, out = task; root = Path(out); torch.set_num_threads(1)
    key = f"{spec['scene']}_{spec['seed']}"; path = root/'episodes'/f'{key}.json.gz'
    if path.exists(): return load(root/'episodes'/f'{key}_parity.json')
    old = cn.FixedRateRepairLink; entries = []; seen = set()
    class CaptureLink(SimpleRepairLink):
        def offer(self, sender, state, receiver, receiver_state, tick):
            label = (sender.agent_id, receiver.stages[-1].id, sender.stages[-1].id)
            if label not in seen:
                seen.add(label); p = root/'snapshots'/f'{key}_{len(entries):03d}.pkl.gz'
                snapshot = copy.deepcopy((sender, state, receiver, receiver_state, spec, tick))
                with gzip.open(p, 'wb') as f: pickle.dump(snapshot, f)
                entries.append(dict(path=str(p.relative_to(root)), sha256=sha(p), key=key,
                    sender=sender.agent_id, receiver=receiver.agent_id, time=sender.now, stage_pair=label[1:]))
            return super().offer(sender, state, receiver, receiver_state, tick)
    try:
        cn.FixedRateRepairLink = CaptureLink
        result = cn.execute(spec, 'reciprocal_fixed', 'rotation', byte_budget=4000000)
    finally: cn.FixedRateRepairLink = old
    archived = load(ROOT/f"results/simple_fixes_v1/communication/episodes/{key}_rotation_4000000.json.gz")
    parity = compare_operations(result, archived)
    dump(path, result); dump(root/'snapshots'/f'{key}_inventory.json', entries)
    row = dict(key=key, parity=parity, snapshots=len(entries))
    dump(root/'episodes'/f'{key}_parity.json', row); return row


def truth_values(points, spec):
    points = np.asarray(points); value = np.zeros(len(points))
    for x, y, sx, sy, amplitude in spec['patches']:
        value += amplitude*np.exp(-.5*(((points[:, 0]-x)/sx)**2+((points[:, 1]-y)/sy)**2))
    return value


def actual_route(agent, state):
    context = copy.deepcopy(agent.navigation)
    _, meta = maze_route(agent, tensor(state), context, 0)
    return meta.get('route_nodes', [])


def true_route_cost(ad, nodes, true_costs):
    if not nodes: return None
    indices = []
    for i, j in zip([-1]+list(nodes[:-1]), nodes):
        matches = [k for k, r in enumerate(ad.records) if (r['i'], r['j']) == (i, j)
            or (i != -1 and (r['i'], r['j']) == (j, i))]
        if not matches: return None
        indices.append(matches[0])
    return float(sum(true_costs[k] for k in indices))


def audit_job(task):
    entry, out = task; root = Path(out); output = root/'audits'/(Path(entry['path']).name+'.json')
    if output.exists(): return load(output)
    torch.set_num_threads(1); began = time.perf_counter()
    path = root/entry['path']; assert sha(path) == entry['sha256']
    with gzip.open(path, 'rb') as f: sender, x, receiver, rx, spec, tick = pickle.load(f)
    original = digest((sender, receiver)); measurements = digest([(s.readings,s.transitions) for s in receiver.stages])
    physical = copy.deepcopy(receiver.current_state)
    # Clock-only hypothetical intervention, disclosed in the frozen protocol.
    now = max(sender.now, receiver.now); sender.now = now; receiver.now = now
    ad = advertise(receiver, rx, sender.agent_id)
    # Force a real serialized round-trip; sender receives no receiver object.
    body = json.loads(__import__('zlib').decompress(ad.wire())); body['records'] = tuple(body['records'])
    ad = type(ad)(**body)
    packets, export_work = candidates(sender, x, receiver.agent_id)
    selected, prediction = rank(sender, x, receiver.agent_id, ad, packets)
    coverage_sender = copy.deepcopy(sender)
    coverage, coverage_meta = novel_payload(coverage_sender, x, receiver.agent_id,
        route_advertisement(receiver, rx, sender.agent_id))
    rotation = copy.deepcopy(sender).export(x, receiver.agent_id)
    rotation_stage = rotation.stage if rotation else None
    choices = dict(rotation=rotation_stage, coverage=coverage.stage if coverage else rotation_stage,
                   decision=selected.stage if selected else rotation_stage)
    # All three methods select from identical per-stage packets. The clock/export
    # seed and candidate contents are identical; sequence counters remain on copies.
    true = costs(ad, [truth_values(r['points'], spec) for r in ad.records])
    predicted_by_stage = {r['stage']: r for r in prediction['candidates']}
    branches = []; live_route = actual_route(receiver, rx)
    ad_route = shortest(ad, costs(ad))
    for packet in [None]+packets:
        a = copy.deepcopy(receiver); event = None if packet is None else a.assimilate(packet)
        before_fit = a.message_fit_work; before_id = a.identification_steps
        report = a.update(rx.copy()); nodes = actual_route(a, rx); u, _ = a.action(rx.copy())
        assert digest([(s.readings,s.transitions) for s in a.stages]) == measurements
        assert a.current_state == physical
        assert report['counterfactual_steps'] <= 1024
        branches.append(dict(stage=None if packet is None else packet.stage,
            packet=None if packet is None else packet.identity, event=event, nodes=nodes,
            true_route_cost=true_route_cost(ad, nodes, true), control=u.tolist(),
            model_accepted=report['accepted'], update_reason=report['reason'], version=a.version,
            flow=report['counterfactual_steps'], fitting_work=a.message_fit_work-before_fit,
            identification_work=a.identification_steps-before_id))
    baseline = branches[0]['true_route_cost']; useful = []
    for b in branches[1:]:
        b['actual_benefit'] = None if baseline is None or b['true_route_cost'] is None else baseline-b['true_route_cost']
        pred = predicted_by_stage[b['stage']]
        b['predicted_benefit'] = pred['benefit']; b['predicted_nodes'] = pred['route']['nodes'] if pred['route'] else []
        b['route_prediction_agrees'] = b['nodes'] == b['predicted_nodes']
        if b['true_route_cost'] is not None: useful.append(b['true_route_cost'])
    # Restore the clock-only changes before checking the complete captured state.
    with gzip.open(path, 'rb') as f: saved = pickle.load(f)
    sender.now = saved[0].now; receiver.now = saved[2].now
    assert digest((sender, receiver)) == original
    selection = {}
    for name, stage in choices.items():
        branch = next((b for b in branches[1:] if b['stage'] == stage), None)
        selection[name] = dict(stage=stage, actual_benefit=branch['actual_benefit'] if branch else None,
            regret=branch['true_route_cost']-min(useful) if branch and useful and branch['true_route_cost'] is not None else None,
            payload_bytes=next((r['payload_bytes'] for r in prediction['candidates'] if r['stage']==stage), None))
    result = dict(entry, scene=spec['scene'], seed=spec['seed'], candidates=len(packets), selection=selection,
        prediction=prediction, branches=branches, coverage=coverage_meta,
        production_route_parity=ad_route is not None and live_route==ad_route['nodes'],
        live_route=live_route, advertised_route=ad_route, state_isolation=True,
        clock_advance=now-saved[2].now, export_work=export_work,
        coverage_export_work=coverage_sender.export_work-sender.export_work,
        seconds=time.perf_counter()-began)
    dump(output, result); return result


def suite(out, workers=4):
    m = freeze(out); root = Path(out)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parity = list(pool.map(capture_job, [(s, str(root)) for s in m['specs']]))
    dump(root/'baseline_parity.json', parity)
    if not all(r['parity']['passed'] for r in parity):
        raise RuntimeError('baseline operational parity failed; stop before packet comparison')
    entries = [e for p in sorted((root/'snapshots').glob('*_inventory.json')) for e in load(p)]
    dump(root/'snapshot_inventory.json', entries)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = []
        for row in pool.map(audit_job, [(e, str(root)) for e in entries]):
            rows.append(row); print(len(rows), '/', len(entries), row['key'], row['candidates'], flush=True)
    dump(root/'rows.json', rows); return report(root)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(out); rows = load(root/'rows.json')
    eligible = [r for r in rows if r['candidates'] >= 2 and all(r['selection'][n]['regret'] is not None for n in ('rotation','coverage','decision'))]
    metrics = {}
    for name in ('rotation','coverage','decision'):
        rr = [r['selection'][name] for r in eligible]
        metrics[name] = dict(snapshots=len(rr), mean_regret=float(np.mean([r['regret'] for r in rr])) if rr else None,
            mean_benefit=float(np.mean([r['actual_benefit'] for r in rr])) if rr else None,
            beneficial=sum(r['actual_benefit']>1e-10 for r in rr), harmful=sum(r['actual_benefit']< -1e-10 for r in rr))
    candidates_all = [b for r in rows for b in r['branches'][1:]]
    positives = [b for b in candidates_all if b['predicted_benefit'] is not None and b['predicted_benefit']>1e-10]
    mean_status = float(np.mean([r['prediction'].get('status_bytes',0) for r in rows]))
    scene_wins = sorted({r['scene'] for r in eligible if r['selection']['decision']['actual_benefit']>1e-10})
    integrity = all(r['production_route_parity'] and r['state_isolation'] for r in rows)
    gate = bool(eligible and integrity and len(scene_wins)>=2 and all(
        metrics['decision']['mean_regret'] < metrics[n]['mean_regret']-1e-10 for n in ('rotation','coverage')))
    summary = dict(snapshots=len(rows), common_multi_candidate=len(eligible), metrics=metrics,
        candidates=len(candidates_all), predicted_positive=len(positives), positive_route_agreement=sum(b['route_prediction_agrees'] for b in positives),
        positive_actual_improvement=sum(b['actual_benefit'] is not None and b['actual_benefit']>1e-10 for b in positives),
        production_route_parity=sum(r['production_route_parity'] for r in rows), integrity=integrity,
        mean_full_status_bytes=mean_status, min_status_bytes=min(r['prediction'].get('status_bytes',0) for r in rows),
        max_status_bytes=max(r['prediction'].get('status_bytes',0) for r in rows),
        useful_scenes=scene_wins, numerical_readiness_gate=gate, navigation_treatments=0,
        diagnostic_seconds=sum(r['seconds'] for r in rows),
        flow=sum(b['flow'] for r in rows for b in r['branches']),
        error='none' if integrity else 'production-route parity or isolation failure; do not interpret performance')
    dump(root/'summary.json', summary)
    paired = [dict(key=r['key'], time=r['time'], sender=r['sender'], candidates=r['candidates'],
        **{n:r['selection'][n] for n in ('rotation','coverage','decision')}) for r in eligible]
    dump(root/'paired.json', paired)
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    valid = [b for b in candidates_all if b['predicted_benefit'] is not None and b['actual_benefit'] is not None]
    axes[0].scatter([b['predicted_benefit'] for b in valid],[b['actual_benefit'] for b in valid],s=12,alpha=.5)
    axes[0].axhline(0,color='gray'); axes[0].axvline(0,color='gray')
    axes[0].set(xlabel='Predicted route benefit',ylabel='True quadrature benefit after actual update')
    axes[1].bar(list(metrics),[v['mean_regret'] or 0 for v in metrics.values()]); axes[1].set(ylabel='Mean one-packet selection regret',title=f'Common snapshots: {len(eligible)}')
    axes[2].hist([r['prediction'].get('status_bytes',0) for r in rows]); axes[2].set(xlabel='Full compressed advertisement bytes',ylabel='Snapshots')
    fig.tight_layout(); fig.savefig(root/'figures/decision_value.png'); plt.close(fig)
    lines = ['# Decision-based packet ranking: saved-state experiment','',
        f"{len(rows)} snapshots; {len(eligible)} common snapshots with at least two candidate packets. Production route parity {summary['production_route_parity']}/{len(rows)}. Numerical readiness gate: **{'PASS' if gate else 'FAIL'}**.",'',
        '| Selector | Mean selection regret | Mean true route benefit | Beneficial / harmful |','|---|---:|---:|---:|']
    lines += [f"| {n} | {v['mean_regret']} | {v['mean_benefit']} | {v['beneficial']} / {v['harmful']} |" for n,v in metrics.items()]
    lines += ['',f"Across {len(candidates_all)} packet candidates, {len(positives)} have predicted positive benefit; {summary['positive_actual_improvement']} actually improve the post-update spatial route score and {summary['positive_route_agreement']} predict the exact post-update route. Denominators include both directions and repeated dependent states; they are not independent experimental trials.",
        '',f"Standalone compressed decision advertisements average {mean_status:.1f} bytes (range {summary['min_status_bytes']}–{summary['max_status_bytes']}). This is a measured format size, not a charged live-channel campaign or a cached/delta overhead claim.",
        '', 'The sender scores ordinary local-only field summaries using advertised current values/support on competing graph edges and receiver task preferences. The reference estimate replaces receiver values only within sender measured support. Missing knowledge remains conditional receiver estimates. Candidate packet prediction approximates the conservative anchor envelope; audits execute real assimilation, fitting, validation, rollback and routing on copies.',
        '', 'The evaluator prices the actual branch route after each copied update using complete static Gaussian truth with production edge/connector quadrature. This measures one-packet decision ranking, not executed travel time, full mission cost, safety, long-term information value or future sensing. Receiver copy clocks advance to the common offer time without new observations; this is an immediate fixed-state audit, not an exact next-boundary delivery simulation.',
        '', 'Eight preserved rotation episodes were regenerated only to capture both learning states, which archived JSON does not contain. Baseline states, controls, model decisions, operational work and events must reproduce before comparison. All snapshot and source hashes are sealed. No navigation treatments, new scenes, cost tuning, learner changes or default changes were run.',
        '', 'A failed gate stops progression: a score that changes a decision according to its own incomplete field is not automatically a reliable estimator of recipient benefit. A pass supports a separately frozen navigation comparison only.',
        '', '[Prediction, selection regret and overhead](figures/decision_value.png). See summary.json, paired.json, per-snapshot audits and baseline_parity.json for all outcomes.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n'); return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['freeze','suite','report']); p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    print({'freeze':lambda:freeze(a.out),'suite':lambda:suite(a.out,a.workers),'report':lambda:report(a.out)}[a.command]())
