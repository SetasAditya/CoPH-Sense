"""New, gated symmetric-fork reconstruction/replay development experiment.

The old decision-sensitive grid is preserved. No gate failure triggers a scene search.
"""
import argparse
import copy
import datetime
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import complementary_reconstruction as cr
from . import decision_sensitive as ds
from .attribution_audits import digest, fixed_state
from .cli import archive_sources, source_hashes
from .cooperative import CommunicationConfig
from .crossing_demo import configuration
from .decision_trace import path_reach
from .evaluation import dump, load
from .hazard_demo import communication
from .maze_family import NetworkWorld
from .model import tensor
from .receiver_corrections import CorrectionLearner, errors
from .receiver_reconstruction import fingerprint, file_hash
from .reconstruction_diagnostics import eligibility, field_queries, ranking
from .route_field import hypothetical_context, values as route_values
from .staggered_demo import execute

NODES = ((0., 0.), (1.5, 0.), (2.5, .6), (2.5, -.6), (3.5, 0.), (5., 0.))
EDGES = ((0, 1), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5))
BRANCHES = (((1, 2), (2, 4)), ((1, 3), (3, 4)))
CASES = ('upper_risk', 'lower_risk', 'clear_control')
SEEDS = tuple(range(82400, 82408))
ARMS = {f'{fit}_{replay}': (fit == 'fit', replay)
        for fit in ('no_fit', 'fit') for replay in ('none', 'current', 'historical')}


def specification(case, seed):
    if case not in CASES:
        raise ValueError(case)
    s = cr.specification('complementary', seed)
    s.update(name=case, scene=case, network_nodes=NODES, network_edges=EDGES,
             goals=[[5., 0.]] * 2, goal=[5., 0.], steps=500,
             bounds=[-1., 6., -1.5, 1.5],
             patches=[] if case == 'clear_control' else
             [[1.7, .45 if case == 'upper_risk' else -.45, .6, .6, .1]],
             prior='New symmetric surveyed fork; geometry known, risk unknown. Independent plants.',
             experiment='development eligibility screen; no outcome-selected geometry')
    return s


def factory_for(spec, fit=False, replay='none'):
    def factory(cfg, x, goal, agent_id, communication):
        a = CorrectionLearner(cfg, x, goal, agent_id=agent_id, communication=communication,
            mode='evidence' if replay == 'none' else 'full', fit_received=fit,
            receiver_fit_mode='precision_bounded', receiver_validation_mode='relative_local',
            receiver_solver_mode='active_set', receiver_search_mode='diagnostic',
            network=(NODES, EDGES), risk_weight=spec['risk_weights'][int(agent_id)],
            unknown_risk=spec['unknown_risk'])
        a.replay_start_mode = replay
        return a
    return factory


def entered(q):
    q = np.asarray(q)
    for name, edge in zip(('upper', 'lower'), ((1, 2), (1, 3))):
        a, b = np.asarray(NODES)[list(edge)]
        d = b-a
        along = float((q-a) @ d / np.linalg.norm(d))
        if .25 < along <= np.linalg.norm(d)+.25 and np.linalg.norm(q-a-along*d/np.linalg.norm(d)) < .25:
            return name
    return None


def branch_scores(agent, x, context=None):
    r = ranking(agent, x, context, True)
    edges = {tuple(sorted(e['edge'])): e for e in r['quadrature'] if -1 not in e['edge']}
    rows = [edges[tuple(sorted(e))] for branch in BRANCHES for e in branch]
    costs = [sum(edges[tuple(sorted(e))]['cost'] for e in b) for b in BRANCHES]
    outgoing = [edges[e] for e in ((1, 2), (1, 3))]
    return dict(upper=costs[0], lower=costs[1], margin=costs[0]-costs[1], quadrature=rows,
        support=[dict(edge=e['edge'], supported=sum(e['route_field']['field_support']), queries=len(e['points'])) for e in outgoing],
        both_supported=all(any(e['route_field']['field_support']) for e in outgoing))


def conditional_range(agent, x, seed, draws=128):
    before = digest(agent)
    design = ds.bootstrap_design(agent)
    blocks = np.unique(design['blocks'])
    score = branch_scores(agent, x)
    if len(blocks) < 2:
        return dict(available=False, reason='fewer_than_two_time_blocks', blocks=len(blocks))
    context = copy.deepcopy(agent.navigation)
    def margin(weights):
        c = copy.deepcopy(context)
        by_id = dict(zip([s.id for s in agent.stages], weights.split(design['n'])))
        for stage in c['route_field_snapshot']['stages']:
            stage['weights'] = by_id[stage['id']].tolist()
        costs = [float(np.sum(np.asarray(e['lengths'])*(1+24*route_values(e['points'], c)))) for e in score['quadrature']]
        return sum(costs[:2])-sum(costs[2:])
    reference = margin(ds.bootstrap_fit(design, np.ones(len(design['blocks']))))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample = rng.choice(blocks, len(blocks), replace=True)
        multiplicity = [int(np.sum(sample == b)) for b in design['blocks']]
        values.append(margin(ds.bootstrap_fit(design, multiplicity)))
    finite = bool(np.isfinite(values).all())
    interval = (score['margin'] + np.quantile(np.asarray(values)-reference, [.025, .975])).tolist() if finite else None
    assert digest(agent) == before
    return dict(available=finite, interval=interval, straddles=bool(finite and interval[0] <= 0 <= interval[1]),
        draws=draws, blocks=len(blocks), linear_solves=draws+1, seed=seed,
        semantics='Uncalibrated conditional range; fixed measured support, geometry and fallback.')


def matched_audit(ctx, packet):
    a, x, world = ctx['agents'][1], ctx['states'][1], ctx['worlds'][1]
    before = digest((a, x, world))
    base = copy.deepcopy(a)
    event = base.assimilate(packet)
    anchors = copy.deepcopy(base.navigation.get('terrain_anchors', []))
    points = np.asarray([p for e in branch_scores(base, x)['quadrature'] for p in e['points']])
    truth = np.column_stack((world.risk_truth(points), world.risk_gradient(points)))
    original, _ = field_queries(base, points)
    measured = digest([(s.readings, s.transitions) for s in base.stages])
    result = {}
    from .replay_budget import prepare
    for name, (fit, replay) in ARMS.items():
        b = copy.deepcopy(base)
        b.fit_received = fit
        b.mode = 'evidence' if replay == 'none' else 'full'
        b.replay_start_mode = replay
        start = b.counterfactual_steps
        d = b.update(x.copy())
        pred, meta = field_queries(b, points)
        mask = np.isfinite(original).all(1) & np.isfinite(pred).all(1)
        c = hypothetical_context(b)
        c['terrain_anchors'] = copy.deepcopy(anchors)
        state = fixed_state(b, x)
        # Separate evaluator-only frozen incumbent replay reach from the update's work.
        probe = copy.deepcopy(b)
        probe.mode = 'full'
        probe.replay_start_mode = 'historical' if replay == 'none' else replay
        prepare(probe, [s.weights for s in probe.stages], probe.damping)
        cost, paths = probe.replay(probe.policy, [s.weights for s in probe.stages], probe.damping)
        reach = [dict(stage=s, **path_reach(p, probe.navigation)) for s, p, _ in paths]
        assert digest([(s.readings, s.transitions) for s in b.stages]) == measured
        assert b.counterfactual_steps-start <= 1024
        result[name] = dict(accepted=d['accepted'], reason=d['reason'], diagnostic=d,
            version=b.version, policy=b.policy.tolist(), control=state['control'],
            fields=[s.weights.tolist() for s in b.stages], errors=errors(pred, truth, mask), support=meta,
            frozen_anchor_score=branch_scores(b, x, c), natural_score=branch_scores(b, x),
            update_flow=b.counterfactual_steps-start, fitting_work=b.message_fit_work-base.message_fit_work,
            replay_probe=dict(valid=cost is not None, reach=reach, eligibility=getattr(probe, 'replay_eligibility', []),
                horizons=getattr(probe, 'replay_plan', {}), handoffs=getattr(probe, 'replay_stage_events', []),
                diagnostic_flow=probe.counterfactual_steps-b.counterfactual_steps))
    assert digest((a, x, world)) == before
    return dict(event=event, packet=packet.identity, sources=packet.sources, branches=result,
        physical_and_operational_state_unchanged=True, measurements_unchanged=True,
        truth_evaluator_only=True, shared_remote_anchors=True)


class Capture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.acquisition = None
        self.last = None
        self.entry = None
        self.cursor = 0
    def boundary(self, ctx):
        branch = entered(ctx['states'][1][:2])
        if branch:
            self.entry = dict(time=ctx['time'], branch=branch)
            raise ds.ScreeningComplete()
        if abs(ctx['time']-.96) < 1e-9:
            self.acquisition = copy.deepcopy(ctx)
        tick = next((t for _, i, t in ctx['batch'] if i == 1), None)
        if tick is None or tick % self.cfg.update_every or ctx['time'] < .96-1e-9:
            return
        sender = copy.deepcopy(ctx['agents'][0])
        sender._ledger_turn = self.cursor
        self.cursor += 1
        packet = sender.export(ctx['states'][0].copy(), '1')
        self.last = copy.deepcopy(ctx), packet


def manifest():
    cfg = replace(configuration(500), route_field_mode='supported_field')
    sources = {k: v for k, v in source_hashes().items()
               if k.startswith('coph_stagewise/') and
               ('campaign' not in k or k.endswith('campaign_reconstruction.py'))}
    m = dict(schema=1, development_only=True, seeds=SEEDS, animation_seed=82400,
        specs={f'{c}_{s}': specification(c, s) for c in CASES for s in SEEDS}, arms=ARMS,
        config=asdict(cfg), communication=asdict(communication('field')), source_hashes=sources,
        gate=dict(before='first actual outgoing entry', support='both outgoing edges have locally supported committed quadrature',
            evidence='valid rectangle-eligible packet with nonzero fitting work on isolated copy',
            sensitivity='abs upper-minus-lower margin <= .5 and 128-draw conditional range contains zero',
            replay='at least one historical incumbent supported replay reaches an outgoing branch',
            all_seeds=True, failure='stop comparisons; do not alter scene or support',
            clear_control_exempt=['sensitivity', 'replay']),
        rationale='Equal geometric branch lengths remove the old 3.4 m detour advantage; no outcome-dependent choice.',
        numerical=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            threads=1, import_order='numpy_then_torch', environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        held_out_executed=False, legacy_grid='results/decision_sensitive_v1; preserved without modification')
    return dict(m, fingerprint=fingerprint(m))


def freeze(root):
    root = Path(root)
    m = manifest()
    if (root/'manifest.json').exists():
        old = load(root/'manifest.json')
        if old['fingerprint'] != m['fingerprint']:
            raise ValueError('Frozen source/configuration mismatch')
        return old
    root.mkdir(parents=True, exist_ok=True)
    for sub in ('screening','snapshots','episodes','status','figures'):
        (root/sub).mkdir(exist_ok=True)
    dump(root/'manifest.json', dict(m, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    archive_sources(root)
    return m


def screen_job(task):
    root, key = task
    root = Path(root)
    m = load(root/'manifest.json')
    spec = m['specs'][key]
    from .records import Config
    cfg, comm = Config(**m['config']), CommunicationConfig(**m['communication'])
    torch.set_num_threads(1)
    began = time.perf_counter()
    capture = Capture(cfg)
    try:
        try:
            execute(spec, 'disabled', comm, audit=False, learner_factory=factory_for(spec),
                    world_factory=NetworkWorld, config=cfg, boundary_callback=capture.boundary)
        except ds.ScreeningComplete:
            pass
        row = dict(id=key, case=spec['scene'], seed=spec['seed'], complete=True, passed=False,
                   entry=capture.entry, acquisition_available=capture.acquisition is not None)
        if capture.last and capture.entry:
            ctx, packet = capture.last
            agent, x = ctx['agents'][1], ctx['states'][1]
            before = digest(ctx)
            score = branch_scores(agent, x)
            bootstrap = conditional_range(agent, x, 182400+spec['seed'])
            audit = matched_audit(ctx, packet) if packet else None
            eligible = eligibility(agent, packet) if packet else []
            fitting = audit['branches']['fit_none']['fitting_work'] if audit else 0
            reach = audit['branches']['no_fit_historical']['replay_probe'] if audit else {}
            checks = dict(support=score['both_supported'], evidence=bool(audit and audit['event']['reason']=='eligible' and fitting>0 and any(e['rectangle'] for e in eligible)),
                sensitivity=abs(score['margin']) <= .5 and bootstrap.get('straddles', False),
                replay=bool(reach.get('valid') and any(p['reaches_decision'] for p in reach.get('reach', []))))
            required = ('support','evidence') if spec['scene']=='clear_control' else tuple(checks)
            row.update(time=ctx['time'], score=score, bootstrap=bootstrap, checks=checks,
                passed=all(checks[k] for k in required), reasons=[k for k in required if not checks[k]],
                snapshot=cr.save_snapshot(root/'snapshots'/f'{key}.pkl.gz', (ctx, packet)), audit=audit)
            assert digest(ctx) == before
        else:
            row['reasons'] = ['missing_prebranch_snapshot']
        row['seconds'] = time.perf_counter()-began
    except Exception as exc:
        import traceback
        row = dict(id=key, case=spec['scene'], seed=spec['seed'], complete=False, passed=False,
                   reasons=['screen_exception'], error=str(exc), traceback=traceback.format_exc(), seconds=time.perf_counter()-began)
    dump(root/'screening'/f'{key}.json.gz', row)
    return dict(id=key, complete=row['complete'], passed=row['passed'], reasons=row.get('reasons', []))


def gate(root, workers=4):
    root = Path(root)
    m = freeze(root)
    tasks = [(str(root), k) for k in m['specs'] if not (root/'screening'/f'{k}.json.gz').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(screen_job, t) for t in tasks]):
            print(json.dumps(f.result()), flush=True)
    rows = [load(root/'screening'/f'{k}.json.gz') for k in m['specs']]
    selected = [c for c in CASES if c != 'clear_control' and all(r['passed'] for r in rows if r['case']==c)]
    result = dict(manifest_fingerprint=m['fingerprint'], complete=all(r['complete'] for r in rows),
        selected=selected, stop=not selected, rows=[{k:v for k,v in r.items() if k not in ('audit','score')} for r in rows],
        hashes={r['id']:file_hash(root/'screening'/f"{r['id']}.json.gz") for r in rows})
    if (root/'gate.json').exists() and fingerprint(load(root/'gate.json')) != fingerprint(result):
        raise ValueError('Gate changed after sealing')
    dump(root/'gate.json', result)
    dump(root/'eligibility_seal.json', dict(manifest=file_hash(root/'manifest.json'), gate=file_hash(root/'gate.json'), selected=selected))
    report(root)
    return result


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    g = load(root/'gate.json')
    rows = [load(p) for p in sorted((root/'screening').glob('*.json.gz'))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, case in enumerate(CASES):
        rr = [r for r in rows if r['case']==case and 'score' in r]
        if not rr:
            continue
        xx = np.arange(len(rr)) + i*9
        margins = np.array([r['score']['margin'] for r in rr])
        intervals = np.array([r['bootstrap']['interval'] for r in rr])
        ax[0].errorbar(xx, margins, yerr=np.maximum(0,np.stack((margins-intervals[:,0], intervals[:,1]-margins))), fmt='o', label=case)
        ax[1].bar(i, sum(r.get('checks', {}).get('replay', False) for r in rr), label=case)
    ax[0].axhline(0, color='k', lw=.7)
    ax[0].set(xlabel='Frozen case / noise seed', ylabel='Upper-minus-lower estimated cost', title='Conditional screening ranges (uncalibrated)')
    ax[0].legend(fontsize=8)
    ax[1].set(xticks=range(3), xticklabels=CASES, ylabel='Historical replay reaches branch / 8', ylim=(0,8.5))
    fig.tight_layout()
    fig.savefig(root/'figures/gate.png', dpi=160)
    plt.close(fig)
    inventory = [{k:v for k,v in r.items() if k not in ('audit','score')} for r in rows]
    dump(root/'gate_inventory.json', inventory)
    checks = {k:sum(r.get('checks', {}).get(k, False) for r in rows) for k in ('support','evidence','sensitivity','replay')}
    text = '# Reconstruction / replay campaign\n\n'
    text += f"Frozen screens: {len(rows)}; completed: {sum(r['complete'] for r in rows)}. Eligible risk cases: {g['selected']}.\n\n"
    text += '| Gate | Passing / screened |\n|---|---:|\n' + ''.join(f'| {k} | {v}/{len(rows)} |\n' for k,v in checks.items())
    text += '\nThe symmetric fork is a new development fixture, preserving the old failed 160-row grid. Seeds vary sensor noise only; this is three designed cases, not 24 independent environments.\n\n'
    if g['stop']:
        text += '**Eligibility failed: no navigation treatments or treatment GIFs were run.** No scene, support, learner, or controller settings were adjusted after screening. This result does not establish absence of reconstruction or replay benefit.\n\n'
    text += 'Matched six-arm updates use isolated complete receiver copies and identical incoming packets. Remote route anchors are held fixed for the attribution scores; natural regenerated scores are logged separately. Evaluator truth appears only in error calculations. Replay reach probes are diagnostic work, separate from each local update’s 1,024-flow ceiling. All snapshots and exclusions are retained.\n\n![Gate margins and replay reach](figures/gate.png)\n'
    (root/'REPORT.md').write_text(text)
    (root/'FINDINGS.md').write_text(text)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('freeze','gate','report'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args(argv)
    if args.command=='freeze': freeze(args.out)
    elif args.command=='gate': gate(args.out, args.workers)
    else: report(args.out)


if __name__ == '__main__':
    main()
