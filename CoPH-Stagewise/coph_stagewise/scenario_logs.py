"""Read-only presentation adapters for preserved CoPH experiment logs.

No learner is updated here. Missing historical signals remain None/unavailable.
Native encounter paths were subsampled *after* integration; their first sample is
at 0.1 s, followed by 0.5 s, not at 0.0 s. Uncertainty frames are saved-state
queries and must never be presented as a physical time sequence.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def source_record(path, root=None):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(chunk)
    return {'path': str(path.relative_to(root)) if root is not None else str(path),
            'sha256': sha.hexdigest(), 'bytes': path.stat().st_size}


def _index(times, time):
    return max(0, min(len(times) - 1, int(np.searchsorted(times, time, side='right') - 1)))


def validate_times(times):
    values = np.asarray(times, float)
    if not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError('Expected finite, strictly increasing logged timestamps')
    return values


def load_stagewise_episode(path):
    """Return the original decoded episode after structural checks; no inference."""
    episode = read_json(path)
    for seq in episode['trajectories']:
        validate_times([row['time'] for row in seq])
        for row in seq:
            if row.get('control_model', {}).get('version') != row['version']:
                raise ValueError('Control-time model/version mismatch')
            if not np.isfinite(row['x'] + row['u']).all():
                raise ValueError('Nonfinite logged state/control')
    return episode


def stagewise_frame(episode, time):
    """Latest logged state at/before time; no positional interpolation.

    An active agent's control_model is the exact snapshot used for the held-input
    command. Its support mask is still required when rendering a surface.
    """
    agents = []
    for ident, seq in enumerate(episode['trajectories']):
        if time < seq[0]['time']:
            agents.append({'agent': ident, 'active': False, 'state': episode['spec']['initials'][ident],
                           'model': None, 'control': None, 'navigation': None})
            continue
        idx = _index([row['time'] for row in seq], time)
        row = seq[idx]
        ended = time >= seq[-1]['end_time']
        obs = [o for o in episode['observations'][ident] if o['time'] <= time + 1e-10]
        agents.append({'agent': ident, 'active': not ended,
                       'state': seq[-1]['after'] if ended else row['x'],
                       'logged_time': seq[-1]['end_time'] if ended else row['time'],
                       'model': row['control_model'], 'model_version': row['version'],
                       'control': None if ended else row['u'], 'nominal': row['nominal'],
                       'navigation': row['navigation'], 'observation': obs[-1] if obs else None,
                       'trajectory_index': idx})
    return {'time': float(time), 'agents': agents,
            'events': [e for e in episode['events'] if e.get('time', np.inf) <= time + 1e-10],
            'world': episode['spec'], 'outcome': episode['agents']}


def load_drag(archive, seed=87000, after_drag=.45,
              arms=('coph_online_static_protected', 'coph_known_static_protected',
                    'phmarl_assigned_static_protected')):
    root = Path(archive) / 'native_drag_shift_v1'
    rows = read_json(root / 'rows.json')
    result = {'scenario': 'unexpected_dynamics', 'seed': seed, 'dt': .1,
              'switch_step': 200, 'switch_seconds': 20., 'after_drag': after_drag,
              'arms': {}, 'sources': [source_record(root / 'manifest.json', archive),
                                     source_record(root / 'rows.json', archive)],
              'unavailable': ['native velocities', 'force decomposition', 'spatial environmental field'],
              'selection': 'lowest declared seed; positive drag-change direction selected before visualization'}
    for arm in arms:
        row = next(r for r in rows if r['seed'] == seed and r['protocol'] == arm and r['d2'] == after_drag)
        tag = f'{arm}_seed{seed}_d{round(after_drag*100):03d}'
        stem = root / f'{arm}_{tag}'
        npz = Path(str(stem) + '.npz')
        trace_path = Path(str(stem) + '_trace.json.gz')
        with np.load(npz, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        trace = read_json(trace_path)
        if arrays['positions'].shape != (401, 4, 2) or len(trace) != 400:
            raise ValueError('Unexpected native 400-step record shape')
        if not np.isfinite(arrays['positions']).all():
            raise ValueError('Nonfinite native trajectory')
        drag = np.full((401, 4), .25 if 'known' in arm else -np.expm1(-.06))
        if arm.startswith('phmarl'):
            drag = None
        else:
            for update in row.get('drag_updates', []):
                drag[update['step']:, update['agent']] = update['committed']
        # The trace records per-agent assigned goals, hence the full landmark set.
        first_goals = [t['goal'] for t in trace[0]]
        second_goals = [t['goal'] for t in trace[200]]
        result['arms'][arm] = {'arrays': arrays, 'times': np.arange(401) * .1,
                              'trace': trace, 'drag_estimate': drag, 'updates': row.get('drag_updates'),
                              'phase1_goals': first_goals, 'phase2_goals': second_goals,
                              'outcome': row}
        result['sources'] += [source_record(npz, archive), source_record(trace_path, archive)]
    return result


def drag_frame(log, step):
    step = min(400, max(0, int(step)))
    arms = {}
    for name, arm in log['arms'].items():
        k = min(step, 399)
        a = arm['arrays']
        trace = arm['trace'][k]
        arms[name] = {'positions': a['positions'][step],
                      'goals': arm['phase2_goals'] if step >= 200 else arm['phase1_goals'],
                      'control': a['actions'][k].reshape(4, 2) if step < 400 else None,
                      'nominal': np.array([r['nominal'] for r in trace]),
                      'targets': np.array([r['target'] for r in trace]),
                      'drag_estimate': arm['drag_estimate'][step] if arm['drag_estimate'] is not None else None,
                      'mean_cumulative_return': float(a['rewards'][:step].reshape(step, 4).mean(1).sum()) if step else 0.,
                      'trace': trace, 'outcome': arm['outcome']}
    return {'time': step * .1, 'step': step, 'true_drag': .25 if step < 200 else log['after_drag'], 'arms': arms}


def load_encounter(archive, seed=300, family='circle_swap', arm='coph_online'):
    root = Path(archive) / 'native_final_layer_v1'
    rows = read_json(root / 'liveness_rows.json')
    chosen = {layer: next(r for r in rows if r['layer'] == layer and r['family'] == family
                         and r['seed'] == seed and r['arm'] == arm) for layer in ('base', 'final')}
    for row in chosen.values():
        q = np.asarray(row['path'], float)
        if q.shape != (100, 4, 2) or not np.isfinite(q).all():
            raise ValueError('Expected logged 4-step-subsampled encounter trajectory')
        row['positions'] = np.concatenate((np.asarray(row['starts'])[None], q))
    times = np.r_[0., .1 + .4 * np.arange(100)]
    validate_times(times)
    return {'scenario': 'symmetric_encounter', 'seed': seed, 'family': family,
            'arms': chosen, 'times': times,
            'unavailable': ['nominal/applied controls', 'yield events', 'model revisions', 'velocities'],
            'semantics': 'Programmed priority/protection; not learned yielding. Sampled separation is not continuous minimum separation.',
            'selection': 'lowest declared seed and four-agent swap family fixed by presentation protocol',
            'sources': [source_record(root / 'manifest.json', archive), source_record(root / 'liveness_rows.json', archive)]}


def encounter_frame(log, sample):
    sample = min(100, max(0, int(sample)))
    return {'time': float(log['times'][sample]), 'sample': sample,
            'arms': {name: {'positions': row['positions'][sample], 'goals': row['goals'],
                            'outcome': {k: row[k] for k in ('success', 'new', 'time')}}
                     for name, row in log['arms'].items()}}


def load_uncertainty(archive, group='complementary_79100_0', pairs=24):
    """Saved-state query sequence. Index is not mission time; no path is invented."""
    archive = Path(archive)
    decisions = read_json(archive / 'reconstruction_decision_v1/rows.json')
    certificates = read_json(archive / 'route_certificate_v1/rows.json')
    index = {(r['group'], r['pair']): r for r in certificates}
    rows = []
    for row in decisions:
        if row['group'] == group and row['pair'] < pairs:
            rows.append({'decision': row, 'certificate': index[(row['group'], row['pair'])]})
    if len(rows) != pairs:
        raise ValueError('Requested saved-state query block unavailable')
    return {'scenario': 'route_uncertainty', 'group': group, 'rows': rows,
            'time_semantics': 'query index; no physical time or navigation trajectory',
            'unavailable': ['archived sampled cost-gap distribution', 'archived path coordinates'],
            'selection': 'first declared acquisition group, contiguous first queries; no outcome filtering',
            'sources': [source_record(archive / f'{name}/rows.json', archive)
                        for name in ('reconstruction_decision_v1', 'route_certificate_v1')]}


def regenerate_uncertainty_geometry(archive, pairs=12):
    """Recover first-group paths with frozen RNG; assert all saved cost estimates.

    Reads trusted local pickle snapshots. No learner update or uncertainty
    resampling occurs; uncertainty flags come directly from the saved archive.
    """
    from .route_certificate_suite import packets
    from .reconstruction_decision_suite import path, cost
    from .terrain_warning import terrain_values
    root = Path(archive)
    acquisition = sorted((root / 'complementary_reconstruction_v1/snapshots').glob('*_acquisition.pkl.gz'))
    if not acquisition:
        raise FileNotFoundError('Trusted preserved acquisition snapshots required')
    source = acquisition[0]
    parts = source.name[:-7].split('_')
    entry = {'path': str(source), 'meta': {'seed': int(parts[1]), 'case': parts[0], 'group': 'acquisition'}}
    group = packets(entry)[0]
    saved = load_uncertainty(root, group['group'], pairs)
    rng = np.random.RandomState(187000)
    geometry = []
    for k in range(pairs):
        paths = [path(rng, group['reps']['field'][0]) for _ in range(2)]
        true = [cost(p, lambda q: group['world'].risk_truth(q)) for p in paths]
        np.testing.assert_allclose(true, saved['rows'][k]['decision']['true'], rtol=0, atol=1e-12)
        reps = {}
        for name, (anchors, gradient) in group['reps'].items():
            estimated = [cost(p, lambda q: terrain_values(q, list(map(tuple, anchors)), unknown=.02)) for p in paths]
            np.testing.assert_allclose(estimated, saved['rows'][k]['decision'][name]['est'], rtol=0, atol=1e-12)
            reps[name] = {'anchors': anchors, 'estimated': estimated,
                          'certificate': saved['rows'][k]['certificate'][name]}
        geometry.append({'pair': k, 'paths': paths, 'true': true, 'reps': reps})
    return {'group': group['group'], 'queries': geometry,
            'patches': [list(p) for p in getattr(group['world'], 'patches', [])],
            'semantics': 'paths regenerated from original seed/snapshot; all displayed costs checked against saved rows; uncertainty flags read directly, no Gaussian resampling',
            'source': source_record(source, root)}


def load_crossing(archive, arms=('disabled', 'reciprocal_fixed'), seed=76000):
    root = Path(archive) / 'crossing_traverses_v1'
    manifest = read_json(root / 'manifest.json')
    if seed != manifest['animation_seed']:
        raise ValueError('Presentation must retain the manifest-selected crossing animation seed')
    paths = {arm: root / 'episodes' / f'{seed}_{arm}.json.gz' for arm in arms}
    return {'scenario': 'reciprocal_discovery', 'seed': seed,
            'episodes': {arm: load_stagewise_episode(path) for arm, path in paths.items()},
            'sources': [source_record(root / 'manifest.json', archive)] + [source_record(p, archive) for p in paths.values()],
            'unavailable': [], 'semantics': 'Independent planar plants; surveyed network, unknown closures; model surface is shaping energy.'}


def load_hazards(archive, arms=('field', 'decision'), scene='double_mild', seed=79000):
    root = Path(archive) / 'hazard_pilot_v2' / scene
    manifest = read_json(root / 'manifest.json')
    if seed != manifest['animation_seed']:
        raise ValueError('Presentation must retain the manifest-selected hazard animation seed')
    paths = {arm: root / 'episodes' / f'{seed}_{arm}.json.gz' for arm in arms}
    return {'scenario': 'heterogeneous_hazards', 'seed': seed, 'scene': scene,
            'episodes': {arm: load_stagewise_episode(path) for arm, path in paths.items()},
            'sources': [source_record(root / 'manifest.json', archive)] + [source_record(p, archive) for p in paths.values()],
            'unavailable': [], 'semantics': 'Scout risk weight .25, carrier6; 25-second stagger; field is shared route evidence, physical terrain remains planar.'}


load_dynamics = load_drag
load_encounters = load_encounter


def audit(archive):
    """Inventory actual signals and all recorded failures; no new experiment runs."""
    import gc
    root = Path(archive)
    records = {'schema': 1, 'archive_mutated': False, 'new_experiments': False,
               'scenarios': {}, 'failure_inventory': [], 'verification': {},
               'selection_rule': 'Existing animation seeds where declared; otherwise lowest declared seed. No outcome-based seed replacement.'}
    native = load_drag(root)
    records['scenarios']['dynamics'] = {
        'selected': {'seed': native['seed'], 'drag_after': native['after_drag']},
        'selection_status': 'Original manifest declares seeds but no animation seed; lowest seed chosen for presentation.',
        'available': ['401 physical positions at .1 s', '400 actual actions/rewards/contact flags',
                      '400 nominal/applied/target protection traces', 'per-agent drag updates',
                      'assigned true goals in both phases'],
        'unavailable': native['unavailable'], 'sources': native['sources'],
        'time': 'State t=k*.1; command/reward k spans [k*.1,(k+1)*.1]; switch20s.'}
    for scenario, loader in [('hazards', load_hazards), ('reciprocity', load_crossing)]:
        log = loader(root)
        counts = {}
        for arm, ep in log['episodes'].items():
            counts[arm] = {'trajectory_lengths': [len(q) for q in ep['trajectories']],
                           'packet_count': len(ep['packets']), 'event_count': len(ep['events']),
                           'update_counts': [len(q) for q in ep['updates']]}
            for ident, rows in enumerate(ep['trajectories']):
                if stagewise_frame(ep, rows[-1]['end_time'])['agents'][ident]['state'] != rows[-1]['after']:
                    raise ValueError('Terminal state did not match exact log')
        records['scenarios'][scenario] = {
            'selected': {'seed': log['seed'], 'scene': log.get('scene')},
            'selection_status': 'Existing manifest animation_seed retained.',
            'available': ['physical q,v', 'nominal/applied controls', 'attraction/shaping forces',
                          'exact control-time field coefficients/centers/target/policy/support',
                          'observed geometry/sensor footprint', 'route/waypoint', 'packets/delivery decisions',
                          'before/after committed models', 'version IDs', 'per-step cost components'],
            'unavailable': [], 'sources': log['sources'], 'counts': counts,
            'time': 'Logged absolute mission seconds; no interpolation. Pre-activation state marked inactive; terminal state held from logged after.'}
        del log
        gc.collect()
    encounter = load_encounter(root)
    records['scenarios']['encounters'] = {
        'selected': {'seed': 300, 'family': 'circle_swap', 'controller': 'coph_online'},
        'selection_status': 'Original manifest declares seeds but no animation seed; lowest seed and four-agent-swap family chosen for presentation.',
        'available': ['logged starts/goals', '100 post-step path samples', 'success/contact count/time-to-goals'],
        'unavailable': encounter['unavailable'], 'sources': encounter['sources'],
        'time': 'Prepend t=0 start; samples q[::4] have times .1,.5,...39.7 s. No interpolation or invented force traces.',
        'comparison': 'base versus final protection; attraction and CoPH are encounter baselines, not phMARL.',
        'learning': 'Yielding is a programmed execution rule, not learned strategy discovery.'}
    uncertainty = load_uncertainty(root, pairs=12)
    geometry = regenerate_uncertainty_geometry(root, pairs=12)
    records['scenarios']['uncertainty'] = {
        'selected': {'group': uncertainty['group'], 'pairs': list(range(12))},
        'selection_status': 'First original group and first12 contiguous query pairs; not selected for favorable outcomes.',
        'available': ['saved true/predicted costs and regret', 'saved certified/flagged/correct indicators',
                      'recovered original candidate paths with all cost estimates checked to1e-12',
                      'matched anchors reconstructed from same original source measurements'],
        'unavailable': ['archived 400-sample cost-gap distributions', 'physical mission time/navigation trajectory'],
        'sources': uncertainty['sources'] + [geometry['source']],
        'time': 'Saved-state query index, not simulation time.',
        'regenerated': geometry['semantics']}
    # Inventory all available episodes, not just the displayed favorable example.
    for row in read_json(root / 'native_drag_shift_v1/rows.json'):
        if row['phase2_coverage'] < 1 or row['new_contact_onsets']:
            records['failure_inventory'].append({'study': 'native_drag_shift_v1',
                **{k: row[k] for k in ('protocol', 'seed', 'd2', 'phase2_coverage', 'new_contact_onsets')}})
    for row in read_json(root / 'native_final_layer_v1/liveness_rows.json'):
        if not row['success'] or row['new']:
            records['failure_inventory'].append({'study': 'native_final_layer_v1/liveness',
                **{k: row[k] for k in ('layer', 'family', 'seed', 'arm', 'success', 'new', 'time')}})
    for row in read_json(root / 'native_final_layer_v1/native_rows.json'):
        if not row['coverage'] or row['new']:
            records['failure_inventory'].append({'study': 'native_final_layer_v1/native',
                **{k: row[k] for k in ('layer', 'seed', 'arm', 'coverage', 'new', 'avoidable')}})
    records['completion_inventory'] = {}
    for study in ('hazard_pilot_v2', 'crossing_traverses_v1'):
        completed = read_json(root / study / 'completion.json')
        records['completion_inventory'][study] = {'episodes': len(completed),
            'agent_goals': sum(r['goals'] for r in completed),
            'incomplete_or_failed': [r for r in completed if r['goals'] != 2],
            'source': source_record(root / study / 'completion.json', root)}
        for row in completed:
            if row['goals'] != 2:
                records['failure_inventory'].append({'study': study, **row})
    records['verification'] = {
        'native_state_shapes': '401x4x2 positions,400 controls and trace entries',
        'control_model_versions': 'Every selected stagewise physical step matches its logged control_model version.',
        'terminal_states': 'Both agents in every selected stagewise arm use exact logged after state.',
        'uncertainty_geometry': '12 original paths/pairs; truth and all3 representations match saved estimates within1e-12.',
        'uncertainty_resampled': False,
        'force_reconstruction_encounter': False}
    return records


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    result = audit(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'scenario_count': len(result['scenarios']),
                      'failure_records_preserved': len(result['failure_inventory']),
                      'out': str(args.out)}))


if __name__ == '__main__':
    main()
