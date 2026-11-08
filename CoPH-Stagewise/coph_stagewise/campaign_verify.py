"""Independent artifact, pairing, budget and operational-source checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .campaign import DEFAULT_OUT, ROOT, verify
from .evaluation import load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_communication(root, name):
    folder = root / name
    protocol = load(folder / 'manifest.json')['protocol']
    rows = load(folder / 'completion.json')
    assert len(rows) == protocol['expected_episodes']
    assert len({row['key'] for row in rows}) == len(rows)
    expected = {(s['name'], s['seed'], a, b)
                for s in protocol['specs'] for a in protocol['arms']
                for b in (protocol['budgets'] if name == 'representation' else [protocol['byte_budget']])}
    actual = {(r['scene'], r['seed'], r['arm'], r['budget']) for r in rows}
    assert actual == expected
    first = {}
    goals = collisions = timeouts = 0
    hashes = {}
    max_work = 0
    for row in rows:
        assert 'error' not in row, row
        path = folder / 'episodes' / (row['key'] + '.json.gz')
        r = load(path)
        identity = (row['scene'], row['seed'])
        initial = np.asarray([samples[0]['x'] for samples in r['samples']])
        if identity in first:
            assert np.array_equal(initial, first[identity]), identity
        else:
            first[identity] = initial
        for a in r['agents']:
            assert sum(a['bytes'].values()) <= row['budget'], (row['key'], a['bytes'])
            assert a['max_update_work'] <= 1024, row['key']
            max_work = max(max_work, a['max_update_work'])
            goals += int(a['success'])
            collisions += int(a['collision'])
            timeouts += int(a['timeout'])
        hashes[str(path.relative_to(root))] = sha(path)
    animation = load(folder / 'animation_verification.json')
    assert animation['timestamps_verified']
    assert animation['actual_frames'] == animation['expected_frames']
    return dict(episodes=len(rows), missions=2*len(rows), goals=goals,
                collisions=collisions, timeouts=timeouts, initial_state_pairing=True,
                per_agent_byte_ceiling=True, maximum_update_work=max_work,
                complete_spec_arm_budget_cross_product=True,
                animation_timestamps_verified=True, episode_sha256=hashes)


def run(root):
    root = Path(root)
    out = {'protocol': verify(root)}
    baseline = load(root / 'operational_baseline.json')
    source_checks = {p: sha(ROOT / p) == h for p, h in baseline['operational_source_hashes'].items()}
    assert all(source_checks.values())
    out['operational_modules_unchanged'] = source_checks
    for name in ('representation', 'reciprocity'):
        out[name] = check_communication(root, name)
    native = load(root / 'native/results.json')
    manifest = load(root / 'native/manifest.json')
    assert native['pairing_verified']
    assert len(native['rows']) == len(manifest['arms']) * len(manifest['seeds'])
    assert {(r['protocol'], r['case']) for r in native['rows']} == {
        (a, f'seed{s}') for a in manifest['arms'] for s in manifest['seeds']}
    out['native'] = dict(episodes=len(native['rows']), starts=len(manifest['seeds']),
                         initial_states_and_landmarks_paired=True,
                         failure_inventory=len(load(root / 'native/failure_inventory.json')))
    faults = load(root / 'faults/audit_rows.json')
    assert len(faults) == 84
    assert all(r['measurement_immutable'] and r['physical_state_unchanged'] and r['operational_unchanged'] for r in faults)
    assert all(r['work'] <= 1024 for r in faults)
    out['faults'] = dict(audits=len(faults), isolation=True,
                         maximum_update_work=max(r['work'] for r in faults))
    # Reconstruction has a failed gate: verify evidence exists without assigning a
    # fabricated navigation outcome or a zero treatment effect.
    gate = load(root / 'reconstruction/gate.json')
    assert gate['complete'] and len(gate['rows']) == 24
    assert len({r['id'] for r in gate['rows']}) == 24
    for row in gate['rows']:
        assert row['complete']
        snapshot = row.get('snapshot')
        if snapshot:
            assert sha(root / 'reconstruction/snapshots' / snapshot['path']) == snapshot['sha256']
    if gate['stop']:
        assert not gate['selected']
        assert not list((root / 'reconstruction/episodes').glob('*.json.gz'))
    out['reconstruction'] = dict(screens=len(gate['rows']),
                                 navigation_treatment_files=len(list((root / 'reconstruction/episodes').glob('*.json.gz'))),
                                 gate_report=str(root / 'reconstruction/FINDINGS.md'),
                                 interpretation='Eligibility and matched audits; navigation effect unavailable.')
    out['reports'] = {name: (root / name / 'REPORT.md').is_file()
                      for name in ('physical', 'native', 'representation', 'reconstruction', 'reciprocity', 'faults')}
    assert all(out['reports'].values())
    out['passed'] = True
    (root / 'artifact_checks.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = run(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'operational_modules_unchanged'}, indent=2))


if __name__ == '__main__':
    main()
