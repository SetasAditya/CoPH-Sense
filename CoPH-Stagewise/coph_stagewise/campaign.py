"""Inventory and immutable coordination record for the current-algorithm campaign.

This module reads suite artifacts; it never turns missing evidence into a zero
effect or treats a gate failure as a navigation failure.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/current_campaign_v1'
SUITES = ('physical', 'native', 'representation', 'reconstruction', 'reciprocity', 'faults')
CLAIMS = {
    'physical': 'Numerical energy accounting, overlap continuity and gradient consistency.',
    'native': 'Online identification versus incorrect and correct fixed dynamics under common protection.',
    'representation': 'Graded reusable information versus decisions, readings and costmaps across byte budgets.',
    'reconstruction': 'Additional influence of received reconstruction and current/historical policy replay.',
    'reciprocity': 'Reciprocal discovery on unselected layouts versus both one-way directions.',
    'faults': 'Evidence-interface behavior under duplicates, echoes, age, ordering, bias and correlation.',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_new(path, value):
    """Exclusive creation: a frozen file can never be silently rewritten."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze(out, gpu_check=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'version': 1, 'scope': 'current algorithm, development evidence',
        'suites': {name: CLAIMS[name] for name in SUITES},
        'defaults_preserved': True,
        'deferred': ['temporal environmental estimator', 'adaptive communication',
                     'physical inter-agent coordination', 'unknown-topology mapping'],
        'constraints': ['freeze each suite before treatment', 'preserve failed gates',
                        'no post-outcome tuning or fixture substitution',
                        'truth and hypothetical branches remain evaluator-only',
                        'record actual computation and communication',
                        'preserve held-out and final-test namespaces'],
        'worker_limit': 3,
        'runtime_module_sha256': sha(__file__),
    }
    path = out / 'campaign_protocol.json'
    if path.exists():
        old = read(path)
        if canonical(old['protocol']) != canonical(protocol):
            raise ValueError('Campaign protocol differs; use a new campaign directory.')
        return old
    value = {'protocol': protocol,
             'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'host': platform.node(), 'python': sys.version,
             'thread_environment': {k: os.environ.get(k) for k in
                                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
             'gpu_probe': gpu_check,
             'source_policy': 'Each suite independently archives and fingerprints its executable sources before treatment.'}
    write_new(path, value)
    write_new(out / 'campaign_protocol.sha256.json', {'sha256': sha(path)})
    return value


def verify(out):
    out = Path(out)
    expected = read(out / 'campaign_protocol.sha256.json')['sha256']
    actual = sha(out / 'campaign_protocol.json')
    if actual != expected:
        raise ValueError('Campaign protocol hash mismatch.')
    return {'protocol_hash_valid': True, 'sha256': actual}


def inventory(out):
    out = Path(out)
    rows = []
    for name in SUITES:
        directory = out / name
        files = sorted(p for p in directory.rglob('*') if p.is_file()) if directory.exists() else []
        reports = [p for p in files if p.suffix == '.md' and p.name.upper() in ('REPORT.MD', 'FINDINGS.MD')]
        manifests = [p for p in files if 'manifest' in p.name.lower() and p.suffix == '.json']
        errors = [p for p in files if p.name.endswith('.error.json')]
        plots = [p for p in files if p.suffix.lower() in ('.png', '.pdf', '.gif')]
        # Presence of a report is an artifact status, never an acceptance verdict.
        rows.append({'suite': name, 'claim': CLAIMS[name],
                     'artifact_status': 'report_available' if reports else 'artifacts_in_progress' if files else 'not_started',
                     'file_count': len(files), 'bytes_on_disk': sum(p.stat().st_size for p in files),
                     'manifests': [str(p.relative_to(out)) for p in manifests],
                     'reports': [str(p.relative_to(out)) for p in reports],
                     'error_records': [str(p.relative_to(out)) for p in errors],
                     'plot_count': len(plots),
                     'animations': [str(p.relative_to(out)) for p in plots if p.suffix.lower() == '.gif']})
    return rows


def report(out):
    out = Path(out)
    checks = verify(out)
    rows = inventory(out)
    # These are derived views; the immutable protocol and suite manifests are untouched.
    (out / 'inventory.json').write_text(json.dumps({'verification': checks, 'suites': rows}, indent=2) + '\n')
    lines = ['# Current-algorithm experiment campaign', '',
             'Development evidence. Each suite owns its frozen protocol, exclusions and acceptance verdict. '
             'A report being available does not mean its mechanism improved navigation.', '',
             '| Suite | Artifact status | Files | Plots | Error records |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['suite']} | {row['artifact_status']} | {row['file_count']} | {row['plot_count']} | {len(row['error_records'])} |")
    for row in rows:
        lines.extend(['', f"## {row['suite'].capitalize()}", '', row['claim'], ''])
        lines.extend(f'- [{p}]({p})' for p in row['reports'])
        if not row['reports']:
            lines.append('Results unavailable or still running; no outcome inferred.')
    lines.extend(['', '## Interpretation boundaries', '',
                  '- Physical accounting does not prove switched-system stability or passivity of filtering.',
                  '- Native identification evaluates a controller transfer, not the full field-sharing pipeline.',
                  '- Shared route evidence, receiver fitting and policy replay require separate causal conclusions.',
                  '- A failed eligibility gate is not a negative treatment effect.',
                  '- Sensor-noise repetitions are not independent environments.',
                  '- Temporal estimation, adaptive scheduling, inter-agent coordination and unknown-map navigation are deferred.', ''])
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('freeze', 'status', 'verify', 'report'))
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--gpu-check', type=Path, help='Previously recorded JSON hardware probe; no device discovery changes numerical execution.')
    args = ap.parse_args(argv)
    if args.command == 'freeze':
        result = freeze(args.out, read(args.gpu_check) if args.gpu_check else None)
    elif args.command == 'verify':
        result = verify(args.out)
    elif args.command == 'report':
        result = report(args.out)
    else:
        result = inventory(args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
