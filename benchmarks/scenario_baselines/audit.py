"""Verify baseline source records and refuse comparisons before native reproduction.

This module does not implement either published method. It seals comparison intent
and makes the source/native-example admission gate executable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINES = ('distgp', 'ugv_exposure')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    seal = json.loads((ROOT / 'protocol_intent_seal.json').read_text())
    for name, expected in seal['sha256'].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Baseline record changed after intent seal: {name}')
    return seal


def admission(record):
    """Admission concerns reproduction integrity, never relative CoPH performance."""
    problems = []
    if record.get('baseline') not in BASELINES:
        problems.append('unknown_baseline')
    if record.get('status') != 'native_reproduced':
        problems.append('native_example_not_reproduced')
    source = record.get('source', {})
    if not source.get('official_link_verified') or not source.get('repository_url'):
        problems.append('official_source_unverified')
    revision = source.get('commit') or ''
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        problems.append('missing_exact_source_commit')
    if not source.get('license'):
        problems.append('software_license_unknown')
    example = record.get('native_example', {})
    for field in ('command', 'input_hashes', 'environment', 'log_sha256', 'expected_checks'):
        if not example.get(field):
            problems.append('native_example_missing_' + field)
    if example.get('return_code') != 0 or example.get('checks_passed') is not True:
        problems.append('native_example_checks_not_passed')
    if record.get('adaptation_used_in_native_example') is not False:
        problems.append('native_example_is_not_an_unadapted_reproduction')
    return {'eligible': not problems, 'reasons': problems}


def freeze(baseline, native_record, out):
    """Final protocol publication is blocked until official native checks pass."""
    verify()
    if baseline not in BASELINES:
        raise ValueError('Unknown baseline')
    record = json.loads(Path(native_record).read_text())
    gate = admission(record)
    if record.get('baseline') != baseline:
        gate = {'eligible': False, 'reasons': gate['reasons'] + ['baseline_identity_mismatch']}
    if not gate['eligible']:
        raise ValueError('Comparison gate blocked: ' + ', '.join(gate['reasons']))
    intent = json.loads((ROOT / 'matched_protocol_intent.json').read_text())['protocols'][baseline]
    protocol = dict(status='frozen_for_matched_evaluation', baseline=baseline,
                    native_record_sha256=digest(native_record), native_record=record,
                    protocol=intent, baseline_performance_used_for_selection=False)
    protocol['fingerprint'] = hashlib.sha256(canonical(protocol)).hexdigest()
    out = Path(out)
    if out.exists():
        if json.loads(out.read_text()) != protocol:
            raise ValueError('Refuse to replace a different frozen protocol')
        return protocol
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, indent=2) + '\n')
    return protocol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('status'); sub.add_parser('verify')
    p = sub.add_parser('freeze')
    p.add_argument('--baseline', choices=BASELINES, required=True)
    p.add_argument('--native-record', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'status':
        verify()
        result = json.loads((ROOT / 'native_reproduction.json').read_text())
        print(json.dumps(result, indent=2))
    elif args.command == 'verify':
        print(json.dumps({'verified': True, 'seal': verify()}, indent=2))
    else:
        print(json.dumps(freeze(args.baseline, args.native_record, args.out), indent=2))


if __name__ == '__main__':
    main()
