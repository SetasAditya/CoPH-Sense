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
    """Admit documented native reproduction, including independent implementations.

    Primitive checks are deliberately insufficient. Metadata must identify a fully
    specified native problem and source-grounded reference checks; author code is
    one provenance route, not a scientific prerequisite.
    """
    problems = []
    baseline = record.get('baseline')
    if baseline not in BASELINES:
        problems.append('unknown_baseline')
    if record.get('status') != 'native_reproduced':
        problems.append('native_example_not_reproduced')
    source = record.get('source', {})
    kind = source.get('kind', 'official_repository')
    valid_hash = lambda h, n: isinstance(h,str) and len(h)==n and all(c in '0123456789abcdef' for c in h)
    if kind == 'official_repository':
        if not source.get('official_link_verified') or not source.get('repository_url'):
            problems.append('official_source_unverified')
        if not valid_hash(source.get('commit'),40):
            problems.append('missing_exact_source_commit')
    elif kind == 'independent_reimplementation':
        expected = {'distgp':'arXiv:2603.07351v1', 'ugv_exposure':'arXiv:2606.12306v3'}
        if source.get('paper_version') != expected.get(baseline) or not source.get('paper_url'):
            problems.append('independent_source_missing_pinned_paper')
        if not source.get('implementation_path') or not valid_hash(source.get('implementation_sha256'),64):
            problems.append('independent_source_missing_own_source_hash')
        if source.get('equations_verified') is not True:
            problems.append('independent_equations_not_verified')
    else:
        problems.append('unknown_source_kind')
    if not source.get('license'):
        problems.append('software_license_unknown')
    example = record.get('native_example', {})
    for field in ('command', 'input_hashes', 'environment', 'log_sha256', 'expected_checks'):
        if not example.get(field):
            problems.append('native_example_missing_' + field)
    if example.get('return_code') != 0 or example.get('checks_passed') is not True:
        problems.append('native_example_checks_not_passed')
    # A publication-setting implementation is allowed; transplanting the method
    # into CoPH's task is not itself a native reproduction.
    if record.get('adaptation_used_in_native_example') is not False:
        problems.append('native_example_is_a_task_adaptation')
    problem = record.get('native_problem', {})
    if problem.get('scope') != 'published_native_setting' or problem.get('primitive_only') is not False:
        problems.append('primitive_or_unspecified_problem_is_not_native')
    for field in ('dataset_or_map', 'observation_model', 'model_parameters', 'motion_and_sensing',
                  'inference_or_planner', 'evaluation_endpoints', 'seed_split'):
        if not problem.get(field):
            problems.append('native_problem_missing_' + field)
    if problem.get('unresolved_inputs') != [] or problem.get('source_consistency_checked') is not True:
        problems.append('native_problem_specification_unresolved')
    references = problem.get('reference_checks', [])
    if not references or any(r.get('scope')!='native_reference' or r.get('passed') is not True
                             or not r.get('reference_source') for r in references):
        problems.append('native_reference_checks_missing')
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
    source = record.get('source', {})
    if source.get('kind') == 'independent_reimplementation':
        own = Path(source['implementation_path'])
        if not own.is_absolute(): own = ROOT / own
        if not own.is_file() or digest(own) != source['implementation_sha256']:
            raise ValueError('Independent implementation source hash mismatch')
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
