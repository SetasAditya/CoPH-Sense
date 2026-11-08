"""Summarize completed paired development runs without selecting favorable seeds."""
import argparse
import json
from pathlib import Path
import statistics


def render(paths):
    lines=['# Development results', '',
           'Single-agent controlled fixtures only. These seeds were inspected during development; no confirmatory generalization or cooperation claim is supported.', '',
           '## Numerical and mechanism checks', '',
           'Run `python -m coph_stagewise.cli check` for the numerical and contract suite. Checks cover flow/work, damping identification, force and multistep replay gradients, overlap correction, validation exclusion, rollback, stage eviction, provenance boundaries and deterministic pairing.', '']
    for path in paths:
        rows=json.loads((path/'summary.json').read_text())
        lines.extend([f'## {path.name}', '',
            '| Arm | Success | Mean cost | Mean field MSE | Mean CF steps |',
            '|---|---:|---:|---:|---:|'])
        for arm in dict.fromkeys(r['arm'] for r in rows):
            group=[r for r in rows if r['arm']==arm]
            mean=lambda key:statistics.mean(r[key] for r in group)
            lines.append(f"| {arm} | {sum(r['success'] for r in group)}/{len(group)} | {mean('mission_cost'):.3f} | {mean('model_field_mse'):.5f} | {mean('counterfactual_steps'):.1f} |")
        lines.extend(['',f"Maximum absolute physical energy residual: {max(r['max_energy_residual'] for r in rows):.3g}.",
                      f"Recorded collisions: {sum(r['collision'] for r in rows)}; timeouts: {sum(r['timeout'] for r in rows)} (counts across all arms).",''])
        base={r['seed']:r['mission_cost'] for r in rows if r['mode']=='frozen'}
        full=[r for r in rows if r['arm']=='full_memory2_reconcile1']
        if full:
            delta=statistics.mean(r['mission_cost']-base[r['seed']] for r in full)
            lines.append(f'Full minus frozen paired mean cost: **{delta:+.3f}** (positive is worse).')
            lines.append('')
    lines.extend(['## Interpretation and progression', '',
        'Improved model estimates and valid gradients do not guarantee improved closed-loop cost. All unsuccessful missions retain the failure penalty. Short-horizon potential optimization can still stall; these results should be diagnosed before expanding to cooperative experiments.', '',
        'The no-history and no-reconciliation ablations are descriptive, not evidence of necessity. The short scenes do not meaningfully distinguish retaining two versus three prior stages. Curvature uncertainty remains uncalibrated.', '',
        'Reciprocal communication, dynamic environments, learned scheduling and theorems about cooperative advantage are not implemented or evaluated in this milestone.'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.write_text(render(args.runs))

if __name__=='__main__':main()
