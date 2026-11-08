# Development results

Single-agent controlled fixtures only. These seeds were inspected during development; no confirmatory generalization or cooperation claim is supported.

## Numerical and mechanism checks

Run `python -m coph_stagewise.cli check` for the numerical and contract suite. Checks cover flow/work, damping identification, force and multistep replay gradients, overlap correction, validation exclusion, rollback, stage eviction, provenance boundaries and deterministic pairing.

## risk_validated

| Arm | Success | Mean cost | Mean field MSE | Mean CF steps |
|---|---:|---:|---:|---:|
| frozen_memory2_reconcile1 | 3/3 | 24.568 | 0.03612 | 0.0 |
| evidence_memory2_reconcile1 | 1/3 | 63.042 | 0.00028 | 0.0 |
| full_memory2_reconcile1 | 2/3 | 48.149 | 0.00057 | 632.0 |
| full_memory0_reconcile1 | 2/3 | 48.236 | 0.00018 | 440.0 |
| full_memory3_reconcile1 | 2/3 | 48.149 | 0.00057 | 632.0 |
| full_memory2_reconcile0 | 2/3 | 48.147 | 0.00057 | 632.0 |

Maximum absolute physical energy residual: 9.41e-16.
Recorded collisions: 0; timeouts: 6 (counts across all arms).

Full minus frozen paired mean cost: **+23.581** (positive is worse).

## obstacle_validated

| Arm | Success | Mean cost | Mean field MSE | Mean CF steps |
|---|---:|---:|---:|---:|
| frozen_memory2_reconcile1 | 2/2 | 25.459 | 0.03213 | 0.0 |
| evidence_memory2_reconcile1 | 1/2 | 52.763 | 0.00008 | 0.0 |
| full_memory2_reconcile1 | 1/2 | 54.889 | 0.00053 | 624.0 |
| full_memory0_reconcile1 | 1/2 | 55.010 | 0.00026 | 444.0 |
| full_memory3_reconcile1 | 1/2 | 54.889 | 0.00053 | 624.0 |
| full_memory2_reconcile0 | 1/2 | 54.886 | 0.00053 | 624.0 |

Maximum absolute physical energy residual: 5.64e-16.
Recorded collisions: 0; timeouts: 5 (counts across all arms).

Full minus frozen paired mean cost: **+29.429** (positive is worse).

## Interpretation and progression

Improved model estimates and valid gradients do not guarantee improved closed-loop cost. All unsuccessful missions retain the failure penalty. Short-horizon potential optimization can still stall; these results should be diagnosed before expanding to cooperative experiments.

The no-history and no-reconciliation ablations are descriptive, not evidence of necessity. The short scenes do not meaningfully distinguish retaining two versus three prior stages. Curvature uncertainty remains uncalibrated.

Reciprocal communication, dynamic environments, learned scheduling and theorems about cooperative advantage are not implemented or evaluated in this milestone.
