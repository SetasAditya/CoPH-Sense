# Diagnostic evaluation findings

The suite completed **126/126 smoke** and **504/504 development** episodes without execution errors. These are development measurements of within-mission adaptation, with fresh initialization in every episode. Training rules, failure penalties and earlier result directories were preserved.

Read the [development report](results/diagnostic_development/REPORT.md), [smoke report](results/diagnostic_smoke/REPORT.md), and [reproduction protocol](EVALUATION.md). The reports link the plots, numerical rows and complete failure galleries.

## Mission outcomes

Development contains eight fixed seeds for each of nine scenarios, or 72 episodes per arm.

| Arm | Successes | Mean complete cost | Paired change from frozen |
|---|---:|---:|---:|
| Frozen | 72/72 | 33.373 | 0 |
| Evidence only | 43/72 | 58.687 | +25.314 |
| Full replay | 49/72 | 56.114 | +22.740 |
| No history | 49/72 | 56.080 | +22.707 |
| Three previous stages | 49/72 | 56.100 | +22.727 |
| No reconciliation | 49/72 | 56.016 | +22.643 |
| Current-state replay | 46/72 | 56.665 | +23.292 |

Positive cost differences are worse. All **147 failed development episodes were timeouts**, with no collisions. Full learning spent a mean 6.655 seconds nearly stationary away from the goal, versus 0.120 seconds for frozen control. This is evidence of mission progress problems despite useful model learning; it does not identify a single causal implementation defect.

Full replay improves on evidence-only in aggregate: six additional successes and 2.573 lower mean cost. On the long corridor, full succeeds 8/8 versus 6/8 for evidence-only and current-state replay. However, frozen also succeeds 8/8 there, and succeeds on every development episode. The suite therefore does **not** support a superiority claim for the full method.

## What the mechanisms establish

- **Physical accounting works numerically:** the maximum exact kinetic/work/dissipation residual across development is approximately 3.49e-15. Model and context shaping jumps are separate from physical energy.
- **Identification changes the model:** mean absolute damping error falls from 0.4181 for frozen to 0.0390 for full. The out-of-range regime is intentionally included; reconciliation cannot identify damping 1.85 inside a [0.15,1.5] box. This identification improvement does not guarantee better control.
- **Supported field reconstruction improves:** on the common supported probes of each full/frozen pair, mean field-value MSE is 0.00155 versus 0.04366 and gradient MSE is 0.01841 versus 0.14606. Full covers only 38.8% of the fixed evaluator probes on average; these are not global-map accuracy claims.
- **History is exercised, but its mission benefit is not established:** the long scene spans eight nominal stages and full traverses them. Peak retained stages are three for full, one for no-history and four for three-history. Those three arms nevertheless have identical aggregate success counts and almost identical complete costs.
- **Reconciliation is not the dominant failure explanation here:** removing it changes mean cost only slightly and does not change the aggregate success count. This is not evidence that constraints are unnecessary outside these scenes.
- **Useful replay and charged computation differ:** full and current-state replay each execute 50,400 charged model transitions across development. Current-state replay includes 1,752 explicitly marked non-learning padding transitions (3.48%). Its result cannot be interpreted as equal useful optimization or identical update scheduling.
- **Short replay calibration and mission performance are separate:** the suite records 1,112 first-eligible stage audits, all domain-valid. Predicted and realized improvement signs agree in 1,100/1,111 nonzero-effect audits (99.0%). The calibration plot and machine-readable audit rows use the same short replay objective. A valid short-horizon improvement is not a complete-mission improvement.

## Evidence still missing

The two- versus three-history comparison is now mechanically exercised but shows little aggregate mission separation. These fixed scenes do not establish when retained-entry replay is necessary. Stronger support would require independently specified cases that distinguish the history mechanisms, without selecting cases after observing favorable outcomes.

There are no collision failures in these executed runs, so they provide little empirical discrimination about collision robustness. They also do not establish benefit under changing fields, communication, delayed evidence, or across-episode learning. Validation and test seed namespaces remain unexecuted.

The current-state comparator matches the episode-level charged replay budget, with padding disclosed. A study seeking strictly equal *useful* optimization work must specify how budget is allocated when episodes terminate at different times or replay domains reject branches; these results do not resolve that design choice.

The smoke suite uses intentionally shorter horizons and has 92 failures. Those are preserved rather than replaced by the longer-horizon development outcomes. The predeclared animation seed is 10000 for smoke and 20000 for development, always using the full arm. Every recorded failure appears in its suite's gallery.
