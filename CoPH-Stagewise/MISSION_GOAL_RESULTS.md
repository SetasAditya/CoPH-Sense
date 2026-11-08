# Mission-goal attraction correction

The isolated correction eliminated all failures in the fixed smoke and development suites: **126/126 smoke and 504/504 development episodes succeeded**, with no collisions, timeouts or execution errors. These are previously examined development seeds, not held-out validation.

Execution now attracts toward the known mission goal. Risk and obstacle forces, clipping, damping, sensing, stage geometry, replay objectives, acceptance rules, episode horizons and mission-cost coefficients remain unchanged. Shaping-energy accounting was updated to use the executed potential. Replay still uses stage-local targets, so its audits do not validate the corrected execution policy.

| Development arm | Success before → after | Mean complete cost before → after |
|---|---:|---:|
| Frozen | 72/72 → 72/72 | 33.373 → 14.280 |
| Evidence only | 43/72 → 72/72 | 58.687 → 13.795 |
| Full replay | 49/72 → 72/72 | 56.114 → 13.846 |
| No history | 49/72 → 72/72 | 56.080 → 13.695 |
| Three previous stages | 49/72 → 72/72 | 56.100 → 13.855 |
| No reconciliation | 49/72 → 72/72 | 56.016 → 13.844 |
| Current-state replay | 46/72 → 72/72 | 56.665 → 13.827 |

Full replay recovered all **23 previous failures** and preserved all **49 previous successes**, starting each episode from its original initial state. Mean time nearly stationary away from the goal fell from **6.655 to 0.120 seconds**; the remaining time is the initial resting timestep. Every full-method scenario now succeeds on all eight seeds.

This establishes a correction to the observed navigation failures. It does not establish an advantage for counterfactual learning: evidence-only costs 0.051 less than full replay on average, and no-history has the lowest mean cost in this suite. Replay's local objective and unsupported-entry skipping remain unresolved. Current-state replay retains its disclosed non-learning padding budget.

Verification includes **22 numerical/contract tests**, exact reproduction of an original 180-step trajectory using `--navigation-target stage`, and per-episode pairing, cost, memory, rollback, audit and source-fingerprint checks. The maximum physical energy residual in corrected development is approximately **6.0e-15**. Existing result directories were preserved.

- [Development before/after comparison](results/mission_goal_development/BEFORE_AFTER.md)
- [Development detailed report](results/mission_goal_development/REPORT.md)
- [Smoke before/after comparison](results/mission_goal_smoke/BEFORE_AFTER.md)
- [Reproduction commands](EVALUATION.md)
- [Corrected overlap-revelation animation](results/mission_goal_development/figures/overlap_revelation_seed20000.gif)
- [Corrected long-corridor animation](results/mission_goal_development/figures/long_corridor_seed20000.gif)

Each corrected suite has nine predeclared full-method animations. Empty failure galleries explicitly reflect zero failures; all prior failure images remain in the original suites.
