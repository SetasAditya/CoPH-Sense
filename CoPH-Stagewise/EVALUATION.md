# Diagnostic evaluation protocol

The mission-goal correction changes executed attraction only. New runs use `--navigation-target mission` (the default); `stage` selects the original behavior. All seven arms receive the same choice. Shaping-energy accounting follows the executed potential, while replay and evaluator replay audits still use the original local targets. Those audits therefore do not validate the corrected execution policy. Existing `diagnostic_*` artifacts retain their original stage-target semantics.

Reproduce the isolated comparison in new directories:

```bash
python -m coph_stagewise.cli suite --profile smoke --navigation-target mission --out results/mission_goal_smoke --workers 4
python -m coph_stagewise.cli suite --profile development --navigation-target mission --out results/mission_goal_development --workers 4
python -m coph_stagewise.cli report --run results/mission_goal_development
python -m coph_stagewise.comparison --baseline results/diagnostic_development --run results/mission_goal_development
python -m coph_stagewise.cli render --run results/mission_goal_development
```

The paired comparison checks matching scenes, seeds, horizons, arm sets, mission/replay costs and all other learner settings. It reports recovered failures, newly failed successes, collisions and complete costs in `BEFORE_AFTER.md` and `before_after.json`. The numerical suite now contains 22 tests.

Run from `CoPH-Stagewise/`. Install the package dependencies from `pyproject.toml` (NumPy, PyTorch, Matplotlib and Pillow). All measurements are development evidence about **within-mission adaptation**. Each episode initializes a fresh learner. No communication or across-episode parameter transfer occurs.

```bash
python -m coph_stagewise.cli check
python -m coph_stagewise.cli suite --profile smoke --out results/diagnostic_smoke --workers 4
python -m coph_stagewise.cli report --run results/diagnostic_smoke
python -m coph_stagewise.cli render --run results/diagnostic_smoke
python -m coph_stagewise.cli suite --profile development --out results/diagnostic_development --workers 4
python -m coph_stagewise.cli report --run results/diagnostic_development
python -m coph_stagewise.cli render --run results/diagnostic_development
```

Use new suite output directories; existing results are never overwritten. `report` and `render` regenerate derived artifacts in their selected run. On machines with a read-only home directory set `MPLCONFIGDIR=/tmp/coph-mpl`. The legacy `rollout`, `compare`, and `check` interfaces remain available.

## Frozen design

`scenarios.specification(name, seed, profile)` produces plain-data scene specifications, independently of learners and outcomes. `ScenarioWorld` accepts these specifications, including explicit risk patches, circular obstacles, initial state, goal, damping and horizon. Custom studies can construct specifications and call `evaluation.execute`; the CLI uses the fixed design below.

| Scenario | Diagnostic purpose |
|---|---|
| clear | Constant background risk, no obstacles; unnecessary adaptation and delay |
| risk_passage | Off-center Gaussian risk with room for lateral avoidance |
| blocked_avoidance | Same risk and damping draw; obstacles obstruct the lower lateral route |
| overlap_revelation | Narrow risk peak ahead of the initial footprint; subsequent local sensing reveals it |
| long_corridor | Repeated risk and obstacles to x=16; eight nominal stages, enough geometric extent for eviction |
| damping_in | Risk-passage geometry with damping 1.35, inside [0.15,1.5] |
| damping_out | Identical geometry with damping 1.85, outside the declared fitting range |
| risk, obstacle | Exact original regression fixtures and RNG ordering |

Risk is smooth and therefore has nonzero tails; overlap revelation does not conceal all earlier evidence artificially. All scenes use the original local sensor, with no privileged observation injection. Geometry does not guarantee that a policy reaches later stages. Actual visited stages, eviction events and retained memory are recorded to expose this limitation.

Smoke uses seeds 10000–10001 and 96 physical steps for distance 5. Development uses seeds 20000–20007 and 180 steps. Horizons are scaled by goal distance before running any arm: long-corridor horizons are 308 and 576 respectively. Validation seeds 100000–199999 and test seeds 200000–299999 are reserved and unexecuted. `manifest.json` fixes all scene parameters, costs, configuration, seeds, animation selections, source SHA-256 hashes, configuration fingerprint and software versions.

Each scene/seed runs frozen, evidence, full, no_history, three_history, no_reconciliation and current_replay. Fitting, initialization, cost definitions and acceptance logic use the existing learner. Instrumentation calls the learner's update implementation. A regression test requires bit-identical executed states with the corresponding direct learner rollout. The stage-target compatibility option also reproduced all 180 transitions in a saved original failure.

The current-state comparator replaces retained entry starts with the actual update state and the current stage's field and target. It retains measured history for fitting. It receives the paired full arm's episode-level replay-transition budget. A baseline/gradient reservation prevents budget exhaustion between a valid baseline and its gradient evaluation. Remaining budget executes explicitly labeled non-learning transitions from the current state. Both **useful replay** and **padding** are reported; equality of charged transitions is not equality of useful optimization, update opportunities or wall time. This limitation must accompany interpretation of this comparator.

## Measurements and audits

Compressed episode records contain executed states/inputs, timestamped observations and visible geometry, pre/post update snapshots, stage bounds and weights, fixed-probe support masks, field proposals, policy replay trials, reasons, and stage/context events. Unsupported replay trials retain their reason and work count; the unchanged core discards their partial paths, so these are not displayed as complete valid trajectories. The raw damping proposal and reconciled damping are logged separately. Only measured observations enter the learner.

Complete mission cost is the sum of time, risk, effort and failure components. Collision and timeout retain the existing failure cost 40. Additional summaries include goal gap, path length, near-stationary time (speed below 0.05 away from goal), transition MSE, damping error, exact work/dissipation/residual, clipping work, separate model/context shaping jumps, update latency, identification/replay/audit work and memory peaks.

Evaluator probes use fixed spacing 0.5 in x and nine y coordinates over recorded scene bounds. A prediction is available only within a retained stage and 0.65 of a fitting observation. The newest supported stage supplies it. Missing predictions stay null; they are never counted as zero error. Common-probe error compares each arm against frozen on the intersection of supported probes, reporting each coverage separately. Errors on different supports are not directly comparable measures of global map accuracy.

At the first eligible replay update in a visited retained stage, the evaluator records four fixed-horizon branches: incumbent/candidate in the reconstructed model and in the true simulator. All four start from the same stage entry, use the same control policy form, target and horizon, and use the original replay objective coefficients. True scoring uses true risk and geometry; controls still use reconstructed fields and sensed geometry. Collisions are flagged but branches continue to preserve equal horizons. This is a replay-objective calibration audit, not an alternative mission evaluation. Domain-invalid audits remain in the denominator and are not retried. Audit work and time are separate and no branch is observed, fitted or committed by the learner. Tests verify snapshot and counter invariance.

## Artifacts and interpretation

- `episodes/*.json.gz`: full records; `*.error.json`: incomplete episodes with tracebacks.
- `manifest.json`, `completion.json`: frozen contract and completion accounting.
- `metrics.csv`, `summary.json`, `paired.json`, `common_probes.json`, `audits.json`: machine-readable results.
- `figures/*_paired.png`: identical worlds and axes, common tick markers every 40 steps.
- `figures/*_learning.png`: field and gradient error, damping, policy and replay traces.
- `figures/*.gif`: four synchronized panels for the predeclared first seed and full arm in every scenario. Estimate/replay panels label their latest logged snapshot tick; they are not interpolated or reconstructed from final models.
- `figures/failures/index.html`: every recorded failed episode, without selection.
- `REPORT.md`: completion counts, paired costs, mechanisms, audit validity and limitations.

The physical truth and estimated-field panels are deliberately distinct. Force arrows show the negative risk gradient, not the complete damped/clipped control. The animation title labels the episode's final outcome, even in earlier frames. Legacy snapshots not present in older artifacts are shown as unavailable. Missing episodes are separate from recorded mission failures.

Do not infer mechanism superiority from replay cost reduction alone. Check actual complete cost and failures, audit calibration, domain failures, field coverage, actual stage traversal and useful computation together. These runs do not establish general navigation safety or cooperative performance.

For independent artifact-contract checks after execution and rendering:

```bash
PYTHONPATH=. python tests/validate_suite.py results/diagnostic_smoke
PYTHONPATH=. python tests/validate_suite.py results/diagnostic_development
```

This verifies source fingerprints, scene pairing, exact arm coverage, cost sums, rollback snapshots, support/null consistency, memory bounds, audit starts/horizons, per-stage audit uniqueness, replay-budget accounting, and—when present—complete failure galleries and multiframe animations.

`PYTHONPATH=. python tests/validate_render.py results/diagnostic_development` additionally re-renders the first and last logged state of each predeclared animation and compares the saved GIF frames pixel-wise, allowing only a small mean error for palette quantization.
