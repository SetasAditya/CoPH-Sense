# Receiver fitting and validation corrections

This opt-in development experiment separates measurement precision, bounded iterative fitting, and validation. Defaults remain `receiver_fit_mode="legacy"` and `receiver_validation_mode="legacy"`; environmental packet formats are unchanged.

## Runtime options

Pass the independent options to `CooperativeLearner` (or its reconstruction subclasses):

```python
receiver_fit_mode="precision_bounded"
receiver_validation_mode="relative_local"
```

The local-only point estimate $w_0$ is unchanged. With received residuals, the corrected fit minimizes

$$
J(w)=\tfrac12(w-w_0)^\top H_{local}(w-w_0)
 +\tfrac12\sum_m(f_m(w)-z_m)^\top C_m^{-1}(f_m(w)-z_m),
\qquad \|w-w_0\|_\infty\leq0.5.
$$

Local measurement rows use frozen delta-method precision at $w_0$, with sensor standard deviation $0.003$. Existing overlap rows, ridge regularization, and inverse marginal-block construction remain. This is local information, not a calibrated posterior; clipped readings and repeated stage evidence retain their existing treatment. Message covariance floors, age terms and provenance checks are unchanged.

The solver uses projected Gauss–Newton, projected-gradient fallback, Armijo constant $10^{-4}$, at most 20 iterations and 12 halvings per iteration. Every candidate is projected into the same box about $w_0$. Convergence requires projected-gradient infinity norm divided by $1+\|g_0\|_\infty$ at most $10^{-6}$. Exhaustion, failed searches and nonfinite results reject the entire update.

The independent relative-local gate compares message and local-only proposals using identical reserved measurements:

$$
E_{message}\leq E_{local}+0.05\max(E_{local},0.003^2)+10^{-12}.
$$

It applies only when eligible received residuals are fitted. Missing reserved field measurements cause rejection. Existing whole-model checks remain; rejection preserves the entire old committed version and routing snapshot. It does not commit the local-only proposal as a fallback.

## Reproduction

From `CoPH-Stagewise/`, with dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m coph_stagewise.cli check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m coph_stagewise.receiver_corrections suite --out results/receiver_corrections_reproduction --workers 4
python -m coph_stagewise.receiver_corrections analyze --out results/receiver_corrections_v1
python -m coph_stagewise.receiver_corrections report --out results/receiver_corrections_v1
python -m coph_stagewise.receiver_corrections render --out results/receiver_corrections_v1
```

The main CLI also dispatches `receiver-corrections`. The suite freezes specifications, solver/validator settings, all input snapshot hashes and Python source files before execution. Resume requires matching fingerprints; use the source archive to reproduce after later code changes. The 160 saved snapshots are trusted local evaluator artifacts, never runtime inputs.

Default reproduction must match archived errors, committed fields, decisions and versions. All variants run on isolated copies; validation modes must see identical proposals. Solver/validation outcomes and proposed/committed estimates are separate records.

Only after default reproduction and acquisition eligibility pass does the suite run three unchanged complementary-coverage cases, seeds 79200–79201, and five arms (30 episodes, 60 missions). All arms use supported-field routing with anchor fallback and no policy replay. Animation seed 79200 was selected in the manifest. The 1,024-flow ceiling is unchanged; residuals, Jacobians, solves, information inversions, routing and diagnostics have separate work counters.

Results include `REPORT.md`, proposal/audit records, validator confusion tables, episode and paired outcomes, lineage, source archive, synchronized GIFs, and complete failure/incomplete inventories. Query counts are correlated quadrature occurrences, not independent trials. No decision-sensitive fixture or held-out evaluation is included. The archived routed-validation source-hash test is reported separately if current development sources no longer match that earlier archive.

## Preserved run and evaluator revision

The recorded experiment is `results/receiver_corrections_v1`. Its original `sources.zip` and manifest preserve all experiment code. After the episodes finished, an evaluator-only correction made proposed-field queries use the stage selection that would be published on commitment. Previously, those queries used the old overlap selection, which could disagree with committed predictions even when coefficients matched. No operational learner, controller, packet or episode was changed. The patch, regression test, and revised source archive are in `verification/evaluator_revision/`.

Original offline and episode records remain untouched. Derived proposal errors and confusion tables are corrected from their saved fields and trusted snapshots, with accepted-proposal/commitment equality verified. Original generated diagnostics are in `prepublication_diagnostics/`. To regenerate the recorded run's corrected analysis after invoking its `report` command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl PYTHONPATH=. python results/receiver_corrections_v1/verification/receiver_corrections_postprocess.py.txt
```

Fresh suites use the corrected evaluator directly. The frozen experiment cannot be resumed with changed sources; use its original source archive or a new output directory. `load_saved_snapshot` handles the trusted `CorrectionLearner` snapshots written by the original `python -m` entry point.
