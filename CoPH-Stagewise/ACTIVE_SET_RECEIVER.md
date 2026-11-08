# Active-set receiver fitting

This is an opt-in numerical reliability experiment. The default receiver fit, search, objective, validation and controller behavior remain unchanged.

## Runtime interface

Use `receiver_fit_mode="precision_bounded"` with independent learner options:

- `receiver_solver_mode="projected"` (default) or `"active_set"`.
- `receiver_search_mode="legacy"` (default) or `"diagnostic"`.

Nondefault solver/search options require precision-bounded fitting. Old snapshots use the defaults. New learner snapshots record selected nondefault options, and manifests record every resolved combination. Packets are unchanged.

The active-set step fixes gradient-binding variables, solves the free subsystem including contributions from fixed components, fixes newly violated bounds, and re-solves. It releases variables when their gradient no longer binds them. Non-descent directions fall back to the diagnosed free-variable step and then projected gradient. The box is always centered on the same local-only proposal; corrections do not accumulate beyond ±0.5.

Both searches use the same normalized projected-gradient stationarity threshold of 1e-6 and at most 20 outer iterations. Legacy search permits the initial trial and 12 halvings with strict Armijo. Diagnostic search permits 40 trials and the exact recorded rounding exception (`10 * float64 epsilon * abs(objective)`). The exception permits rounding-level objective increases; it does not waive stationarity certification. Final active-set certification follows the diagnostic's autograd objective-gradient arithmetic; the original default retains its residual-gradient arithmetic. These are the same mathematical gradient but can differ numerically near the threshold.

Runtime code is NumPy/PyTorch float64 and does not import SciPy or diagnostic modules. Recorded work distinguishes residual rows, Jacobian rows, final gradient rows, objective evaluations, reduced-system solves, refinements, fallbacks and line-search trials. The 20-iteration limit is not an equal-computation claim. Solver failure rejects the entire update through the existing gate; neither incumbent replacement by the local-only proposal nor partial commitment is introduced.

## Reproduction and staged gates

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_active_set.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl python -m coph_stagewise.active_set_suite suite --out results/active_set_receiver_v2
MPLCONFIGDIR=/tmp/coph-mpl python -m coph_stagewise.active_set_suite report --out results/active_set_receiver_v2
```

The suite freezes sources/configurations and all 409 audit snapshot hashes. It runs the full tests, permitting only the explicitly reported pre-existing archived-source-hash error. It captures the 697 numerical problems once under NumPy-first imports, hashes and freezes those arrays, and supplies the same arrays to every worker. It launches eight isolated single-thread processes: four step/search combinations under NumPy-first and Torch-first imports. Original 697-stage reproduction and archived active-set reproduction must pass under the operational import order.

Selection prefers active-set/legacy, then active-set/diagnostic. A candidate must satisfy bounds/finite-result/certification checks, lose no originally converged problem below half the tolerance, and reduce failures under **both** import orders. Selection never uses route-query error or mission outcomes. Near-tolerance results (0.5–2 times the threshold), cross-order disagreements and all seven archived exhaustion identities remain explicit.

Only after selection does the suite repeat acquisition gates and run the unchanged 30-episode receiver-corrections protocol, seeds 79200–79201, animation seed 79200. The 18 local-only/legacy/validation-only episodes must reproduce before the 12 selected-solver episodes execute. All arms use supported-field routing, fixed-rate communication and disabled replay. Baseline operational comparison reports numeric/bitwise agreement separately from full serialized and compressed-byte differences, which include instrumentation and timing.

Sources cannot change during a frozen run. Use its `sources.zip` for exact reproduction after development, or choose a fresh output directory. No previous result directory is overwritten. If selection fails, navigation remains unexecuted and is inventoried as such; GIFs are not manufactured.

Reports distinguish stage-solver convergence, whole-update acceptance, proposal/commitment estimation error and navigation. Multi-start agreement and sampled positive Hessians are numerical evidence, not proofs of global uniqueness or convexity. Default promotion, a decision-sensitive fixture, historical replay changes, controller tuning, adaptive rates and held-out evaluation are outside this milestone.

The preserved `active_set_receiver_v1` attempt stopped before selection: rebuilding inputs separately changed 691/697 input hashes across import orders. Version 2 isolates solver sensitivity using canonical captured inputs. Its alternate-order validation branches still use native receiver reconstruction; estimation summaries under that order are numerical stress diagnostics, not additional navigation evidence.

Completed results: [findings](results/active_set_receiver_v2/FINDINGS.md) and [generated report](results/active_set_receiver_v2/REPORT.md). The selected experimental configuration is `receiver_solver_mode="active_set", receiver_search_mode="diagnostic"`; strict legacy search failed the protected-convergence gate. Runtime defaults remain projected/legacy.
