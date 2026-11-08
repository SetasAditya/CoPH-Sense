# Development and reproduction

## Current implementation

`CoPH-Stagewise` is the current package. The older `CoPh-sense` material-sensing prototype remains available with its original source and checkpoint; its root `requirements.txt` targets that legacy Python 3.8 environment. Do not combine it with the current package environment.

The main modules are `model.py` (plant, potentials, work accounting), `records.py` (measured records and stage memory), `learner.py` (fit/replay/reconciliation/commitment), `cooperative.py` (packets and receiver updates), and `maze_learning.py`/`route_field.py` (supported routing). Experiments and diagnostic variants use explicit switches; their recorded outcomes do not imply that every variant is the default.

## Tests

From `CoPH-Stagewise`, run `python -m pytest tests -q`. This scopes collection to the current package and avoids collecting historical source archives or legacy environments. CPU float64 checks cover energy balance, force/replay derivatives, evidence isolation, support, rollback, routing and communication. For reproducible numerical execution, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before starting Python.

Five test classes and the reconstruction-quality class consume preserved snapshots or frozen source archives. They report explicit skips when those inputs are absent. To run them, restore the original `complementary_reconstruction_v1`, `receiver_corrections_v1`, `solver_diagnosis_v1`, and `completion_first_v1` artifacts under `CoPH-Stagewise/results/`, then rerun the same command. Do not download or unpickle untrusted snapshots. The earlier full workspace suite retained an archived-source-hash failure in `test_routed_validation`; omission of the archive does not resolve that historical mismatch.

`python -m coph_stagewise.cli check` runs the unittest-compatible subset; pytest is the complete source test entry point. Native VMAS has a separate environment and test command documented in `benchmarks/phmarl/README.md`.

## Experiments and artifacts

`cli rollout` and `cli compare` regenerate small single-agent development runs. `crossing_demo` regenerates the reciprocal two-agent warning demonstration and produces reports/animations. Full suite protocols and switches are documented in the package's experiment-specific Markdown files. Some later diagnostics use saved snapshots from earlier stages; they cannot run from summaries alone. Output-directory existence and fingerprint checks must be respected.

The source release excludes the 18 GB generated stagewise archive, native environments, third-party checkouts, caches and new binary runs. One small frozen reciprocity manifest is included as a test/scenario fixture. Selected JSON summaries and an artifact inventory are under `docs/results`; they are recorded evidence, not substitutes for the complete trajectories, snapshots, uncertainty samples, or native checkpoints required to recompute them. Legacy artifacts already present in repository history are preserved.

`docs/RELEASE_SOURCES.json` records copied source hashes and any packaging changes. Runtime numerical settings and learning/controller rules are preserved. The benchmark's upstream path is now configurable so it works outside the original workspace; archive-dependent tests skip only when their required files are absent.

## Earlier prototype

`docs/LEGACY_README.md` preserves the previous root README; its relative paths and commands are interpreted from the repository root. `CoPH-Stagewise/DEVELOPMENT_HISTORY.md` and `benchmarks/phmarl/DEVELOPMENT_HISTORY.md` preserve earlier milestone instructions. The Overleaf manuscript is maintained separately.

## Release verification

The source checkout passed 245 tests and 7 subtests, with 41 explicit archive-dependent skips. All 26 native adapter/protection tests passed in the pinned Python 3.8 environment. Editable installation, a successful 46-step single-agent rollout, and a 32-step four-channel crossing smoke run were checked. The short crossing smoke is not a completion test. Details are recorded in `RELEASE_CHECKS.json`; no full scientific campaigns were rerun during packaging.
