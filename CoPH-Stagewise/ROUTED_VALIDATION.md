# Selected-controller validation

Routed-progress is selected after development and before held-out execution. Protected-progress remains a failed ablation. The original four-configuration gate remains failed; this experiment records a separate readiness decision for the selected controller. Navigation readiness does not require communication to outperform communication off.

The runner reuses the original frozen job and the validation specifications in `results/completion_first_v1/manifest.json`. It verifies every original Python file against both its recorded hash and its archived bytes. Only the new runner and its tests are additional Python sources; these have a separate fingerprint and source archive. It also records and verifies a content inventory of the entire reference result directory.

Run from `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl \
  python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl \
  python -m coph_stagewise.routed_validation \
  --out results/routed_validation_v1 --workers 32
```

The selected development gate requires exact pairing and all 64 agent goals in each communication setting, with zero collisions and timeouts. The held-out gate requires all 64 episodes / 128 agent missions to complete without collisions or timeouts. The four scenes and seeds 110000–110007 are copied directly from the frozen manifest. Animation seed 110000 is recorded before execution. Horizons, learner/controller parameters, costs, sensing, byte ceilings, and communication settings remain identical to development.

`--resume` continues only missing jobs and requires unchanged source and reference fingerprints. Failed and incomplete records are retained, not silently retried. `--render-only` regenerates artifacts from existing records. Outputs contain the selection timestamp, resolved manifest, selected-development and held-out gates, full episodes, machine-readable rows, paired secondary costs, complete failure records/gallery, and synchronized GIFs. Displayed geometry and replay come from the latest logged update snapshot; execution uses recorded states and controls.

A pass establishes reliability only for the tested static scenes with independent agents sharing environmental evidence. It does not establish inter-agent collision avoidance, changing-hazard forecasting, or adaptive communication. Existing causal audits establish that reciprocal evidence changes learned models and subsequent controls; any benefit to mission outcomes must be evaluated separately. Failures must be reported and diagnosed without tuning against held-out outcomes. Final-test namespaces remain untouched.

After the runner finishes, verify artifacts and regenerate the overview with:

```bash
bash scripts/check_routed_validation.sh results/routed_validation_v1
bash scripts/summarize_routed_validation.sh results/routed_validation_v1
```

Report byte totals aggregate both agents; the original byte ceiling applies per agent. Episode wall times include both agents and are recorded under the declared worker concurrency. The artifact checker verifies all 64 episode configurations, terminal outcomes, costs, byte ceilings, flow budgets, energy residuals, complete failure coverage, GIF frame counts/timestamps, and preservation of the original experiment.
