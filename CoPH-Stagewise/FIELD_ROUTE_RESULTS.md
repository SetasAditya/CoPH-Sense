# Direct reconstructed-field route scoring

**Complete:** [Findings and limitations](results/field_route_v1/FINDINGS.md). All 24 episodes reached **48/48 goals with zero collisions/timeouts**; all 12 anchor regressions reproduce prior operational records bit-for-bit after excluding additive metadata. Direct evaluation improves same-state supported risk estimates and reduces the leader's time/risk in the direct-risk fork, but slightly slows the carrier. Received field assimilation adds no demonstrated branch or completion-time benefit. The new mode remains opt-in.

This opt-in development experiment compares direct evaluation of a committed environmental field with the existing sparse-anchor route representation. It leaves policy replay disabled in all four primary arms.

## Correction to the earlier interpretation

The earlier decision-bottleneck discussion described field reconstruction as having no direct route-score input. That wording omitted an existing indirect path: `TerrainLearner.update` evaluates each committed/current stage model at locally supported points, produces value/gradient anchors, and passes them to routing. The anchor scorer takes a clipped maximum of supported local and remote Taylor estimates. Reconstruction therefore already influences route scores through local model anchors.

This experiment tests whether direct RBF evaluation avoids sampling, Taylor-approximation, or envelope-aggregation loss. It does **not** demonstrate creation of a previously absent causal connection. The original experimental records and numerical results remain preserved.

## Implemented behavior

`Config.route_field_mode` accepts `anchors` (unchanged default) or `supported_field`. In the new mode, the newest supported retained stage supplies risk values at graph-edge and virtual-connector quadrature points. A routing snapshot is published only on successful model commitment. Rejected proposals retain the previous coefficients; new stages fall back to anchors until committed. Stage/measurement eviction may shrink support, never expand it. Frozen snapshots include coefficients, versions and support coordinates, not raw measured values or validation records.

Outside field support, routing keeps the existing anchor envelope, then unknown risk 1. Risk values retain the existing [0,20] route clipping, weight 4, quadrature spacing, tie-breaking, surveyed geometry and safety rules. Direct values can be lower than the anchor envelope; no conservative-risk guarantee is claimed. Support remains the retained stage rectangle and distance at most 0.65 from a retained, non-validation local measurement. Messages and predictions do not certify new support.

The four arms are anchor baseline, anchor reconstruction, direct local field, and direct cooperative field. All retain remote route anchors, fixed-rate communication, local fitting, damping identification, reconciliation and validation; only field assimilation and route representation vary. The three existing forks, seeds 78000–78001, starts at 0/5 seconds, 500 active steps, four-million-byte ceilings and 300-second lifetime are unchanged. These are two noise seeds on fixed maps, not independent environment samples.

## Reproduction

From `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli field-route-suite suite \
  --reference results/decision_bottleneck_v1 --out results/field_route_v1 --workers 6

MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli field-route-suite report --out results/field_route_v1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli field-route-suite render --out results/field_route_v1

python -m coph_stagewise.cli field-route-suite verify --out results/field_route_v1
```

The suite freezes its source archive/configurations before execution and gates the 12 direct-field episodes on the 12 anchor numerical regressions. Resume rejects different fingerprints. Numerical regression tolerance is atol=1e-12, rtol=1e-10, with exact discrete decisions and operational flow counts; bitwise and actual serialization-byte differences remain reported. Additive `route_field*` diagnostic/configuration metadata is excluded from comparison.

The [generated report](results/field_route_v1/REPORT.md) includes all completed runs, failures, unsupported cases and same-state audits. The episode records contain per-control quadrature traces and packet/model lineage. `aggregates.json` reports common-support error and coverage separately. Routing, diagnostic and evaluator-only field evaluations are charged separately; cache hits avoid counting computations that were not executed.

The startup failure in `results/field_route_runner_preflight_v1` is preserved with its source manifest and log. It completed no episodes; missing output-directory initialization was fixed and covered by a test before the final run.

To reproduce after live source changes, run `bash scripts/reproduce_field_routes.sh results/field_route_v1 results/field_route_reproduction`. It uses the frozen Python archive in an isolated temporary checkout and the original saved diagnostic reference. The shell reproduction helper is also saved with the results; the temporary checkout is retained for inspection.

For the supplementary paired-agent plot and comparison rows, run `MPLCONFIGDIR=/tmp/coph-mpl bash scripts/field_route_pairs.sh results/field_route_v1`. Verify committed snapshots, logged field values and full field-evaluation accounting with `bash scripts/verify_field_route_records.sh results/field_route_v1`. Both helpers record their own source hashes in their output artifacts.
