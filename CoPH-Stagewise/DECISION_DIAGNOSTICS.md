# Decision bottleneck diagnostics

**Completed:** [Findings and next architecture experiment](results/decision_bottleneck_v1/FINDINGS.md). All 30 fork episodes reached 60/60 goals with zero collisions/timeouts; both maze regressions passed numerical equivalence, but neither was bitwise identical. Remote anchors change branch choices; field reconstruction and historical replay add no demonstrated branch/completion-time benefit here. Historical replay reaches stage handoffs but misses the outgoing-branch marker, and safety projection is inactive in the forks. See the report for timing, support, small updates and field-validation rollback.

This opt-in development milestone diagnoses whether shared environmental evidence reaches route choices and applied controls. It preserves routed-progress, fixed-rate exchange, learning rules, support, safety, and the 1,024-flow local-update ceiling. It does not tune the controller or train a new predictor.

`decision-diagnostics` exposes `analyze`, `suite`, `report`, and `render`. From this package:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli decision-diagnostics analyze --reference results/attribution_v1 --out results/decision_legacy_analysis_v1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli decision-diagnostics suite --reference results/attribution_v1 --out results/decision_bottleneck_v1 --workers 6

MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli decision-diagnostics report --out results/decision_bottleneck_v1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.cli decision-diagnostics render --out results/decision_bottleneck_v1
```

The suite freezes two full historical-replay regressions against the previous attribution episodes before running 30 fork episodes: three cases, five arms, and seeds 78000–78001. Both regressions must preserve operational trajectories, controls and packet values numerically (absolute tolerance 1e-12, relative tolerance 1e-10); identities, schedules, model versions, acceptance decisions and executed work remain exact. Timing and diagnostic-only metadata are excluded. Raw JSON byte counts are reported separately because last-bit float differences can alter digit counts. Bitwise failures remain recorded; numerical equivalence is not described as byte-for-byte parity. Source/configuration mismatch blocks resume, and failed attempts are retained.

The fork cases are clear, direct-route risk, and bypass risk on the same surveyed graph. They share a 500-active-step horizon, starts at 0/5 seconds, four-million-byte ceilings per agent, and 300-second packet lifetime. These two noise seeds are not independent map samples. Animation seed 78000 is selected before execution; held-out namespaces remain untouched.

Diagnostics distinguish whole-model acceptance, accepted policy backtracking, and committed policy change. Returned replay paths report length, junction proximity, outgoing-branch entry, and stage handoffs. A rejected batch's unavailable partial paths remain missing, not zero-length trajectories. Repeated baseline/gradient/candidate paths are computational observations, not independent episodes.

At isolated first-eligible-packet audits, exact risk is used only for two route-ranking references: within existing route-anchor support (unknown risk stays 1 elsewhere), and over all surveyed edges. Both preserve quadrature and risk weight 4; neither modifies runtime or expands replay support. Route-ranking references do not claim oracle navigation performance. Ordered route/field/policy decomposition and finite-difference sensitivities at epsilon 1e-4 expose clipping/projection effects; active-constraint changes are flagged rather than interpreted as smooth derivatives.

The first and second route scores exclude duplicate representations of the same departure. The production cost/node Dijkstra tie ordering is retained for the best route; equal-cost alternative candidates use a deterministic node ordering. The graph, collision checks, and pictures all use the declared fork geometry. Landscapes use the exact logged control-time model and mask unsupported environmental predictions; physical kinetic energy remains separate from shaping.

Results are written to a new directory. The [legacy analysis](results/decision_legacy_analysis_v1/legacy_analysis.json) preserves unavailable legacy fields explicitly. The [new report](results/decision_bottleneck_v1/REPORT.md) reports failures and unresolved bottlenecks without requiring an improvement.

For reproduction after live source changes, use `bash scripts/reproduce_decision.sh results/decision_bottleneck_v1 results/decision_bottleneck_reproduction`. It extracts the frozen source archive into an isolated checkout and uses the preserved attribution data as the regression reference. After all runs and GIFs complete, verify records with `bash scripts/verify_decision_artifacts.sh results/decision_bottleneck_v1`.

For the final summary figure and explicit direct/bypass grouping of oracle rankings, run `MPLCONFIGDIR=/tmp/coph-mpl bash scripts/decision_summary_figures.sh results/decision_bottleneck_v1` after `render`/`report`. This read-only artifact postprocessor is fingerprinted separately in `summary_aggregates.json`. Planner node routes can differ only in connector/waypoint selection on the same physical branch; do not interpret every node-route difference as a branch change.

Replay decision reach is a geometric diagnostic: a returned path visits within 0.25 units of a graph junction, then enters an incident edge beyond 0.25 units with perpendicular distance below 0.25. Corner-cutting paths may miss this marker; minimum junction distance and actual branch sequences are retained. It is not proof that an alternative branch was supported or considered. Prebranch evidence timing uses the fork exit markers x > 1.75 (direct) or y > 0.25 with x > 1.25 (bypass). First-eligible-packet-per-stage audits can miss later informative packets in the same stage; operational packet and control traces retain those events.

Two stopped preflights are preserved: `decision_bottleneck_preflight_v1` for off-center route deduplication, and `decision_bitwise_preflight_v1` for the failed bitwise comparison. Neither contributes fork outcomes. The latter records round-off and actual serialization-byte differences explicitly; controller rules, numerical solvers and packet encoding were not changed to force parity.
