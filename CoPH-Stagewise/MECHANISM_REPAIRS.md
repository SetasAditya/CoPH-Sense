# Opt-in mechanism repairs

This milestone implements the six proposed changes separately. Legacy defaults and old experimental records remain available. The native allocation/landmark-target controller is a separate implementation and is not silently changed by these options.

## Runtime interfaces

| Mechanism | Opt-in interface | What changes |
|---|---|---|
| Route-choice prediction | `RouteDecisionLearner(..., route_options=RouteDecisionOptions(mode='prefix', starts='current', terminal='graph'))` | Evaluate bounded route candidates through supported physical prefixes and an explicit approximate remaining-route cost. `starts='historical'` additionally evaluates retained entries. |
| Localized receiver fitting | `receiver_fit_mode='precision_bounded', receiver_solver_mode='active_set', receiver_search_mode='diagnostic', receiver_validation_mode='relative_local', receiver_localization_mode='compact'` | Preserve the local-only stage proposal and fit bounded compact residual coefficients. Acceptance remains transactional. |
| Compact wire coding | `CommunicationRepairOptions(wire_mode='compact_v1')` with `FixedRateRepairLink` | Exact canonical-message compression in a versioned frame; normal decoding/validation; explicit byte accounting. |
| Receiver-relevant payloads | `CommunicationRepairOptions(payload_selection='receiver_relevant')` | Select a retained-stage summary using a bounded explicit route/support advertisement at fixed transmission opportunities. Status overhead is charged. |
| Local consistency check | `CommunicationRepairOptions(consistency_mode='local_residual')` | An opt-in, uncalibrated check against local evidence before normal assimilation. Unsupported checks cannot establish message accuracy. |
| Smooth stage overlap | `Config(..., stage_blend_mode='smooth')` | Differentiate a normalized blend of complete potentials over already-supported retained stages. Requires retained protected/routed control and positive overlap. |

Communication options are a separate transport interface; passing them does not alter the existing cooperative runner automatically. `FixedRateRepairLink` must be used to obtain wire-byte accounting and fixed-rate receiver-relevant selection. The original runner keeps its original packet accounting. The compact basis and blend are integrated into production field/force evaluation, committed routing snapshots, branch copies, receiver summaries and rich control logs.

## Route decision objective and limits

For each frozen candidate route $a$, replay the same execution controller from each frozen eligible start for a fixed horizon $T$ and evaluate

$$
Q(a)=\frac1{|\mathcal E|}\sum_{e\in\mathcal E}
\left[\sum_{t=0}^{T-1}\Delta t\left(1+4\rho\widehat r(q_t)+0.03\|u_t\|^2\right)
+\widehat V_{\mathrm{remaining}}(x_T,a)\right].
$$

The continuation term uses the production graph risk quadrature and fallback rules, interpreted at nominal speed 1 m/s. It is an approximation with no terminal effort estimate, not a calibrated prediction or HJB value. Unknown risk remains the declared prior. Candidate generation is bounded and seeded by the production incumbent connector; execution/terminal conditioning can subsequently rejoin a candidate suffix through any production-eligible connector. This can collapse distinct candidate lists to the same physical suffix and must not be interpreted as a persistent branch commitment.

All candidate starts and horizons are frozen within a batch. The eight-step initial horizon, at most four candidates, actual safety-guard flows and replay transitions share the existing 1,024-flow update ceiling with any prior policy replay. Unsupported candidate batches retain the incumbent; they are not shortened or selectively discarded to make a candidate appear cheaper. New route choices are a decision stage after ordinary model commitment, not a replacement for measurement validation.

## Reproduction

From `CoPH-Stagewise/`, use float64 CPU execution and single-thread numerical libraries:

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
python -m coph_stagewise.route_decision_suite --out results/mechanism_repairs_v1/route_decision --workers 4
python -m coph_stagewise.localized_receiver_evaluation suite --out results/mechanism_repairs_v1/reconstruction --workers 8
python -m coph_stagewise.communication_repair_suite freeze --out results/mechanism_repairs_v1/communication
python -m coph_stagewise.communication_repair_suite payloads --out results/mechanism_repairs_v1/communication
python -m coph_stagewise.communication_repair_suite faults --out results/mechanism_repairs_v1/communication --workers 8
```

Existing output directories are immutable or fingerprint-checked; use fresh paths for a new run. The communication consistency phase has its own frozen protocol, source amendment and reproduction command in its results directory. The preserved communication baseline ran against its exact frozen dependency archive after a concurrent overlap-edge-case correction; do not replace its source hashes with current ones.

See [localized formulation and commands](LOCALIZED_RECEIVER.md) and [overlap formulation and commands](STAGE_BLENDING.md). The full suite must run from the package directory because existing snapshot-based tests use package-relative artifact paths. The known archived-source-hash test remains a recorded failure rather than rewriting its archive.

## Evidence and interpretation

See [results](results/mechanism_repairs_v1/REPORT.md) for the completed counts, paired outcomes and limitations. The experiments isolate mechanisms; they are not a joint tuning campaign or held-out validation. Final-test namespaces remain untouched.

No option by itself establishes a receiver-reconstruction navigation benefit, historical-replay superiority, adaptive-rate advantage, or general switched-system stability/passivity. Smooth fixed-support overlaps can still encounter membership, stage-creation/eviction, model and waypoint jumps. Spatial blending also adds force terms, so the original per-stage radial-progress bound does not automatically extend to the blended potential.
