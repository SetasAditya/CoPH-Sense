# Receiver-side packet evaluation and boundary-reaching replay

Both mechanisms are implemented and opt-in. The development evaluation supports
using actual receiver evidence to reproduce the initial update. It does **not**
establish sufficiently reliable feedback for automatic communication-rate control.
Sender predictor training and information-aware rates remain deferred.

## Implementation and verification

- `CooperativeLearner.evaluate_packet(packet)` evaluates isolated deliver/withhold
  copies of the complete receiver learning state. Initial reconstruction,
  reconciliation, reserved-data validation, and rollback use actual retained
  measurements. Four indexed model-environment draws share randomness across the
  branches, with checkpoints at 8, 16, and 32 steps. Future observations remain
  hypothetical and never enter operational stores.
- `replay_horizon_mode="supported_boundary"` pilots known retained-stage entries,
  freezes horizons reaching the handoff plus four steps (at least eight, at most
  32), and falls back to eight with a recorded reason. Pilots, baseline,
  differentiation, and candidate validation share the per-update transition
  counter. Both comparison arms use a 1,024-transition ceiling.
- Versioned feedback records include identity/version/time, conditional benefit
  estimates, dispersion, validity/exclusions, acceptance decisions, model/force
  revisions, and work. Feedback replaces a basic ACK when it fits the byte
  budget; otherwise the estimate and processing remain logged as unsent.
  Evaluation occurs at the first eligible packet in each receiver stage, followed
  by exactly one operational assimilation/update.
- **50 numerical/contract tests pass.** Controlled checks cover successful
  supported handoffs, finite-difference replay gradients, unsupported frontier
  rejection, pilot failure, exhausted proposal budgets, invalid-update rollback,
  unchanged operational state, and unsent feedback.
- **200 episodes completed:** 40 smoke plus 160 development; smoke repeats the
  first two of eight development seeds. All agents reached their goals in these
  static fixtures, with no collisions, timeouts, or incomplete runs. All 32
  development episodes in the irrelevant-information scenario remain present
  with zero eligible packet opportunities. This suite does not demonstrate
  robustness in harder or changing scenes.
- Artifact checks cover **27,872 operational/prediction/audit updates** and
  **1,178 feedback opportunities** across both profiles. The largest update used
  **334 transitions**, below the ceiling. A separate controlled 32-transition
  case exhausted its proposal budget after baseline and gradient evaluation and
  retained its incumbent policy. The 1,024 ceiling was not binding in development.
- With the new options disabled, saved trajectories, packet events/payloads, and
  final models exactly match the previous `complementary/60000` result.

## Prediction results

Development results below compare compact and exact receiver evaluation at
matching physical states and packet opportunities **within each replay mode**.
Every prediction is compared with its corresponding true-world deliver/withhold
continuation. Those audit branches never supply runtime inputs or training labels.

| Replay | Horizon | Common valid cases | Compact MAE | Receiver MAE |
|---|---:|---:|---:|---:|
| Fixed | 8 | 172 | 0.003114 | 0.001629 |
| Fixed | 16 | 60 | 0.015743 | 0.004441 |
| Fixed | 32 | 60 | 0.027216 | 0.013843 |
| Boundary | 8 | 171 | 0.003164 | 0.001677 |
| Boundary | 16 | 59 | 0.011793 | 0.004282 |
| Boundary | 32 | 59 | 0.029076 | 0.012738 |

Exact receiver initial acceptance agreement is **100%**, versus **69.5%** for
compact/fixed and **69.4%** for compact/boundary. The artifact audit additionally
checks equality of initial validation residuals, policy, and control against the
true-world branch. This isolates the benefit of removing synthetic initial history.

Coverage remains limited: fixed replay supports **60/236** packet opportunities
at 16 steps; boundary replay supports **59/235**. All these valid estimates have
four valid draws. Unsupported predictions remain missing. Receiver benefit-sign
agreement is **26/40 (65%)** for fixed and **25/38 (65.8%)** for boundary at 16
steps, excluding effects within 1e-6 of zero. These denominators are development
cases, not independent held-out evidence.

There are **no matched physical-state cases across fixed and boundary replay at
16 or 32 steps** in this run: boundary replay has already changed the executed
trajectory at those opportunities. Consequently, cross-replay differences in
prediction MAE are descriptive, not a matched-state improvement claim. The
pairwise report preserves the zero denominators.

A remaining failure is `reverse_warning`, seed `60005`, receiver 1, packet
`0:2.880000000:2`, at 32 steps. Exact receiver/fixed predicts benefit **+0.15312**
with dispersion **0.000666**, while the true benefit is **−0.000759**. Both true
branches reach the goal in 22 steps. The first reconstructed draw predicts 23
steps for withholding and 22 for delivery, alongside excessive reconstructed risk.
The initial update is correct; the future environment and termination prediction
are not. Small draw dispersion does not reveal this model bias and must not be
interpreted as confidence.

## Mission behavior and computation

Each row contains 40 development episodes. Replay handoffs count executions of
handoffs across baseline, gradient, and candidate rollouts, including repeated
replay of the same retained entry; they are not distinct physical transitions.

| Variant | Mean team mission cost | Execution handoffs | Replay handoffs | Operational replay transitions | Prediction replay transitions | Actual communication bytes |
|---|---:|---:|---:|---:|---:|---:|
| Compact, fixed | 21.129156 | 224 | 0 | 26,445 | 166,848 | 13,699,965 |
| Receiver, fixed | 21.129156 | 224 | 0 | 26,445 | 160,604 | 8,573,353 |
| Compact, boundary | 21.098864 | 224 | 1,881 | 48,425 | 296,258 | 13,608,359 |
| Receiver, boundary | 21.102104 | 224 | 1,881 | 48,425 | 289,876 | 8,521,296 |

Boundary replay generates **653 successful boundary-horizon plans** per estimator
arm and uses about **1.83×** the operational replay transitions. Receiver/boundary
reduces mean team mission cost by **0.02705 (0.128%)** versus receiver/fixed; all 40
paired differences are negative in these development fixtures. The improvement is
small, and all methods already succeed. It does not establish broad superiority
or a benefit from feedback-driven scheduling: scheduling was fixed-rate throughout.

Receiver feedback itself costs **604,846 bytes** with fixed replay and **605,605
bytes** with boundary replay. It is included in ACK accounting, not added twice.
Compact summary/status replies require more traffic. Payloads, status, ACKs,
feedback, and processing are separately recorded. Different overhead can exhaust
byte budgets at different times, so fixed opportunity rates do not imply identical
actual communication. Feedback does not recover bytes already spent delivering a
packet. The small compact/receiver mission-cost differences arise despite feedback
not controlling rates; the byte ledger can affect subsequent transmissions.

Total suite elapsed time was **428.5 seconds with 16 worker processes**, including
smoke, development, evaluator branches, and report generation. Per-episode and
per-update latency, identification work, pilot work, field-draw work, and evaluator
work remain in the machine-readable artifacts. The update ceiling applies to each
learning update, not the combined work of four multi-update prediction draws.

## Artifacts and reproduction

- [Frozen manifest](results/receiver_feedback_v1/manifest.json),
  [evaluated source archive](results/receiver_feedback_v1/sources.zip), and
  [numerical checks](results/receiver_feedback_v1/numerical_checks.txt).
- [Development report and plots](results/receiver_feedback_v1/development/REPORT.md),
  [matched-state comparisons and diagnostic plots](results/receiver_feedback_v1/development/PAIRWISE.md),
  and [smoke report](results/receiver_feedback_v1/smoke/REPORT.md).
- [Prediction rows](results/receiver_feedback_v1/development/rows.json),
  [episode metrics](results/receiver_feedback_v1/development/episodes.json),
  [complete feedback/audit records](results/receiver_feedback_v1/development/feedback_audits.json.gz),
  and [failures](results/receiver_feedback_v1/development/failures.json).
- [Artifact verification](results/receiver_feedback_v1/artifact_verification.json),
  [controlled mechanisms](results/receiver_feedback_v1/controlled_mechanisms.json),
  and [legacy regression](results/receiver_feedback_v1/legacy_regression.json).
- [Commands and interfaces](COOPERATIVE.md#receiver-side-feedback-and-boundary-reaching-replay).

The frozen rollout source hashes were verified after execution. The read-only
pairwise/contract analysis is archived separately with its own source fingerprint.
Previous results and reproduction commands are preserved. No held-out validation
or final-test seeds were executed, and no sender predictor or default was promoted.
