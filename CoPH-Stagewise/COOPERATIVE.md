# Cooperative Hamiltonian learning

This extension runs two copies of the stagewise learner. It evaluates within-mission reciprocal model adaptation, not across-episode navigation learning. Existing single-agent commands retain their legacy replay objective so previous commands remain reproducible. The cooperative commands explicitly select `Config(replay_objective='mission')`.

## Reproduction

Run from `CoPH-Stagewise/`:

```bash
export MPLCONFIGDIR=/tmp/coph-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m coph_stagewise.cli check
python -m coph_stagewise.cli replay-gate --out results/cooperative_replay_gate_v2 --workers 12
python -m coph_stagewise.cli cooperative-suite --profile smoke --out results/cooperative_smoke_final --workers 12
python tests/validate_cooperative.py results/cooperative_smoke_final
python -m coph_stagewise.cli cooperative-suite --profile development --out results/cooperative_development_final --workers 12
python tests/validate_cooperative.py results/cooperative_development_final
python -m coph_stagewise.cli cooperative-report results/cooperative_development_final
python -m coph_stagewise.cli cooperative-render results/cooperative_development_final
```

Creation commands require a fresh output directory. Rendering/reporting regenerate derived artifacts only. Both communication ceilings, 120,000 and 240,000 bytes per agent per mission, are frozen before evaluation. Smoke contains 252 episodes; development contains 1,008. Seven scenarios, nine arms (including both one-way directions), and two/eight seeds produce these counts. Calibration and scheduler fitting use separate recorded seeds; future validation and test namespaces remain unexecuted.

## Implemented data flow

`CooperativeLearner` accepts local `Observation`/`Transition` records and immutable `Packet` records, through distinct interfaces. It never receives a world object. Agent-prefixed observation identities prevent cross-agent collisions.

A packet carries position, momentum/velocity, navigation Hamiltonian, task potential, goal/context conventions, environmental value/gradient anchors, joint covariance, support geometry, timestamps, version, and source IDs. Here

$$H_i^{nav}(q,p)=\tfrac12\|p\|^2+V_i^{task}(q)+V_i^{shape}(q),\qquad H_i^{physical}(p)=\tfrac12\|p\|^2.$$

The receiver does not copy the sender's goal, momentum, damping, or policy. Task information describes the estimate and supports recipient-value modeling. The actual transferable fit residual is environmental values and gradients. Forces are applied once, through the controller.

Exports are rebuilt from local sensor measurements with a fixed ridge prior, independently of the operational model. Thus a received estimate cannot be silently re-exported as locally measured evidence. A sender's newest usable contribution replaces its previous contribution. Packet and evidence acknowledgments prevent treating an unchanged source set as new information. Returned evidence, expired packets, incompatible units/conventions, reordered summaries, invalid covariance, and unsupported packets are rejected with reasons.

Message fitting uses a Gauss–Newton step in value/gradient units, regularized by local fitting curvature, with a 0.5 per-coefficient step bound. The existing held-out local field/plant validation and policy reconciliation determine commitment. Rejected candidate fields and replay calls remain in update records. Assimilation does not mutate position, momentum, or measured records.

Both agents sense before any delivery. Learning opportunities are matched across arms, including exogenously scheduled stress-channel delivery times. The queue sorts deliveries deterministically. This avoids adding extra optimization rounds only to communicating agents. Communication does not impose fixed scout/carrier learning roles.

## Replay and energy conventions

Aligned replay uses the executed mission-goal attraction, clipped force, exact held-input damped flow, and time/risk/effort running cost. Sensed obstacle collisions incur the existing failure cost. Goal entry terminates a replay trajectory. A short replay horizon is a truncation, not a mission timeout, so it adds neither the old distance penalty nor a fabricated timeout penalty. Gradients are piecewise derivatives away from discrete collision, goal, and domain boundaries.

Historical stage-entry states remain the replay starts. Unsupported starts and domain exits are reported. The supported replay domain remains certified by local measured support: sparse peer anchors do not automatically grant arbitrary extrapolation permission. This conservative choice can limit the benefit of communication and historical replay.

Logs separate actuator work, dissipation, clipping work, physical energy residual, model shaping jumps, and observation/stage context changes. Hamiltonian updates change model energy at a fixed physical state; they are not physical work.

## Uncertainty and scheduling

A bounded eight-member bootstrap resamples entire acquisition-time blocks, preserving the spatial batch of each sensor acquisition. Three calibration scenes determine a covariance scale and three different scenes assess marginal 95% coverage. The stored covariance includes a declared floor. Calibration probes cover a short acquisition prefix, not every position or hazard intensity encountered on a full mission. Passing this small development check is not evidence of a calibrated joint posterior everywhere.

Information gain is computed on the same advertised field variables. Unsupported but in-window receiver probes are marked through larger prior uncertainty; they are not additional observations. A novelty fraction discounts overlapping source sets. This is a declared approximation to correlated information gain, not exact conditional mutual information.

Fixed-rate exchange runs at the existing learner-update interval. Information-gain and recipient-value schedulers solve the scalar bounded-rate objective by derivative bisection and sample sends using index-addressed randomness. All status query/reply bytes, packet payloads, duplicate/return transmissions, and acknowledgments count against hard per-agent budgets. Predictor residual RMSE penalizes recipient-value uncertainty. Resource comparisons use matched ceilings, not identical realized bytes.

The recipient predictor is a regularized linear model trained before evaluation on paired hypothetical deliver/withhold branches reconstructed only from sender packets and receiver advertisements. Synthetic readings belong exclusively to hypothetical learners. Unsupported branches are missing labels, not zero-benefit examples. The frozen predictor uses eight-step labels as an approximation to remaining-mission benefit; it is **not** a learned full-horizon mission-value estimator. This limitation is explicit in the predictor artifact and report.

## Scenarios, audits, and interpretation

Scenarios cover complementary observations, early warning, reverse warning, redundant coverage, irrelevant evidence, changing hazards, and a delay/duplicate/return stress channel. Dynamic fields vary smoothly in amplitude. The dynamic estimator refits a three-second sliding measurement window; two-second packet expiry is separate from temporal estimation. It does not forecast unseen changes.

The physical plants are independent; peer contact dynamics and inter-agent collision avoidance are not modeled. This isolates information cooperation. Neither agent has privileged geometry or field access. Obstacles remain local sensor information; the initial cooperative scenes focus on field transfer and do not transmit obstacle geometry.

Evaluator-only audits clone the complete receiver and true simulator at the first eligible packet opportunity per visited stage. Deliver and withhold branches share starts, horizons, indexed sensor noise, and future communication suppression. They perform the actual receiver update and subsequent sensing/control. Because physical agents are independent, this receiver intervention's cost difference equals the corresponding team difference with the other agent held unchanged. Model-based paired continuations use the same audit horizon; model-domain failures remain in the audit denominator.

Reports separate actual team mission cost, model error/coverage, replay work, scheduler overhead, communication work, and evaluator audit work. They include common supported probes for static scenarios, time-indexed dynamic traces, paired figures, synchronized GIFs, and every failed completed episode. Incomplete episodes are recorded separately. No threshold requires every packet, replay update, or complete arm to improve mission cost.

## Source map

- `cooperative.py`: learner-side packet validation, local-only exports, bootstrap covariance, message residuals, rate optimization.
- `scheduling.py`: legal-information hypothetical continuations and the frozen recipient predictor.
- `cooperative_evaluation.py`: paired scenarios, queues, actual rollouts, truth-only audits, manifests and suites.
- `cooperative_reporting.py`: machine-readable aggregation, static figures, GIFs and failure gallery.
- `replay_gate.py`: frozen-manifest comparison of evidence-only, legacy replay, and aligned replay.

The exact physical flow and work accounting retain the source attribution in `model.py` to CoPH-Sense and CoPH-v2.

Intermediate run directories are preserved. The first replay gate contains 17 serialization failures; `_v2` is the complete replay gate. Early cooperative runs exposed an audit-only export counter mutation. Final runs use cloned status generation and a regression test proving audit-on/off equality of trajectories, model ledgers, communication bytes, and learner work counters.

Work counters are operation proxies, not FLOP counts. `export_work` counts the primary local-export reconstruction; auxiliary status refits are included in wall-clock scheduling latency. Audit branch timing and transition/identification/replay work are separate from learner counters, and total episode wall time includes audit preparation. Agents stop execution and communication when their mission terminates.

## Recipient prediction diagnosis

Run the evaluator-only factorial audit with:

```bash
PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.cli recipient-diagnostics \
  --out CoPH-Stagewise/results/recipient_diagnostics_v1 --seeds 2
```

Use a fresh output directory. The default manifest freezes five static scenes and
existing development seeds 60000–60001, reciprocal fixed-rate exchange, and
8/16/32-step horizons. At the first eligible delivered packet per receiver stage,
the diagnostic compares exact versus advertised receiver state in true versus
frozen reconstructed environments. All four cells use the same continuation
routine, start, control/cost conventions, learning schedule, and indexed sensor
noise. Horizon checkpoints share prefixes. True-world continuations include
normal termination costs; truncation alone is not a mission failure.

The common frozen world uses the incumbent receiver's current-stage field,
sensed obstacles, and damping estimate. It filters hypothetical sensor readings
to certified support and rejects motion outside that support. It is deliberately
a controlled diagnostic environment, **not** the legacy scheduler's sampled
posterior world. Thus this experiment measures reconstruction sensitivity but
does not by itself explain every error of the existing predictor.

Exact snapshots and true-world results never become runtime scheduler inputs or
training labels. The diagnostic callback receives independent copies. Existing
training rules, packet payloads, costs, and schedules remain unchanged. Saved
artifacts include the frozen manifest and source archive, all episode records,
packet payloads, branch paths and updates, validity masks, per-horizon costs,
work counters, wall times, common-valid comparison rows, report, and scatter.
Missing predictions and incomplete runs remain visible. Benefits near zero
(within 1e-6) are excluded only from directional agreement, with denominators
reported. Other metrics retain them.

## Supported-stage prediction and compact summaries

Run the frozen smoke, development, and predictor-validation sequence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl-cache \
PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.cli stagewise-diagnostics \
  --out CoPH-Stagewise/results/stagewise_summary_final --workers 4
```

All manifests, including a conditional training manifest, are written before any
method runs. Smoke repeats the five static scenes on development seeds
60000–60001; development uses 60000–60007. Separate predictor-validation seeds
45000–45001 determine the retraining gate; training seeds 30000–30007 execute
only if that gate passes. Reserved final validation/test namespaces remain unused.

`Config(replay_objective='mission', replay_domain='retained')` opts into historical
replay across the frozen retained-stage collection. `single` remains the default.
Replay follows the same stage-switch threshold and control/potential computation
as execution. Crossing requires a supported overlap and a known consecutive
stage. Missing future stages and unsupported motion reject prediction, including
when only one member of a replay batch fails. This is deliberately conservative.
Gradients are piecewise derivatives through the active model, integrator, and
policy; discrete support/stage boundaries are not claimed to be differentiable.
Stage-switch shaping jumps are logged separately with zero physical-state jump.

`ReceiverSummary` is a separate versioned model-context interface. It contains
state, goal, policy, plant estimate, configuration, retained stage coefficients,
entries, sensed obstacles, support coordinates, and source acknowledgments. It
contains no raw measured values, executed transitions, or held-out validation
records. Local fitting coordinates alone certify support; received environmental
anchors cannot enlarge it. Summaries are not accepted by `assimilate` as new
independent environmental evidence.

Reconstruction creates a hypothetical learner with the advertised policy, damping,
and models. Its synthetic readings and validation samples are explicitly labeled;
its acceptance decisions approximate actual measured-evidence validation. Existing
received-message payloads are not copied: their acknowledged sequence/provenance
is retained and their effect is represented in the operational field prior. This
is an approximation, not an exact checkpoint of past learning.

`SummaryCache` keys model content by sender, recipient, version, and content digest.
Support changes can invalidate content even without a new committed model version.
Every transmitted reference still carries state and time. Stale/reordered summaries
are rejected, the cache is bounded, and a budget failure never grants free access
to the offered model.

For opt-in runtime experiments, the Python entry point is:

```python
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.cooperative_evaluation import execute, specification

result = execute(
    specification('complementary', 60000, 'development'),
    'reciprocal_fixed',
    CommunicationConfig(prediction_variant='both'),
    replay_domain='retained',
)
```

`prediction_variant` defaults to `legacy`. Additional choices are `baseline`,
`compact`, `cross_stage`, and `both`. New variants use direct 16-step model-pair
benefit estimates; they do not silently reuse a predictor trained on legacy
labels. Runtime queries, status, summaries, and payloads consume the same byte
ceiling; summary-generation and prediction wall times are reported separately.
The existing mission-cost components and communication prices are unchanged.

The diagnostic runs all predictions against the same exact-receiver/true-world
reference while leaving live fixed-rate communication unchanged. Each prediction
pair shares one indexed Gaussian common-anchor field draw and indexed sensor
randomness. Value/gradient residuals transport that draw into supported stage
bases, without expanding support. This uses the earlier scheduler’s Gaussian
approximation; it does not claim a calibrated posterior over entire stage
collections. All model histories stay hypothetical.

Four variants distinguish the previous sparse-status reconstruction, the compact
learned-state summary, retained-stage prediction with legacy initial policy and
plant parameters, and the combined method. Retained-stage geometry itself requires
summary traffic, even in the cross-stage ablation. Reported diagnostic traffic is
a separate per-variant ledger at audited opportunities, not total live traffic or
an assertion of matched communication use.

The predeclared gate uses 16-step predictor-validation pairs: at least ten common
valid cases, strictly lower paired absolute error, retention of every baseline-valid
prediction, and no incomplete validation episodes. A failed gate leaves the old
predictor unchanged. A passing gate trains a separate candidate only from legal
model-continuation labels, never from true-world audit benefits; it is not promoted
automatically. Short-horizon benefit remains different from remaining-mission value.

## Receiver-side feedback and boundary-reaching replay

Both mechanisms are opt-in. Existing defaults, packet conventions, penalties,
termination rules, and sender predictor remain unchanged. Static reciprocal
fixed-rate exchange is the supported configuration for this milestone.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=CoPH-Stagewise \
  python -m coph_stagewise.cli check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl-cache \
  PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.cli receiver-feedback-suite \
  --out CoPH-Stagewise/results/receiver_feedback_reproduction --workers 16
```

The command requires a new output directory, freezes the manifest and source
archive before execution, runs two smoke seeds then eight development seeds per
scene, and writes full episodes, feedback/audit branches, prediction rows, plots,
and reports. Smoke repeats the first two development seeds at the same 180-step
horizon. Reserved validation and final-test namespaces are not executed.

```python
from coph_stagewise.cooperative_evaluation import execute, specification
result = execute(
    specification('complementary', 60000, 'development'),
    'reciprocal_fixed', audits=False,
    replay_domain='retained', feedback_mode='receiver',
    horizon_mode='supported_boundary', replay_cap=1024,
    feedback_audits=True,  # evaluator-only truth branches
)
```

`feedback_mode` is `none` (default), `compact`, or `receiver`. The compact arm
requests and pays for a current receiver summary and anchor uncertainty status;
the receiver arm evaluates locally using its actual retained measured and
validation records. `CooperativeLearner.evaluate_packet(packet)` is read-only:
it clones complete learning state, validates the packet, and returns a versioned
`PacketFeedback` plus four paired reconstructed-environment draws. Operational
assimilation and commitment occur once, afterward. Only the first eligible
packet per visited receiver stage is evaluated. Future simulated readings have
hypothetical provenance and remain in branch-local stores.

Feedback includes checkpoint means, sample dispersion, valid-draw counts,
exclusions, initial acceptance decisions, shaping/force/parameter revisions, and
work counts. Partial-draw means are conditional on supported continuations;
missing predictions are never replaced with zero. The Gaussian field-draw proxy
is not a calibrated atlas posterior, and sample dispersion is not confidence.

`Config.replay_horizon_mode` is `fixed` (default) or `supported_boundary`.
Boundary mode requires mission replay over retained models. An incumbent pilot
chooses at least eight steps, extending to the first supported handoff plus four
steps (at most 32); genuine termination still applies. If the pilot cannot reach
that handoff, the horizon remains eight and its reason is logged. Entry states
and horizons stay fixed through baseline, differentiation, and candidate
validation. Unknown stages and unsupported motion invalidate the replay batch.
Pilots log fixed-state shaping jumps separately from physical work.

Boundary mode defaults to a 1,024-transition update ceiling. Both factorial replay
arms explicitly use that ceiling. Baseline and gradient work are reserved before
proposal search; pilots and candidate checks consume the same counter. Work is
counted when a transition is executed, including a transition later rejected for
support. The ceiling is per local update, not per episode or entire four-draw
assessment. Field fitting and communication processing are separately counted.

Receiver feedback replaces a basic ACK when the return payload fits the existing
byte ceiling; otherwise its estimate and computation remain logged as unsent.
Feedback bytes are a subset of ACK bytes. Compact summary/status replies count as
status traffic. Actual payload, status, ACK/feedback, and processing costs are
reported separately. Feedback neither controls this milestone's fixed-rate
opportunities nor recovers bytes already spent. Different overhead can exhaust
budgets at different times, so realized traffic and physical states may differ
between arms. True-world audit computation is excluded from runtime prediction
and recorded separately. No sender predictor is retrained or promoted.

For read-only artifact checks and matched-state pairwise tables after the suite:

```bash
PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.receiver_feedback_audit \
  --root CoPH-Stagewise/results/receiver_feedback_reproduction
```

This verifies the frozen rollout source hashes and saves its own analysis source
fingerprint separately. Pairwise tables compare compact versus exact receiver
estimates within each replay arm; the four-way common cohort may be empty at
longer horizons because boundary replay changes the operational trajectory.

## Complex static environments and synchronized comparisons

See [COMPLEX_FIXED_RATE_RESULTS.md](COMPLEX_FIXED_RATE_RESULTS.md) for four new
layouts, six fixed-frequency arms, reproducible commands, paired GIFs, and the
collision audit. The harder tests expose navigation failures; they do not establish
cooperative superiority or change default learning/control behavior.

## Sensed-geometry navigation comparison

The opt-in four-controller suite, collision projection, route potential, and reproduction commands are documented in [NAVIGATION_FIX.md](NAVIGATION_FIX.md). See [NAVIGATION_FIX_RESULTS.md](NAVIGATION_FIX_RESULTS.md) for results. Existing legacy commands and frozen results remain unchanged.
