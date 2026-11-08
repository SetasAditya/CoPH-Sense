# Supported stage prediction and compact receiver summaries

**Receiver fitting and validation isolation:** [Findings](results/receiver_corrections_v1/FINDINGS.md) · [Report](results/receiver_corrections_v1/REPORT.md) · [Implementation and commands](RECEIVER_CORRECTIONS.md). All 160 legacy snapshots reproduce. Fresh unchanged complementary cases complete **60/60 agent goals with zero collisions/timeouts**. Corrected fitting improves proposed estimates more often, but strict bounded convergence rejects many updates; no branch-choice benefit is demonstrated. Both new options remain opt-in, and the decision-sensitive fixture remains deferred.

**New direct-field routing study:** [Results and architectural correction](FIELD_ROUTE_RESULTS.md). The reconstructed field already influenced routing through local model anchors; this study compares that approximation with direct committed-field evaluation. **48/48 goals, zero collisions/timeouts**, and all 12 anchor regressions reproduce prior operational records. Direct evaluation improves supported risk estimates and the leader's direct-risk run, but slightly slows the carrier; additional received-field assimilation still shows no navigation advantage. Default routing remains unchanged.

**New decision-bottleneck diagnosis:** [Results, evidence, and recommended next experiment](results/decision_bottleneck_v1/FINDINGS.md). All 30 fork episodes completed with **60/60 goals, zero collisions/timeouts**. Remote anchors change the carrier's branch choice; reconstruction/replay add no demonstrated branch or completion-time benefit. Historical replay crosses stages but misses the decision marker; local policy changes are small, field validation sometimes rolls them back, and safety projection does not suppress controls in these forks. Both full maze regressions pass numerical equivalence, with bitwise/byte differences explicitly preserved. [Three synchronized GIFs and commands](DECISION_DIAGNOSTICS.md).

**New attribution study:** [Separate geometry sharing, field reconstruction, and historical replay](ATTRIBUTION_RESULTS.md). The seven-arm blocked/soft-risk maze suite is complete: **224/224 goals, zero collisions or timeouts**, with full artifact checks passed. Shared geometry reduces blocked-branch entries and retreat; field assimilation and historical replay show no additional navigation benefit in these two layouts. This development evidence remains separate from the preserved studies below. See the [verified implementation contract](IMPLEMENTATION_CONTRACT.md) for the mapping from the original formulation to current code.

**New maze demonstration:** [Cooperative maze, Hamiltonian landscapes and force visualization](MAZE_RESULTS.md). A single cooperative animation shows the scout discovering closures and the carrier responding. 8/8 goals, zero wall/gate collisions or timeouts; first-branch avoidance verified in both development seeds.

**New terrain-warning demonstration:** [Scout warning changes the carrier’s route](TERRAIN_WARNING.md). With a shared surveyed route layout and bounded remote terrain memory, the carrier selects the bypass before local hazard detection; 8/8 agent goals across two paired development seeds.

**New demonstration:** [Same-origin, five-second staggered starts](STAGGERED_DEMO_RESULTS.md): 16/16 goals; synchronized packet-timing and model/control-effect GIFs. Communication effects are measurable but small.

**Latest navigation status:** [Selected routed-progress held-out validation](ROUTED_VALIDATION_RESULTS.md) passed: 128/128 agent goals, zero collisions and zero timeouts across 64 episodes, with unchanged controller/learner code. Both communication settings passed independently. The original protected-progress failure is preserved as an ablation.

The receiver-summary prediction study below is an earlier, separate milestone; its failed predictor-validation gate remains unchanged.

The implementation and evaluation are complete. **The combined approximation did
not pass the predictor-validation gate.** It remains opt-in; the existing packet-
value predictor was retained and the conditional training manifest was not run.

## Implemented and verified

- Frozen retained-stage models support historical replay across known overlaps,
  using the execution stage threshold, potential, force clipping, and exact plant
  flow. Unsupported motion and unavailable future stages remain invalid.
- Versioned receiver summaries include learned policy/damping and retained model
  context without raw measured values, executed transitions, or reserved validation
  records. Reconstructed fitting/validation samples are explicitly hypothetical.
- Summaries have bounded version/content caches, stale-version rejection, byte
  accounting, and processing measurements. They are distinct from environmental
  evidence packets and do not enlarge certified support.
- Four prediction variants share indexed common-anchor Gaussian draws and sensor
  randomness. The Gaussian approximation is not claimed to be a calibrated
  posterior over the entire stage collection.
- 45 tests pass. Additional checks confirm retained-replay rollback and rejection
  of an older model version even when its timestamp is newer. A legacy episode
  reproduces its saved trajectories, final models, packets, costs, and work exactly.
- The controlled overlap fixture crosses beyond the original stage boundary with
  a nonzero policy gradient and a separate shaping-energy jump. Removing the next
  stage rejects that trajectory. Physical state/energy does not jump at handoff.

![Controlled overlap](results/stagewise_summary_final/controlled_overlap.png)

## Final evaluation

All 60 episode runs completed: 10 smoke, 40 development, and 10 separate predictor-
validation runs. Smoke repeats the first two development seeds, so these are **50
unique scene/seed pairs**, not 60 independent trials. There are 354 recorded packet
audit cases including the smoke repetitions; irrelevant-evidence scenes have zero
eligible packet cases and remain in the episode denominator. Artifact validation
covers 3,540 branches and 10,620 horizon checkpoints.

Development results below use the same 60 common-valid cases at 16 steps. Valid
coverage is reported separately over all 236 development packet opportunities.

| Variant | Valid predictions | Common-case MAE | Direction agreement | Initial acceptance disagreement |
|---|---:|---:|---:|---:|
| Sparse-status baseline | 102/236 | 0.005522 | 58.5% | 0.0% |
| Compact summary | 108/236 | 0.017841 | 51.2% | 18.3% |
| Cross-stage, legacy policy/plant initialization | 60/236 | 0.014179 | 48.8% | 18.3% |
| Combined | 60/236 | 0.017841 | 51.2% | 18.3% |

Directional agreement uses 41 cases with non-negligible predicted and true benefit.
Acceptance disagreement compares the two branches' initial update decisions;
synthetic validation is not equivalent to actual held-out measurement validation.
The four-way common mask prevents unsupported cases from becoming zero-error rows.

On separate predictor-validation scenes/seeds, the combined method's paired
16-step MAE was **0.011690**, versus **0.001428** for baseline. Coverage fell from
**25 to 15** cases. The gate therefore failed both its accuracy and coverage
requirements. Neither thresholds nor communication penalties were relaxed.

The opt-in execution learners recorded 1,428 replay calls but **zero stage
handoffs** in these development missions at the unchanged eight-step policy-replay
horizon. Team mission costs exactly match the old fixed-rate runs. The controlled
fixture establishes the new handoff capability; this suite does not establish an
operational advantage from that capability.

## What failed and what remains uncertain

A retained atlas supports historical movement through already known stages; it
cannot certify the unobserved forward frontier. Strict rejection at an unavailable
next stage explains the combined method's reduced prediction coverage.

A compact operational model does not reproduce the real receiver's evidence and
validation state. Synthetic validation disagrees with actual update acceptance,
and the reconstructed environment remains inaccurate. Richer status alone is
therefore insufficient.

Some large errors arise near goal termination. In redundant-coverage seed 60007,
a packet's true benefit was approximately 0.00000066, while the combined predictor
estimated 0.16833. The real branches both finished in 12 steps; the hypothetical
deliver/withhold branches finished in 14/15 steps with inflated risk cost. The
termination rule is unchanged, and these cases remain in the aggregate metrics.
See [worst-case branch decompositions](results/stagewise_summary_final/worst_cases.json).

Future work needs a better model of receiver acceptance and supported prediction
near future stages, together with uncertainty calibration. Increasing horizon or
transmitting more model parameters alone has not resolved these errors. The final
held-out namespaces remain unexecuted.

## Reproduction and artifacts

- [Commands and interface details](COOPERATIVE.md#supported-stage-prediction-and-compact-summaries)
- [Final reports and plots](results/stagewise_summary_final/REPORT.md)
- [Predeclared validation gate](results/stagewise_summary_final/gate.json)
- [Artifact checks and computation](results/stagewise_summary_final/contract_checks.json)
- [Mission comparisons](results/stagewise_summary_final/mission_comparison.json)

The final run took 481.2 seconds wall time with four workers. Evaluator and
prediction branches together used 45,871 simulated transitions, 387,283 replay
transitions, and 556,256 identification-work units. Field-draw reconstruction adds
294,720 Jacobian-entry work units. These counters are operation proxies, not FLOPs;
parallel branch wall times must not be interpreted as serial elapsed time.

Development diagnostic status/summary traffic was 1,839,793 bytes for baseline and
5,742,429 bytes for each summary-based variant. These are separate hypothetical
ledgers at audited opportunities, not equal-resource claims or the live episode's
communication totals. Runtime opt-in exchange charges actual serialized summaries
against the mission byte ceiling.

The intermediate point-estimate evaluation remains in `stagewise_summary_v1`.
The final `sources.zip` contains the exact sampled-evaluation code. A separately
recorded runtime-label correction ensures opt-in label collection returns its own
16-step labels and does not reuse legacy predictor residuals; those paths are not
used by the diagnostic suite. `runtime_label_fix.patch`, current source hashes,
and `current_sources.zip` preserve that correction and its passing tests.

## Follow-up: actual receiver evidence and boundary replay

The next milestone is implemented and evaluated in
[RECEIVER_FEEDBACK_RESULTS.md](RECEIVER_FEEDBACK_RESULTS.md). It uses receiver-local
measured validation and bounded boundary-reaching historical replay. Initial update
agreement improves, but long-horizon support and environmental-model errors remain;
fixed-rate exchange and the existing sender predictor are retained.

## Complex fixed-rate environments

[The complex-scene evaluation and GIFs](COMPLEX_FIXED_RATE_RESULTS.md) expose obstacle
storage and collision-avoidance limitations. Fixed-rate cooperation does not provide
a useful navigation benefit in these harder scenes; the failures are preserved.

## Subsequent navigation investigation

The fixed-rate complex-scene obstacle investigation is separate from these receiver-summary results. See [NAVIGATION_FIX_RESULTS.md](NAVIGATION_FIX_RESULTS.md) for the geometry, collision-protection and local-routing comparison, and [NAVIGATION_FIX.md](NAVIGATION_FIX.md) for implementation and reproduction.

## Receiver reconstruction in the existing hazard pilot

The [receiver reconstruction diagnosis](results/receiver_reconstruction_v1/FINDINGS.md) completes the existing-pilot investigation: all 32 saved pilot episodes analyzed, four bitwise operational tracing regressions passed, and twelve new replay-disabled comparisons completed with **24/24 goals, zero collisions and zero timeouts**. See the [generated report](results/receiver_reconstruction_v1/REPORT.md) and [commands](RECEIVER_RECONSTRUCTION.md).

The key result is an eligibility limitation: **none of 368 deliveries across the sixteen traced episodes entered receiver reconstruction**. Remote anchors remained usable for routing, but were outside retained reconstruction rectangles; the reconstruction stores stayed empty. All 36 carrier audits therefore had zero received-field fitting work. Latest-sender replacement was tested but was not the observed bottleneck. Direct-local and direct-cooperative outcomes matched, as did shared-anchor outcomes in these scenes. This is not a negative result about active fusion quality: the pilot did not exercise fusion. A future complementary-coverage fixture must establish eligible received evidence and supported route queries before the decision without enlarging support or tuning for a favorable outcome.

The original pilot, formulation, controller, learning rules and defaults are unchanged. Two post-run reporting-only corrections are recorded separately in `reporting_revision.json`; original archives and episode fingerprints remain preserved. Held-out and final-test namespaces were not used.

## Complementary-coverage receiver reconstruction

The [complementary-coverage diagnosis](results/complementary_reconstruction_v1/FINDINGS.md) passes all four required eligibility checks and completes **18 episodes / 36 agent missions, with zero collisions or timeouts**. All 18 acquisition prefixes match the frozen communication-free reference. Unlike the 25-second non-overlap relay, peer evidence now enters reconstruction before the branch decision and changes committed fields and applied controls.

The result remains mixed: the four intended-direction acquisition audits all worsen supported route-query value error relative to fitting disabled. Across 148 navigation audits, value error improves in 68 and worsens in 80; no selected route or target changes. All arms take the direct branch with equal completion times and small risk/trajectory differences. This establishes an active reconstruction mechanism, not added navigation benefit. The evidence supports investigating reconstruction quality and spatial validation before advancing to historical replay or adaptive communication. See the [report and GIFs](results/complementary_reconstruction_v1/REPORT.md) and [commands](COMPLEMENTARY_RECONSTRUCTION.md). Existing results, defaults, controller and learning rules are preserved.

## Active-set receiver fitting: isolated numerical gate

The [active-set findings](results/active_set_receiver_v2/FINDINGS.md) select **active-set with diagnostic search**, still opt-in. Strict-search active-set improves convergence but loses protected original successes, so it fails the predefined gate. The selected combination reproduces the archived candidate and converges on **690/697** canonical problems under both import orders, with seven preserved exhaustions and explicit near-tolerance records. The diagnostic search's longer allowance and rounding exception are part of the selected configuration.

All **30 navigation episodes / 60 agent missions** complete without collisions or timeouts. The 18 unchanged baseline episodes reproduce operational records bitwise. Selected corrected-fit solver rejections fall from the preserved **66/300 to 3/300**; the combined receiver additionally rejects 15 updates through relative validation. Paired branch choices remain unchanged. This supports numerical reliability, not a demonstrated route-choice benefit or uniformly better estimation. [Commands and option semantics](ACTIVE_SET_RECEIVER.md), [generated report](results/active_set_receiver_v2/REPORT.md), and [navigation GIFs](results/active_set_receiver_v2/navigation/REPORT.md).

The initial v1 attempt remains preserved: reconstructing inputs independently confounded import-order comparisons, so v2 freezes one shared collection of numerical inputs. Full testing reports 187 passing and the existing archived-source-hash error; defaults and previous results remain unchanged.

## Decision-sensitive receiver experiment: gate did not qualify the grid

The [preregistered screening experiment](results/decision_sensitive_v1/FINDINGS.md) completed **160/160 scene-seed screens**. All 144 grid pairs had supported queries on both outgoing branches and valid scout evidence, but **none** met the decision-sensitivity margin or bootstrap-straddling criterion: the estimated direct branch remained **3.16–4.90 cost units cheaper**. Exact route truth, including bypass tails, flips only for the x=3.0 amplitude family. Both controls pass their exempted checks but are withheld because no grid scene qualifies.

Per the frozen stop rule, **no treatment episodes or treatment GIFs were produced**. This does not test the receiver reconstruction route-choice claim. All 18 archived baselines reproduced operational records bitwise; 197 tests pass with the existing archived-source-hash error. Existing runtime source files, defaults and prior results remain unchanged. [Specification and commands](DECISION_SENSITIVE_SPEC.md), [complete gate](results/decision_sensitive_v1/gate.json), and [report](results/decision_sensitive_v1/REPORT.md).

## Current-algorithm campaign: broader controls and negative results

The [six-suite campaign](results/current_campaign_v1/FINDINGS.md) adds physical
accounting/handoff checks, 100 fresh native episodes with a correct-model
control, 48 representation episodes, 32 unselected-layout reciprocity episodes,
24 reconstruction/replay screens, and isolated fault audits. Existing results
are preserved; the new evidence narrows claims rather than replacing failures.

Online native identification improves return over the incorrect fixed model,
but the correct fixed model performs better, and the fresh comparison does not
support superiority over the published phMARL checkpoint. Graded costmap/raw
baselines match or outperform field packets on the four designed hazard scenes.
All 160 representation/reciprocity agent missions complete without collisions
or timeouts; reciprocal path savings occur in 3/8 unselected layouts.

Neither new reconstruction risk case passes the frozen decision gate, so no
navigation treatments execute. Matched fitting changes supported estimates in
both directions, and historical replay reaches a branch in 8/24 probes. Fault
checks reject invalid metadata/order, but biased legal packets still pass and
degrade estimates relative to clean evidence. Physical accounting remains at
floating-point precision while stage shaping/force discontinuities remain.

The full pytest suite records 209 passing tests and the known archived-source-
hash failure; all 10 native tests pass. See [commands](CURRENT_CAMPAIGN.md),
[artifact index](results/current_campaign_v1/REPORT.md), and the verified
[potential/force/communication GIF](results/current_campaign_v1/representation/figures/hamiltonian_pipeline.gif).

## Native assignment diagnosis: task allocation helps, residual goal offsets remain

The [assignment diagnostic](results/native_assignment_v1/FINDINGS.md) completes
120 episodes on the same 20 native starts, comparing index-assigned goals with
centralized reset-only minimum-squared-distance assignment. All 60 legacy runs
reproduce archived arrays bit for bit and decoded decisions, updates and work
exactly. All 14 native tests pass; three synchronized GIFs have verified 51-frame
logs/timestamps.

Full landmark coverage improves from 18/20 to 19/20 with frozen damping, 18/20
to 20/20 with known damping, and 17/20 to 18/20 with online identification.
Every reassigned episode is contact-free. Online mean return improves from
−247.372 to −85.676 (paired difference +161.697, bootstrap 95% interval
[40.756, 333.799]). These are reused development starts and centralized
allocation, not a decentralized superiority result.

Reassignment resolves the highlighted online failures at seeds 82000, 82011
and 82019 but loses coverage at seeds 82004 and 82017. The remaining uncovered
agents settle near protection-shifted targets, respectively 0.151766 and
0.168207 units from their landmarks, outside the unchanged 0.15 radius. Their
nominal commands are already small, unlike the original projection-suppressed
cluster stalls. Goal handling remains unresolved; no recovery waypoints or
safety changes were introduced.

See the [report](results/native_assignment_v1/REPORT.md),
[checks and residual traces](results/native_assignment_v1/diagnostic_checks.json),
[seed-82011 comparison](results/native_assignment_v1/comparison_seed82011.gif),
and [reproduction instructions](../benchmarks/phmarl/ASSIGNMENT_DIAGNOSTIC.md).

## Native landmark geometry: all three CoPH modes cover every landmark

The [static-target experiment](results/native_goal_geometry_v1/FINDINGS.md)
completes **200 episodes** on the same 20 development starts. Every agent
independently reconstructs the reset geometry from its full native observation;
all four assignments agree on all starts. This requires full native observations
and known identity ordering, not a radius-limited sensing interface.

A joint minimum-displacement target construction extends the symmetric pairwise
split to clusters. Targets remain within 0.149999 of their landmarks and have
projected center separation at least 0.32. They are frozen at reset and used
inside the unchanged 0.35 near-goal region. Actual-motion constraints, the 0.15
coverage threshold, controller gains, identification, and horizons are unchanged.
All starts admit a constructive joint witness; no start is excluded.

With minimum-distance assignment held fixed, full coverage changes from
19/20 to **20/20** for frozen damping, 20/20 to **20/20** for known damping,
and 18/20 to **20/20** for online damping. All 60 static-target CoPH runs are
contact-free. The three previously residual failure runs now pass. This is a
successful development diagnosis, not held-out validation.

The phMARL checkpoint also reads index-specific goal slots. Matched wrappers
permute its landmark slots consistently. Unmodified phMARL covers all landmarks
in 10/20 starts (2/20 contact-free); assigned raw phMARL covers 10/20 (17/20
contact-free); assigned with legacy protection covers 11/20 (20/20 contact-free);
assigned with static targets and matched protection covers 9/20 (20/20
contact-free). These are checkpoint evaluations, not retraining.

Online static-target CoPH has mean return −86.586, versus −127.126 for the
same frozen-damping controller, −75.889 for known damping, and −189.635 for
matched static-protected phMARL. The paired online-minus-frozen difference is
+40.540 (95% bootstrap interval [32.817, 48.844]); online-minus-known is
−10.697 [−13.763, −8.109]. Assignment and target planning are common navigation
improvements, separate from the benefit of online identification. Receiver
reconstruction and historical replay are absent from this native adapter.

All **80 archived baselines reproduce bit for bit**, all **20 native tests
pass**, and **128,000** logged static-target selections and executed controls
are checked. All 53 incomplete/contact episodes across the ten arms remain in
the inventory. See the [report](results/native_goal_geometry_v1/REPORT.md),
[artifact checks](results/native_goal_geometry_v1/artifact_checks.json),
[seed-82004 GIF](results/native_goal_geometry_v1/comparison_seed82004.gif), and
[commands and target equations](../benchmarks/phmarl/GOAL_GEOMETRY_DIAGNOSTIC.md).

## Opt-in mechanism repairs: tested improvements and remaining limits

The [repair report](results/mechanism_repairs_v1/REPORT.md) records six new
opt-in mechanisms; [interfaces and commands](MECHANISM_REPAIRS.md) preserve
legacy defaults and previous results. **249 main-suite tests pass; the existing
archived-source-hash check still fails**. Seven subtests and all **20 native
tests** pass. The archive is not rewritten to remove that failure.

Explicit route-candidate prediction now evaluates supported prefixes plus an
approximate remaining-route cost. **24 episodes / 48 agent missions** on the
unchanged symmetric forks all complete without collisions/timeouts. Prefix-only
scoring makes four worse branch decisions; terminal-current and terminal-history
match the graph baseline exactly. There is still no historical navigation gain.
Read-only diagnosis finds that failed prefixes remain within current measured
support but hit the rule requiring an unavailable prospective successor stage;
this is stricter than reaching the true field-support frontier.

Localized compact reconstruction is integrated through fitting, validation,
rollback, control, replay, routing and summaries. **184 snapshots / 552 updates**
verify locality and rollback. On the exact 24 original branch-query sets,
compact improves/worsens in **14/7**, with three unchanged, versus global **12/12**.
On the older 160 full-graph sets, compact is worse overall than corrected global
fitting. Locality does not guarantee better estimation or navigation.

Lossless field-packet coding reduces matched mean bytes **4,442.5 → 1,724.6**;
raw/costmap packets remain smaller. Receiver-relevant selection is operational
and tested at fixed rates. A historical-export bug is fixed and its first pilot
preserved. The corrected **16-episode / 32-mission** pilot completes every goal
without collisions/timeouts but shows **no mission advantage**, adding roughly
638–711 kB in the reciprocal cases. Corrected controlled selection yields
**108/108 nonempty eligible pairs**, rather than counting empty exports as useful
selection changes.

**2,560 corrected/legacy fault updates + 1,280 valid consistency updates** retain
all isolation and work checks. The consistency gate rejects all biased packets
but also 18 clean packets, including 14 useful updates; it remains off by default.
The failed serialization attempt is preserved and separately charged.

Smooth fixed-support overlap blending removes the controlled potential/force
handoff jumps, with replay-gradient error **2.04e−11**. Hard membership, model,
waypoint and stage-creation/eviction changes remain; general stability/passivity
and the original per-stage radial bound are not established for the blend.

See [route GIF](results/mechanism_repairs_v1/route_decision/figures/upper_risk.gif),
[fixed-rate selector GIF](results/mechanism_repairs_v1/communication/navigation_v2/figures/selected_comparison.gif),
and [verification](results/mechanism_repairs_v1/verification.json). These are
isolated development comparisons, not a claim that all six original advantages
are now demonstrated. Final-test namespaces remain untouched.


## Communication accounting reconciliation (development reproduction)

The corrected evaluation reproduces all **8 original hazard runs exactly**. All **72 compact-link episodes / 144 agent missions** reach their goals with zero collisions or timeouts. Each has a matched legacy-accounting rerun; **72/72 operational projections are bitwise identical**, including controls, measurements, decisions, versions, update work and communication events. The evaluation correction therefore changes costs, not navigation.

The old compact-link evaluator used carrier risk weight **1** while its planner used **6**. Corrected carrier cost is time plus the declared weighted risk, with effort and failure also reported separately in full cost. Team-total status bytes are no longer compared with a per-agent allowance. Old artifacts remain preserved as superseded measurements.

| Scene | No communication | Fixed-rate rotation | Receiver-relevant selection |
|---|---:|---:|---:|
| Mild | 70.38 | 70.38 | 70.38 |
| Moderate | 108.28 | 82.99 | 82.99 |
| Severe | 151.91 | 83.79 | 83.79 |
| Two mild hazards | 97.82 | 87.53 | 97.82 |

Table: mean carrier time + weighted risk, two noise seeds, 4,000,000 bytes per agent. Rotation helps in three designed scenes; the previously reported moderate/double-hazard harm was an accounting artifact. Recipient-aware selection has no added benefit and at 8,000 bytes can spend its allowance advertising before useful scout payloads fit. No fitting/replay advantage follows from this correction.

The original hazard channel and replay-enabled learner are reproduced separately from the compact evidence-only runner; channel/learner differences remain explicit. These are fixed scenes with noise-only replication, not generalization evidence.

See [findings](results/communication_reconciliation_v1/FINDINGS.md), [report](results/communication_reconciliation_v1/REPORT.md), [accounting and reproduction contract](COMMUNICATION_RECONCILIATION.md), and [moderate-scene GIF](results/communication_reconciliation_v1/figures/moderate_seed86100_budget4000000.gif). The full tests report **257 passed, 7 subtests passed, and the existing archived-source-hash failure**.


## Targeted simple fixes: communication improvement, safety failure, replay clarification

The [targeted-fixes report](results/simple_fixes_v1/REPORT.md) preserves all previous results and defaults. Independent acknowledgment/route-coverage and compact cached-status/reserve options were evaluated in **168 episodes / 336 successful agent missions**, with zero collisions/timeouts. All 72 baseline repetitions reproduce archived behavior and costs.

Both communication repairs recover rotation performance at 8,000 bytes per agent (mean primary difference +0.00002) and improve mean carrier time-plus-weighted-risk by **0.75507** at 16,000 / 32,000 / 4,000,000 bytes. The improvement comes entirely from the double-hazard case: **87.5258 → 84.5055** over two observation-noise seeds. Other scenes tie. More compute and, at higher budgets, more actual traffic than rotation are required; equal ceilings do not imply equal use. This is fixed-rate payload selection, not adaptive-rate evidence.

The centralized joint-protection candidate failed its development safety gate on **84 episodes / 21 starts**. All CoPH starts complete coverage and online return remains +101.60 above matched phMARL, but each CoPH arm has two new contacts. The candidate introduces a contact at 82014 and retains 85048. Read-only prediction with the public native contact law reduces position error from 0.0611 m to 6.18e-8 m, identifying omitted contact impulses. The failed variant stays opt-in; fresh 85200–85249 validation is not executed.

Code inspection and **24/24 exact reproductions** confirm that original differentiable historical policy replay already uses each entry’s own context. The later historical discrete route voter had the current-route mismatch. The new opt-in composition keeps original shared-policy learning and current-state route choice: 24/24 nonzero gradients, 23/24 committed policy changes, 24/24 finite-difference checks (maximum error 4.36e-11), max 390/1,024 flow work. No new navigation experiments or replay-benefit claim.

Full tests: **270 passed, 7 passing subtests, the existing archived-source-hash failure retained**. Native: **26/26 passed**. Independent arithmetic/state checks cover all 168 communication and 84 native episodes, all 336 communication-agent budgets, and preservation of 108 prior runtime modules. The results establish one scoped communication improvement; they do not establish six superiority wins.

## Decision-based packet scoring: accurate reactions, failed benefit gate

The [new diagnostic](results/decision_packet_v1/FINDINGS.md) reproduces eight
archived rotation episodes bitwise operationally, then evaluates 148 candidate
packets at 64 frozen sender/receiver snapshots. The receiver's actual isolated
post-update route is predicted correctly in **148/148** cases, with immutable
measurements/physical state and maximum update flow 6/1,024.

The proposed selector chooses the same stage as rotation in **64/64** snapshots.
On 50 common multi-candidate states, both have mean true spatial-route selection
regret **0.012769**, versus **0** for the repaired coverage selector. Two small
adverse connector reversals explain the difference. Production edge and connector
quadrature use different grids; resampling the same learned field at spacing .03
changes their predicted benefits from +0.217/+0.186 to −0.302/−0.320. This is a
planner-score integration issue, not a new mission-cost accounting error.

**138/148** packets have zero immediate route effect; this does not establish
zero future or multi-packet value. Standalone compressed decision advertisements
average **4,051 bytes** before payloads. The frozen progression gate fails, and
no navigation treatments are launched. Existing results and defaults remain.
Full tests: **277 passed, 7 passing subtests, the same archived-source-hash failure**.

## Additive integration and packet-sequence diagnostic

The [new findings](results/decision_sequence_v1/FINDINGS.md) establish an opt-in
additive route-cost representation: full edges and partial connectors use one
nonnegative density, and execution follows the scored projection-plus-edge
polyline. Eight legacy runs reproduce bitwise. All 64 saved-state routing checks
pass; canonical geometric post-update predictions agree in 148/148 candidates.
The two earlier false-positive benefits disappear.

All **16 full additive-routing episodes complete 32/32 agent goals**, with zero
collisions/timeouts. This is not a cost improvement: carrier primary cost rises
by 0–2.4397 relative to matched legacy counterparts. The mode changes connectors
from diagonal shortcuts to projection-plus-edge paths and stays opt-in.

Eight first-carrier snapshots produce **56 matched true-simulator packet-order
continuations**. Sequence selection passes its frozen ranking gate: mean
32-step prefix-plus-nominal-tail regret **0.319155 → 0.158784**. The improvement
comes entirely from double_mild seed86101; seed86100 retains a missed opportunity.
Incurred 32-step costs are essentially unchanged. All 24 selected-order branch
directions agree, but nominal predictions exit the junction 2–6 steps too early.
This supports a narrow diagnostic result, not a full-mission scheduling claim.
No live sequence-selector campaign or held-out run was executed.

Correction to the preceding v1 section: its hand-written diagnostic truth helper
omitted background risk 0.02. Repricing the same saved routes with the simulator's
truth gives mean regret **0.018898**, rather than 0.012769, for rotation and the
old decision selector; repaired coverage remains 0. The signs and conclusion are
unchanged. Actual simulator/mission-cost logging already included the background.
Original artifacts remain, with corrected values in
[v1_endpoint_correction.json](results/decision_sequence_v1/v1_endpoint_correction.json).

Verification covers all episode flow/cost components, continuation receipts/bytes,
32 frozen/execution control steps, eight finite-difference checks (maximum error
8.05e-12), and receiver-summary conventions. Full tests: **285 passed, 7 passing
subtests, the existing archived-source-hash failure retained**. Use
[paired_horizon_corrected.json](results/decision_sequence_v1/paired_horizon_corrected.json)
for branch metadata at 8/16/32 steps; its reporting correction leaves all 32-step
gate endpoints unchanged.
