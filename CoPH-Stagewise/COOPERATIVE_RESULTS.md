# Cooperative implementation and development results

Implemented aligned historical replay, two identical local learners, reciprocal Hamiltonian-context packets, separate received-evidence storage, bounded field reconciliation, bootstrap uncertainty, fixed/information-gain/recipient-value scheduling, dynamic-window estimation, causal audits, and synchronized visualization. Existing CoPH-v2, CoPH-Sense, and previous result directories were preserved.

## Verification

- 34 numerical/contract tests pass.
- Replay alignment gate: 216/216 episodes completed, all successful. All 144 legacy/evidence costs and outcomes exactly reproduce prior results.
- Final cooperative smoke: 252/252 completed; every team succeeded; all logged contracts passed.
- Final cooperative development: 1,008/1,008 completed; every team succeeded; all logged contracts passed.
- All 14 final-suite GIFs passed 28 endpoint checks against logged states and model versions.
- Source archives match their recorded hashes. Renderer source and runtime/configuration fingerprints are saved separately.
- Audit isolation now covers learner work counters as well as model/state isolation. Across all 252 smoke episodes, the counter fix leaves trajectories, final models, bytes, replay work, identification work, and message-fit work identical.

## Local replay alignment

| Local arm | Mean complete mission cost |
|---|---:|
| Evidence only | 13.79507 |
| Legacy replay | 13.84622 |
| Aligned replay | 13.78375 |

Aligned replay removes the execution/replay target mismatch. Its short horizon remains a truncation, with no invented terminal-distance or timeout penalty. This small development improvement does not establish general superiority.

## Cooperative outcomes

The following rows use a 240,000-byte ceiling per agent. Each row contains the same 56 scenario/seed pairs. Costs exclude communication charges, which are reported separately and enter the scheduler objective.

| Arm | Mean team mission cost | Mean actual team bytes |
|---|---:|---:|
| disabled | 21.667210 | 0 |
| one_way_01 | 21.675623 | 119550 |
| one_way_10 | 21.658278 | 118798 |
| reciprocal_fixed | 21.666631 | 236475 |
| information_gain | 21.669709 | 215847 |
| recipient_value | 21.667210 | 91731 |
| evidence_only | 21.720598 | 237879 |
| no_history | 21.676902 | 236447 |
| no_reconciliation | 21.666665 | 236453 |

**Communication benefit is not established.** Reciprocal fixed-rate exchange changes mean cost by −0.000579 relative to disabled communication. Its descriptive 95% seed-block bootstrap interval is [−0.02790, +0.02562]. The information-gain arm changes mean cost by +0.002499, with interval [−0.02482, +0.02766]. These intervals use eight seed blocks across seven scenarios; they are development summaries, not confirmatory inference.

**Recipient-value scheduling currently fails to produce useful exchange.** Its predicted benefits do not exceed the declared byte/processing cost and uncertainty penalty, so it sends zero payloads while still spending bytes on status queries/replies. The penalties and scenarios were not adjusted to produce favorable results.

**Packet-benefit prediction is weak.** 1840/4226 model/truth audit pairs are domain-valid; sign agreement is 48.80%. Unsupported audits remain in the denominator and artifacts.

Reciprocal full learning is slightly cheaper than the evidence-only and no-history ablations in this suite. Reconciliation adds essentially no mission-cost benefit on these benign cases. Neither observation proves that those mechanisms are universally necessary.

## Limits and remaining evidence

- The frozen communication predictor uses 35 valid eight-step hypothetical training pairs. It is an initial approximation, not a full remaining-mission value estimator.
- The covariance calibration check covers a short acquisition prefix and reports marginal coverage. Joint uncertainty calibration across the mission remains unestablished.
- Sparse peer anchors do not automatically expand the locally certified replay domain. Unsupported historical and hypothetical paths remain excluded/rejected.
- Two independent physical plants share information; inter-agent contact and mechanical coordination are outside this implementation.
- Dynamic fields use a sliding-window estimator, not a learned forecast of future hazards.
- These are development seeds. Reserved validation/test namespaces were not run; navigation learners reset between missions.

## Artifacts and reproduction

- [Implementation and commands](COOPERATIVE.md)
- [Complete development report](results/cooperative_development_final/REPORT.md)
- [Smoke report](results/cooperative_smoke_final/REPORT.md)
- [Replay gate](results/cooperative_replay_gate_v2/REPORT.md)
- [Paired uncertainty intervals](results/cooperative_development_final/paired_intervals.json)
- [Common supported probes](results/cooperative_development_final/common_probes.json)
- [Message/model effects](results/cooperative_development_final/message_effects.json)
- [Causal audit rows](results/cooperative_development_final/audits.json)
- [Learning traces](results/cooperative_development_final/learning_traces.json)
- [Example synchronized GIF](results/cooperative_development_final/figures/complementary_60000_reciprocal_fixed_b240000.gif)
- [Failure gallery](results/cooperative_development_final/figures/failures/index.html)

Each final suite has seven preselected GIFs, paired world/trajectory figures, model-learning traces, cost/component/computation plots, and a prediction-versus-realization scatter. Empty failure galleries reflect zero failed completed episodes; completion and error records are separate. Intermediate debugging runs remain available, with their source snapshots and limitations documented in COOPERATIVE.md.

A final validation-only guard rejects nonfinite packet timestamps. The source archives retain the exact evaluated code; `postrun_validation.patch` and its regression test record this subsequent hardening. All evaluated packet timestamps are finite, so the guard does not change the recorded experiment paths.

## Follow-up recipient-value diagnosis

The [recipient prediction diagnostic](RECIPIENT_DIAGNOSTIC_RESULTS.md) adds a
four-condition receiver/environment comparison at 8/16/32 steps. Ten development
episodes preserve their original trajectories exactly. The 59 eligible audits
show that richer receiver knowledge alone does not resolve prediction error;
world-model error and supported prediction horizon also need attention.

## Supported-stage and receiver-summary follow-up

[The implementation and final evaluation](STAGEWISE_SUMMARY_RESULTS.md) add frozen
retained-stage prediction and compact receiver summaries. All 60 final episode
runs completed and 45 tests passed. The combined approximation failed separate
predictor validation on both error and coverage, so the existing predictor was
retained and conditional training was not executed.

Receiver-side evaluation and boundary-reaching replay are now implemented and
benchmarked in [RECEIVER_FEEDBACK_RESULTS.md](RECEIVER_FEEDBACK_RESULTS.md).
The new modes remain opt-in; no information-aware rate or predictor is promoted.
