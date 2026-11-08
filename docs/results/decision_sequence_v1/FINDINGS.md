# Findings: integration repaired; sequence value remains preliminary

**The numerical route-integral defect is fixed in the opt-in mode.** Partial connectors and full edges share a nonnegative density, and execution follows the scored projection-plus-edge path. Eight legacy episodes remain bitwise identical. All 64 additive routing advertisements match production routing. The two previous false-positive benefits disappear. Raw post-update node-list predictions agree in 140/148 cases; canonical geometric paths agree in 148/148. The remaining raw differences insert a redundant collinear junction, not a different physical branch.

**The common routing change preserves completion, but increases mission cost in these scenes.** All 16 full episodes complete 32/32 agent goals, with zero collisions/timeouts. The scalar controller/safety settings are unchanged, but additive connectors use projection spurs rather than the old diagonal shortcuts. Carrier time-plus-weighted-risk changes range from 0 to +2.4397 against matched legacy counterparts. Do not describe this engineering correction as a navigation performance improvement.

| Scene | Rotation: mean full-mission primary change | Repaired coverage: mean change |
|---|---:|---:|
| mild | 0.000020 | 0.000020 |
| moderate | 1.492862 | 1.492862 |
| severe | 0.926979 | 0.926979 |
| double_mild | 2.435234 | 2.258896 |

**The sequence predictor passes its frozen ranking gate, narrowly.** Across eight scene/seed snapshots and 56 matched true-simulator continuations, mean 32-step prefix-plus-nominal-tail selection regret is 0.319155 for rotation and coverage-first, versus 0.158784 for sequence selection. This gain comes entirely from double_mild seed86101 (about 1.283 hybrid cost units). Seed86100 retains a missed opportunity. The gate uses an estimator-ranking endpoint and cannot establish a full-mission scheduling benefit.

**Actual incurred costs show essentially no difference over 32 steps.** All hazards are still ahead. The positive differences are in the separately evaluated remaining-route tail, at nominal 1m/s. The sequence predictor uses frozen graph fields and omits actual receiver dynamics and sensing; its branch-exit timing is 2–6 steps early here. All 24 selected-order branch directions match the true continuations, but this does not establish accurate benefit magnitudes.

**Packet timing is visible.** In the double-hazard case the carrier first enters the direct branch at actual step3, before the second packet at step8. In moderate/severe cases useful first packets change the outgoing branch and actual exit occurs at step10. A fixed pool/sequence is a conditional experiment; it is not a forecast of new future scout measurements or continuously refreshed sender models.

Standalone compressed sequence advertisements are about 1.95kB, with actual payload/acknowledgment bytes logged. These are fresh diagnostic-window budgets, not remaining archived mission allowances. This is fixed-rate content ordering, not adaptive-rate optimization. No live sequence-selector navigation campaign was run.

**Correction to v1:** its hand-written diagnostic truth helper omitted background risk 0.02. The original artifacts remain, with repriced planned-route endpoints in v1_endpoint_correction.json. The two old harmful cases remain harmful; the simulator and actual mission-cost logger were already correct.

Verification independently checks all full-episode flow/cost components and all 56 continuation endpoints/bytes/receipts. New-mode frozen/execution controls match on 32 steps, all eight multistep finite-difference checks pass, and receiver summaries preserve the integration convention. See verification.json and the full test log.

**Decision:** keep both new modes opt-in. Integration consistency is established. Sequence ranking has one small development improvement with an idealized tail metric, higher computation, and unresolved timing/calibration limitations. A separately frozen full-mission sequence-selector comparison is required before making a scheduling advantage claim. No controller retuning, scene search, held-out runs, or claim of universal superiority is justified.

[Report](REPORT.md) · [Summary](summary.json) · [Animation](figures/double_mild_86100.gif) · [Plots](figures/results.png)
