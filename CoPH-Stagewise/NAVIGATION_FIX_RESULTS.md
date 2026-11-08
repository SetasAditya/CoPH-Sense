# Navigation fix: implementation and evaluation results

The geometry and collision-protection mechanisms work in the tested cases. The complete routed controller is implemented, but **its navigation performance remains unresolved**: it often stops or progresses too slowly despite having a route. It remains opt-in; legacy defaults are unchanged.

All 62 numerical/contract tests passed. The final frozen suite completed **64/64 smoke and 320/320 development episodes**. Two additional controlled missions with learning and reciprocal communication completed all four agent goals without collisions. No validation/test seed namespace was executed; smoke repeats the first two development seeds.

## Controller comparison

Each row has 32 development episodes (four unchanged scenes × eight seeds), or 64 independent-agent missions. Costs are mean **team mission cost**, with the original time, risk, effort and failure terms. Legacy rows are the preserved frozen references.

| Controller | Communication | Goals / 64 | Collisions | Timeouts | Mean cost | Mean team bytes |
|---|---|---:|---:|---:|---:|---:|
| legacy | disabled | 2 | 61 | 1 | 91.017 | 0 |
| legacy | reciprocal_fixed | 2 | 61 | 1 | 91.016 | 36,548 |
| geometry | disabled | 3 | 59 | 2 | 92.355 | 0 |
| geometry | reciprocal_fixed | 3 | 59 | 2 | 92.372 | 36,550 |
| protected | disabled | 56 | 0 | 8 | 97.873 | 0 |
| protected | reciprocal_fixed | 56 | 0 | 8 | 97.780 | 716,637 |
| routed | disabled | 1 | 0 | 63 | 248.595 | 0 |
| routed | reciprocal_fixed | 1 | 0 | 63 | 248.013 | 2,250,855 |

Protection-only changes reciprocal completion from 2/64 to 56/64 and removes the observed collisions. Its eight timeouts remain failures. Mean cost increases from 91.016 to 97.780 under the unchanged objective: longer safe trajectories accumulate time/risk cost, while early collisions terminate with the fixed failure penalty. This is a completion and collision improvement, not a claim of uniformly lower mission cost.

Adding the specified waypoint potential performs substantially worse than protection-only: reciprocal routed completion is 1/64, with 63 timeouts. Zero collisions does not make these missions successful.

## Cooperation and replay

The following comparisons all use the same complete routed controller. Each arm has 32 episodes / 64 agent missions; all had zero collisions.

| Arm | Goals / 64 | Timeouts | Mean team cost |
|---|---:|---:|---:|
| disabled | 1 | 63 | 248.595 |
| one_way_01 | 1 | 63 | 247.853 |
| one_way_10 | 2 | 62 | 247.228 |
| evidence_only | 0 | 64 | 259.475 |
| reciprocal_fixed | 1 | 63 | 248.013 |
| short_replay | 1 | 63 | 248.015 |

Reciprocal communication changes mean cost by −0.093 for protection-only and −0.582 for routed, relative to each controller’s paired communication-off runs. Goal counts do not improve in either comparison. Cost decreases in 20/32 protection-only pairs and 18/32 routed pairs, and increases in the remaining pairs. These small, mixed development effects do not establish a robust cooperation benefit. Communication has the same byte ceilings, with actual bytes reported rather than equal traffic claimed.

With routed control, evidence-only costs 259.475 versus 248.013 for historical replay, but completion remains poor. Fixed eight-step replay and boundary-reaching replay differ by only about 0.002 in mean cost (248.015 versus 248.013), with identical aggregate goal counts. This experiment does not show a meaningful boundary-horizon benefit.

## Verified failure mechanism

For blocked passage, seed 61000, reciprocal routed agent 0 ends over seven units from its goal. Its logged waypoint attraction is `(0.868, 0.296)`, learned risk force `(-0.829, -0.311)`, and barrier force `(-0.036, 0.016)`. Including velocity feedback gives nominal force `(0.00159, 0.00079)`. The applied control equals nominal; it never brakes in this episode. The reconstructed force exactly matches the logged force. This verifies attraction–risk cancellation in this case; it is not evidence that every timeout has the same cause.

[Force-balance audit](results/navigation_fix_v3/smoke/force_balance_audit.json). The next navigation change needs to address progress under the scalar route potential. No gains, penalties, horizons, sensing rules, scene layouts or communication costs were tuned after this finding.

## Computation, accounting and validation

- Maximum observed local-update work: **891 / 1,024** flow evaluations, including pilots, guards, gradients, validation and diagnostic controls. Routing expansions and projection candidates are recorded separately.
- Maximum physical energy-accounting residual: **8.66e−15**. Route/model/context jumps and safety-modification work remain separate from physical work.
- Zero operational infeasible-safety events in these development runs. The deliberately infeasible late-discovery unit test records the event correctly; this is not a guarantee for unseen obstacles or out-of-range dynamics.
- Reciprocal protection-only: 6,593 interventions, 245 braking steps, stage IDs through 9, and 3,031 logged replay handoffs across all updates/recomputations. Reciprocal routed: 467 interventions, 104 braking steps, 3,039 route replans, stage IDs through 6, and 312 logged replay handoffs.
- Mean development episode wall time with communication off: 2.38 s geometry, 47.43 s protection, 114.09 s routed; reciprocal: 2.54 s, 50.57 s, 111.89 s respectively. These are observed parallel-run wall times, not normalized CPU benchmarks.
- Development totals: 3,663,637 update flow evaluations and 3,252,924 guard evaluations across execution and hypothetical controls (overlapping accounting categories; do not add them). Sum of episode wall times is 22,285.64 s across 32 workers; root completion.json separately records elapsed suite time, including rendering.
- All 16 rerun legacy smoke cases reproduce the original trajectories and mission costs exactly. Sensor footprints certify geometric coverage only; environmental support is unchanged.
- A routed receiver-side evaluation executed 40 hypothetical branch updates without changing the operational receiver’s serialized state. See [isolation audit](results/navigation_fix_v3/controlled/receiver_isolation.json).

All **272 failed development episodes / 516 failed agent missions** remain in the new failure gallery, including safe timeouts. Preliminary v1 (56 complete smoke runs, eight execution/logging errors) and v2 (51 collected, 13 interrupted before collection) are preserved and excluded from final comparisons because they used superseded source versions. v1 exposed constant-policy braking gradients and Boolean log serialization; v2 was stopped before development to complete route-context contracts. The final v3 includes every planned comparison, with no incomplete runs.

## Artifacts and reproduction

- [Implementation, equations and commands](NAVIGATION_FIX.md)
- [Frozen source/configuration manifest](results/navigation_fix_v3/manifest.json) · [source archive](results/navigation_fix_v3/sources.zip) · [verification](results/navigation_fix_v3/verification.json)
- [Supplementary renderer fingerprint](results/navigation_fix_v3/rendering_manifest.json) · [rendering script](scripts/render_navigation_ablations.sh)
- [Full development report](results/navigation_fix_v3/development/REPORT.md) · [machine-readable metrics](results/navigation_fix_v3/development/metrics.csv) · [paired differences](results/navigation_fix_v3/development/paired.json)
- [Every failed development episode](results/navigation_fix_v3/development/figures/failures/index.html) · [preserved legacy results](results/complex_fixed_v1/development/REPORT.md)
- [Controlled disk GIF](results/navigation_fix_v3/controlled/figures/controlled_disk.gif) · [controlled passage GIF](results/navigation_fix_v3/controlled/figures/controlled_passage.gif)

Four-controller GIFs below use the preselected seed 61000, reciprocal fixed-rate communication, common scales and synchronized timestamps. They show legacy, geometry repair, protection-only and complete routing; finished agents retain their terminal outcomes. Legacy routing/guard data are explicitly unavailable. The full development report also contains communication-off versus reciprocal routed animations.

- [hazard slalom GIF](results/navigation_fix_v3/development/figures/hazard_slalom_controllers.gif)
- [blocked risk passage GIF](results/navigation_fix_v3/development/figures/blocked_risk_passage_controllers.gif)
- [asymmetric coverage GIF](results/navigation_fix_v3/development/figures/asymmetric_coverage_controllers.gif)
- [long obstacle corridor GIF](results/navigation_fix_v3/development/figures/long_obstacle_corridor_controllers.gif)
