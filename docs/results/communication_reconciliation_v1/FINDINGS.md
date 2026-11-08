# Reconciliation findings

Original reproductions: {'complete': True, 'passed': True}. New-channel legacy/corrected accounting parity: {'complete': True, 'passed': True}. Corrected episodes: 72/72; 144/144 goals, 0 collisions, 0 timeouts. Numerical comparison tolerance is 1e-10; discrete decisions and serialized differences are reported separately in the regression records.

**Communication at 4 MB per agent.** Means across the two sensor-noise seeds; primary is time plus declared-weight risk. FULL additionally includes effort and failure.

| Scene | Off primary | Rotation primary | Relevant primary | Off FULL | Rotation FULL | Relevant FULL |
|---|---:|---:|---:|---:|---:|---:|
| mild | 70.3779 | 70.3779 | 70.3779 | 70.4677 | 70.4677 | 70.4677 |
| moderate | 108.2808 | 82.9867 | 82.9867 | 108.3706 | 83.1413 | 83.1413 |
| severe | 151.9060 | 83.7916 | 83.7916 | 152.1552 | 83.9463 | 83.9463 |
| double_mild | 97.8182 | 87.5258 | 97.8181 | 97.9080 | 87.7005 | 97.9079 |

**Was communication harmful?** The old bandwidth evaluation priced carrier risk with weight one, although its declared preference was six. For moderate and double-mild hazards this underpriced the risky direct route and made the detour look harmful. The table below shows matched rotation-minus-off differences; negative favors rotation. This attribution is valid only when the operational parity check passes.

| Scene | Old incorrect FULL difference | Corrected primary difference | Corrected FULL difference |
|---|---:|---:|---:|
| moderate | +12.04915 | -25.29406 | -25.22924 |
| double_mild | +17.06953 | -10.29239 | -10.20746 |

**The remaining selector limitation is real.** Receiver-relevant selection must be compared with rotation under the same corrected metric, independently of the above artifact. In the double-mild case inspect its primary cost, actual path and patch-arrival records; an absent first-patch summary is distinct from a late second-patch summary. At 4 MB, budget refusal cannot explain a case with no refusals. These observational timing records explain the information pattern without substituting an oracle success claim.
- Double-mild seed86100 rotation: first received anchor times for the two patch ellipses [27.88, 25.96]; actual interior edges [[1, 4], [1, 2], [2, 5], [4, 5]]; backward x travel 0.7472916021708214 m.
- Double-mild seed86100 receiver_relevant: first received anchor times for the two patch ellipses [None, 25.96]; actual interior edges [[1, 4]]; backward x travel 0.0 m.

Implementation limitation, separate from the measured timing: advertisements acknowledge sources in receiver.received, while accepted remote-route-only information can live solely in remote_terrain/remote_cells. Already useful route evidence may therefore remain absent from the acknowledgment set. This can misrepresent novelty. The double-mild logs repeatedly select the later stage while omitting the first patch; this is evidence of a payload-selection/interface limitation, not a controlled proof that acknowledgment alone causes it. No selection policy was changed during this reconciliation.

**At an 8,000-byte ceiling, analyze each payer separately.** Status advertisements count against the receiver advertising them. A sender may be blocked because its intended recipient cannot afford that status. This does not mean either agent exceeded its own allowance. Offered-direction and charged-agent refusals are separate in episode_diagnostics.json.
- rotation: 0/16 agents sent no payload; 0/16 spent status bytes while sending no payload. All own-budget checks pass: True.
- receiver_relevant: 8/16 agents sent no payload; 8/16 spent status bytes while sending no payload. All own-budget checks pass: True.

Concrete moderate/seed86100 chronology: the first reverse-direction send at t=25.0 charges [{'agent': 0, 'component': 'status', 'bytes': 6151}, {'agent': 1, 'component': 'payload', 'bytes': 844}]. At the first payload refusal t=25.92, charges are [{'agent': 1, 'component': 'status', 'bytes': 5616}]; per-agent remaining balances are [1710, 1540]. The reconstructed charge trace matches the final logged agent totals: True.
In the matched 16k run, the first scout packet at t=25.92 is 3,432 bytes (4/4 anchors near the advertised receiver route). In the 8k trace the scout already paid 6,151 status bytes and 139 acknowledgment bytes, leaving 1,710 bytes. The forward payload cannot fit even though no single status advertisement exceeds8k. Later status refusals keep that direction silent. This is reverse-status-before-forward-payload overhead, not a violation of either agent's budget.

**Readiness and limits.** Completion/collisions remain separate from cost improvements. Inter-agent contact dynamics are absent. Four designed terrains and two noise seeds do not establish generalization, and repeated budgets are not independent terrain samples. Original full-replay/uncompressed-channel reproductions and new evidence-only/compact-channel results are not pooled as one treatment.

**Tests.** The scoped pytest run reported 257 passed and 7 subtests passed with the existing archived-source-hash failure retained. The numerical CLI check retained the same known failure (225/226 passed). The initial unscoped pytest additionally discovered archived external-source tests with missing dependencies; its collection errors remain logged. No full-suite-all-passing claim is made.

See REPORT.md for all rows and separate endpoints; paired_rows.json for selector and off-reference differences; failure_inventory.json for failed/incomplete episodes; reporting_verification.json for report-source/input/log fingerprints; animation_inventory.json for synchronized rendering availability.

## Plots and synchronized animations

- [4 MB cost components](figures/cost_components_4000000.png)
- [16k cost components](figures/cost_components_16000.png)
- [Time/risk and budget comparison](figures/risk_time.png)
- [Status/payload/ack and per-agent ceilings](figures/bytes.png)
- [Moderate warning and budget timeline](figures/moderate_timing.png)
- [Double-hazard warning and budget timeline](figures/double_mild_timing.png)
- [mild three-arm GIF, seed86100, 4,000,000/agent](figures/mild_seed86100_budget4000000.gif)
- [moderate three-arm GIF, seed86100, 4,000,000/agent](figures/moderate_seed86100_budget4000000.gif)
- [severe three-arm GIF, seed86100, 4,000,000/agent](figures/severe_seed86100_budget4000000.gif)
- [double_mild three-arm GIF, seed86100, 4,000,000/agent](figures/double_mild_seed86100_budget4000000.gif)
