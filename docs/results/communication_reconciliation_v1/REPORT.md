# Communication evaluation reconciliation

Available corrected episodes: 72/72. Goals: 144/144; collisions: 0; timeouts: 0. Complete: True.

**Endpoints.** The primary endpoint is time plus receiver-weighted risk. Full logged cost additionally includes effort and failure. Earlier bandwidth accounting omitted the declared agent risk weights; corrected accounting changes reported risk costs while operational parity is checked separately. Team bytes sum both endpoints; each agent has its own ceiling. Status, payload and acknowledgments all count.

**Channel and learner distinction.** Original hazard reproductions use the original uncompressed channel, query-specific receiver statuses and full policy replay. The newer compact channel uses evidence-only learning and route advertisements for receiver-relevant selection. Queue/update details and realized trajectories can differ. Original-versus-new differences cannot be attributed solely to effort or to the accounting correction. Only the paired legacy-versus-corrected accounting runs certify unchanged operations.

Original reproduction checks: {'complete': True, 'passed': True}. Accounting operational parity: {'complete': True, 'passed': True}. Inspect regression.json/parity.json for numerical, discrete and serialized comparisons; unavailable checks remain unavailable.

| Scene | Seed | Selector | Ceiling / agent | Carrier time+risk | Carrier FULL | Carrier effort | Team actual bytes | Goals |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| double_mild | 86100 | receiver_relevant | 8,000 | 97.8178 | 97.9075 | 0.0897 | 12,673 | 2 |
| double_mild | 86100 | rotation | 8,000 | 97.8176 | 97.9073 | 0.0897 | 14,213 | 2 |
| double_mild | 86100 | receiver_relevant | 16,000 | 97.8176 | 97.9073 | 0.0897 | 24,993 | 2 |
| double_mild | 86100 | rotation | 16,000 | 87.5279 | 87.7024 | 0.1745 | 28,608 | 2 |
| double_mild | 86100 | receiver_relevant | 32,000 | 97.8176 | 97.9073 | 0.0897 | 56,815 | 2 |
| double_mild | 86100 | rotation | 32,000 | 87.5279 | 87.7024 | 0.1745 | 57,866 | 2 |
| double_mild | 86100 | disabled | 4,000,000 | 97.8178 | 97.9075 | 0.0897 | 0 | 2 |
| double_mild | 86100 | receiver_relevant | 4,000,000 | 97.8176 | 97.9073 | 0.0897 | 227,873 | 2 |
| double_mild | 86100 | rotation | 4,000,000 | 87.5279 | 87.7024 | 0.1745 | 79,717 | 2 |
| double_mild | 86101 | receiver_relevant | 8,000 | 97.8186 | 97.9085 | 0.0899 | 12,806 | 2 |
| double_mild | 86101 | rotation | 8,000 | 97.8186 | 97.9085 | 0.0899 | 14,381 | 2 |
| double_mild | 86101 | receiver_relevant | 16,000 | 97.8186 | 97.9085 | 0.0899 | 25,043 | 2 |
| double_mild | 86101 | rotation | 16,000 | 87.5237 | 87.6987 | 0.1750 | 28,758 | 2 |
| double_mild | 86101 | receiver_relevant | 32,000 | 97.8186 | 97.9085 | 0.0899 | 56,955 | 2 |
| double_mild | 86101 | rotation | 32,000 | 87.5237 | 87.6987 | 0.1750 | 57,989 | 2 |
| double_mild | 86101 | disabled | 4,000,000 | 97.8186 | 97.9085 | 0.0899 | 0 | 2 |
| double_mild | 86101 | receiver_relevant | 4,000,000 | 97.8186 | 97.9085 | 0.0899 | 228,303 | 2 |
| double_mild | 86101 | rotation | 4,000,000 | 87.5237 | 87.6987 | 0.1750 | 79,869 | 2 |
| mild | 86100 | receiver_relevant | 8,000 | 70.3779 | 70.4676 | 0.0897 | 12,742 | 2 |
| mild | 86100 | rotation | 8,000 | 70.3779 | 70.4676 | 0.0897 | 14,331 | 2 |
| mild | 86100 | receiver_relevant | 16,000 | 70.3779 | 70.4676 | 0.0897 | 25,088 | 2 |
| mild | 86100 | rotation | 16,000 | 70.3779 | 70.4676 | 0.0897 | 28,075 | 2 |
| mild | 86100 | receiver_relevant | 32,000 | 70.3779 | 70.4676 | 0.0897 | 56,872 | 2 |
| mild | 86100 | rotation | 32,000 | 70.3779 | 70.4676 | 0.0897 | 60,814 | 2 |
| mild | 86100 | disabled | 4,000,000 | 70.3779 | 70.4676 | 0.0897 | 0 | 2 |
| mild | 86100 | receiver_relevant | 4,000,000 | 70.3779 | 70.4676 | 0.0897 | 228,139 | 2 |
| mild | 86100 | rotation | 4,000,000 | 70.3779 | 70.4676 | 0.0897 | 78,624 | 2 |
| mild | 86101 | receiver_relevant | 8,000 | 70.3778 | 70.4678 | 0.0899 | 12,632 | 2 |
| mild | 86101 | rotation | 8,000 | 70.3778 | 70.4678 | 0.0899 | 14,189 | 2 |
| mild | 86101 | receiver_relevant | 16,000 | 70.3778 | 70.4678 | 0.0899 | 25,006 | 2 |
| mild | 86101 | rotation | 16,000 | 70.3778 | 70.4678 | 0.0899 | 27,890 | 2 |
| mild | 86101 | receiver_relevant | 32,000 | 70.3778 | 70.4678 | 0.0899 | 56,779 | 2 |
| mild | 86101 | rotation | 32,000 | 70.3778 | 70.4678 | 0.0899 | 60,653 | 2 |
| mild | 86101 | disabled | 4,000,000 | 70.3778 | 70.4678 | 0.0899 | 0 | 2 |
| mild | 86101 | receiver_relevant | 4,000,000 | 70.3778 | 70.4678 | 0.0899 | 228,158 | 2 |
| mild | 86101 | rotation | 4,000,000 | 70.3778 | 70.4678 | 0.0899 | 78,521 | 2 |
| moderate | 86100 | receiver_relevant | 8,000 | 108.2801 | 108.3699 | 0.0897 | 12,750 | 2 |
| moderate | 86100 | rotation | 8,000 | 82.9867 | 83.1412 | 0.1545 | 14,341 | 2 |
| moderate | 86100 | receiver_relevant | 16,000 | 82.9867 | 83.1412 | 0.1545 | 31,913 | 2 |
| moderate | 86100 | rotation | 16,000 | 82.9867 | 83.1412 | 0.1545 | 28,910 | 2 |
| moderate | 86100 | receiver_relevant | 32,000 | 82.9867 | 83.1412 | 0.1545 | 55,329 | 2 |
| moderate | 86100 | rotation | 32,000 | 82.9867 | 83.1412 | 0.1545 | 57,844 | 2 |
| moderate | 86100 | disabled | 4,000,000 | 108.2801 | 108.3699 | 0.0897 | 0 | 2 |
| moderate | 86100 | receiver_relevant | 4,000,000 | 82.9867 | 83.1412 | 0.1545 | 222,795 | 2 |
| moderate | 86100 | rotation | 4,000,000 | 82.9867 | 83.1412 | 0.1545 | 80,456 | 2 |
| moderate | 86101 | receiver_relevant | 8,000 | 108.2814 | 108.3713 | 0.0899 | 12,657 | 2 |
| moderate | 86101 | rotation | 8,000 | 82.9867 | 83.1415 | 0.1548 | 14,159 | 2 |
| moderate | 86101 | receiver_relevant | 16,000 | 82.9867 | 83.1415 | 0.1548 | 31,822 | 2 |
| moderate | 86101 | rotation | 16,000 | 82.9867 | 83.1415 | 0.1548 | 28,799 | 2 |
| moderate | 86101 | receiver_relevant | 32,000 | 82.9867 | 83.1415 | 0.1548 | 55,245 | 2 |
| moderate | 86101 | rotation | 32,000 | 82.9867 | 83.1415 | 0.1548 | 57,738 | 2 |
| moderate | 86101 | disabled | 4,000,000 | 108.2814 | 108.3713 | 0.0899 | 0 | 2 |
| moderate | 86101 | receiver_relevant | 4,000,000 | 82.9867 | 83.1415 | 0.1548 | 222,383 | 2 |
| moderate | 86101 | rotation | 4,000,000 | 82.9867 | 83.1415 | 0.1548 | 80,303 | 2 |
| severe | 86100 | receiver_relevant | 8,000 | 151.9039 | 152.1530 | 0.2492 | 12,775 | 2 |
| severe | 86100 | rotation | 8,000 | 83.7916 | 83.9461 | 0.1545 | 14,288 | 2 |
| severe | 86100 | receiver_relevant | 16,000 | 83.7916 | 83.9461 | 0.1545 | 31,898 | 2 |
| severe | 86100 | rotation | 16,000 | 83.7916 | 83.9461 | 0.1545 | 28,840 | 2 |
| severe | 86100 | receiver_relevant | 32,000 | 83.7916 | 83.9461 | 0.1545 | 55,289 | 2 |
| severe | 86100 | rotation | 32,000 | 83.7916 | 83.9461 | 0.1545 | 57,625 | 2 |
| severe | 86100 | disabled | 4,000,000 | 151.9039 | 152.1530 | 0.2492 | 0 | 2 |
| severe | 86100 | receiver_relevant | 4,000,000 | 83.7916 | 83.9461 | 0.1545 | 222,734 | 2 |
| severe | 86100 | rotation | 4,000,000 | 83.7916 | 83.9461 | 0.1545 | 80,212 | 2 |
| severe | 86101 | receiver_relevant | 8,000 | 151.9081 | 152.1575 | 0.2493 | 12,619 | 2 |
| severe | 86101 | rotation | 8,000 | 83.7916 | 83.9464 | 0.1548 | 14,144 | 2 |
| severe | 86101 | receiver_relevant | 16,000 | 83.7916 | 83.9464 | 0.1548 | 31,747 | 2 |
| severe | 86101 | rotation | 16,000 | 83.7916 | 83.9464 | 0.1548 | 28,726 | 2 |
| severe | 86101 | receiver_relevant | 32,000 | 83.7916 | 83.9464 | 0.1548 | 55,128 | 2 |
| severe | 86101 | rotation | 32,000 | 83.7916 | 83.9464 | 0.1548 | 57,528 | 2 |
| severe | 86101 | disabled | 4,000,000 | 151.9081 | 152.1575 | 0.2493 | 0 | 2 |
| severe | 86101 | receiver_relevant | 4,000,000 | 83.7916 | 83.9464 | 0.1548 | 222,182 | 2 |
| severe | 86101 | rotation | 4,000,000 | 83.7916 | 83.9464 | 0.1548 | 80,113 | 2 |

## Original-channel endpoint reproduction

| Scene | Original arm | Time+risk | FULL | Effort |
|---|---|---:|---:|---:|
| mild | none | 70.3777 | 70.4675 | 0.0897 |
| mild | field | 70.3777 | 70.4675 | 0.0897 |
| moderate | none | 108.2837 | 108.3735 | 0.0897 |
| moderate | field | 82.9867 | 83.1412 | 0.1545 |
| severe | none | 151.8986 | 152.1478 | 0.2492 |
| severe | field | 83.7916 | 83.9461 | 0.1545 |
| double_mild | none | 97.8172 | 97.9069 | 0.0897 |
| double_mild | field | 87.5225 | 87.6971 | 0.1746 |

## New-channel communication benefit at 4 MB

Negative differences favor communication. The disabled reference is the same scene and seed; its cost is independent of a channel budget because communication is off. Means below summarize the two noise seeds descriptively.

| Scene | Rotation − off time+risk | Relevant − off time+risk | Rotation − off FULL | Relevant − off FULL |
|---|---:|---:|---:|---:|
| mild | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| moderate | -25.2941 | -25.2941 | -25.2292 | -25.2292 |
| severe | -68.1144 | -68.1144 | -68.2090 | -68.2090 |
| double_mild | -10.2924 | -0.0001 | -10.2075 | -0.0001 |

## Payload selection by budget

Relevant minus rotation uses the identical scene, seed, per-agent allowance and evidence-only learner. Negative favors receiver-relevant selection. This is separate from whether either communication arm beats disabled.

| Ceiling / agent | Paired episodes | Relevant − rotation time+risk | Relevant − rotation FULL | Relevant − rotation TEAM bytes |
|---|---:|---:|---:|---:|
| 8,000 | 8 | +23.3521 | +23.3596 | -1549.0000 |
| 16,000 | 8 | +2.5731 | +2.5518 | -137.0000 |
| 32,000 | 8 | +2.5731 | +2.5518 | -2455.6250 |
| 4,000,000 | 8 | +2.5731 | +2.5518 | +145594.0000 |

## Before/after evaluation accounting

The old bandwidth table reported full cost with evaluation risk weight one for both agents. The corrected table uses the declared scout/carrier weights. Only successful operational parity allows this difference to be attributed to accounting. The primary endpoint also omits effort, so its name is shown explicitly.

| Ceiling / agent | Selector | Old incorrectly weighted FULL | Corrected time+risk | Corrected FULL |
|---|---|---:|---:|---:|
| 8,000 | rotation | 52.5295 | 83.7436 | 83.8658 |
| 8,000 | receiver_relevant | 59.5040 | 107.0957 | 107.2254 |
| 16,000 | rotation | 56.7969 | 81.1705 | 81.3140 |
| 16,000 | receiver_relevant | 52.5295 | 83.7436 | 83.8658 |
| 32,000 | rotation | 56.7969 | 81.1705 | 81.3140 |
| 32,000 | receiver_relevant | 52.5295 | 83.7436 | 83.8658 |
| 4,000,000 | disabled | 59.5040 | 107.0957 | 107.2254 |
| 4,000,000 | rotation | 56.7969 | 81.1705 | 81.3140 |
| 4,000,000 | receiver_relevant | 52.5295 | 83.7436 | 83.8658 |

### Moderate and double-hazard communication check

These rows compare rotation with disabled within the new channel/learner setting at the same seed and four-million-byte ceiling. The old scalar is the incorrectly weighted full endpoint; the corrected primary and full endpoints are both shown. Operational parity is checked independently.

| Scene | Seed | Arm | Old incorrectly weighted FULL | Corrected time+risk | Corrected FULL |
|---|---:|---|---:|---:|---:|
| moderate | 86100 | disabled | 48.43643 | 108.28014 | 108.36988 |
| moderate | 86100 | rotation | 60.48562 | 82.98669 | 83.14120 |
| moderate | 86101 | disabled | 48.43681 | 108.28137 | 108.37129 |
| moderate | 86101 | rotation | 60.48592 | 82.98669 | 83.14150 |
| double_mild | 86100 | disabled | 46.79269 | 97.81776 | 97.90749 |
| double_mild | 86100 | rotation | 63.86246 | 87.52790 | 87.70237 |
| double_mild | 86101 | disabled | 46.79301 | 97.81860 | 97.90851 |
| double_mild | 86101 | rotation | 63.86232 | 87.52367 | 87.69872 |

## Timing, routes, and budget starvation

Patch timing maps received anchors, local readings and actual carrier positions to each true patch ellipse for evaluator diagnostics only. A missing local encounter may mean the route avoided that area; it is not an invented discovery time. Planned route changes are distinct from actual interior-edge entries. Backward x travel is recorded without calling every detour wrong.

| Scene | Selector | Ceiling | Patch | Carrier receives anchor | Locally senses | Actual entry |
|---|---|---:|---:|---:|---:|
| double_mild | disabled | 4000000 | 1 | None | 36.4 | 40.24 |
| double_mild | disabled | 4000000 | 2 | None | 45.52 | 49.24 |
| double_mild | receiver_relevant | 16000 | 1 | None | 36.4 | 40.24 |
| double_mild | receiver_relevant | 16000 | 2 | 25.96 | 45.52 | 49.24 |
| double_mild | receiver_relevant | 32000 | 1 | None | 36.4 | 40.24 |
| double_mild | receiver_relevant | 32000 | 2 | 25.96 | 45.52 | 49.24 |
| double_mild | receiver_relevant | 4000000 | 1 | None | 36.4 | 40.24 |
| double_mild | receiver_relevant | 4000000 | 2 | 25.96 | 45.52 | 49.24 |
| double_mild | receiver_relevant | 8000 | 1 | None | 36.4 | 40.24 |
| double_mild | receiver_relevant | 8000 | 2 | None | 45.52 | 49.24 |
| double_mild | rotation | 16000 | 1 | 27.88 | None | None |
| double_mild | rotation | 16000 | 2 | 25.96 | None | None |
| double_mild | rotation | 32000 | 1 | 27.88 | None | None |
| double_mild | rotation | 32000 | 2 | 25.96 | None | None |
| double_mild | rotation | 4000000 | 1 | 27.88 | None | None |
| double_mild | rotation | 4000000 | 2 | 25.96 | None | None |
| double_mild | rotation | 8000 | 1 | None | 36.4 | 40.24 |
| double_mild | rotation | 8000 | 2 | 25.96 | 45.52 | 49.24 |
| moderate | disabled | 4000000 | 1 | None | 40.96 | 44.8 |
| moderate | receiver_relevant | 16000 | 1 | 25.96 | None | None |
| moderate | receiver_relevant | 32000 | 1 | 25.96 | None | None |
| moderate | receiver_relevant | 4000000 | 1 | 25.96 | None | None |
| moderate | receiver_relevant | 8000 | 1 | None | 40.96 | 44.8 |
| moderate | rotation | 16000 | 1 | 25.96 | None | None |
| moderate | rotation | 32000 | 1 | 25.96 | None | None |
| moderate | rotation | 4000000 | 1 | 25.96 | None | None |
| moderate | rotation | 8000 | 1 | 25.96 | None | None |

Per-agent traffic records in episode_diagnostics.json separate offered-payload-direction refusals from charged-agent budget refusals. A status_budget refusal belongs to event.receiver, which advertises its route; payload_budget belongs to event.sender. The first_budget_refusal uses the charged agent, with first_offer_budget_refusal recorded separately. Last payload and any later payload after refusal are logged. No team-total allowance is substituted for per-agent budgets.

These four designed layouts use two sensor-noise seeds. Repeated budgets, quadrature points and noise seeds are not independent terrain trials. Results are descriptive; failures and incomplete/missing episodes remain in failure_inventory.json and unavailable_information.json. Original and new channels are not pooled as one controlled treatment.

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
