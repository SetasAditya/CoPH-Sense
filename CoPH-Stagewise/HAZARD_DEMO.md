# Share the environmental field, not the decision

[Pilot report](results/hazard_pilot_v2/REPORT.md) · [Moderate-hazard animation](results/hazard_pilot_v2/moderate/figures/decision_vs_field.gif) · paper figures `paper_figures/output/fig7_terrain.png` and `fig8_belief.png`

> In four heterogeneous-agent development scenarios, sharing a learned graded environmental field allowed the receiver to apply its own risk preferences, achieving lower worst-case excess cost than the tested binary summaries and sender-specific advice. Raw measurements matched its navigation outcomes with higher transmitted bytes. Receiver field assimilation provided no additional navigation benefit in this pilot.

The demonstrated causal chain is

scout observations → learned field r̂(q) → packet → carrier route scores w_carrier·r̂(q) → different behavior.

The scout transmits what it learned without imposing its own preferences on the carrier. This pilot does not establish the stronger chain of receiver reconstruction → historical replay → improved learned policy.

## Setup

- **Agents and route.** A risk-tolerant scout (risk weight 0.25) and a risk-averse carrier (risk weight 6) both travel from the west end of the crossing network's middle row to its east end. The carrier starts 25 s later.
- **Hazards.** The middle row holds traversable Gaussian risk bumps spanning the corridor.
- **Cost.** Each agent's mission cost and planner price risk with its own weight: per step, `dt·(1 + 4·w·risk)`.
- **Settings.** One noise seed on four designed scenes. The unknown-terrain prior is 0.02 (the background risk), declared in each manifest; with the default prior of 1, unvisited corridors look costlier than known hazards. Agents are independent plants with no inter-agent contact dynamics.

All arms use the same fixed-rate reciprocal channel, packet validation and byte accounting (`coph_stagewise/sharing.py`). Only the environmental content of the anchors differs:

| Arm | Content |
|---|---|
| field | the sender's learned graded environmental field: values, gradients and bootstrap covariance at 4 anchors |
| raw | the sender's raw sensor readings at 12 of its most hazardous supported points; no model, no gradients |
| no_learning | field packets reach the route store but never enter the receiver's own field fit |
| occupancy τ | the field's anchor points, values quantized to blocked/free at τ ∈ {0.1, 0.25, 1} |
| decision | occupancy at the sender's own avoidance threshold, 1/(4·w_sender) = 1 |

**Export.** All learners use ledger rotation: each packet summarizes one retained stage in turn, newest first. This is one implementation of ledger export. The formulation's Algorithm B encodes the committed ledger but prescribes no rotation schedule.

The first pilot (`results/hazard_pilot`, preserved) exported only the newest stage. With two patches, the carrier heard about only one, and the field arm's excess cost in that scene was 14.9. The default exporter is unchanged for every other experiment.

## Pilot result

The table gives the carrier's **excess cost relative to the best tested arm in each scene**, measured, with the frozen oracle value in brackets. This is not regret against a globally optimal policy. All 64 agent missions reached their goals with no collisions.

| Scene | none | field | raw | no_learning | occ. 0.1 | occ. 0.25 | occ. 1 | decision |
|---|---|---|---|---|---|---|---|---|
| one mild bump | 0 (0) | 0 (0) | 0 | 0 | 12.3 (16.5) | 0 | 0 | 0 |
| two mild bumps | 14.9 (13.5) | 4.6 (0) | 4.6 | 4.6 | 0 (0) | 14.9 | 14.9 | 14.9 |
| moderate | 25.3 (25.5) | 0 (0) | 0 | 0 | 0 | 0 | 25.3 | 25.3 |
| severe | 68.1 (42.4) | 0 (0) | 0 | 0 | 0 | 0 | 0 | 0 |
| **worst-case excess cost** | 68.1 | **4.6** | **4.6** | **4.6** | 12.3 | 14.9 | 25.3 | 25.3 |

- **Lowest worst-case excess cost.** The learned field ties raw readings and no-receiver-fitting at 4.6.
- **All tested binary schemes do worse.** The three occupancy thresholds and the scout's threshold advice each have one or two scenes costing 12–25 units more than the best tested arm. Recipient-aware binary advice and richer quantization were not tested.
- **The field's 4.6 in the two-bump scene is a timing cost.** Rotation delivers the second stage summary about 2 s after the first, when the carrier is already 0.8 m into the row.
- **Bytes.** Raw readings match the field's navigation outcomes, with the scout sending 176 kB against 149 kB (18% more). Raw packets carry 12 anchors against 4, so this difference describes these two codecs, not an intrinsic compression advantage of a learned summary.
- **No receiver-learning benefit.** The no_learning arm matches the field arm. In this relay the benefit reaches the receiver through route scoring of the shared field. This agrees with the separate direct-field-routing experiment ([FIELD_ROUTE_RESULTS.md](FIELD_ROUTE_RESULTS.md)).
- **Wording.** The comparisons don't show that transmitting H, q, p, Hamiltonian structure or gradients specifically causes the advantage, so "graded environmental field" is the accurate term.
- **Oracle agreement is qualitative, not exact.** The mismatches are 12.3 vs 16.5, 14.9 vs 13.5, 25.3 vs 25.5, 4.6 vs 0 and 68.1 vs 42.4. The oracle idealizes constant speed and immediate full transfer.

## Evidence against the project objectives

| Objective | Evidence from this pilot |
|---|---|
| Ahead agent provides useful information to follower | Demonstrated |
| Information remains useful under different agent preferences | Demonstrated for the tested weights and scenes |
| Learned summary reduces communication relative to raw data | Modest measured reduction for these packet formats |
| Receiver learning adds value beyond using received information | Not demonstrated |
| Backward stage correction improves decisions | Not demonstrated |
| Follower helps leader through fresh or complementary evidence | Not established |
| Adaptive rates, changing environments, generalization | Separate work |

## Reproduce

```bash
python -m coph_stagewise.hazard_demo --table                        # oracle predictions only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m coph_stagewise.hazard_demo --out results/hazard_reproduction --workers 32
python -m coph_stagewise.hazard_demo --out results/hazard_reproduction --render
python -m paper_figures.make_all --terrain results/hazard_reproduction
```
