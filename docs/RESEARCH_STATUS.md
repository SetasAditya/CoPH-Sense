# CoPH-Sense: formulation and research status

**Status: 22 September 2026.** The causal sensing/sharing/memory mechanism and material-aware pH integration have been demonstrated in restricted tasks. A complete learned A3→NEED→A4→A5→pH policy has **not** passed the acquisition gate, and no confirmatory unknown-terrain evaluation has been run.

## Problem Prof. Bajaj posed

Two agents move through uncertain terrain with partially overlapping sensing footprints and different progress along a route. One may look ahead, but its information only helps the other if it is **new to that recipient**, delivered before a decision remains physically actionable, and worth the sensing, travel, and radio cost. A follower should use earlier evidence to explore a *different unresolved area*, rather than repeat the leader's sensing. Either agent may become the informative one. The physical execution layer remains material-aware and port-Hamiltonian (pH); information changes the local risk belief that enters its Hamiltonian.

This is distinct from claiming a new pH plant-controller interface. The proposed learning contribution is **decision-valued cooperative acquisition and selective sharing**, with a pH/MHRF realization for the terrain mission.

## Formulation

Let the latent terrain map be `M=(G,R)`, with public static geometry `G` in the current E2 generator and hidden surface/traction state `R`. Agent `i` has legal local history `h_t^i`: its motion, observations, acquired readings, delivered messages, ACKs, public teammate trajectory, and clock. The policy never reads unobserved terrain or a teammate's private state.

At a decision point, a local acquisition policy selects a feasible set `D_i`, including the empty set. Sensing physically executes and produces evidence `e_i`. A receiver may first transmit a charged NEED summary `q_j` (region, modality, uncertainty/relevance, and useful deadline). The sender then selects HOLD or a compact innovation message. A4's SEND value must include the receiver's *subsequent acquisition response*, not just its immediate route response. Delivered evidence persists in the receiver's memory and updates its local belief `b_i`; A5 then chooses the next sensing set from that new state.

The relevant conditional values are

```text
Q_sense(h,D) = C_sense(h,D)
             + E[downstream team cost | legal history h, execute D,
                 frozen sharing/continuation policy]

Delta_send(h_i,q_j,m) = E[J_HOLD - J_SEND | h_i,q_j,m]
```

The expectation is over *compatible hidden worlds and continuation randomness*. A positive realized-world difference alone does not mean SEND was rational from the sender's information. The team minimizes expected mission cost, with success reported separately:

```text
J_team = E[C_travel + lambda_R C_risk + C_sensing
           + C_communication + C_delay + C_failure].
```

Risk-tail `CVaR_0.95(C_risk)` is a secondary metric. Costs charge both agents, including scout detours, carrier delay, sensing dwell, packets, ACKs, and the executed mission. A finite deadline, resource caps, and delivery timing constrain feasible actions. A message is usable for a maneuver only if it arrives **before the latest physically feasible route switch**, which may precede formal route commitment.

The deployed material-pH executor receives only `b_i` and public geometry. Its schematic form is

```text
H_i(q_i,p_i;b_i) = 1/2 p_i^T M_i^{-1} p_i
                  + V_goal(q_i) + V_obstacle(q_i)
                  + V_material(q_i;b_i)

(q_dot_i, p_dot_i) = pH dynamics + damping + bounded input.
```

The included adapter uses the frozen learned material coefficients, belief-derived risk patches, the original force/integration source, and the controller-only calibration. True traction affects the plant, not the actor's force proposal. This implementation has **not** established a general passivity or safety certificate for arbitrary belief, graph, or communication changes.

## Why the work was broken into gates

The first finite oracle checks separated **what to sense (A3), what realized evidence to share (A4), what to sense after receiving memory (A5), and whether information can arrive while actionable (A6)**. These were deliberately small enough to enumerate and debug legal information access. A moving VMAS bridge then checked whether those decisions still mattered after physical viewpoint motion, packet delivery, and route timing. The material-pH campaign replaced the scripted executor with the intended physical backbone. This progression preserves the professor's causal chain; the finite tasks are diagnostics, not substitutes for the final terrain experiment.

| Gate | What is established | Limit |
| --- | --- | --- |
| Finite A2–A6 | Exact restricted acquisition/sharing references; learned set acquisition, selective sharing, persistent memory, and timing behavior in finite cases. | Small enumerated worlds and restricted protocols. |
| Moving two-fork A5 | With scout evidence retained, the carrier changes its physical sensing choice from R1 to R2; canonical paired cost **2.116 vs 3.089** with memory reset. [GIF](../CoPh-sense/coph_fork/results/a5_moving_bridge/physical_search/frozen_two_fork_information.gif). | Scripted VMAS bridge, one canonical paired replay. |
| pH integration | Belief updates change the material-aware pH force proposal; a limited scripted 16-world ablation showed lower cost from CoPH information under both direct and split integration. | Direct and split used the same pH vector field, so this is **not** pH-vs-non-pH evidence. |
| Natural look-ahead A4 | Compact innovation can change receiver sensing, but a legal-history audit found no reproducibly positive conditional SEND context in that task; 43% of individual worlds in the closest context benefited in hindsight while its conditional mean stayed negative. | HOLD is supported for that specific information structure; this is not a universal communication result. |
| Bidirectional NEED A4 | A charged receiver NEED packet makes positive and negative conditional SEND contexts identifiable in both directions. Learned history+NEED A4: **0.001419** mean test regret on 384 controlled cases; request-match: **0.001736**. [Frozen result](../CoPh-sense/coph_terrain/results/lookahead_bidirectional_a4/learning_frozen_v1.json). | The controlled requests are simple; learned A4 nearly matches a strong heuristic. It is not yet an end-to-end E2 gain. |
| Material-pH Gate 1.1 | Seed-disjoint Natural/Complementarity/Moving-A5 teacher mixture was run under the calibrated executor. Validation exact rate **90%**, mean realized regret **0.0203418**. | The strict regret gate `<0.02` failed; validation had **0 pair-optimal** and **0 memory-switch** cases. |
| A3/A5 micro-overfit | Same critic architecture fits the two-state R1→R2 memory switch exactly. | It fits only **25/32** full training decisions after 3,000 steps (mean regret **0.34094**); no deployable A3/A5 checkpoint exists. |
| Conditional-scout geometry | Attributed frozen benchmark: **97.65%** analytic decision accuracy, **0.002680** analytic regret, and 48/48 successful collision-free rollouts with recovery. | Physical execution prefers IDLE in all three analytically useful cases. It is a kinematic diagnostic, not material-pH evidence. |
| Material-pH dispatch integration | Carrier-side legal-history record; complete finite terrain prior with clipped-Gaussian appearance conditioning and causal replay; paired complete continuations; recovery/packet ledger; exact delivery intervention; and fail-closed gate are implemented. The frozen admission split contained **1 useful / 13 unnecessary / 10 infeasible** tasks. | The restricted gate failed. Test regret **0.01907** met the scalar threshold, but the learner selected IDLE in all 11 states: **72.7%** agreement and **0/3** useful recall. Causal, fully charged dispatch, recovery, and known-canonical subgates therefore did not pass. |

The decisive current record is [`CAMPAIGN_LOG.md`](../CoPh-sense/coph_terrain/results/material_campaign/CAMPAIGN_LOG.md), with the [Gate 1.1 report](../CoPh-sense/coph_terrain/results/material_campaign/gate11/gate11_report.json) and [micro-overfit report](../CoPh-sense/coph_terrain/results/material_campaign/gate11/micro_overfit_full_report.json). The included failed candidate checkpoint is retained for diagnosis only.

The conditional-scout formulation and gate are recorded separately in
[`CONDITIONAL_SCOUT_STATUS.md`](../CoPh-sense/coph_terrain/CONDITIONAL_SCOUT_STATUS.md),
with the complete geometric result under
[`conditional_scout_geometric_v1`](../CoPh-sense/coph_terrain/results/conditional_scout_geometric_v1/)
and the corrected finite-prior result under
[`conditional_scout_material_finite_v2_admission`](../CoPh-sense/coph_terrain/results/conditional_scout_material_finite_v2_admission/).

## Current roadblock and next decision

The full pipeline is present in code, but neither the A3/A5 learner nor the
conditional-dispatch learner has earned the right to drive a confirmatory
rollout. The repaired dispatch teacher now finds legally supported decisions
of both signs and avoids the old map-splicing error. The small frozen training
set nevertheless contains very few useful dispatches, and the fitted critic
learns the safe IDLE policy. Its low mean regret hides complete failure on the
three useful held-out decisions. The complete acceptance gate correctly turns
that into a failed result rather than a claimed success.

Two specific training-target questions remain: (1) Natural and Moving-A5 labels currently use a **single realized-world** continuation where deployment requires a belief-conditional value; (2) the critic fits absolute mission values and risk quantiles while action selection uses the mission-plus-CVaR ordering. These are hypotheses about the failure, not established causes.

The next controlled decision is whether to expand the predeclared finite-prior
training coverage while leaving the teacher, physics, prices, layouts, and
untouched evaluation fixed, or to close this restricted dispatch route as a
negative result. Any retry must be justified from training coverage alone and
must retain the current test manifest. Only a policy that passes useful recall,
causality, recovery, and fully charged cost can proceed to full-stack and E2
evaluation. No final-evaluation result is claimed.
