# Claim tests with pre-declared gates (2026-11-03)

Five tests, each with its gate written into the script docstring and the run manifest before execution. Earlier results are untouched.

| # | Claim | Result | Gate |
|---|---|---|---|
| 1 | Native controller holds on unused starts | 50/50 coverage in all three CoPH arms; online beats phMARL+same assignment+same protection by 112 return (CI 102–123), 50/50 wins | **Failed on one criterion**: one contact (seed 85048), shared by every protected arm |
| 2 | Historical replay improves navigation | Hold rule removes every `unavailable_stage` exit (current starts 8/24 → 24/24 valid; episodes 672 → 768/768). No cost change. The graph planner is already at or below the best forced branch, so headroom is ≤ 0 | **Failed**. Stop rule applies: report as a negative result |
| 3 | Field packets beat raw/costmap for decisions | Regret 0.048 for field vs 0.101 raw and 0.106 costmap; field better in 12/12 groups and in every offset stratum | **Passed** (saved state, 12 groups; ~1% of path cost) |
| 4 | Receiver-relevant selection helps under tight bandwidth | 8 kB: advertisements (11.8 kB) starve payloads and it loses on severe. 16 kB and above: it avoids a harmful double_mild detour, equal elsewhere. Pooled tight difference −1.35 (CI −9.9 to 5.5) | **Failed** |
| 5 | Smooth overlap is passive | Lemma: passivity between events with storage $K+V_M$; membership, revision and target jumps are recorded | Proof, see [STAGE_BLENDING_LEMMA.md](STAGE_BLENDING_LEMMA.md) |

## 1. Native held-out validation

Script: `benchmarks/phmarl/heldout_campaign.py`. Output: `results/native_heldout_v1/summary.json`.

- **Setup.** Seeds 85000–85049 had not been used by any experiment. All 50 pass the reset geometry and independent-assignment gate, so none are excluded. Controller, assignment, targets and protection are imported unchanged from `goal_geometry_campaign`.
- **CoPH.** Full coverage: frozen 50/50, known 50/50, online 50/50. phMARL variants: 31–33/50.
- **Mean return:**

  | Arm | Mean return |
  |---|---:|
  | online | −70.4 |
  | known | −58.5 |
  | frozen | −106.2 |
  | phMARL | −207.1 |
  | phMARL + assignment | −184.6 |
  | phMARL + assignment + static targets + protection | −182.8 |

- **Paired comparisons for online:**

  | Online vs | Mean difference | 95% CI | Wins |
  |---|---:|---|---:|
  | phMARL, fully matched | +112.4 | 102.4–122.8 | 50/50 |
  | frozen | +35.8 | 32.5–39.1 | 50/50 |
  | known | −11.9 | −15.1 to −9.2 | 1/50 |

- **The contact on seed 85048.**
  - The start has agents 0 and 2 overlapping at 0.237. Resolving that overlap pushes agents 1 and 2 to 0.287 by step 3.
  - It is identical in every protected arm, including phMARL protected. It is a shared protection-layer edge case: initial-overlap recovery does not guard third parties.
  - Do not fix it and rerun on these seeds. Any fix needs fresh seeds.

## 2. Stage-transition hold and historical replay

Code: `FrozenStages.transition_mode='hold'`, opt-in through `agent.stage_transition_mode`; the default is unchanged. Suite: `coph_stagewise/replay_hold_suite.py`. Output: `results/replay_hold_v1/`.

- **Setup.** Fresh noise seeds 86000–86007, three fork cases, five arms plus two forced-branch references. 168 episodes; all 336 missions succeed with zero collisions.
- **The repair works for current starts.** No `unavailable_stage` exit remains.
- **Historical starts remain mostly invalid** (145/768). The remaining reason is `route_prefix_safety_or_route`: route candidates are conditioned on the current position, so replaying them from old entries is ill-posed.
- **Headroom:**

  | Case | Graph | Best single forced branch |
  |---|---:|---:|
  | upper | 42.39 | 42.62 |
  | lower | 45.82 | 45.82 |
  | clear | 36.98 | 37.34 |

  The graph planner is already optimal on these scenes, so no route-decision method can beat it here.
- **Conclusion.** Historical replay gives no navigation benefit on these scenes, and they leave nothing to recover. Per the stop rule, no further replay repair.

## 3. Decision-level reconstruction

Suite: `coph_stagewise/reconstruction_decision_suite.py`. Output: `results/reconstruction_decision_v1/`.

- **Setup.** Matched evidence, built exactly as in `communication_repair_suite.matched_payloads`. 12 sender/snapshot groups, 400 random pairs of supported straight paths per group. Paths are priced with the production anchor scorer at w = 6.

| Stratum (mean path–anchor offset) | Field regret | Raw | Costmap | Raw − field 95% CI |
|---|---:|---:|---:|---|
| All | 0.048 (91.9% correct) | 0.101 (88.1%) | 0.106 (88.0%) | 0.047–0.060 |
| Near (≤ 0.33 m) | 0.025 | 0.075 | 0.079 | 0.042–0.057 |
| Between | 0.050 | 0.084 | 0.082 | 0.025–0.046 |
| Far (> 0.40 m) | 0.070 | 0.145 | 0.156 | 0.057–0.094 |

- **Caveats.**
  - The mean optimal path cost is 5.0, so the regret reduction is about 1% of path cost.
  - These are saved-state decisions, not navigation trials.

## 4. Bandwidth-limited selection

Suite: `coph_stagewise/bandwidth_selection_suite.py`. Output: `results/bandwidth_selection_v1/`.

- **Setup.** Four graded-hazard scenes, per-agent byte ceilings of 8 k, 16 k, 32 k and 4 M. Noise seeds 86100 and 86101 give identical outcomes, so the effective n is 4 scenes.

**Carrier cost by scene (rotation / receiver-relevant):**

| Scene | Off | 8 k | ≥ 16 k |
|---|---:|---|---|
| mild | 42.22 | 42.22 / 42.22 | 42.22 / 42.22 |
| moderate | 48.44 | 60.49 / 48.44 (starved) | 60.49 / 60.49 |
| severe | 100.57 | 60.62 / 100.57 (starved) | 60.62 / 60.62 |
| double_mild | 46.79 | 46.79 / 46.79 | 63.86 / 46.79 |

**Findings:**
- The 8 kB advertisement cost exceeds the budget, so selection fails exactly where it was meant to help.
- Where it "wins" (double_mild), it does so by withholding packets that cause a harmful detour, not by delivering more relevant evidence.
- **Side finding that needs follow-up.** In this fixed-rate-link harness (legacy receiver fitting, compact link), field sharing *hurts* the carrier in moderate (60.49 vs 48.44 with communication off) and, under rotation, in double_mild. The paper's hazard table, produced with the original channel, shows zero excess for field in every scene. The hazard claim therefore depends on the channel and receiver configuration.

## 5. Passivity

See [STAGE_BLENDING_LEMMA.md](STAGE_BLENDING_LEMMA.md). The lemma is proved: with fixed membership, the blended potential is one $C^1$ function, and Proposition `prop:energy` applies to it. The corollary bounds the storage by the sum of the positive recorded jumps. Convergence and stability are not claimed.

## 6. Mid-mission payload change (companion priority 1: "robustness to later drag changes")

Script: `benchmarks/phmarl/drag_shift_campaign.py`. Output: `results/native_drag_shift_v1/`. Figure: `paper_figures/fig11_drag_shift.py`.

- **Setup.**
  - At step 200 the native drag switches from 0.25 to 0.45 or to 0.10, and all four landmarks are redrawn.
  - Allocation and static targets are recomputed with the reset procedure.
  - 30 fresh seeds (87000–87029) × 2 conditions × 4 arms = 240 episodes.
- **Gate:** passed in both conditions.

| Drag after change | Online − fixed 0.25 | Online − prior 0.058 | Online − matched phMARL |
|---|---|---|---|
| 0.45 | +33.3 [30.6, 36.0], 30/30 | +75.1 [69.8, 80.6], 30/30 | +57.1 [50.7, 63.6], 30/30 |
| 0.10 | +29.7 [24.8, 34.8], 30/30 | +17.7 [15.0, 20.5], 30/30 | +46.9 [41.8, 52.1], 30/30 |

- **Before the change**, the fixed 0.25 model beats online by 9.4, as in every earlier native run.
- **After the change**, the fixed model goes stale. At 0.10 it oscillates and has 7 new contacts, against 1 for online. Online re-identifies the new drag within about 3 updates.
- **Coverage:** online reaches full phase-2 coverage on 30/30 starts in both conditions; matched phMARL reaches it on 14/30 (→ 0.10) and 12/30 (→ 0.45).
- **Caveats.**
  - The new contact on the shared seed appears in every CoPH arm and in phMARL.
  - One task, one change magnitude in each direction.

## 7. Route certificates from the packet covariance (companion: "bridge field error to decision regret")

Suite: `coph_stagewise/route_certificate_suite.py`. Output: `results/route_certificate_v1/`. Figure: `paper_figures/fig13_route_certificate.py`.

- **Setup.** Same matched evidence and path pairs as test 3. Each packet's own covariance is sampled 400 times. A route choice is *certified* when it wins in at least 95% of the samples.

| Packet | Certified | Wrong when certified | Wrong when flagged | True gap inside stated 90% interval |
|---|---:|---:|---:|---:|
| field | 66% | **0.7%** | 22% | **93%** |
| costmap | 69% | 5.0% | 28% | 72% |
| raw | 71% | 4.8% | 29% | 74% |

- **Result.** The field's gradients and covariance make its uncertainty honest: about 1 certified choice in 140 is wrong, and the true cost gap falls inside the 90% interval 93% of the time. The value-only packets are overconfident.
- **Use.** The 34% of choices the field flags as uncertain are where an agent should request a measurement or take the conservative route.
- **Caveat.** Saved-state study, 12 groups; not navigation trials.

## 8. Liveness in symmetric encounters (companion priority 6)

Script: `benchmarks/phmarl/liveness_campaign.py`. Output: `results/native_liveness_v1/`. Figure: `paper_figures/fig14_liveness.py`.

**Development (seeds 0–9).**
- A gyroscopic term $u \mathrel{+}= 3Sv$ was tried first. It does no work, but it vanishes once the projection has stopped the agents.
- A 0.3 keep-right waypoint shift was tried next.
- Both left 10/10 four-agent swaps gridlocked as a square at the centre.
- Deterministic priority cleared 10/10. That rule: an agent whose path is blocked by a visible lower-index peer sidesteps (target $q + 0.5\,\text{right} - 0.2\,\text{forward}$).
- Settings were then frozen.

**Evaluation.** Seeds 100–129, four families, 480 episodes.

| Family | Attraction | + priority | CoPH online | + priority |
|---|---:|---:|---:|---:|
| Exact head-on | 28 | 30 | 29 (8 contacts) | 30 (9 contacts) |
| Offset head-on | 28 | 30 | 29 (1) | 30 (1) |
| 4-agent swap | **0** | **30** | **0** | **30** |
| Crossing | 30 | 30 | 29 | 30 |

- **Gate: failed** on the zero-new-contact criterion for CoPH + priority in the head-on families.
- **Cause of the contacts.** The same contacts occur without the priority rule (8 and 1), so the rule does not cause them.
  - CoPH cruises faster than attraction: peak speed ≈ 0.49 m/s against 0.39.
  - Peers are visible only within 0.45 m, and a peer's velocity is unknown on first sighting.
  - Two closing CoPH agents cannot brake within the remaining 0.15 m. This is the companion's limited-visibility case: sensing range must exceed reaction plus braking distance.
- **Bug note.** The run's own gate expression filtered arms by a stale suffix and checked nothing. The gate was recomputed with the declared rule, and `summary.json` records the correction.
- **Liveness result.** Priority turns 0/30 into 30/30 in the gridlock family for both controllers, loses no successes elsewhere, and adds no contacts. It costs time: median time to goals in crossings rises from 77 to 119 steps for CoPH.

## 9. Liveness fix: braking-derived interaction radius (fresh seeds)

Scripts: `benchmarks/phmarl/liveness_range_campaign.py` and `benchmarks/phmarl/range_regression_campaign.py`. Outputs: `results/native_liveness_range_v1/` and `results/native_range_regression_v1/`. The figure `fig14_liveness` now shows this run.

- **Change.** `protection.braking_radius()` = 0.641 m replaces the shared layer's 0.45 m peer radius.
  - It is derived from native physics, not tuned: top speed with |u_x|,|u_y| ≤ 1, unit braking, and one reaction step for the unknown first-sighting velocity.
  - Opt-in: the default stays 0.45, so earlier results are unaffected. 26/26 native tests pass.
- **Liveness, seeds 200–229. Gate passed.**

  | Family | Attraction | + fix | CoPH online | + fix |
  |---|---:|---:|---:|---:|
  | Exact head-on | 29 | 30 | 30 (6 contacts) | 30 (0) |
  | Offset head-on | 29 | 30 | 30 (3 contacts) | 30 (0) |
  | 4-agent swap | 0 | 30 | 0 | 30 |
  | Crossing | 30 | 30 | 30 | 30 |

  Minimum separation for CoPH + fix is 0.321 m or more in every family. The cost is time: CoPH median time to goals is 141 steps against 77 in crossings, and 142 in 4-agent swaps.
- **Regression on the main benchmark, seeds 88000–88049. Criteria passed.**
  - Online CoPH: 50/50 coverage. Return change versus 0.45 m: +0.04 [−0.005, 0.10].
  - Online CoPH beats matched phMARL under the same layer by +104 [95, 113], winning 50/50.
  - No new contacts in starts that begin separated.
- **Still open.**
  - 21/50 native starts begin with two agents overlapping. In those starts, 4–5 new contacts occur in every arm, whatever the radius. Separating the initial pair pushes one of them into a third agent; this is the seed-85048 mechanism.
  - The criterion excluded these starts by declaration, so this is not fixed.

## 10. Final shared protection layer: initial-overlap contacts (fresh seeds)

Scripts: `benchmarks/phmarl/final_layer_campaign.py` and `benchmarks/phmarl/contact_certificate.py`. Output: `results/native_final_layer_v1/`.

**Diagnosis (seed 85048).** VMAS's penalty contact force ejects a spawn-overlapped agent at up to about 2 m/s. The penalty is 100 × penetration over one step. The old layer did not model this, so a bystander kept closing on the ejected agent.

**Final layer.** Opt-in; the defaults are unchanged and 26/26 tests pass. It combines:
- the braking-derived 0.641 m radius;
- contact-impulse prediction from public VMAS physics, applied to the agent itself and to its peers;
- a least-violation fallback when the constraints are infeasible, instead of plain braking;
- deterministic priority.

**Avoidability certificate.**
- The episode is replayed with native physics, contact forces included. The contacting pair applies full input directly away from each other for 10 steps past the onset; everyone else replays their logged actions.
- If the pair still touches, the contact is certified unavoidable by any admissible control.
- The escape is a best-effort strategy, not a proven optimum.

**Development (200 used episodes).** Avoidable contacts: base layer 3, radius only 2, radius + contact prediction 3, full layer 0.

**Evaluation, seeds 90000–90099 native and 300–329 liveness. Gate passed, all four parts.**

| Check | Result |
|---|---|
| Online CoPH coverage | 99/100 |
| Avoidable contacts | 0 for online CoPH and for matched phMARL |
| Online CoPH vs matched phMARL | +105 return [98, 113], 100/100 wins |
| Final layer vs base, online CoPH | +0.21 [0.10, 0.35] |
| Liveness | 30/30 in every family for both controllers, 0 contacts. The base layer gets 0/30 on 4-agent swaps and has 6 and 3 CoPH head-on contacts |

**Caveats.**
- One contact remains in each arm, on seed 90029. It is identical under both layers and certified unavoidable, with a best-escape distance of 0.296 against the 0.30 contact distance. That margin is thin.
- On these fresh native seeds the base layer also had no avoidable contacts. The native improvement from contact prediction therefore rests on the development set; on the evaluation set it is shown only for liveness.
- Priority costs time: median time to goals for CoPH in crossings is 140 steps against 81.
