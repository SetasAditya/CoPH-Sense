# A1: CoPH-Fork dynamics

CoPH-Fork is a two-agent, single-world VMAS diagnostic for the first
scout-carrier experiment. It separates four objects:

1. **Latent physical terrain:** binary route clearance and continuous route
   traction affect collision geometry and executed motion.
2. **Acquired evidence:** geometry or traction is revealed only by a charged
   sensing action taken within range of a route probe.
3. **Delivered evidence:** only evidence already held by the sender can enter a
   range-limited, delayed, lossy packet queue. Observation identifiers retain
   provenance through relaying.
4. **Physical execution and accounting:** VMAS integrates motion after the
   acquisition/delivery phase. The ledger charges time, commanded effort,
   low-traction exposure, sensing attempts, transmitted bytes, collisions,
   invalid actions, and deadline failure.

The actor observation contains kinematics, public time/budget state, locally
acquired evidence, and actually delivered evidence. It does not contain the
latent terrain or undelivered packets. `ForkAction` implements

```text
(motion force, optional route/modality probe, optional evidence packet)
```

The timestamp order in one transition is

```text
deliver due packets -> acquire requested evidence -> enqueue transmissions
-> deliver zero-delay packets -> scale force by true traction -> VMAS physics
-> commit route if the carrier crosses the decision plane -> charge costs
```

Consequently, a message can affect the current maneuver only if delivery
precedes the route commitment. A packet delivered later remains an observed
event but is marked `usable_before_decision=false` for that decision.

The team objective exposed to later learners is the negative incremental cost,

```text
C = C_time + C_motion + C_risk + C_sense + C_comm
  + C_collision + C_invalid + C_terminal_failure,
reward = -C.
```

Every term is nonnegative and the transition reports both its incremental
ledger and the cumulative episode ledger. Communication charges an attempt and
all 32 header plus 16 payload bytes, including failed or out-of-range sends.
On delivered evidence, the receiver attempts an 8-byte ACK with its own
32-byte header and attempt/byte charge when the return link is in range. The
sender marks evidence acknowledged only when that ACK arrives; ACK loss or
delay is visible in deterministic replay.
Sensing charges every attempt, including a probe requested from an invalid
location. The numerical contract is frozen in `a1_config.json`.

The initial A1 implementation uses batch size one and noiseless probes so that
causal semantics and deterministic replay can be audited exactly. Packet loss
is already parameterized and driven by a seeded RNG. Vectorization, noisy
sensors, learned policies, and the pH actor are later milestones.

The included scripted witness is an environment verification policy, not a
learning result or baseline: it probes the top route, sends both evidence
items, waits for delivery, and then drives both agents through the admitted
route. Its privileged summary plot displays the true traction fields for human
audit; those fields are absent from actor observations.

## A2 exact acquisition oracle

`oracle.py` defines a finite, exactly enumerable information problem on top of
the A1 action semantics. The default prior has four equally likely worlds:

```text
top geometry  in {blocked, open}
top traction  in {low, sufficient}
bottom route  known open with sufficient traction
```

The safe top route costs 1, the known fallback costs 3, and an unsafe selected
route costs 5. A geometry or traction probe costs 0.04. Each attempted packet
costs 0.0124 and is delivered before commitment with probability 0.9.

For sensing subset `D` and decentralized sharing rule `sigma`, the evaluator
computes the finite sum

```text
Q(D, sigma) = sensing cost(D)
            + E[communication cost(sigma, world)]
            + sum_h min_{route certified under h}
                    sum_world P(world, delivered history h) C(route, world).
```

The receiver's decision is indexed only by delivered history `h`; private
scout readings never enter it. An absent packet is marginalized as either an
untriggered send rule or channel failure. The default certified oracle admits a
route only if it is safe in every positive-probability world compatible with
`h`. A separately labeled `expected_cost` mode omits this certification gate.

The solver enumerates:

- all 16 subsets of the four route/modality measurements;
- four executable send rules per acquired item: never, always, if bad, if good;
- 625 complete sensing/sharing policies in total;
- every packet-delivery outcome; and
- both terminal route decisions at each delivered history.

The report contains all 625 `Q_share` action values, the optimal
`Q_share_star` continuation for every sensing subset, and all 16
`Q_sense_star` values. These are exact for this declared finite policy class;
they are not claimed to solve arbitrary continuous VMAS motion or unrestricted
decentralized policy trees. In particular, the terminal costs `1/3/5` are a
**normalized route-cost abstraction**. They have not been calibrated to the A1
VMAS time, force, collision, and failure ledger. The A2 oracle shares the A1
measurement identities, acquisition price, packet format, and causal delivery
order; it is a diagnostic information oracle, not a continuous-control oracle.

Run it with:

```bash
CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_oracle.py
```

## A3 learned set-value diagnostic

`train_critic.py` trains three matched-input critics on independently generated
pre-acquisition states. Each state varies top-route priors, sensing and packet
prices, timely-delivery probability, and route costs. Its 16 sensing subsets
receive exact A2 targets. The three critics are:

- additive `q0 + sum q1`;
- DeepSets with pooled candidate embeddings; and
- explicit pairwise `q0 + sum q1 + sum q2`.

All receive the same public state and candidate metadata. No future sensor
reading or latent world realization enters an input. Each predicts execution,
sensing, and communication components, and is evaluated by exact acquisition
regret after selection. Validation regret selects the checkpoint; an independent
test-state split is reported once. The supplied run uses three seeds to expose
seed variation. This is supervised learning of the finite oracle, not yet
learned navigation or an improvement over a matched pH control baseline.

The frozen three-seed diagnostic uses 320 train, 80 validation, and 120 test
information states per seed, with all 16 acquisition values labeled at each
state. Mean test results are:

| Critic | Mean exact regret | Top-1 accuracy | Pair recall | Harmful sensing rate |
|---|---:|---:|---:|---:|
| Additive | 0.01455 | 0.781 | 0.901 | 0.178 |
| DeepSets | 0.00641 | 0.856 | 0.849 | 0.056 |
| Explicit pairwise | **0.00423** | 0.850 | 0.826 | **0.047** |

The pairwise model has the lowest regret on all three seeds but does not win
every metric. DeepSets has slightly better top-1 accuracy and pair-selection
recall. The additive model often selects the correct pair yet gives the
individually harmful singleton probes falsely favorable values in the canonical
A2 state. Thus the result supports a set-valued critic over an additive value
decomposition, but does not establish that explicit pair supervision dominates
all generic set models or generalizes to unseen measurement operators.

```bash
CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/train_critic.py

CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/summarize_a3.py
```

The frozen aggregate is in `results/a3/summary.json` and its plot is
`results/a3/a3_three_seed_comparison.png`.

## A4 exact post-observation sharing and learned critic

`share_oracle.py` solves a finite sharing decision after the sender has acquired
and observed evidence. The decision variable is a **set of evidence packets**,
including the empty set. The sender may condition on realized readings and
charged ACKs. The receiver's certified route depends only on acknowledged
evidence it already holds and packets delivered before the decision deadline.
For each set, the oracle enumerates all independent packet-delivery outcomes
and compatible latent worlds. It charges evidence attempts and expected ACK
attempts, then reports exact expected downstream route cost. Silence is not
decoded: no packet and a dropped packet produce the same receiver history.

This is an exact information-decision oracle for the finite normalized route
model, not exact optimization of VMAS trajectories. The A4 ACK abstraction
assumes the return link remains in range when a timely evidence packet arrives;
the A1 environment checks that range explicitly. A3's labels still assume
A2-optimal sharing and have not yet been recomputed under learned A4 sharing,
so an end-to-end A3→A4 claim is premature.

`train_share_critic.py` fits additive, DeepSets, and pairwise critics on all
feasible message-set values at each sampled post-observation state. Train,
validation, and test states are independently generated and balanced across
complementary, duplicate, costly/irrelevant, novel-useful, and stale cases.
Validation exact regret selects each checkpoint. The fixed heuristics are
never send, always send, send all novel, and an uncertainty threshold. The
primary metric is exact sharing regret; the report also contains top-1 set
accuracy, wasteful and missed sends, expected transmitted bytes including ACKs,
and category-wise results. `summarize_a4.py` aggregates three frozen seeds and
audits canonical counterfactual states. The learned A4 result remains a
diagnostic until the learned policies reliably pass those interventions.

After ACK accounting was aligned, the frozen three-seed diagnostic (600
training, 150 validation, 200 test states per seed) gave mean exact sharing
regret 0.0234 for additive, 0.0325 for DeepSets, and 0.0284 for pairwise,
versus 0.0973 for send-all-novel and 0.3498 for never-send. Pairwise has the
lowest wasteful-send rate among the learned critics (0.038), but additive has
the lowest mean regret. All three learned models choose the useful good-evidence
pair and the message that completes an ACKed pair on all three canonical
seeds. The pairwise model also suppresses duplicates, stale evidence, and an
in-range costly/low-link case on all three seeds. None of its seeds sends the
useful bad-terrain warning in the declared expected-cost case, so the learned
A4 completion criterion is **not met**. The per-seed reports, aggregate JSON,
and comparison plot are in `results/a4/`.

The frozen-input audit in `results/a4/input_and_warning_audit.json` found only
13–19 bad-warning examples per 600 training states. It also found a genuine
representation collision outside the original independent-prior generator:
two different joint terrain priors have identical old critic inputs but
opposite exact sharing choices. The old input exposed only geometry and
traction marginals; expected-cost decisions can depend on their joint law.

`train_share_critic_v2.py` is a separate development revision. It adds the
full four-world public prior, the receiver's no-message route and route-cost
margin, and the effective evidence-plus-ACK price. It reserves 120 of 600
training and 30 of 150 validation states for constructed bad warnings, then
tests on the unchanged five-category stratified test generator and an
independently seeded 100-state warning set. It compares additive with and
without consequence weighting and weighted pairwise on the same data. The
old frozen checkpoints remain under `results/a4/`; this revision is under
`results/a4_v2/`. `a4_v2_controls.py` separately tests the old input with the
warning curriculum and the revised input without that curriculum to attribute
the improvement.

The A4-v2 weighted pairwise critic is the finite-task default. Across three
seeds, it obtains mean exact regret **0.0159**, wasteful-send rate **0.065**,
mean expected bytes **52.0**, and critical-message recall **0.993** on the
unchanged stratified test generator. It sends all 100 useful bad warnings in
each independently seeded warning test and chooses the exact oracle set on all
seven canonical interventions in each seed. This meets the A4 engineering
gate for the declared finite independent-prior task. It is not a claim about
correlated-prior zero-shot transfer or continuous VMAS control.

The controls matter for attribution. The old input plus warning curriculum
also obtains 100% held-out warning recall in each seed, while the revised
input without that curriculum obtains 42%, 100%, and 98%. Thus warning
coverage, rather than extra pairwise capacity, explains robust warning
recovery here. The revised input is still needed to remove the demonstrated
joint-prior and ACK-pricing collisions. With the same revised representation
and curriculum, adding consequence weighting changes additive mean regret
from 0.0202 to 0.0173 and lowers its missed-useful value, but raises its
wasteful-send rate. A tiny 32-state capacity audit fits all 16 training
warnings and succeeds on its separate 100-warning diagnostic; it is not an
independent method benchmark.

The following section checks the A3→A4 continuation gap: A3's original labels
presume oracle-optimal sharing, whereas deployed A4 uses this learned critic.

## Nested A3→A4 consistency check

`nested_oracle.py` enumerates every realized reading for each of the 16 A3
sensing sets, then evaluates the same post-observation A4 message-set problem
under either exact or learned sharing. It uses one unified protocol for both
continuations: no inference from silence and charged evidence/ACK attempts.
The original A2 labels use content-conditional send rules and omit ACK cost,
so `evaluate_nested.py` reports that protocol bridge shift separately instead
of attributing it to the learned A4 policy.

On the original 360 held-out A3 contexts, the learned A4 continuation has a
nonnegative gap at every sensing set. Mean gap across sets is **0.0229**;
the original A3 critic's mean acquisition regret rises from **0.0045** under
exact A4 sharing to **0.0070** under learned A4 sharing. The learned
continuation changes the exact-A4 optimal sensing set in 27 contexts and
preserves the strict geometry–traction complementarity ordering in 197 of 217
contexts where exact A4 sharing has it. The canonical ordering survives.

Because the continuation changes some decisions, `finetune_a3_for_a4.py`
regenerates A3 component labels with the **frozen learned A4** continuation.
It starts from the original A3 pairwise checkpoints, selects an update on
validation nested regret (including epoch zero), and compares on new seeds
2101–2103. Across the three paired confirmation runs, mean nested regret falls
from **0.0127 to 0.0063** and irrelevant bottom-route sensing falls from
**6.9% to 2.2%**. Each seed improves regret. All revised checkpoints choose
the canonical geometry–traction pair. The normalized executed cost decreases
from 3.2364 to 3.2301. These are finite diagnostic results; the A3/A4 stack
has not yet been connected to persistent cooperative memory or pH execution.
On the fresh complementary-state subsets, mean nested regret falls from
**0.0090 to 0.0046**; two seeds improve and one worsens slightly. The
learned A4 continuation itself still loses the exact strict pair ordering in
20 of the original 217 complementary diagnostic contexts, so this result
supports continuation-aware A3 decisions rather than perfect A4 sharing.

The original and revised checkpoints remain separate. Exact context-level
values and the first audit are in `results/nested_a3_a4/contexts.json` and
`summary.json`; the fresh-seed comparison is in `finetune_summary.json` and
`finetune_comparison.png` there.

## A5 cooperative-memory diagnostic

`memory_a5.py` adds an exactly enumerable **finite two-region information
task**. Each region has hidden geometry and traction; the scout measures both
Region-1 variables and the frozen A4-v2 critic chooses which readings to send.
Only delivered packets enter the carrier's evidence ledger, with region,
modality, value, uncertainty, timestamps, and source. Successful delivery
incurs a charged ACK. The carrier then chooses at most two measurements before
a fixed, conservative route executor makes the two route decisions. All 16
hidden worlds and packet-delivery branches are enumerated exactly. This is a
diagnostic abstraction, **not** a two-region VMAS simulation or pH execution.

The one-fork A3 checkpoint has no persistent-memory input, so the frozen test
uses an explicit adapter: it conditions each region's prior on delivered
evidence, scores the original A3 sensing sets separately in each region, and
selects the largest predicted gain. No A3 or A4 weights are updated. The exact
memory-aware acquisition reference uses the same restricted carrier action
menu; it is an acquisition-decision reference, not an optimal decentralized
scout–carrier policy. The `shared_reset` ablation delivers evidence for route
execution but clears it before the later acquisition decision. `shuffled`
assigns delivered Region-1 records to Region 2.

Across three paired frozen checkpoints, expected team cost is 5.6600 with no
sharing, 5.7006 with reset memory, 5.4306 with persistent memory, 5.3230 with
the exact acquisition reference, and 6.0156 with shuffled memory. Persistent
memory causes useful Region-2 sensing in 54.0% of exact branches, versus 0%
with reset memory, but it still duplicates a delivered measurement in 28.1%
of carrier measurements and selects a nonpositive-value batch in 20.3% of
branches with a carrier acquisition. Its mean acquisition regret is 0.1076.
In the all-good world with both Region-1 messages delivered, one frozen seed
remeasures the known Region-1 pair (exact marginal value -0.08), while the
exact reference measures the Region-2 pair (value +0.42). **A5 has not
passed.** The frozen model shows a partial memory benefit but has not learned
reliable avoidance of known evidence. This exposes the need for an explicit
memory-aware acquisition representation and training support before claiming
the complete cooperative behavior. In particular, A3 was trained with
geometry/traction prior probabilities in [0.15, 0.85], whereas an exact
delivered reading puts the adapter at 0 or 1; this is a genuine out-of-support
test, so the diagnostic does not isolate architecture from training coverage.
The `independent` condition has zero
information-theoretic redundancy because it receives no scout evidence;
physical measurement overlap is reported separately.

The exact branch ledger, per-seed results, canonical trace, and plot are in
`results/a5_memory/`. Run the frozen diagnostic with:

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_memory_a5.py
```

### A5-v2: memory-aware acquisition critic

`memory_critic_a5.py` replaces the one-region adapter with a full 16-world
posterior, four explicit evidence-status fields, and region/modality candidate
features. `train_memory_a5.py` checks all 81 noiseless known/unknown/value
patterns for contradictory exact labels, then trains additive, DeepSets, and
pairwise set critics on exact carrier-action costs. Already-known Region-1
states are deliberately represented in training. A4-v2 remains frozen.

Before finalizing the A5-v2 comparison, we corrected the finite reference:
because A4 sends selectively and packets can fail, the set of messages
received by the decision deadline is informative, including silence. The
carrier's exact 16-world posterior now enumerates the frozen A4 send rule and
all packet outcomes. The route executor remains fixed and uses **explicit**
delivered/acquired evidence to authorize a shortcut; a posterior implication
alone does not change its route. The original A5 figures above remain an
archived evidence-only diagnostic, while `results/a5_memory_v2/comparison.json`
uses the corrected A4-consistent reference for all nested-regret comparisons.
There is no second A4 sharing stage after carrier acquisition in this finite
protocol.

Across three paired checkpoints, the new additive, DeepSets, and pairwise
critics all attain the corrected restricted acquisition reference on the
enumerated delivery branches: expected team cost **5.2551**, exact nested
acquisition regret **0**, duplicate delivered measurements **0%**, and useful
Region-2 sensing about **92%**. The frozen prior A3 adapter costs **5.4306**
with **28.1%** duplicate measurements and corrected nested regret **0.1755**.
Every A5-v2 model selects the unresolved Region-2 pair in the canonical
fully delivered all-good case for all three seeds. On separately sampled
posterior/cost states, mean regret is about **0.002–0.004** depending on model
and seed. The additive model performs as well as the richer models on the
executed finite task, so this result supports memory sufficiency/training
coverage, **not** a pairwise interaction advantage. The training set explicitly
contains all A4-reachable delivery signatures; matching this finite reference
is an engineering checkpoint, not evidence of transfer to unseen protocols.

**A5-v2 passes the declared finite cooperative-memory checkpoint.** This does
not yet extend VMAS geometry, establish continuous pH execution, or test noisy
and stale evidence. Reproduce the training and corrected comparison with:

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/train_memory_a5.py

PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/summarize_memory_a5_v2.py
```

`make_a5_memory_gif.py` produces `results/a5_memory_v2/a5_memory_story.gif`
and a final-frame PNG. It loads the frozen seed-1701 models and exact all-good,
both-packets-delivered branch, then animates scout inspection, A4 delivery,
the divergent carrier acquisitions, and the charged route outcome. The GIF
labels this single branch separately from the three-seed expected team costs.

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/make_a5_memory_gif.py
```

## A6 exact communication-timing diagnostic

`timing_a6.py` adds a common irreversible route commitment, a one-dimensional
scout–carrier separation, communication range, optional scout return into
range, fixed network delay, independent packet delivery, and a complete cost
ledger. Delivery is usable only when
`acquisition_duration + return_duration + network_delay < decision_slack`;
arrival exactly at commitment is too late. The frozen A4-v2 message actor
receives the effective timely-delivery probability, and the frozen additive
A5-v2 carrier actor uses only evidence delivered before commitment. Late
packets can still incur transmission and ACK charges. The exact reference
enumerates **inspect both Region-1 variables** versus **skip inspection**
under those same frozen downstream actors; it is not a global decentralized
optimum. A waiting scout cannot improve range in this static-position finite
model, so only return into range and no return are meaningful physical choices.

`run_timing_a6.py` checks 270 range/delay/reliability/slack contexts for each
of three paired checkpoints, including the requested packet probabilities
1.0, 0.9, 0.7, and 0.5 plus low-reliability stress points 0.2 and 0.1. At
slack 0.4 s and range 1.0, the exact gate inspects for delays through 0.25 s
at packet success 0.5–1.0, then skips at 0.5 s. At range 0.6, returning into
range already misses this commitment, so it skips. With success probability
0.1 it also skips at in-range early delays; at 0.2 the three checkpoints
disagree in one third of cases. Across the 306 post-deadline seed-contexts,
the exact gate skips every time; the frozen always-inspect scout incurs mean
avoidable cost 0.0977. On the exact same all-good world with reliable delivery,
the early and near-deadline packets change the carrier's later sensing and
route, while a packet arriving at commitment cannot.

The frozen A5 scout schedule still inspects every time. `train_timing_gate_a6.py`
adds a separate local scout inspect/skip actor using only public positions,
range, return time/effort, packet reliability, delay, slack, and the exact
public deadline-feasibility flag. It trains on exact two-choice values under
the **same frozen A4-v2 and A5-v2 continuation**; neither checkpoint is
updated. Training, validation, and confirmation use disjoint sampled timing
contexts. On 720 fresh seed-context evaluations, the selected gate agrees
with the exact restricted reference **97.4%** of the time, has mean decision
regret **0.0018** versus **0.0588** for the fixed always-inspect scout, and
never inspects after the hard deadline. On the previously inspected 810-case
grid, accuracy is **96.5%** with mean regret **0.0026**. The hard deadline
boundary is supplied as public input, so this is evidence of learning the
remaining cost/reliability trade-off, not discovering the feasibility rule
from pixels. Additive A5 remains the carrier model throughout.

**A6 passes as a finite timing/range engineering checkpoint**, with a local
scout gate and exact comparison under the declared fixed-position protocol.
This abstraction does not include moving-carrier rendezvous, multiple
commitment times, VMAS motion, or pH execution. Full exact-grid results,
fresh-context gate results, and phase plots are in `results/a6_timing/`.

## A6.5 moving-agent bridge: diagnostic, not passed

`run_moving_bridge_a65.py` places the **frozen** A4-v2 sharing actor and A6
inspect/skip gate in the A1 one-fork VMAS world. The scout travels to a real
probe, senses through the environment API, optionally sends acquired evidence
through the charged packet queue, and both agents move with fixed scripted
controllers. The carrier commits when its simulated trajectory crosses the
fork; delivered evidence changes its route only before that event. The pilot
stores three deterministic replays and a trajectory plot in
`results/a65_moving_bridge/`. This bridge does **not** run A3/A5 (the VMAS
world still has only one fork), and the motion controller is not pH/MHRF.

`run_moving_bridge_a65_sweep.py` compares frozen gate, always inspect, and
never inspect on all four combinations of top clearance and traction, with
bottom traction 0.50. For 0.15 s packet delay, the gate inspected in all four
worlds and mean team cost was **1.3862**, versus **1.2302** for never inspect;
all four episodes succeeded for both. For 7 s delay the gate skipped in all
four and matched never inspect at **1.2302**, while always inspect cost
**1.3330**. There were no collision costs in these runs. Thus the finite
timing gate does suppress clearly late sensing, but its acquisition value does
not transfer to this physical team-cost ledger: the upper route takes longer
even when it is safe, and the lower route is already cheap. These worlds are
a deterministic diagnostic, not a statistical performance estimate.

The public online commitment estimate is also inaccurate: at the pilot gate
state it predicts **4.30 s**, while the carrier actually commits **6.80 s**
later when taking the uninformed lower route. At a 5 s message delay, a
message sent after the two physical acquisitions could have arrived before
that commitment, but the frozen A4 actor sees the predicted deadline and
selects no message. This is a false feasibility rejection. The next repair
is the trajectory-conditioned forecast and paired VMAS acquisition-value audit
below. The world must then be extended to two forks to test A5
cooperative-memory behavior before the pH/MHRF factorial. No pH benefit is
claimed here.

### A6.5-v2 paired rollout audit

`run_moving_oracle_a65_v2.py` now clones the physical VMAS state at the scout's
inspection point. A public provisional-policy rollout predicts the carrier's
commitment time; the controller uses only current kinematics and legitimately
delivered evidence. Across all four top-terrain worlds, the forecast is
**6.80 s** and matches the subsequent uninformed carrier's commitment time.
The matching forecasts across hidden worlds are checked in the integration
test. This is an exact forecast for the *scripted provisional continuation in
this one-fork setting*, not a general prediction guarantee.

From each cloned state, the script executes both physical choices through
the same frozen A4 continuation and complete VMAS ledger. The oracle chooses
one ex-ante action under a **uniform prior over four top geometry/traction
worlds**; it does not peek at which hidden world was drawn. Costs below exclude
the common pre-decision prefix:

| Packet delay | Inspect | Skip | Ex-ante oracle | Frozen gate with rollout forecast |
| --- | ---: | ---: | --- | --- |
| 0.15 s | 1.2543 | 1.0983 | skip | inspect; regret 0.1560 |
| 5.00 s | 1.5373 | 1.0983 | skip | inspect; regret 0.4389 |
| 7.00 s | 1.2011 | 1.0983 | skip | skip; regret 0 |

At 5 s delay, the corrected forecast permits two packets to arrive before
the actual maneuver commitment in the all-good world. The original local
estimate would have suppressed them. **Timing repair therefore works, but
the acquisition-value mismatch remains and becomes more visible.** In this
specific geometry the lower route is cheap and the upper route is slower even
when safe; fitting a gate that always skips here would be a vacuous learning
result. The benchmark needs physically consequential route trade-offs, and
the information policy must be valued with the moving ledger, before A6.5
can pass. The exact paired results are in
`results/a65_moving_bridge/v2_world_rollouts.json` and
`v2_oracle_summary.json`.

### A6.5-v3 physical value-of-information family

`run_moving_oracle_a65_v3.py` parameterizes the VMAS start positions and the
two carrier corridors. In this family, the carrier starts at y=0.05, takes a
certified upper corridor at y=0.32, or takes the lower backup corridor at
y=-0.65. The goal sits at y=0.20, so the backup route makes a visible detour
before returning to it. The scout's lower return path is separated from the
carrier's.
Commitment occurs at x=-0.38, the last tested safe turning region before the
island. The existing VMAS traction multiplier slows low-traction motion; no
route reward or abstract route cost is added. In the checked all-good world,
the carrier's measured upper path is **2.098 m** versus **2.359 m** on the
lower route, and takes **515** versus **1,510** steps when lower traction is
0.15. Both geometry and the declared traction-dependent force scaling
contribute to the completion-time saving. Both
agents reach the goal, with zero collision cost.

The exact *restricted* acquisition reference clones the VMAS state at the
scout viewpoint and evaluates inspect-both versus skip through frozen A4 and
the same scripted motion continuation. It selects one action before learning
the hidden terrain, averaging equally over the four top-clearance/traction
worlds. Table entries are post-decision team costs; the common approach cost
is removed. All 32 deterministic paired rollouts succeeded with zero collision
cost.

| Physical condition | Inspect | Skip | Oracle |
| --- | ---: | ---: | --- |
| Lower route traction 0.50, 0.15 s delay | 1.363 | **1.335** | skip |
| Lower route traction 0.15, 0.15 s delay | **2.636** | 3.032 | inspect |
| Same useful route, probe cost 0.40 per modality | 3.356 | **3.032** | skip |
| Same useful route, 7 s packet delay | 3.131 | **3.032** | skip |

The paired outcomes also make the simple threshold interpretable. At the
standard probe price, an unsuccessful inspection adds about **0.129** to the
backup-route cost. In the certified-safe world the post-decision cost falls
to **1.061**. With safe-top probability 1/4, the physical shortcut saving is
too small when the backup costs 1.335, but exceeds inspection expenditure
when the backup costs 3.032. The table uses the executed ledger directly;
this threshold calculation is a diagnostic approximation to it.

Thus the physical oracle has both actions and changes choice because of
route progress, probe expenditure, and usable delivery time. The frozen A6
gate chooses inspect in the useful case and skip at 7 s, but still inspects
in the low-value and expensive-probe cases. These are **physical decision
regrets**, not evidence that the frozen finite gate has generalized broadly.

`run_moving_comm_grid_a65_v3.py` adds two communication slices. With 20%
delivery probability per evidence packet, it enumerates the four independent
two-packet outcomes and weights their VMAS costs exactly under the declared
packet model: inspect **3.108**, skip **3.032**, so the oracle skips while the
frozen gate inspects. At range 0.45 with reliable delivery, inspect **2.650**
versus skip **3.032**; moving back into range has a cost but does not reverse
the choice. ACK randomness does not alter this scripted continuation, which
does not resend based on ACK status. Much shorter ranges currently expose a
return-to-range controller collision and are not included as clean oracle
conditions. The result and physical-route figures are
`results/a65_moving_bridge/v3_oracle_decisions.png` and
`v3_physical_routes.png`.

These results establish a **nondegenerate VMAS benchmark and exact restricted
decision labels**, not a passed A6.5 learned-gate checkpoint. A VMAS-grounded
gate still needs to learn the boundary over more layouts and held-out
conditions, with low physical regret and no hidden-world input. The two-fork
A5 bridge and pH/MHRF executor remain subsequent milestones.

### A6.5 VMAS-grounded acquisition gate

`vmas_gate_data_a65.py` now enumerates paired `INSPECT` and `SKIP` VMAS
continuations from the same pre-acquisition snapshot. Each label averages the
post-decision ledger over the declared four-world top-terrain prior and the
four joint delivery outcomes for two evidence packets; its target is the physical advantage
`J_skip - J_inspect`. The gate sees public task metadata, local/public agent
kinematics, a public backup-traction **prior**, route geometry, sensing cost,
communication parameters, and a no-information commitment forecast. It does
not see hidden top geometry or traction. The backup prior is exact for the
current deterministic backup slices, so transfer to uncertain backup terrain
remains untested. The feature audit checks equality across all hidden top
worlds, including the forecast.

The 21-to-128-to-128-to-64-to-1 MLP is trained with Huber regression. The
ordinary model regresses advantage directly. A second, structured diagnostic
regresses advantage *before* the known two-probe sensing charge, then subtracts
that exact charge at inference. The latter is a disclosed revision after
inspecting the first OOD failures, not a predeclared independent method.
Training/validation/OOD development sets have 46/11/9 contexts. The data
generation deliberately includes useful shortcuts and near-boundary examples;
the OOD development set varies parameter combinations and extrapolates on
some ranges. The five canonical interventions were also used diagnostically.

Mean physical decision regret (`J(chosen)-min(J_inspect,J_skip)`):

| Method | OOD development (9) | Canonical development (6) | Untouched range-composition set (12) |
| --- | ---: | ---: | ---: |
| Raw advantage MLP | 0.1146 | 0.0000 | 0.0000 |
| Structured advantage MLP | 0.0185 | 0.0636 | 0.0000 |
| Always inspect | 0.2988 | 0.0879 | 0.2058 |
| Never inspect | 0.1486 | 0.1296 | 0.4283 |
| Deadline only | 0.0218 | 0.0714 | 0.0977 |
| Finite A6 gate | 0.0218 | 0.0714 | 0.0977 |
| Hand-calibrated analytic VoI | 0.0185 | 0.0000 | 0.0000 |
| Paired VMAS oracle | 0.0000 | 0.0000 | 0.0000 |

The raw model selects all six canonical actions correctly but has 55.6%
accuracy on OOD development contexts. Subtracting the exact known sensing
charge removes its costly-probe failure and reduces OOD regret, but the
structured model still misses a range-dependent canonical and ties the
hand-calibrated analytic baseline on OOD. Both frozen models achieve zero
regret on the later untouched 12-context set, which has large action margins
and therefore does not resolve near-boundary generalization. An earlier
12-context test was inspected before the structured revision and is recorded
as development in `structured_gate_report.json`.

**At this checkpoint, the A6.5 learned-gate result remained open.** The then-current
evidence supports physical value labels and causal public inputs, but not
robust learned decision value over the range/return boundary or an advantage
over the analytic VoI baseline. The next focused revision should represent
trajectory-dependent return-to-range cost and test it on a genuinely untouched
near-boundary composition set before proceeding to the two-fork A5 bridge.
This one-fork VMAS continuation uses frozen A4-v2 sharing and a scripted
carrier motion policy; it does **not** deploy the finite A5-v2 carrier policy.
The pH/MHRF executor and 2-by-2 control factorial have not yet been run.

Reproducible labels, checkpoints, and reports are in
`results/a65_moving_bridge/gate_data/`. Generation scripts are
`vmas_gate_data_a65.py`, `augment_vmas_gate_data_a65.py`, and
`augment_vmas_gate_boundary_a65.py`; training scripts are
`train_vmas_gate_a65.py` and `train_vmas_gate_structured_a65.py`. The final
range test is specified in `final_vmas_gate_range_test_a65.py` and evaluated
by `eval_range_final_vmas_gate_a65.py`. These exact-oracle labels incur full
VMAS continuation work and are a diagnostic privilege, not a cost-free
deployment signal. The analytic heuristic was calibrated on preliminary
physical-route values and should be treated as a strong development baseline.

### Trajectory-conditioned residual diagnostic

`vmas_gate_trajectory_a65.py` clones the public pre-probe state and simulates
two probe ticks followed by the scout return controller, without measuring or
sending. It exposes reconnect time, scout path length, commanded motion
charge, elapsed-time charge, radio margin, and final separation. A
hidden-world invariance test checks these features across all four top-terrain
realizations. `train_vmas_gate_residual_a65.py` fits
`analytic_VoI + learned_residual` with a small 64/32 MLP and a matched
residual lacking the new trajectory features.

Before fitting either residual, `freeze_vmas_gate_boundary_test_a65.py`
created and labeled a separate 21-context physical boundary set. Contexts
were selected using exact labels to center their *sensing costs* near the
indifference boundary, then frozen before model fitting. Ten favor inspection;
eleven favor skipping, all with `|J_skip-J_inspect|=0.025`. It contains five
range, seven probe-cost, eight packet-reliability, and only one deadline case;
the deadline stratum is too small for a generalization claim. Its contexts
and labels are in `gate_data/untouched_boundary_*.json`.

| Method | Canonical regret / correct | OOD development regret | Untouched boundary regret / correct |
| --- | ---: | ---: | ---: |
| Analytic VoI + trajectory residual | 0 / 6 of 6 | 0.0266 | 0.0048 / 17 of 21 |
| Analytic VoI + residual without trajectory | 0 / 6 of 6 | 0.0185 | 0.0048 / 17 of 21 |
| Analytic VoI alone | 0 / 6 of 6 | 0.0185 | **0.0024 / 19 of 21** |
| Deadline only | 0.0714 / 3 of 6 | 0.0218 | 0.0131 / 10 of 21 |
| Paired VMAS oracle | 0 / 6 of 6 | 0 | 0 / 21 of 21 |

Both residuals correct the earlier range-dependent canonical miss, but the
trajectory features do not improve the new near-boundary decisions. The
trajectory residual misses two of five range cases, one of seven probe-cost
cases, and the sole deadline case. On that deadline case the forecasted
packet arrival precedes route commitment by 0.35 s, yet inspection has
negative physical value. This suggests that the carrier needs maneuver lead
time *before* formal route commitment; delivery just before commitment can
be causally timely but physically too late to exploit. This is an
interpretation of the observed continuation, not a proven general rule.
The learned gate **does not beat the analytic baseline** on the untouched
boundary set, so A6.5 remained open at that checkpoint. The set has now been inspected and must
be treated as development data for any subsequent revision. A next test must
be newly frozen before fitting a revised representation of physical maneuver
lead time and reconnect cost.

### Final actionability revision

`vmas_gate_trajectory_a65.py` now estimates the latest carrier-only route
switch that still reaches the alternate corridor under the declared scripted
controller. It parks the scout during this *carrier feasibility forecast* so
that the scout's provisional no-information path does not create an unrelated
collision; the paired VMAS outcome oracle still evaluates both moving agents
and all actual costs. The search uses only the pre-acquisition public state
and a forced top-route control branch, never hidden top readings. Its output
is an actionability margin (latest feasible switch minus projected evidence
arrival), distinct from nominal route-commitment slack.

`freeze_vmas_gate_actionability_a65.py` froze a new 25-context set before
fitting: five matched early/late deadline pairs, plus five contexts each for
range, probe cost, and packet reliability. In each pair, only network delay
changes; both arrivals precede formal route commitment but straddle the
carrier-only latest switch. Exact labels are in
`gate_data/final_actionability_labels.json`.

`train_vmas_gate_actionable_a65.py` applies a known-cost/actionability
factorization: it uses the analytic VoI estimate while evidence can still
change the carrier route, otherwise assigns no carrier information benefit
and charges the two probes. A 64/32 MLP learns a residual only on actionable
training states. This hard gate assumes late evidence has no net scout-only
benefit in this restricted one-fork task; paired-label evaluation checks that
assumption. Full results are written to `gate_data/actionability_report.json`.

The frozen actionability test has 13 inspect-optimal and 12 skip-optimal
contexts. All ten paired deadline cases have **positive formal commitment
margin** (0.30--0.55 s). All five early deliveries make inspection worthwhile;
all five late deliveries make it harmful. The old analytic VoI inspects every
late case, whereas both actionability-aware methods select all ten correctly.

| Method | Fresh 25-case regret | Correct / 25 | Canonical regret | OOD development regret |
| --- | ---: | ---: | ---: | ---: |
| Actionability-aware analytic gate | **0.0030** | **22** | **0** | 0.0185 |
| Actionability-aware learned residual | **0.0030** | **22** | 0.0636 | 0.0185 |
| Previous analytic VoI | 0.0216 | 17 | 0 | 0.0185 |
| Deadline-only / finite A6 | 0.0256 | 13 | 0.0714 | 0.0218 |
| Paired VMAS oracle | 0 | 25 | 0 | 0 |

The analytic actionability gate's three mistakes are small-margin
packet-reliability cases (`+0.025` physical advantage). The learned residual's
three mistakes are small-margin range cases; it also skips the useful
short-range canonical. The residual therefore **matches the actionability
analytic gate on aggregate fresh regret but has no demonstrated benefit over
it**. The actionability timing fix passes as a restricted embodiment sanity
check, and the structured analytic gate is the frozen timing guard for the
next bridge. The *learned residual* is not promoted as the default. We stop
one-fork refinement here and proceed to moving two-fork cooperative memory,
where the acquisition learner's added value can be tested directly. The
carrier-only feasibility forecast is specific to the scripted controller;
it is not a pH controller certificate or a general actionability theorem.

```bash
MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_moving_oracle_a65_v3.py

MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_moving_comm_grid_a65_v3.py

MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/plot_a65_v3.py

MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/plot_a65_v3_paths.py
```

```bash
MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_moving_oracle_a65_v2.py
```

```bash
MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_moving_bridge_a65.py

MPLCONFIGDIR=/tmp/mpl-a65 PYTHONPATH=CoPh-sense \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_moving_bridge_a65_sweep.py
```

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_timing_a6.py

PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/train_timing_gate_a6.py
```

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/evaluate_nested.py

PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/finetune_a3_for_a4.py \
  --a3-seed 1701 --a4-seed 1801 \
  --output-dir CoPh-sense/coph_fork/results/nested_a3_a4/finetune1701
```

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/train_share_critic.py \
  --seed 1801 --output-dir CoPh-sense/coph_fork/results/a4/seed1801

PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/summarize_a4.py

PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/train_share_critic_v2.py \
  --seed 1801 --output-dir CoPh-sense/coph_fork/results/a4_v2/seed1801

PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/summarize_a4_v2.py
```

Run the verification and demo with the isolated pH-MARL environment:

```bash
PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python -m unittest \
  discover -s CoPh-sense/coph_fork/tests -v

CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/run_demo.py
```

## Moving two-fork A5 bridge: prototype and negative oracle check

`two_fork_scenario.py` and `two_fork_environment.py` add two physical VMAS
forks with four hidden terrain bits. The scout measures Region 1; the carrier
chooses `skip`, a Region-1 measurement set, or a Region-2 lookahead measurement
set **before Fork 1**, when both viewpoints are reachable. Acquisition,
packet/ACK, motion, risk, collision, and timeout costs use the VMAS ledger.
The Region-2 lookahead is a remote measurement of the second corridor, not a
contact traction probe. A4-v2 and A5-v2 weights are loaded unchanged.

The adapter constructs carrier memory only from own acquisitions and delivered
packets. Its posterior includes the likelihood of A4 sending nothing under
the fixed public protocol. `shared_reset` discards received evidence only for
the acquisition decision; `shuffled` corrupts its region label as a negative
control. The physical controller is scripted here. The pre-fork schedule
keeps both candidates actionable, but the A6.5 actionability rollout is not
yet integrated into this two-fork controller. This prototype therefore does
not establish the full moving A5/A6.5 or pH result.

The restricted exact reference enumerates all seven carrier acquisition sets
in all 16 worlds, using paired physical continuations and grouping by the
*actually delivered* memory signature. All 112 continuations in
`physical_oracle_all_groups_symmetric_v2.json` reached the goal without
collision. Yet the best acquisition set was `skip` for every signature. With
safe Region-1 evidence, expected downstream cost was 2.6050 for `skip` and
2.6464 for the Region-2 geometry/traction pair. Thus the present symmetric
corridors do **not** satisfy the required physical witness that persistent
memory makes Region-2 sensing worthwhile. The all-safe canonical trajectory
does show the intended action switch (reset chooses Region 1; persistent
chooses Region 2), but its lower realized cost is conditional on a favorable
hidden world and is not a valid expected-cost win. A trial with longer lower
bypasses still did not establish the witness and was not promoted.
Under the four equally likely delivered R1 signatures, frozen A5-v2 selects
the Region-2 pair every time. Its mean conditional downstream cost is 2.8302
versus 2.7390 for the restricted physical oracle, a mean regret of 0.0912.
The matched no-memory probe reference also prefers `skip`: over all 16 worlds,
its downstream costs are 2.7854 for skip, 2.9735 for the Region-1 pair, and
2.8911 for the Region-2 pair. This is a restricted three-action audit; it is
not an optimization over every possible decentralized policy.
An exact repricing of the same trajectories over traction-risk coefficients
0.12, 1, 4, 8, 16, and 32 confirms that this is not fixed by changing one
objective weight: the three-action no-memory reference chooses skip through
4 and Region 2 from 8 onward; it never chooses Region 1. This is a
development sensitivity audit, not a held-out result. The next map needs a
physical Region-1 decision whose value is substantial before scout evidence
and redundant after its delivery.

The next world revision must be frozen only after an exact no-memory witness
prefers Region 1 and its matched valid-memory witness prefers Region 2, with
all sensing and route costs charged. It should then be evaluated on held-out
worlds before adding pH/MHRF execution.

```bash
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.run_two_fork_memory_bridge --tag local_run

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.two_fork_physical_oracle --jobs 8 --tag new_run

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m unittest coph_fork.tests.test_two_fork_bridge -q

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.summarize_two_fork_bridge

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.analyze_two_fork_risk_sweep
```

## Oracle-selected asymmetric two-fork regime (frozen after physical search)

The prior symmetric bridge above was kept as a negative result. We then
searched only VMAS route geometry, lower-corridor traction, sensing price,
and declared independent per-feature safe priors. The physical oracle used
matched pre-Fork-1 snapshots: the M0 clone masks carrier memory, while the
M1 clone retains two delivered, valid Region-1 readings. Their position,
velocity, clock, budget, packet/ACK state, and cost prefix are identical.
The lower bypass parameter now extends the corresponding VMAS island, so a
larger detour represents actual geometric clearance rather than just a
scripted waypoint preference. No A5 output was consulted before the selected
layout was frozen in `physical_search/frozen_physical_layout_v1.json`.

The frozen layout uses lower waypoints -0.72/-0.50, backup tractions
0.18/0.25, sensing cost 0.02 per reading, and independent safe-feature priors
0.65/0.60 for Regions 1/2. In the full seven-action restricted physical
oracle, the expected downstream costs are:

| Memory at matched staging snapshot | Skip | Region-1 pair | Region-2 pair | Physical optimum |
| --- | ---: | ---: | ---: | --- |
| M0: no received readings | 3.6468 | **3.4705** | 3.6377 | Region 1 |
| M1: valid safe Region-1 pair | 2.7996 | 2.9209 | **2.6343** | Region 2 |

All four singleton choices are more expensive than the selected pair in
their respective states. Every enumerated continuation succeeded without
collision. The four required pairwise margins are 0.1764, 0.1672, 0.1653,
and 0.2866; the declared acceptance margin was 0.05. All 27 nearby
prior/sensing-price combinations passed (worst margin 0.0855), and a deeper
neighboring physical layout also passed. These are *development-family*
checks, not held-out map generalization.

Only after that freeze did we evaluate the unchanged A5-v2 checkpoint. It
chooses Region 1 at M0 and Region 2 at M1, with zero regret against the
restricted physical oracle in both states. On the same R1-safe hidden worlds
and identical staging prefix, its memory-conditioned action reduces expected
downstream cost from 2.9209 (R1 repeated without memory) to 2.6343 (R2
sensed with valid memory), a reduction of 0.2866. Repeating Region 1 after
delivery has negative physical marginal value: 2.7996 - 2.9209 = -0.1213.

The frozen A4/packet/ACK/A5 canonical VMAS run also switches acquisition:
independent and shared-reset choose Region 1, while persistent memory chooses
Region 2. Its all-safe raw total costs are 3.0366, 3.0894, and 2.1163;
the shuffled control chooses Region 1 and costs 3.0894. These raw totals
include different communication prefixes and are not an oracle-optimality
comparison. The persistent replay matches its saved SHA-256 hash. The
structured A6.5 actionability rollout and pH/MHRF controller are not yet
integrated in this two-fork bridge; the acquisition decision is held at an
actionable pre-Fork-1 staging point.

```bash
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.evaluate_frozen_a5_on_physical_layout

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.run_frozen_two_fork_causal

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.plot_frozen_two_fork
```

The figure is saved as `physical_search/frozen_two_fork_marginal_value.pdf`
and `.png`. The full CoPH-Fork suite passes 64 tests for this revision.

`make_frozen_two_fork_info_gif.py` animates the archived canonical persistent
VMAS replay without rerunning or modifying the policy. It shows the actual
scout/carrier positions, both R1 acquisition events, the two charged packets
in transit, and separate local geometry/traction beliefs updating only when
the agents acquire or receive evidence. The right-hand comparison uses the
matched executed shared-reset replay: its carrier repeats R1, whereas the
persistent-memory carrier inspects R2. Terrain colors depict evaluator truth,
not agent observations. The artifact is illustrative for this restricted
scripted bridge and is not a held-out E2 result.

```bash
CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/coph_fork/make_frozen_two_fork_info_gif.py
```

Outputs are `physical_search/frozen_two_fork_information.gif`, a final-frame
preview PNG, and a six-keyframe contact sheet PNG.

## First pH/control factorial on the frozen two-fork layout

`ph_factorial_controller.py` and `ph_factorial_environment.py` add a matched
direct versus split port-Hamiltonian executor. Both use the same goal,
public-geometry IPC, and evidence-conditioned Gaussian traction-risk
potentials, the same damping and force/speed limits, and the same actual
latent traction plant gain. The direct cell uses semi-implicit Euler;
the pH cell uses momentum half-kick, position drift, and second half-kick.
The belief is fixed within each integration step. Terrain geometry and
traction enter the force field only after local acquisition or packet
delivery; the hidden map is used only by the VMAS plant/collision ledger.
The model is a minimal GRL-SNAM-style numerical realization, **not** a
trained GRL-SNAM controller or a passivity/safety certificate. The effect
of traction on the conservative force is an external plant port, and
evidence/target changes can alter stored potential between steps.

The factorial uses the frozen A4/A5 information checkpoints. Ordinary
information predeclares a Region-1 pair for the carrier and broadcasts
both scout readings. CoPH information uses selective A4 sharing and
memory-aware A5 acquisition. The common route rule was repaired *before*
the final paired run so both executors can complete a lane change before
the second island and finish at distinct positions inside the goal region.
The public actionability forecast runs the actual cell's integrator to
its selected viewpoint under unit-traction prior dynamics; the realized
route still uses the actual controller and checks pre-commitment arrival.
This forecast is a staging-point feasibility check, not a full online
latest-usable-switch calculation.

On all 16 enumerated terrain worlds, paired by world and packet seed:

| Cell | Mean team cost | Success | Collision cost |
| --- | ---: | ---: | ---: |
| A: ordinary + direct | 0.6871 | 16/16 | 0 |
| B: ordinary + pH | 0.7051 | 16/16 | 0 |
| C: CoPH + direct | 0.6661 | 16/16 | 0 |
| D: CoPH + pH | 0.6779 | 16/16 | 0 |

The paired information effect C−A is −0.0209 on average and helps in
10/16 worlds; B−A is +0.0180, so this split executor does not beat the
matched direct executor on team cost. The interaction D−C−B+A is −0.0062.
Mean sensing and radio costs are the same in all four cells (0.08 and
0.0488); the differences arise from motion, time, and terrain risk.
The direct and pH cells deliberately use the same conservative force and
damping, so their continuous-time vector fields are mathematically the
same pH system. Thus B−A isolates **numerical execution**, not the
benefit of pH structure over a genuinely non-pH controller. A later
structural comparison needs a matched non-pH policy with the same
information, actuator limits, and safety interface. This is a single-layout
scripted-route integration test, not a held-out navigation result.
`results/ph_factorial/paired_factorial.json` contains per-world ledgers;
`summary.json` and `paired_factorial.pdf` show the aggregate and paired
effects. pH-MARL is an external benchmark and is not one of these cells.

```bash
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.run_ph_factorial --calibrate --worlds 8

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.run_ph_factorial --worlds 16

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_fork.summarize_ph_factorial
```
