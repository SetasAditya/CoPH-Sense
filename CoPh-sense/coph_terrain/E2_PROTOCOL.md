# CoPH-Terrain E2 protocol — pretraining specification

This file fixes the scientific decision rules before fitting a procedural E2
policy. The map generator and split manifests are already versioned by SHA-256.
The final pre-freeze hard stop and fresh audit are specified in
`MISSION_INTEGRATED_STOP_RULE.md`. Disposable `E2-dev` engineering runs on
training maps may proceed now; their weights and scores will not be used as
confirmatory E2 results.
The audit has now failed its sensing-prevalence and pair-optimality gates.
`E2_SPLIT_DECISION.md` freezes the natural benchmark and separates a future
controlled complementarity challenge from natural-distribution evaluation.
The starred physical resource defaults below remain **provisional** until a
blind environment/oracle pilot on separate `pilot` seeds verifies that sensing,
skipping, complementarity, and communication scarcity all occur. Record any
change and its reason before model training; do not use E2 test maps for it.

## Task and risk

Two planar agents, a scout and payload-carrying carrier, start in a shared
12 m × 12 m continuous world. Both must reach distinct positions in the goal
region before the provisional 65 s deadline* with no collision or immobilization. Map
Static occupancy/topology, start, and goal geometry are public to both agents.
Local surface support/roughness and traction remain latent; only the
simulator/evaluator sees their complete realization. Initial actor
information includes short-range noisy material appearance.

At each 0.05 s transition, the evaluator computes

\[
r_{\rm true}(x)=\operatorname{clip}\!\left(
  0.4g(x)+0.4t(x)+0.2g(x)t(x),0,1\right),\quad
t(x)=\operatorname{clip}\!\left(\frac{0.55-\mu(x)}{0.40},0,1\right),\qquad
C_R=\frac{0.05}{T_{\max}}\sum_{t<T}\left[
  r_{\rm true}(x^s_t)+r_{\rm true}(x^c_t)\right].
\]

Scout and carrier have equal exposure weight. The risk sum stops at goal or
failure; it does not include collisions or stalled time after termination.
Here \(T_{\max}=\Delta t\,H\) is the declared physical horizon, so
\(0\le C_R\le2\) for any horizon \(H\); the quantile head's [0,2] support
remains valid when the deadline changes.
Collision, immobilization, and deadline are separately reported failures and
receive a terminal penalty in nonrisk mission cost (C_0). A policy cannot
establish the primary claim by stopping safely because completion
noninferiority is a prerequisite. Both team and per-agent exposure are saved.
Here \(g\in[0,1]\) is latent surface support/roughness risk and
\(\mu\in[0.15,0.95]\) is traction. Surface patches are generated
independently of traction-severity patches on each realized map. The
physical traction port uses \(\mu(1-0.2g)\), so surface condition affects
both the declared exposure cost and progress. Passive appearance is a noisy
proxy for joint risk, not a direct reading of either latent field. No
separate synthetic hazard field is added in E2-v1.

Pre-freeze audit correction: the initial failure penalty of 3 allowed an
early immobilization to score below a later deadline failure, so a restricted
physical oracle preferred failure in a calibration state. Set the provisional
failure penalty to 10 before the confirmatory resource audit. At the declared
1300-step provisional horizon, the other bounded resource charges are below this
penalty; completion remains a separate success endpoint. This is a benchmark
accounting correction, not a CoPH-driven tuning result.

The normalized nonrisk cost (C_0) charges elapsed time, simulated actuator
effort, active sensing, attempted packets including headers/ACKs, and a
terminal failure cost. These are simulated resource units, not measured
joules. The deployed pre-acquisition decision rule minimizes the fixed score

\[
S(\mathcal I,D)=\mathbb E[C_0\mid\mathcal I,D]
 +\operatorname{CVaR}_{.95}(C_R\mid\mathcal I,D),
\]

over resource-feasible, physically actionable sets (D). Thus the fixed risk
coefficient is one after the above normalization. A3/A5 predicts expected
nonrisk cost and a distribution of *complete episode* material exposure, not
only a scalar expected risk. A4 uses the same downstream score after a reading
is realized. Static episode CVaR uses accumulated exposure and a precommitment
tail threshold in the information state; it is not silently replaced by a
nested conditional CVaR. Completion is additionally enforced through the
failure cost during training and as a separate checkpoint/evaluation gate.

Fit the bounded, monotone risk-quantile head from **individual rollout
returns** using quantile Huber regression; do not regress a high-tail CVaR
estimated from a handful of continuations. Use paired random worlds/network
seeds across candidate sets. The empty/single/pair interactions form a latent
set embedding, which is decoded into one complete-return distribution;
quantile vectors are not summed as though quantiles were additive.
The frozen candidate grid for the upper-tail head is
\(\{.05,.10,\ldots,.80,.85,.90,.925,.95,.96,.97,.98,.99,.995\}\);
the head constrains every predicted quantile to \([0,2]\).

## Physical and information interface

| Capability | Scout | Carrier |
| --- | ---: | ---: |
| Speed cap | 0.75 m/s | 0.55 m/s |
| Body radius | 0.065 m | 0.10 m |
| Public static occupancy + default local material appearance | full map + 0.6 m* | full map + 0.6 m* |
| Active line-of-sight surface-geometry scan | 1.5 m, 0.2 s, cost 0.02* | 1.0 m, 0.2 s, cost 0.02* |
| Local traction probe | 0.25 m, 0.4 s, cost 0.04* | 0.25 m, 0.4 s, cost 0.04* |
| Maximum active measurements | 4* | 4* |

At most four evidence packets* may be attempted across the team; each
packet/ACK uses the charged CoPH-Fork header/payload convention. Communication
is range-limited, delayed, and seeded. A request never reveals a reading;
evidence is usable only after actual delivery and before the controller's
latest physically actionable route switch. All methods receive the same
candidate generator, belief map, packet protocol, global planner, frozen split
pH executor, actuator limits, and observation-limited collision filter.

The candidate generator proposes 8–16 reachable viewpoint–modality actions
from local/public belief, uncertain terrain regions, current route alternatives,
budgets, and public physical forecasts. It can inspect public occupancy but
not true surface condition, traction, or undelivered packets. The acquisition critic considers empty,
single, and affordable pair sets; its network accepts a variable candidate
and evidence-memory set. A regression test must hold actor history fixed,
change hidden terrain, and verify identical candidates **and candidate
features**. A separate test checks actor/action invariance before delivery.

Pre-freeze interface correction: the route-only candidate generator offered
a common geometry viewpoint to both agents on just 2/120 excluded initial
maps (none in bottleneck or labyrinth), making scout-to-carrier reuse nearly
unselectable. The revised generator reserves up to two public task-corridor
viewpoints, then fills the remaining slots from each agent's observed route.
It uses only public goal/team kinematics and each agent's own observed
reachability. A new 120-parent excluded check found shared public viewpoints
in every topology; all methods receive the same revised menu. This is an
action-interface feasibility correction, not a CoPH policy result.

The order within a decision is: choose physical acquisition, travel/dwell,
realize the local reading, choose evidence to send, transmit/deliver, update
recipient memory/belief, replan, then execute the pH motion proposal. Default
sensor operation is accounted for through time/effort; no resource is charged
twice. The safety filter sees public static occupancy and local contact,
never latent surface condition or traction.

Passive appearance is sampled on entry to each 0.2 m map cell from
that cell's center, with a 0.6 m line-of-sight footprint. Revisiting the same
cell does not generate a new independent noisy appearance sample. This
cell-quantized sensor is part of the E2-v1 observation model and is applied
equally to every method.

The known-topology revision was chosen **before E2 training** after the
excluded-parent paired root-cause diagnosis in
`results/pilot/NAVIGATION_ROOT_CAUSE.md`. The previous hidden-topology
resource and memory counts are archived diagnostics and do not transfer to
this revised task. Before any renewed acquisition audit, run the
no-active-acquisition pH navigation check on 90 fresh excluded parents
(830000--830089, 30 per ID family), with the 65 s horizon and unchanged
resource settings. Its predeclared sufficiency gate is at least 95% success
overall and at least 90% in each family. Only if that passes should the
acquisition/resource audit resume.

That check passed: 89/90 overall, with 30/30 open, 30/30 bottleneck, and
29/30 shallow labyrinth completions. The next fresh resource audit uses
parents 840000--840089, decision steps 120 and 360, the public eight-per-agent
menu, every singleton, and eight stratified pairs, with the same 95% overall
and 90% per-family continuation-coverage guard. A passing navigation result
does not itself freeze sensing/radio settings.

The first revised resource audit covered all 180 scheduled states, but
its 161 skip, 13 singleton, and zero pair winners among 174 evaluated
decisions used a **straight-line sensing detour** in the scripted
continuation. That detour could stall against known walls before measuring.
It has been repaired to route through public occupancy. On a matched
40-parent subset, the corrected continuation covered 80/80 scheduled
states and had 69 skip, eight singleton, and zero pair winners among 77
evaluated decisions. The corrected full 90-parent resource distribution
has not been rerun, so the original class counts must not be used as
current estimates. A pre-repair forced ten-query script hit the packet
cap on 12/12 parents and both measurement caps on 11/12; cap binding also
needed rechecking. The matched post-repair stress script reached all three
declared caps on 12/12 parents; extra packets did not lower cost on any of
the twelve paired successful runs. This remains a forced-script result.
The E2 resource freeze remains **on hold**.

In a pre-repair paired post-delivery memory pilot on 24 more excluded parents,
19 scout packets were physically delivered and all 19 resulting carrier
menus had successful comparisons. The carrier's no-memory best choice never
overlapped the scout's measured region; zero cases changed from a valuable
repeat to an alternative after retaining the packet. Two best actions
changed, both from skip to an additional surface scan. This pilot therefore
does **not** establish the intended memory-reallocation mechanism for the
revised task. Its physical cost comparisons need repeating after the
route-to-viewpoint repair. No E2 training or manifest regeneration should
follow until candidate relevance and terrain-information value have been
re-audited globally on fresh excluded parents.
An evaluator-only decomposition on 40 already-excluded development parents
(840000--840039, steps 120 and 360) now separates ideal information value
from executable sensing. A denser public-route screen found an ideal paired
reveal worth more than 0.05 in 17/73 nonempty evaluated states, but the
actual point-traction footprint, dwell, travel, and communication reduced
this to one physically positive pair among those 17; that pair did not beat
its geometry singleton. The public carrier menu covered the high-value
region with geometry in 9/17 cases and with colocated traction in 0/17.
A natural scout-to-carrier geometry-memory case was found in one bottleneck
world: the carrier's best post-delivery action changed from a useful scan
when evidence was withheld to skip when the same packet was retained.
The scout's whole physical acquisition and delivery also reduced realized
team cost in that world (1.7157 to 1.5278). This is a single-world restricted
continuation diagnostic, not a learned-policy or prevalence result.
Details and limitations are in `results/pilot/VALUE_DECOMPOSITION.md`.
The E2 resource freeze remains on hold pending a physically relevant
candidate/operator decision and a fresh validation audit.
The first public acquisition-interface revision now ranks route-relevant,
uncertain regions and proposes geometry and traction together at each chosen
viewpoint. The terrain, risk formula, point-local traction sensor, prices,
caps, and horizon were unchanged. On the disjoint fresh subset of 28 parents
(850012--850039), all 56 scheduled physical states were covered, but the 54
evaluated decisions favored 51 skips, three singletons, and zero tested
pairs. Eight of 51 nonempty value screens had ideal paired value above 0.05;
the preliminary proximity metric marked three of those eight top regions
near traction candidates, but it overcounted a neighboring-cell point probe
as coverage. Actual contact coverage must require the exact target cell;
no top physical pair had value above 0.01. See
`results/pilot/INTERFACE_REVISION.md`. This revision does not pass the
resource-nondegeneracy gate; E2 training remains on hold while the point
probe's inference and physical sensing path are investigated without
changing benchmark prices or terrain semantics.
The next physical decomposition found that travel alone can erase most of
the remaining point-sensor value, and dwell and radio/rendezvous add further
cost. The modality-specific target/pose and public joint-tour revision was
then audited on new parents. Its first execution revealed that scans started
while moving could finish outside the validation radius; this was corrected
by settling before dwell and tested on another disjoint 20-parent range
(890000--890019). The corrected audit covered 40/40 scheduled states but
selected skip in 35, a singleton in three, and a pair in zero of 38 evaluated
decisions. A privileged evaluator-selected high-value region still yielded
no positive physical pair for either agent across eight high-value states,
with or without transmission. Allowing the evaluator to withhold each
acquired reading changes the 38-decision tally only to 32 skip, six
singleton, and zero pair winners; the mean always-skip gap remains 0.0097
(0.59%). The old menu-proximity metric was also
corrected: a neighboring-cell traction probe does **not** cover a target
cell. Details are in `results/pilot/PHYSICAL_DETOUR_ANALYSIS.md` and
`results/pilot/POSE_TOUR_AUDIT.md`. The active-sensing task remains
degenerate under this scripted continuation; no E2 resource freeze or
teacher training follows from these results.
The separate 30-parent menu audit found only 42.7% of candidates within
0.6 m of the actor's observed route and a median 1.18 m viewpoint travel
distance; see `results/pilot/KNOWN_TOPOLOGY_REVISION.md`. This motivates a
candidate-value diagnosis, not an untested menu change.
The subsequent evaluator-only decomposition on 40 already-excluded
development maps sampled 682 route/alternative regions. Ideal paired
information worth more than 0.05 existed in 17/73 nonempty states and
strong all-successful-branch colocated interactions in eight states. Yet
the actual point-probe footprint, dwell, travel, and packets reduced
physically positive pair acquisitions to one of those 17 high-value
regions; its geometry singleton was better. See
`results/pilot/VALUE_DECOMPOSITION.md` for the staged values and caveats.

## Splits, rivals, and primary result

The archived `results/manifests_v2/` records 500 train, 100 validation, 100 calibration,
200 in-distribution test, and 200 OOD parent map groups. Groups fix geometry
and material-patch support; five test realizations vary latent patch severity,
sensor noise, and radio randomness under each parent. Test groups use open,
bottleneck, and shallow labyrinth families. Deep labyrinth and irregular
families occur only in OOD test. The ID splits select equal low/medium/high
shortest-path material-risk strata **within each topology family**, using
cutoffs 0.10 and 0.30 calibrated on excluded generator-audit parents and the
mean over five provisional material realizations. Half of each OOD family is
stratum-matched (the `topology` slice); the other half follows the natural
generator distribution (the `compound` slice). These slices are declared
before training. The original unstratified manifests remain archived in
`results/manifests/` and are not used for E2-v2. The generator file hash, map-group hashes,
and manifest hashes must verify before training and evaluation. **The
known-topology/surface-risk generator revision invalidates the v2 generator
hash and risk strata. The archived v2 manifests must not be used for training
or evaluation. A new manifest version must be generated and verified after
the resource configuration is frozen.**

The predeclared primary rival pool is the Cartesian product of four
acquisition rules (fixed, entropy, singleton decision value, generic recurrent)
and three complete communication rules (broadcast, uncertainty-sparse,
generic recurrent). Every composite receives the same message bytes and
acquisition rights. Select one family/checkpoint by aggregate validation
\(\operatorname{CVaR}_{.95}(C_R)\), among those satisfying the same completion
criterion; break effective ties by expected team cost. Freeze that choice,
all thresholds, and all method checkpoints before opening the test manifests.
Mechanistic comparisons hold communication fixed while changing acquisition,
or acquisition fixed while changing communication.

Train five independent CoPH seeds and five independent seeds for learned
rivals. Each seed is evaluated on 200 test groups × five paired realizations
per primary condition. The estimand is the mean across independently trained
seeds of each seed's empirical test CVaR and success difference. Resample
training seeds and parent map groups hierarchically, preserving the five
realizations and method pairing, and recompute CVaR inside each of 10,000
bootstrap replicates. The claim passes only if:

1. the one-sided 95% lower bound for CoPH minus rival completion exceeds
   −0.02; and
2. conditional on (1), the upper endpoint of the two-sided 95% interval for
   CoPH minus rival \(\operatorname{CVaR}_{.95}(C_R)\) is below zero.

Report absolute and relative CVaR, expected full team cost, success/failure
subclasses, \(\operatorname{CVaR}_{.90/.99}\), per-agent risk, sensing and
radio expenditure, duplicate acquisition with physical marginal-value
labels on audit states, message timeliness, and restricted acquisition regret.
At least 20% lower CVaR remains a stated research target, never a substituted
result. If the confirmatory sample size is not reached, label the run
exploratory.

Save evaluator-only `metadata.json`, `events.jsonl`, `states.npz`, and
`truth_map.npz` for every recorded episode. The offline renderer reads those
artifacts; no rendered truth map enters an actor. Before test evaluation,
preselect the first, middle, and last parent group within each topology
family for qualitative figures. A median paired-effect episode may be added
by a deterministic rule with ties broken by group ID. Do not select episodes
for figures because their visual outcome favors CoPH.

## Before the freeze

Use only separate pilot parent seeds and environment/physical-oracle policies
to audit the starred deadline and budgets. Require feasible completion and
nontrivial inspect/skip choices, complementary measurements, and cases where
broadcasting all acquired evidence exceeds the radio cap. Freeze the resulting
resource/deadline configuration and record its hash **before any E2 model is
trained**. No benchmark condition is selected using CoPH policy performance.

The first optimized pilot used nine excluded parent maps, three each from
open, bottleneck, and shallow-labyrinth families. A planner given the full
map for feasibility testing completed all nine in 380--731 steps. The
observation-only, no-active-sensing planner completed six of nine: it
immobilized twice and missed the 1200-step deadline once. On the six paired
successes, full-map and local-only paths sometimes differ substantially in
material exposure; in one open map the local-only path had *lower* exposure.
These are navigation feasibility controls, not acquisition-policy results.
They show that the tentative deadline admits feasible solutions and missing
information can matter. They do not establish a nondegenerate active-sensing
tradeoff, complementarity, or radio scarcity. All starred defaults therefore
remain provisional.

After replacing Python Dijkstra with SciPy's exact weighted shortest-path
solver, equal-cost path ties resolve differently, so earlier pilot costs are
not pooled with the new planner version. Repeating the same nine excluded
parent maps with the new planner gave 9/9 feasible full-map-planner finishes
(380--700 steps) and 8/9 observation-only finishes; the one local failure
was immobilization. This is still a feasibility control, not evidence for
CoPH acquisition. The planner implementation must be hashed in the final
freeze artifact.

The expanded generator-only audit used 100 excluded parents per family.
Reachable area with risk above 0.5 averaged 0.13--0.15 across families.
The geometric shortest path's mean risk was 0.24 (open), 0.28 (bottleneck),
0.16 (shallow labyrinth), 0.14 (deep labyrinth), and 0.25 (irregular), with
broadly overlapping distributions. Under six predeclared risk weights, maps
had a mean of 2.3--2.9 materially distinct sampled route classes. Risk-aware
path-length inflation was about 0.09 in shallow labyrinth and 0.24 in open
and bottleneck; those figures are route proxies, not control rollouts.
The balanced ID strata address the possible topology-risk prior without
altering the terrain generator or equalizing natural OOD difficulty.
These statistics do not establish active-sensing value.

## Physical resource audit outcome (excluded parents)

The full audit and its scripts are in
`results/pilot/PHYSICAL_RESOURCE_AUDIT.md`. Under the still-provisional 60 s
deadline, 200 excluded parent maps produced 112 successful skip winners, 45
singleton winners, nine pair winners, and 34 maps with no successful set in
the restricted empty/single/pair physical menu (11 sets for 199 parents,
four sets for one parent). Two maps met the strong successful-branch
cost-complementarity criterion. The 4+4 measurement and four-packet caps
physically bound a separate ten-query script on all 12 excluded stress maps;
relaxing the packet cap rescued one deadline failure, but increased cost on
the nine paired successful runs. Two 12-parent delivered-evidence probes did
not identify a carrier repeat measurement that was valuable without memory.

The full E2 resource freeze is therefore **on hold**. Twenty-nine of the 34
uncovered maps are shallow labyrinths. Extending only their empty-set
continuation to 90 s completed 20/29; this is a diagnosis, not a replacement
for the full acquisition audit. A separately predeclared rule—the 95th
percentile of successful full-map completion on 120 new excluded parents,
plus 30 s for a map-width detour and sensing/communication slack, rounded up
to five seconds—gave a provisional **65 s** horizon. The first reference
implementation had seven deadline failures caused by terminal waypoints too
near the edge of the goal disc. After moving the two terminal waypoints
inward, the same 120 excluded full-map references completed 118/120; the
successful-time 95th percentile was 34.45 s and the rule still yielded 65 s.
Two shallow-labyrinth references immobilized. The earlier result is archived,
not pooled with the corrected continuation. The quantile rule and full-map
feasibility do not by themselves show that restricted acquisition choices
pass the fresh coverage gate.
The 65 s, 4+4, and four-packet starred values remain provisional. No E2
acquisition model may be trained on the final manifests as though these
settings were frozen. A fresh full acquisition audit and a nondegenerate
memory opportunity are required before freeze.

The fresh 65 s check uses 90 new excluded parents, two public decision times
(steps 120 and 360), eight candidates per agent (16 joint), every singleton, and eight
stratified pairs. A scheduled state counts covered if a tested branch
completes, or if the prefix already completed; early prefix failure counts
uncovered. The predeclared feasibility gate is at least 95% coverage overall
and at least 90% for each ID topology family. This is a restricted physical
continuation gate, not an exact decentralized-policy guarantee. Candidate
class frequencies, complementarity, and cap usage are reported without
forcing arbitrary class balance.
The environment-only revision, corrected full-map feasibility result, cap
stress, and memory-search caveats are recorded in
`results/pilot/PHYSICAL_RESOURCE_REVISION.md`.

The completed fresh shared-menu audit on parents 820000--820089 did **not**
pass this gate: 154/180 scheduled states were covered (85.6%), with 96.7%
open, 95.0% bottleneck, and 65.0% shallow-labyrinth coverage. Its 175
evaluated states had 80 skip, 70 singleton, and two pair winners; 23 had no
successful tested set. The memory pilot found physically selectable
Region-1-to-Region-2 acquisition switches, but it cannot override this
feasibility failure. The 65 s horizon and resource prices remain provisional;
no E2 resource freeze or final-manifest teacher training is authorized by
this audit. A global information-limited continuation revision and a fresh
excluded-parent audit are required.

For the 65 s candidate, the maximum successful nonfailure decision score is
conservatively bounded by 7.3864: time at most 2.6, two pH forces of norm at
most four contribute at most 2.08, eight probes at most 0.32, four maximum
footprint evidence packets and four ACKs at most 0.3864, and normalized risk
at most two. This is below the terminal failure charge of 10, which remains
provisional but needs no increase for the 65 s candidate. This bound assumes
the existing four-packet and 4+4 measurement caps.
