# Terrain information value decomposition (development audit)

**Status:** no terrain, prices, passive noise, caps, horizon, or manifests
changed. These are evaluator-only, single-realization diagnostics on already
excluded development parents 840000--840039 at steps 120 and 360. They are
not belief-expected \(Q^*\), a learned-policy result, or a new resource
freeze. An oracle reveal changes only the carrier's information state;
later route and pH motion still execute in the same hidden world.

## A physical-continuation bug found and repaired

The first physical tier sent an agent **straight toward the sensing cell**
instead of using the public occupancy planner. On maps with intervening
walls, this falsely labeled a reachable viewpoint as costly or impossible;
some episodes failed before acquiring evidence. The shared scripted
continuation now routes to sensing and radio destinations using only the
actor's public occupancy and local belief. The pre-repair physical results
remain archived in `value_decomposition_excluded40_v1.json` but must not be
used as physical value estimates. Ideal cost-free reveals in that file are
unaffected. A separate corrected physical artifact is
`physical_value_recheck_route_to_viewpoint_v2.json`.

The correction is consequential but does not by itself create a pair result.
On a matched 40-parent resource subset, the old physical continuation had
72 skip and five singleton winners among 77 evaluated states; the corrected
one has **69 skip, eight singleton, and zero pair winners**. Both cover all
80 scheduled states including three early successful prefixes. These are
paired development outcomes, not the final E2 resource distribution.

The matched twelve-parent forced-resource script was also rerun after the
repair. It reached both measurement caps and the four-packet cap on **all
12** parents; relaxing either cap caused extra physical use on all 12.
Extra packets made none of the twelve paired successful runs cheaper. This
confirms physical scarcity for the forced script, not value for an optimal
information policy. Artifact:
`cap_stress_audit_known_topology_route_repair_same12_v2.json`.

## Initial route-region screen and cost stages

The first screen sampled 314 public route, alternative-route, and
divergence regions over 73 evaluated states (seven other scheduled states
completed before the decision). It compared empty, surface, traction, and
colocated surface+traction reveals. The *ideal* traction reveal covers a
1 m region; the deployed traction probe measures **one point**. This
difference makes the ideal tier an information upper bound, not an executable
measurement. The surface scan is already noiseless within its footprint, so
the ideal and travel-free actual surface readings coincide. Negative values
can occur for a realized hidden world under the fixed heuristic continuation.

| Diagnostic over 73 states | Count |
| --- | ---: |
| Any sampled region with ideal \(V_G>0.05\) | 10 |
| Any sampled region with ideal \(V_T>0.05\) | 10 |
| Any sampled region with ideal \(V_{GT}>0.05\) | **13** |
| Ideal \(V_{GT}>0.01\) | 36 |
| Colocated \(S_{GT}>0.05\), \(V_{GT}>0.05\), all four branches successful | five regions in four states |

At the **highest ideal-value region** in each state, the median pair value
relative to skip falls through the staged audit:

| Stage | Median \(Q(\varnothing)-Q(G,T)\) | States above +0.01 |
| --- | ---: | ---: |
| Ideal 1 m truth reveal, no acquisition cost | +0.0096 | 36/73 |
| Actual surface footprint + point traction, no travel/dwell | +0.0011 | 15/73 |
| Same readings after a carrier hold/dwell and explicit sensor fees | −0.0697 | 4/73 |
| Corrected physical route, dwell, sensing, and packet protocol | **−0.152** | **1/73** |

The dwell tier issues a hold-position pH target while the scout continues;
it is a diagnostic intervention rather than a perfectly frozen carrier.
The corrected physical pair continuation succeeds in all 73 states, starts
measurements in 72, and delivers at least one packet in 70. The large
physical penalty is therefore no longer an artifact of generally unreachable
viewpoints. It reflects mainly the gap between small downstream decision
benefits and the time/motion/sensing/communication needed to obtain them.

The one state with a >0.01 corrected physical pair benefit was seed 840032,
step 120. Its highest ideal-value region was absent from the public menu;
the menu nevertheless contained another useful singleton in the paired
resource rerun. This is evidence of a **candidate-placement limitation**,
but also a warning against treating ideal top-region coverage as the sole
measure of a menu's usefulness.

## Candidate relevance and passive appearance

Among the 31 sampled region instances with ideal pair value above 0.05,
five were covered by a public surface candidate and **none** by a public
traction candidate under the current eight-action carrier menu. Among the
13 states' highest ideal-value regions above 0.05, four had a surface
candidate and none had a traction candidate. These are footprint-overlap
checks; an alternative menu action can still be useful.

An independent 30,000-free-cell descriptive audit used twenty equal-width
bins for appearance and each active property. Its empirical within-bin
risk variance was 0.04763 without conditioning, 0.02562 given appearance,
0.01474 given appearance+surface, and 0.01057 given appearance+traction.
Risk is deterministic given exact surface+traction. The binned statistics
ignore spatial dependence and are **not a calibrated belief posterior**.
They indicate that appearance has not removed all pixel-level uncertainty;
the physical rollout shows that much of the remaining uncertainty has small
acquisition value at the present decision states.

## Cooperative-memory implication

The scout had a shorter public path-time proxy than the carrier to the
highest ideal-value region in 54/73 states, but only six of the 13 states
with ideal pair value above 0.05. Under the corrected physical scout
continuation, all six tested scout acquisitions completed and delivered;
only **one** scout traction probe improved team cost (by 0.036). At that
same state, the carrier's own traction acquisition had negative value.
Thus the *initial six-region screen* did not identify a state where the
carrier would profitably remeasure a region the scout can first report.
The denser screen below did identify a natural candidate, audited under
"Value-conditioned memory case". The initial negative search remains a
useful warning against manufacturing an M0/M1 contrast by forcing a probe.

## Interpretation and limits

The task contains some causal information value and a few strong *ideal*
colocated interactions. The main suppressors in this screen are the actual
point-probe footprint, small return improvements relative to dwell/physical
cost, and incomplete candidate coverage of high-value regions. This does
**not** prove that no useful pair exists elsewhere: the initial region set
is a screening grid, and all values use one realized world and a fixed
scripted continuation. A denser cost-free screen is recorded separately
before changing any environment parameter.

Artifacts: `value_decomposition_excluded40_v1.json`,
`physical_value_recheck_route_to_viewpoint_v2.json`,
`rich_resource_audit_route_repair_same40_v2.json`,
`passive_appearance_variance_excluded40_v1.json`, and
`scout_access_value_candidates6_v1.json`.

## Denser public-route screen and decisive table

The broader screen used up to twelve regions per state, adding
appearance-uncertain route patches and more samples along alternative and
diverging routes. It tested 682 regions across 77 evaluated scheduled states;
four of those states had no region passing the public route-spacing rule, so
the value rates below use **73 nonempty states**. This screen still does not
exhaust every cell or every future information policy.

| Quantity | Result |
| --- | ---: |
| States with a sampled ideal pair reveal of positive value | 67/73 |
| States with ideal \(V_G>0.05\) somewhere | 11/73 |
| States with ideal \(V_T>0.05\) somewhere | 12/73 |
| States with ideal \(V_{GT}>0.05\) somewhere | **17/73** |
| Colocated ideal \(S_{GT}>0.05\) and \(V_{GT}>0.05\), all branches successful | 12 regions in eight states |
| High-value top regions covered by the carrier's public surface menu | 9/17 |
| High-value top regions covered by a colocated public traction probe | **0/17** |
| Top high-value regions with actual point-probe pair value >0.01 before dwell/travel | 10/17 |
| Top high-value regions with pair value >0.01 after hold/dwell and sensor fees | 3/17 |
| Top high-value regions with corrected physical pair value >0.01 | **1/17** |

For these 17 ideal high-value regions, median pair value was +0.081 under
the 1 m perfect reveal, +0.032 with actual sensing footprints but no time or
charge, −0.065 after the hold/dwell and sensor fees, and **−0.171** after
physical acquisition and communication. All 17 physical pair continuations
completed. The one pair-positive state (840037, step 120) still preferred
its geometry singleton: +0.206 for geometry versus +0.107 for the pair.
Thus the ideal interaction is real in the sampled model, but it is not yet
a **physically preferred set acquisition**.

For a coarse decision-relevance check, we compared the planned route's
crossing at three fixed map columns, requiring a change of at least three
cells (0.6 m) to count. Ideal paired evidence changed this corridor
signature in 10/17 high-ideal-value states, versus 7/56 other states.
This avoids treating every one-cell path difference as a new route, though
it still does not measure executed route value by itself. Artifact:
`route_choice_change_dense40_v1.json`.

These results separate three issues. The public candidate menu omits
colocated traction probes at the ideal high-value regions. The actual
traction probe's one-cell footprint is much weaker than the ideal 1 m
region reveal. Finally, modest information gains are often outweighed by
dwell/travel and communication costs even after route-to-viewpoint planning
is corrected. Passive appearance is informative but does not eliminate
pixel-level risk uncertainty; the bottleneck is the conversion of that
uncertainty into timely, decision-changing *physical* evidence. No sensing
price, noise level, field correlation, or manifest was changed in this audit.

Dense-screen artifacts: `dense_information_screen_excluded40_v1.json` and
`dense_top_cost_recheck_excluded40_v1.json`.

## Value-conditioned memory case

The denser screen found a scout-accessible carrier-valued geometry region
at cell (23, 26) in excluded bottleneck parent 840037, step 120. The scout
physically acquired geometry and delivered its evidence at step 143. We then
cloned that *same* physical delivery state. In M0, the carrier's delivered
surface cells were restored to their prior unknown values; in M1, they were
retained. Positions, clock, incurred sensing/radio costs, hidden world,
appearance, and remaining protocol were identical. The carrier's public
eight-action menu was unchanged by the intervention.

| Post-delivery state | Skip cost | Best menu cost | Best action |
| --- | ---: | ---: | --- |
| M0, evidence withheld | 1.7479 | **1.5820** | Carrier geometry at (28, 30) |
| M1, evidence retained | **1.4998** | 1.4998 | Skip |

In a separate whole-continuation comparison from the same step-120 state,
skipping acquisition costs 1.7157 while the scout's physical geometry
acquisition and delivery costs **1.5278**; both succeed, and the latter
includes one measurement and one delivered packet. The 0.1879 reduction is
a single realized-world effect under the restricted scripted continuation,
not an expected-policy estimate.

The scout's footprint and the carrier's otherwise useful surface-scan
footprint overlap substantially (approximately 45/81 cells under the
viewpoint-centered footprint comparison). This is a *natural, restricted
post-delivery* memory-induced acquisition switch: a measurement that helps
without the message is no longer worth buying after the message. The
whole-continuation comparison also shows a net physical benefit in this
one world. Neither check establishes prevalence or a learned A5 policy
result. Artifact:
`value_conditioned_memory_840037_v1.json`.
