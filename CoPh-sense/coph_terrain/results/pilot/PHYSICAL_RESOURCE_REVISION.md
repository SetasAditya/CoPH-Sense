# E2 resource revision after the initial hold

Status: **65 s is a provisional environment-only candidate; no full E2
resource freeze or E2 model training has occurred.** This note separates
diagnostics performed with the old terminal waypoint from those using the
corrected physical continuation.

## Deadline and accounting

Before the new full-map sample, the rule was fixed as

\[
T_{\max}=5\left\lceil
\frac{q_{.95}(T_{\rm full\ map,success})+30\,\mathrm{s}}{5\,\mathrm{s}}
\right\rceil\mathrm{s}.
\]

The 30 s fixed slack permits approximately one 12 m map-width detour at
carrier speed (12/0.55 ≈ 22 s), up to eight sensing dwells, and delivery.
On 120 excluded parent maps (700000–700119), the corrected full-map pH
reference completed 118/120; its successful-time 95th percentile was
34.45 s, so the rule gives **65 s (1,300 steps)**. The two remaining
failures were shallow-labyrinth immobilizations. The uncorrected reference
completed 111/120; seven apparent deadline failures were caused by terminal
waypoints too close to the shared goal disc's edge. The corrected terminal
targets are separated and lie well inside the disc. The old reference
artifact remains archived and is not pooled with the corrected one.

For every declared horizon \(H\), the environment now computes

\[
C_R=\frac{1}{H}\sum_{t<H}(r_s(t)+r_c(t))\in[0,2].
\]

Thus the [0,2] return-quantile support remains valid at 65 s. With
\(H=1300\), the maximum successful nonfailure score is conservatively
bounded by 2.6 time + 2.08 motion + 0.32 sensing + 0.3864 packet/ACK +
2 risk = **7.3864**, below the terminal failure charge of 10. This bound
uses the force norm cap of four, the 4+4 measurement cap, four packet
attempts, and a 19×19 upper bound on a geometry-scan footprint. It is
a cost-accounting bound, not a controller success guarantee.

## Fresh cap stress at the 65 s configuration

On 12 new excluded parents (740000–740011), a forced ten-query physical
script reached both the 4+4 measurement caps and the four-packet cap on
every map. Relaxing packets from four to ten allowed eight transmissions
in all 12 and rescued one deadline failure (6/12 versus 7/12 completion).
On the six paired successful parents, extra packets increased score by
0.205 on average. Relaxing measurements allowed ten acquisitions on 11/12;
the other run ended before completing all ten. These are **forced-script
resource checks**, not evidence that an optimal information policy would
exhaust either budget.

After adding shared task-corridor candidates, the same first-five-query
stress script reached the packet cap on 7/12 and both measurement caps on
6/12 further excluded maps (810000–810011). The public corridor candidates
appear earlier in the proposal order and can consume more travel time before
the fifth query. A new **nearest-five public viewpoints** script on fresh
parents (811000–811011) reached both measurement caps and the four-packet
cap on all 12. Relaxing packets allowed extra transmissions on all 12 and
raised score by 0.181 on the ten paired successful parents; it did not
rescue a failure in that sample. The lower saturation in the previous
first-five run was script ordering rather than an unreachable resource cap.
Neither script is a policy-value estimate.

## Memory search

The forced-repeat diagnostic tried carrier-route viewpoints at ranks
1, 3, 5, and 7 on excluded moving-carrier trajectories. Many measurements
were physically delivered, and some forced repeats changed value when
the packet was withheld. A candidate-availability check showed that most
such repeat actions were absent from the carrier's public menu by delivery
time. In the rank-7 sample, one of 25 fully successful paired states had
the repeat in both 16-candidate public menus, and none had the desired
publicly executable repeat-to-skip flip. Forced repeats outside the menu
do not establish the E2 deployed-policy claim.

An independent 120-parent public-menu audit exposed the cause: with the
route-only proposal list, just 2/120 parents offered a common initial
geometry viewpoint to both agents (0/40 bottleneck and 0/40 labyrinth).
The candidate generator was therefore revised **globally**, without changing
terrain: it reserves up to two viewpoints along the public start-to-goal
task corridor, then fills remaining slots from each agent's own observed
route. It uses only the public goal, observable team kinematics, and each
agent's observed reachability. On 120 further excluded parents, both agents
had a common public geometry viewpoint in all 120 (40/40 in each topology).
All acquisition baselines will receive this same revised candidate menu.
Overlap establishes executability of a shared sensing choice, not its value.

A paired whole-menu check used the same physical state and budget, erased
only packet-borne carrier knowledge, and evaluated the carrier's public
acquisitions plus skip. With four candidates, 23/24 states received the
packet, 19 had successful choices under both memory conditions, and two
changed their best action. One strong case (parent 721002) changed from
a geometry scan overlapping scout evidence, worth 0.276 relative to skip
without memory, to skip with memory. The second change had a 0.005 margin.
Neither showed reallocation to a different useful region in that
four-candidate menu. With eight candidates on the same pilot parents, six
of 19 comparable states changed their best choice and five moved from a
scan overlapping scout evidence to skip or another choice. The source was
still selected outside the scout's public action menu, so this is an
information-state diagnostic, not a locally executable policy result.
The new shared-menu run selects a source in **both** initial public menus;
these pilot seeds are development data, not confirmatory prevalence estimates.

With the revised shared menu, 21/24 scripted scout acquisitions physically
delivered evidence and 19 states had successful carrier choices in both
memory conditions. Six of those 19 changed their best public action. Parent
791011 is a clean Region-1 to Region-2 switch: with evidence withheld,
geometry at Region 1 costs 1.599 versus 1.622 at Region 2; after delivery,
Region 2 costs 1.622 versus 1.668 at Region 1. All four branches succeed.
Parent 791004 switches from repeating geometry in the already reported
region to a traction acquisition. These examples establish the physical,
locally selectable memory mechanism at pilot scope. They do not establish
held-out prevalence or a learned policy advantage.
Both public source anchors selected in this 24-state experiment were verified
to be in **both** agents' deployed eight-candidate menus on all 24 initial
states; the scout action is not supplied by a privileged carrier-only menu.

## Freeze decision

The deadline rule, risk normalization, failure-cost bound, and caps are
implemented. The corrected-terminal **route-only** audit on 90 fresh parents
and two decision states covered 87.8% of scheduled states: 98.3% open,
93.3% bottleneck, and 71.7% shallow labyrinth. Among 169 evaluated states,
skip won 95, a singleton 41, and a pair 16; 17 had no successful tested set.
Two bottleneck decision states met the stronger all-successful-branch
complementarity test: neither singleton improved on skip within 0.01, while
a pair reduced cost by more than 0.05. This confirms nondegenerate pair
value exists, but the route-only task still fails feasibility.
It failed the predeclared feasibility gate, so its mixed acquisition choices
do not justify a freeze. Before the revised shared-menu audit, the gate was
set to at least 95% scheduled-state coverage overall and 90% per ID topology;
early prefix failures count uncovered. A full E2 freeze requires that gate
**and** a natural public-menu memory opportunity.

The final shared-menu audit used 90 *different* excluded parents
(820000--820089), two scheduled states per parent (steps 120 and 360), and
up to eight public candidates per agent: empty set, every singleton, and
eight stratified pairs. Of 180 scheduled states, 175 reached the decision
and five terminated earlier (two successes, three failures). Among the 175
evaluated states, skip won 80, a singleton 70, a pair two, and 23 had no
successful tested set. The resulting coverage was **85.6% overall**, with
**96.7% open, 95.0% bottleneck, and 65.0% shallow labyrinth**. It therefore
**fails both the overall and per-family predeclared gates**. The 21
no-success labyrinth states contained 309 immobilized and 216 deadline
failed branches; the two no-success bottleneck states contained seven
immobilized and 27 deadline failed branches. These branch counts are not
independent maps or policy replications.

Two pair-optimal states and the physically selectable memory switches show
that acquisition and team memory can matter, but pair winners are rare in
this restricted audit. The memory search is an existence check on pilot
states, not a prevalence estimate. The failed coverage is decisive:
**do not freeze the 65 s horizon/caps/prices or train E2 models yet**. The
next global revision must address the information-limited scripted/pH
continuation in shallow labyrinths, then repeat the same declared gate on
another fresh excluded sample. Increasing the deadline alone is not
justified by these mixed immobilization and deadline failures. Do not tune
individual maps or CoPH weights against these pilots.

Artifacts: `deadline_calibration_excluded120_terminal_v2.json`,
`cap_stress_audit_fresh12_65s_terminal_v1.json`,
`cap_stress_audit_fresh12_65s_sharedmenu_v2.json`,
`candidate_overlap_shared120_v1.json`,
`candidate_overlap_joint16_shared120_v2.json`,
`rich_resource_audit_fresh90_65s_terminal_v2.json`,
`memory_menu_audit_excluded24_terminal65_v1.json`,
`memory_menu_audit_excluded24_terminal65_v2.json`,
`memory_menu_audit_shared_excluded24_terminal65_v3.json`,
`cap_stress_audit_fresh12_65s_sharedmenu_distance_v3.json`, and
`rich_resource_audit_fresh90_65s_joint16_v4.json`.
