# E2 known-topology revision and pre-freeze outcome

**Decision:** static obstacle occupancy is public; local surface
support/roughness and traction remain latent. This was changed before E2
training after the paired hidden-topology root-cause diagnosis. The previous
hidden-topology resource and memory artifacts are archived, not pooled with
this revision. The 65 s horizon, prices, and budgets remain provisional.

**Later continuation correction:** the acquisition branches reported below
used a straight-line detour to viewpoints. A public route-to-viewpoint
repair changed some winners on a matched 40-parent subset. Treat the
resource-class and post-delivery memory counts below as **pre-repair
diagnostics**, not the current freeze evidence. The corrected result and
value decomposition are in `VALUE_DECOMPOSITION.md`.

The new geometry scan measures a high-fidelity local surface field, not wall
locations. Traction probes retain their previous meaning. Passive appearance
is a noisy proxy for the joint terrain risk

\[
r(x)=\operatorname{clip}[0.4g(x)+0.4t(x)+0.2g(x)t(x),0,1],\quad
t(x)=\operatorname{clip}[(0.55-\mu(x))/0.40,0,1].
\]

Surface condition also changes the plant traction port to
\(\mu(x)(1-0.2g(x))\). Both fields are evaluator-only until locally observed
or delivered. The public occupancy map is used consistently by the planner,
pH obstacle potential, and collision shield. The critic's map encoder now
has separate masked channels for public occupancy, appearance, measured
surface, and measured traction. The old v2 manifests are invalidated by the
generator/risk change; no E2 weights have been trained.

## Stage 1: fresh navigation sufficiency

The predeclared no-active-acquisition pH check used 90 new excluded parents
(830000--830089), 30 per ID family, at the same provisional 65 s horizon.
It required at least 95% success overall and 90% per family. Results:

| Family | Completed |
| --- | ---: |
| Open | 30/30 |
| Bottleneck | 30/30 |
| Shallow labyrinth | 29/30 |
| **Overall** | **89/90** |

The guard **passes**. The prior topology-specific navigation collapse is
removed on this fresh sample without changing the pH controller or horizon.
Artifact: `navigation_sufficiency_known_topology_fresh90_v1.json`.

## Stage 2: physical resource and information checks

On another 90 fresh excluded parents (840000--840089), two scheduled
decision states per parent used the actual eight-per-agent public menu,
every singleton, and eight stratified pairs. The restricted physical
continuation covered **180/180** scheduled states (100% in each ID family),
passing the predeclared 95% overall/90% family feasibility gate. Six states
terminated successfully before their scheduled decision. Among the other
174 states, the successful single-world best set was:

| Set class | States |
| --- | ---: |
| Skip | 161 |
| Singleton | 13 |
| Tested pair | **0** |

Nine singleton winners were in bottlenecks, three in shallow labyrinths,
and one in open maps. The mean singleton advantage over skip on those 13
states was 0.083 in the audit's single-world score; five exceeded 0.05.
Seven tested pairs improved skip by more than 0.01, but none beat the best
singleton or met the stronger successful-branch complementarity criterion.
This is a restricted candidate/continuation audit, **not** a claim that no
complementary pair exists in the full action space or under belief-expected
value. It shows that the declared eight sampled pairs do not establish the
paper's set-value mechanism at the audited decision states.

Artifact: `rich_resource_audit_known_topology_fresh90_v1.json`.

A separate forced ten-query script on 12 fresh excluded parents
(850000--850011) reached the four-packet cap in 12/12 and both agents'
measurement caps in 11/12. Relaxing either cap led to more physical usage
on 12/12. This establishes that the caps bind for a feasible stress script;
it says nothing about whether an optimal policy should exhaust them.
Artifact: `cap_stress_audit_known_topology_fresh12_v1.json`.

In a paired scout-to-carrier memory pilot on 24 further excluded parents,
19 packets were physically delivered and 19 carrier menus had successful
choices in both retained- and withheld-evidence states. **No no-memory
best acquisition overlapped the scout's reported surface region**, so there
was no repeat-to-new-region or repeat-to-skip reallocation. Two best actions
changed, both from skip to a different surface scan after delivery. This
pilot does not establish the intended A5 mechanism under the revised terrain
semantics. It is an existence/diagnostic search, not a prevalence estimate.
Artifact: `memory_menu_audit_known_topology_fresh24_v1.json`.

A separate menu-only check on 30 more excluded parents (870000--870029)
found a mean of 7.68 candidates per agent-state. Only 42.7% of offered
actions were within 0.6 m of that agent's observed planned route; among
the first four slots, the fraction was 36.8%. The median travel distance to
a viewpoint was 1.18 m. Median high-fidelity surface unknown fraction was
1.0 because no scans had yet occurred, so that attribute cannot distinguish
the initial candidates. These are geometry/relevance descriptors, not
physical value estimates. They make the public corridor anchors and
route-alternative placement a concrete next diagnostic target; they do not
prove that candidate placement alone caused the skip-heavy result.
Artifact: `candidate_relevance_known_topology_fresh30_v1.json`.

## Freeze decision

**Do not freeze E2 or train the critic.** Navigation and physical resource
binding now pass, but the information decision distribution is skip-heavy,
tested pair winners are absent, and the paired memory intervention is not
the desired sensing reallocation. The next work should inspect candidate
placement and the downstream value of measured surface/traction *before*
any global change to prices, passive noise, or terrain correlations. Any
revision must apply to all methods, be documented before a fresh excluded
audit, and be followed by new split manifests; the archived v2 risk strata
and hashes are no longer valid.
