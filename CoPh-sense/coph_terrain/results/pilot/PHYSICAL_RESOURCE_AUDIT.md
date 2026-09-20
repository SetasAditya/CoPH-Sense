# E2 physical resource and nondegeneracy audit

Status: **completed as an audit; full E2 resource freeze is on hold.** No E2
acquisition model was trained or selected using these parents.

## Scope and method

The main sample is 200 excluded parent maps (seeds 671000–671199), cycling
through open, bottleneck, and shallow-labyrinth topologies. At physical step
120, the observation-only pH rollout offers four public candidate actions:
geometry and traction at one scout viewpoint, and geometry and traction at one
carrier viewpoint. The evaluator rolls out the empty set, all available
singletons, and all available pairs. In 199 parents, all four candidates and
11 sets were available; one parent had only two candidates and four sets,
for 2,193 branches total. Travel to the viewpoint, dwell, packet delivery, ACK,
motion, failure, and material exposure are charged by the VMAS environment.
All branches of a parent share its realized world and random seed. A winner
must complete; failure branches are never called low-cost optima.

The score here is the *realized-world* `mission_cost + material_exposure` under
one scripted continuation. It is a resource sanity check, **not** the
belief-conditional expected-cost-plus-CVaR objective, a global decentralized
optimum, or a demonstration of learned-policy performance. In particular,
positive pair interactions in this audit do not establish that a learned set
critic identifies them.

## Main physical result

| Topology | Parents | Skip best | Singleton best | Pair best | No successful set |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open | 67 | 58 | 7 | 0 | 2 |
| Bottleneck | 67 | 36 | 24 | 4 | 3 |
| Shallow labyrinth | 66 | 18 | 14 | 5 | 29 |
| **Total** | **200** | **112** | **45** | **9** | **34** |

Across the 147 parents where skip completed, choosing the best successful
set improved realized score by 0.0724 on average. Of 133 parents with at
least one four-corner rectangle in which every branch completed, 78 had a
positive pair interaction contrast above 0.05; there were 168 such
rectangles in total. Two parents (671091, 671149) met the stronger
complementarity test: both singleton costs were no better than skip within
0.01, while the pair improved cost by more than 0.05. Among the nine
pair-optimal parents, five have all four relevant branches successful. Do
not confuse pair-only success rescue with this successful-branch cost test.

The principal weakness is that 34/200 parents have no successful action in
this 11-set menu. Twenty-nine are shallow-labyrinth maps, predominantly
deadline failures (253 of 319 failed branches), with the rest immobilizing.
Thus the present 60 s scripted continuation does not reliably cover that
topology; those failures are retained in all summaries.

As a diagnosis, the same empty-set continuation was extended to 90 s on
those 34 uncovered parents. It completed 22/34, including 20/29 shallow
labyrinth parents and 2/3 bottleneck parents. This is evidence that the
60 s deadline explains much of the missing coverage. It is **not** an
acquisition comparison at 90 s, so no 90 s score or winner class is inferred.

## Budget and memory interventions

A separate 12-parent forced-query physical stress test (seeds
680000–680011) proposed five public geometry viewpoints per agent. With
the declared 4+4 measurement and four-team-packet caps, all 12 scripts
reached both caps. Relaxing the packet cap allowed eight sends on all 12;
relaxing measurements allowed ten measurements. The packet-relaxed branch
rescued one bottleneck deadline failure. On the nine parents where both
packet settings succeeded, the extra four sends **increased** score by
0.219 on average (packet-relaxed minus declared). The caps therefore bind
physically, and selective sharing matters; this forced script does not
prove that an optimal policy would exhaust either budget. The corrected
continuation stops attempting measurements once an agent's cap is reached.

The paired memory probe physically acquired and delivered scout geometry,
then evaluated carrier skip versus repeating that region with the carrier's
delivered evidence retained or cleared, keeping physical state and sunk
cost fixed. It used 12 near-start and 12 farther-viewpoint excluded parents.
All 24 delivered evidence that included cells previously unknown to the
carrier. Nineteen had four successful paired continuations. In **none** of
those 19 was the repeat action beneficial even with memory cleared. Hence
these E2 viewpoints do **not** yet supply a nondegenerate memory-induced
redundancy test. This does not invalidate the earlier finite/moving A5
bridge; it limits this E2 audit.

## Decision and limits

The audit supports feasible skip, singleton, and pair decisions, some genuine
cost complementarity, and physical measurement/radio scarcity. It does
**not** justify freezing the full E2 benchmark: shallow-labyrinth coverage is
poor and the tested E2 memory counterfactual is degenerate. A 90 s horizon
is a plausible documented revision, but it needs a fresh full resource audit
before adoption. Choose a carrier repeat opportunity with
positive no-memory physical marginal value *before* retraining or claiming
an E2 memory effect. Any revised resource configuration needs a written
reason and a fresh excluded-parent check; these seeds are development data.

Artifacts:

- `resource_audit_excluded200_v1.json`: all 2,193 physical branches.
- `cap_stress_audit_excluded12_corrected_v2.json`: declared/relaxed cap rollouts.
- `memory_resource_audit_excluded12_v2.json`: near-start memory probe.
- `memory_resource_audit_rank2_excluded12_v1.json`: farther-viewpoint probe.
- `deadline_probe_no_success_90s_v1.json`: focused deadline diagnosis (see its
  summary; it does not rerun all sensing sets).

The final 1,100 train/validation/calibration/test map groups remain disjoint;
the manifest verifier passed. The 22 E2 unit tests passed after the audit
code changes. The legacy cap-stress v1 artifact used an over-cap continuation
that continued targeting its fifth query; use the corrected v2 artifact for
conclusions.
