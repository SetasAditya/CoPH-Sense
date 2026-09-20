# Modality-specific pose and joint-tour audit

**Status: E2 resource freeze remains on hold.** The terrain/layout generator,
latent fields, risk formula, passive appearance, 65 s horizon, sensing/radio
prices, and resource caps were unchanged. The public acquisition interface
now represents an information **target region** separately from the physical
sensing pose. Geometry chooses a reachable public-line-of-sight pose within
its scan range; traction still requires contact with the target cell. For a
common-region pair, a public-cost planner searches feasible geometry poses
and both visit orders through the contact pose and mission goal. The critic's
candidate features now include both target and pose coordinates (12 inputs).
No E2 critic was trained or checkpoint reused across this feature change.

The physical continuation also exposed an execution error. It previously
started the scan as soon as the robot entered the request radius while still
moving. Since sensor validity is checked **at completion**, many scans
finished outside that radius. The corrected continuation waits until it is
within half the request radius and moving at at most 0.12 m/s, holds the
current pose through dwell, then resumes the route immediately. The 0.12
m/s threshold is an executor-start condition, not a change to the sensor
range, dwell duration, price, or plant. A regression test reproduces the
prior moving-scan failure and confirms both physical measurements cover
their advertised target.

Two earlier runs are retained only as implementation diagnostics:

| Run | Evaluated | Skip | Singleton | Pair | Limitation |
| --- | ---: | ---: | ---: | ---: | --- |
| 860000--860039, separate poses/order | 77 | 68 | 9 | 0 | Before joint pose optimization and settling |
| 880000--880019, joint tour | 37 | 32 | 5 | 0 | Before settling; many scans missed the target |

The **fresh corrected audit** uses disjoint parents 890000--890019 at steps
120 and 360. It tests all legal singletons, all same-region G/T pairs, and
the existing stratified cross-agent pairs under restricted scripted pH
continuations. All 40 scheduled states have a successful tested branch.
Among 38 evaluated decisions, **35 select skip, three a singleton, and zero
a pair**. Only two singleton improvements exceed 0.01; the third is 0.0009.
Mean skip cost is 1.6393 versus 1.6318 for the best successful tested set,
a mean gap of only **0.0074 (0.45%)** on these realized worlds. Thus an
always-skip policy remains competitive under this restricted reference.
Among successful same-region pair runs, 193/198 physically acquired both
advertised target readings. The corrected sensing executor therefore makes
the measurement semantics credible, but sensing remains too rare for the
main active-acquisition benchmark. Artifact:
`settled_joint_pose_fresh20_physical_v1.json`.

The separate fresh value funnel finds ideal pair value above 0.05 in 8/37
screened decisions and strong ideal interaction in two. Five of those eight
retain positive value under the actual point-sensor footprint before
physical costs. Exact footprint coverage of the top ideal target is three
geometry candidates and **zero contact-probe candidates**. None of the
old common-cell full physical pairs is positive. An actual-menu pose
recheck confirms three executable geometry observations, no traction
observation, and no positive geometry acquisition at those top targets.
Artifacts: `settled_joint_pose_fresh20_value_funnel_v1.json` and
`settled_joint_pose_fresh20_top_recheck_v1.json`.

To distinguish candidate omission from acquisition cost, an evaluator chose
each high-ideal-value target but used only public, feasible poses and the
same pH execution. Across eight targets, **zero carrier G/T pairs and zero
scout G/T pairs** improved team cost, whether evidence was sent or held.
This privileged-region result cannot be deployed and does not prove that
no other region is useful. It does show that simply inserting these omitted
top regions into the menu would not resolve the current physical-cost
bottleneck. Artifacts: `settled_joint_pose_fresh20_oracle_region_v1.json` and
`settled_joint_pose_fresh20_oracle_region_scout_v1.json`.

The matched no-send audit evaluates every tested acquisition set with and
without transmission. Giving the evaluator the cheaper branch changes the
38-decision tally to **32 skip, six singleton, zero pair**; five winners
withhold evidence and one sends it. Only three improvements exceed 0.01.
Mean skip cost remains 1.6393 versus 1.6296 for the best selective branch,
a **0.0097 (0.59%)** gap. Avoiding unnecessary broadcast helps, but does
not make pair acquisition useful or make always-skip clearly inferior.
This is a restricted evaluator reference, not the learned A4 policy.
Artifact: `settled_joint_pose_fresh20_selective_share_v1.json`.

The pre-pose physical decomposition is in `PHYSICAL_DETOUR_ANALYSIS.md`.
Together the audits indicate that the next controlled change should address
**when and along which mission path sensing occurs**, rather than enlarging
the point probe, changing prices, altering terrain, or oversampling test
episodes. A posterior may be useful later, but it cannot by itself remove
the measured detour and dwell opportunity cost.
