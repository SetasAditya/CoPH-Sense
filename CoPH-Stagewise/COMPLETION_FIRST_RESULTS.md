# Completion-first cooperative navigation: results

Implemented and evaluated 32 smoke and 128 development episodes. Development gate: **FAIL**. Held-out validation executed: **False**. Prior results and final-test namespaces remain untouched.

| Controller | Communication | Previous goals / 64 | Progress goals / 64 | Collisions | Timeouts |
|---|---|---|---|---|---|
| protected | disabled | 56 | 48 | 0 | 16 |
| protected | reciprocal_fixed | 56 | 48 | 0 | 16 |
| routed | disabled | 1 | 64 | 0 | 0 |
| routed | reciprocal_fixed | 1 | 64 | 0 | 0 |

The potential, replay, energy accounting and receiver-summary changes passed 77 numerical/contract tests. Controlled disk and passage cases completed with both frozen policies and learning enabled; strong-risk goal capture also passed. Recorded checks cover all eight old near-goal failures and the old waypoint-cancellation state.

The completion gate is intentionally stricter than numerical correctness. The fixed-target radial bound cannot guarantee progress after safety projection and stopping guards. The failed-agent records in [analysis.json](results/completion_first_v1/analysis.json) expose the final nominal/projected/applied controls and stopping-clearance margins. A projection-feasible command can still fail the subsequent stopping guard, leading to repeated braking. These failures are not counted as successful navigation.

Reciprocal causal audits used identical complete receiver states and isolated deliver/withhold continuations at 8, 16 and 32 steps. Both directions changed the initial reconstructed field and control; operational states and measurements remained untouched. The observed revisions are:

- Receiver 0: field coefficient change 0.147288; applied-control change 4.51043e-06.
- Receiver 1: field coefficient change 0.108182; applied-control change 3.41652e-06.

All failed development agents are in hazard slalom. Protected-progress repairs the previous asymmetric-coverage goal-capture failures but introduces repeated braking near slalom obstacles. Routed-progress completes all four scenes, including stage 9 in the long corridor; it records no braking or stagnation replans, and only 0.24 seconds of near-stationary time per episode on average (including startup).

This establishes an active information-to-model-to-control mechanism, not a demonstrated completion advantage from communication. Compare communication off and reciprocal within the same controller; gains over old routed control arise from the common navigation fix. Mission costs remain secondary diagnostics.

The 1,024-flow ceiling includes replay, pilots, safety guards and update diagnostic controls. Routing and projection work are logged separately; actual compute and bytes differ across arms. No penalties, horizons, safety margins, goal thresholds or communication rates were tuned.

Maximum recorded update work was 758/1,024 flow evaluations; the largest physical energy residual was 9.76e-15. Four synchronized 32-frame GIFs and all 16 failed development episodes are preserved. Full evaluation and rendering took about 13.2 minutes with 32 workers. The protected-controller failure blocks the overall working-version claim and the predeclared 64 held-out episodes.

[Frozen manifest and fingerprints](results/completion_first_v1/manifest.json) · [Source archive](results/completion_first_v1/sources.zip) · [Controlled completion figure](results/completion_first_v1/controlled_completion.png)

[Development table and synchronized GIFs](results/completion_first_v1/development/REPORT.md) · [Complete failure gallery](results/completion_first_v1/development/figures/failures/index.html) · [Implementation and reproduction](COMPLETION_FIRST.md)
