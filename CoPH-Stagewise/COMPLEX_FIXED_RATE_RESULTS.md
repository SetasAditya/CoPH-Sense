# Fixed-frequency cooperation in complex static environments

**The fixed-rate learning loop is implemented, but robust navigation is not complete,
and these tests do not demonstrate a useful cooperation benefit.** The earlier
simple-scene results established learning/update consistency and a small replay
benefit. Obstacles reveal limitations that communication does not resolve.

## Evaluation

Four new layouts have smooth risk fields, locally sensed circular obstacles, and
geometrically feasible routes: hazard slalom, blocked risk passage, asymmetric
coverage, and a 20-unit obstacle corridor. A grid connectivity check verifies route
existence without supplying routes or hidden geometry to the learner.

Six arms share each scene/seed: communication off; one-way 0→1; one-way 1→0;
reciprocal evidence-only; reciprocal learning with supported-boundary replay; and
reciprocal learning with fixed eight-step replay. Opportunities occur every **0.96
seconds**. The manifest freezes a **2,000,000-byte ceiling per agent** to accommodate
long missions; per-byte prices, learning rules, sensing, and penalties are unchanged.
This is a larger capacity than the earlier suite, not a reduction of communication
cost. Episode horizons scale with goal distance before execution: 300 steps for
10-unit scenes and 600 for the long corridor. Prediction-based rates remain disabled.

**52 tests pass. All 240 episodes completed:** 48 smoke and 192 development. Smoke
repeats the first two of eight development seeds per scene. Reserved validation/test
namespaces remain untouched. Evaluation and rendering took about 124 seconds with
16 workers on this machine. Source/configuration fingerprints and full logs are saved.

## Outcomes

No method completed a two-agent mission in any development scene. Individual
successes occurred, but **375 of 384 agent missions failed** across all six arms.
Failures are retained, including every failed episode's static figure.

| Development arm | Team successes / episodes | Agent collisions | Agent timeouts | Mean team mission cost |
|---|---:|---:|---:|---:|
| Communication off, boundary replay | 0 / 32 | 61 | 1 | 91.016574 |
| Reciprocal, boundary replay | 0 / 32 | 61 | 1 | 91.016036 |
| Reciprocal, evidence-only | 0 / 32 | 61 | 3 | 93.634240 |
| Reciprocal, eight-step replay | 0 / 32 | 61 | 2 | 92.312692 |

Reciprocal communication changes mean mission cost by only **−0.000538** relative
to communication off, with identical collision/timeout counts. That is not a
practical navigation benefit. Boundary replay performs better than the replay
ablations on some asymmetric cases, but does not solve the obstacle failures.
The report retains both one-way directions and every per-scene paired difference.

The long layout spans more than eight nominal stages, but agents collide before
visiting beyond stage 2. Therefore it supplies **no evidence of successful long-range
navigation or sustained eviction behavior**. Early termination is not hidden by
showing the intended route or extending the recorded trajectory.

## What breaks

A read-only audit traces collision obstacles to recorded sensor observations and
reconstructs the last force from the final model. Force decomposition is used only
when it matches the recorded pre-clipping command; mismatches remain explicitly
marked in `collision_audit.json`.

1. **Stage-local obstacle storage can discard relevant sensed geometry.**
   `Learner.observe` retains an obstacle only when its center is inside the stage
   rectangle. In the blocked-passage reciprocal arm, all 16 collision obstacles
   were observed before impact, but **12 were absent from the final active-stage
   obstacle set**. An obstacle surface can intersect the agent's path while its
   center is outside the stage window. Center containment is not an adequate rule
   for collision-relevant geometry.
2. **The smooth obstacle penalty does not enforce collision avoidance.**
   In the reciprocal slalom cases, the reconstructed mean goal-attraction norm at
   impact is **6.82**, versus **0.86** for obstacle repulsion (16 matched cases).
   In the long corridor the corresponding norms are **16.64** and **0.81** (16
   matched cases). The mission-goal quadratic grows with distance. Recorded controls
   continue toward obstacles despite prior sensing; positive barrier coefficients
   and learning-update feasibility do not constitute a clearance guarantee.

These observations do not require changing a penalty or selectively removing failed
benchmarks. The learner, controller, and communication logic were left unchanged
for the evaluation. The present plants also have no inter-agent contact model or
joint mechanical planner; sharing information is the implemented cooperation.

The next implementation step is to preserve sensed obstacles whose inflated geometry
intersects the active/predicted domain and to add an explicit collision-avoidance
constraint accounting for velocity and stopping distance. Its applied control must
be identical in execution and replay, with physical work accounted for. Then rerun
this frozen suite before claiming a cooperation benefit. More communication alone
cannot supply that missing behavior.

## Visualizations

Each GIF synchronizes communication off, reciprocal evidence-only, and full
reciprocal learning. The fourth panel displays the most recent logged field/stage
snapshot and retained counterfactual trajectories, with its snapshot time labeled.
Snapshots can lag execution; they are not silently reconstructed as current facts.
Blue and green identify agents; stars are goals; gray disks are obstacles; dotted
outlines include the agent radius; red shading is evaluator-only true risk; purple
crosses identify recently sent anchors. Finished trajectories are held at their
recorded endpoints. All GIF seeds were chosen in the manifest before execution.

### Hazard slalom

![Hazard slalom](results/complex_fixed_v1/development/figures/hazard_slalom.gif)

### Blocked risk passage

![Blocked risk passage](results/complex_fixed_v1/development/figures/blocked_risk_passage.gif)

### Asymmetric coverage

![Asymmetric coverage](results/complex_fixed_v1/development/figures/asymmetric_coverage.gif)

### Long obstacle corridor

![Long obstacle corridor](results/complex_fixed_v1/development/figures/long_obstacle_corridor.gif)

[All six-arm comparisons and metrics](results/complex_fixed_v1/development/REPORT.md)
· [Every failed episode](results/complex_fixed_v1/development/figures/failures/index.html)
· [Collision audit](results/complex_fixed_v1/collision_audit.json)
· [Force comparison](results/complex_fixed_v1/development/figures/collision_forces.png)
· [Manifest](results/complex_fixed_v1/manifest.json)
· [Checks](results/complex_fixed_v1/numerical_checks.txt)
· [Artifact verification](results/complex_fixed_v1/verification.json)

## Reproduction

Run from the repository root, using a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=CoPH-Stagewise \
  python -m coph_stagewise.cli check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl-cache \
  PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.complex_environments \
  --out CoPH-Stagewise/results/complex_fixed_reproduction --workers 16
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl-cache \
  PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.complex_collision_audit \
  CoPH-Stagewise/results/complex_fixed_reproduction
```

The collision audit is separate postprocessing with its own archived source and
fingerprint. Existing implementations, defaults, and previous results are preserved.
