# Sensed-geometry routing and collision protection

This opt-in controller extends the cooperative Hamiltonian learner. Communication remains fixed-rate. Existing commands default to `controller_mode="legacy"`; the new suite explicitly selects `legacy`, `geometry`, `protected`, or `routed`.

| Mode | Geometry retention | Control projection and guards | Observed-space routing |
|---|---|---|---|
| `legacy` | Original center containment | Original force clipping | None |
| `geometry` | Inflated disk/rectangle intersection | Emergency brake on geometry overflow only | None |
| `protected` | Repaired | Enabled | None |
| `routed` | Repaired | Enabled | Enabled |

## Geometry and support

Each sensed obstacle is a disk inflated by the 0.12 agent radius and 0.05 clearance. Intersection with a retained stage rectangle, rather than center containment, determines stage membership. The independent geometry store also retains obstacles near the current stopping region. Its default cap is 256 deduplicated disks. Overflow latches an explicit braking condition for the mission: dropping excess disks does not silently restore a nominal safety claim.

Sensor footprints retain center, radius and original timestamp, with at most 120 per stage. Their union certifies observed geometry under the existing static, complete obstacle sensor. These records do **not** expand the environmental-field support used by reconstruction and replay. Frozen stage models, hypothetical branch copies and schema-2 receiver summaries carry geometry, footprints and routing state. Legacy schema-1 summaries remain loadable for the legacy controller; they cannot substitute for missing routing context in a new-controller evaluation.

## Scalar potential and routing

The routed controller selects a waypoint $w$ using deterministic eight-connected A* on a 0.1-unit grid inside retained stage rectangles and observed free space. Edges forbid diagonal corner cutting and intersecting inflated disks. Node erosion by 0.075 units certifies each short grid edge within the union of sensing disks; source and waypoint connectors use exact segment coverage. Grid ordering resolves remaining ties.

The route objective sums edge lengths multiplied by $1+4\bar r$, where $\bar r$ is the mean endpoint risk. Supported reconstructed field values provide risk; unsupported risk is explicitly assigned the fixed planning value 1. It is never stored as a sensor measurement. The target is the reachable goal, or the reachable observed point nearest the goal. Replanning occurs at update intervals, when a route segment is obstructed, or after 0.25-unit path deviation. A fixed grid waypoint up to one unit ahead advances within 0.2 units. No route causes braking.

The nominal control remains the gradient of a scalar potential:

$$
V(q)=\tfrac12\|q-w\|^2+\theta_r\widehat r(q)+\theta_b B(q),\qquad
u_0=-\nabla_q V(q)-1.4v.
$$

The mission goal and 0.25-unit success threshold do not change. Waypoint changes contribute the fixed-state jump

$$
\Delta V_{\mathrm{route}}=\tfrac12\|q-w^+\|^2-\tfrac12\|q-w^-\|^2,
$$

separate from physical work, field/model changes and stage changes. Routing alone does not remove every local minimum in this risk-shaped potential.

## Projection and held-input checks

For each obstacle define $z=q-c$ and $h=\|z\|^2-(r+0.17)^2$. The two-dimensional projection minimizes $\|u-\nu_0\|^2$ subject to $\|u\|\le3$, predicted endpoint speed at most 1, and the gain-2 second-order barrier condition at both damping endpoints:

$$
2z^T u\ge \max_{d\in\{d_{\min},d_{\max}\}}2d z^T v
 -2\|v\|^2-8z^T v-4h.
$$

Exact damped flow gives $v^+=e^{-d\Delta t}v+(1-e^{-d\Delta t})u/d$, so each endpoint speed constraint is a disk in control space. Deterministic PyTorch enumeration considers interior, line, circle, and pairwise boundary candidates. Derivatives are preserved within a fixed active set; route switches, feasibility changes and active-set changes are nonsmooth.

At estimated damping and both fitting endpoints, the held-input guard checks swept segment clearance, endpoint speed and a conservative stopping ball of radius $\|v^+\|^2/(2u_{\max})$. Failed projection or guard triggers maximum opposing braking, reduced near zero velocity to avoid reversal. A failing braking guard records `infeasible=True`; this is not a safety certificate. Unseen obstacles and out-of-range dynamics remain outside the claim.

Execution, retained replay, boundary pilots and packet continuations call the same controller. Each hypothetical branch owns its routing state. Diagnostic force evaluations restore routing state afterward. Plant fitting uses the executed input.

## Work and energy accounting

The 1,024-flow update ceiling includes physical replay transitions, boundary pilots, guard flow evaluations, gradient evaluation, candidate validation and diagnostic controls. Protected replay reserves up to seven flow evaluations per transition (three candidate guards, three fallback guards, one transition); unused reservations are not reported as executed work. Routing expansions and projection candidates are separate counters. Measurements, hypothetical observations and evaluator truth retain their existing separation.

Logs record raw nominal, clipped nominal, projected and executed controls. Clipping work is $(u_{clip}-u_0)^T\Delta q$; safety-modification work is $(u_{exec}-u_{clip})^T\Delta q$. Exact physical energy balance uses $u_{exec}$. Neither quantity absorbs route or model energy jumps.

## Reproduction

From `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl \
  python -m coph_stagewise.navigation_evaluation \
  --out results/navigation_fix_v3 --workers 32
bash scripts/render_navigation_ablations.sh results/navigation_fix_v3
```

The supplementary rendering command adds the four-controller comparison GIFs and controlled-case GIFs from existing logs. Its separate source fingerprint is recorded in `rendering_manifest.json`; it does not change the frozen simulation.

Use a fresh directory for a fresh run. `--resume` requires the frozen source fingerprints to match; completed episodes are retained. `--render-only` regenerates reports and GIFs from saved logs. `--reference` defaults to `results/complex_fixed_v1`; its manifest supplies unchanged scenes, seeds, horizons, communication configuration and reserved namespaces.

Smoke: four modes × communication off/reciprocal × four scenes × two seeds (64 runs). Development: routed across six original arms (192 runs), plus geometry/protected with off/reciprocal (128 runs). Old legacy development results remain separate references. All failed episodes receive a static gallery entry. GIFs use seed 61000 chosen before execution, common timestamps and original world scales; retained-model snapshot times are labeled. Legacy routing/constraint logs do not exist and are not reconstructed.

See [the results report](NAVIGATION_FIX_RESULTS.md). No inter-agent collision avoidance, global map, learned scheduling or predictor retraining is included.
