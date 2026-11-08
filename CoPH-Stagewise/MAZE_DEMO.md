# Cooperative maze: sensed closures and individual Hamiltonians

This opt-in demonstration scales the terrain example to a **14-junction, 19-connection maze** with three initially unknown physical gates. Both agents share the surveyed corridor layout, start at the same point, and have the same goal and learner. A0 starts at 0 s; A1 starts at 5 s. “Scout” and “carrier” identify their initial roles: either agent can discover a later obstruction and warn the other.

The main animation shows **one reciprocal-communication run**, rather than side-by-side communication-off/on worlds. Communication-off runs remain a backend check of what information changed behavior.

## What changed

- Surveyed corridors form a bidirectional graph with real free-space boundaries. Hidden gate locations are accessible only to simulator sensing, collision checks, and evaluator graphics. The graph planner receives no gate truth. A detected obstacle that intersects a surveyed centerline makes that graph connection unavailable; this is conservative route-network planning, not a proof that every conceivable continuous detour within that corridor is impossible.
- Actual gates span a corridor. Each is represented by three overlapping radius-0.34 disks, consistent with the existing obstacle model. Walls use the union of radius-0.85 corridor capsules. Swept motion outside that free-space footprint is a collision. The controller checks held-input motion and stopping clearance against the same public corridor footprint, in addition to existing obstacle protection.
- Versioned geometry extensions carry the locally sensed disk, sensor footprint, observation time, and sender-owned evidence ID alongside the existing H/state/environment packet. Receivers validate these fields, preserve physical state, retain received geometry separately, and remove blocked graph connections. Received geometry is never re-exported as independently sensed evidence.
- The geometry stores are bounded at 32 local and 32 remote disks; overflow latches braking. Reports expire. Environmental observations and validation records remain local measurements. Receiving geometry does not certify remote environmental replay support.
- Maze stages move horizontally **or vertically**, including backtracking. Adjacent stage centers shift by 70% of the selected window dimension, preserving nominal 30% overlap. Current plus two previous stages are retained. Replay uses the matching spatial handoff rule and frozen route/geometry context.
- Virtual-start connections and graph edges use consistent risk-integral costs. Waypoints lie on a fixed 0.1 m grid; the existing scalar potential, learned dynamics, clipping, projection, guards, reconciliation and rollback remain active. The route planner chooses paths; it does not directly apply steering forces.

## Visualization

The large panel displays world geometry, both actual trajectories, current route plans, waypoints, and packet deliveries. Gray gate sections are evaluator truth not yet sensed by either agent; red sections have been sensed. This distinction does not grant either learner access to the evaluator map.

The two Hamiltonian panels evaluate the **exact logged control-time model**, at fixed logged momentum:

$$
\widehat H_i(q,p_i(t))=\tfrac12\|p_i(t)\|^2+\widehat V_i(q;w_i(t)).
$$

Each panel identifies the stage, model version, and logging time. Color scales are shared. Unsupported or wall regions are gray. White arrows show the conservative field $-\nabla_q\widehat V_i$ at 0.25 m per N. Colored arrows show attraction, full soft shaping, controller damping feedback ($-1.4p$), and applied control at 1 m per N. Soft shaping includes the spatial taper and environmental/barrier terms; it is not necessarily purely repulsive.

The lower panels show force magnitudes, shaping Hamiltonian versus physical kinetic energy, and timestamped discovery/report events. Changes in shaping energy are not actuator work. Hard corridor constraints are applied separately from the scalar landscape. No model values are reconstructed retrospectively or interpolated between model versions.

## Reproduction and checks

From `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.maze_demo --out results/cooperative_maze_reproduction --workers 2
```

The frozen development configuration uses seeds 75000 and 75001, communication disabled and reciprocal fixed-rate, a 180 s active horizon per agent, and four million bytes per agent. Static packet lifetime is 300 s. Seed 75000 is selected for the main reciprocal animation before execution. These are new development conditions, not the previous frozen validation benchmark.

92 regression tests passed, including six maze checks: geometry provenance and non-echoing, warning-driven route changes before local sensing, real wall/gate collisions, spatial stage transitions, execution/replay agreement, finite-difference gradients, and landscape agreement with the actual scalar potential, and exact feasible-nominal projection equivalence. Archived routed-validation source-selection gates are not applicable to modified source files and remain preserved separately.

[Run report](results/cooperative_maze_v2/REPORT.md) · [Cooperative animation](results/cooperative_maze_v2/figures/cooperative_maze.gif)

Execution sources are frozen before the v2 run. Rendering-only memory refinements and the first-warning still-frame selection are separately fingerprinted under `render_sources/`; the archived override reproduces the delivered visuals. The v1 batch terminated with exit status 143 before saving complete episodes and is marked incomplete. V2 adds bounded caching of unchanged graph costs, an exact fast path when the nominal input already satisfies all projection constraints, and progress/resource logging. No scene parameters, horizons, costs or control decisions were intentionally changed by these computation improvements.

This milestone remains static and uses independent plants. Inter-agent collision avoidance, unsurveyed-map exploration, moving obstructions, and adaptive communication remain outside this demonstration. Remote-only geometry affects routing and protection, not receiver damping or task-goal ownership. Preserve failures and incomplete episodes rather than treating an informative animation as proof of navigation success.

The post-run `scripts/verify_maze_artifacts.sh` check validates packet geometry against actual sensor records, model/version alignment, pairing, byte and flow budgets, physical energy residuals, and GIF frame counts. It also writes a fixed-state shaping-jump ledger from successive exact logged models. The ledger separates target changes from model/geometry changes by an explicit telescoping decomposition; these are not actuator work or uniquely attributable single-packet effects.

This experiment demonstrates geometry sharing and the resulting route/waypoint revision. It does not independently establish that counterfactual policy optimization improves maze navigation over a frozen policy. Some reported closures can be redundant or irrelevant to the selected route; they are retained rather than forced to produce an effect.
