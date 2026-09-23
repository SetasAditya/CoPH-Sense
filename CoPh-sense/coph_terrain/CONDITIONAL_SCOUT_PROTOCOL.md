# Conditional Scout Protocol

The geometric protocol, controller, feature representation, generated sensing
sites, and original tests were contributed in `CoPH-Sense-yiwang` and ported
here with attribution. CoPH-Sense adds the paired benchmark report, explicit
analytic/physical separation, recovery-complete carrier semantics, and the
material-pH integration.

This experiment changes the mission semantics from two continuously active
mission agents to **carrier-primary navigation with an optional scout**.

The carrier is always active.  The scout stays at its safe home/staging pose
unless the carrier-side acquisition value exceeds the complete dispatch cost.
The causal sequence is

```text
carrier belief -> NEED/dispatch decision -> scout travel -> acquisition
-> report -> carrier belief update -> carrier replan / pH potential reshaping
-> scout return
```

## Objective

For a candidate scout task `d`, compare the complete paired continuations

`Q_idle(b)` and `Q_dispatch(b,d)`.

Dispatch only if

`Q_idle - Q_dispatch > margin`.

The dispatch branch includes scout travel, sensing dwell/cost, radio cost and
delay, carrier waiting or lost progress, material exposure, and downstream
navigation.  This is an **ex-ante** decision and is therefore different from
the historical A4 post-reading SEND/HOLD decision, where sensing cost is
already sunk.

## Mission semantics

Set `TerrainConfig(carrier_primary=True)`.  Only carrier goal attainment is
required for terminal success.  Scout motion, sensing, communication, collision
and material exposure remain in the team ledger.  A scout immobilization is
logged but does not by itself make carrier delivery fail; collisions remain
team safety failures.

## Exact admission benchmark

Run:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_benchmark
```

The suite contains:

1. **Static hidden block, useful scout.**  The carrier cannot visit the remote
   observation site before route commitment; the scout can query it in parallel.
2. **Locally known state, no need.**  The same route variable is already known by
   the carrier, so dispatch should remain off.
3. **Static hidden block, too late.**  Remote information is useful in principle
   but arrives after the route commitment deadline.
4. **Moving blocker prediction.**  A remote dynamic object crosses or clears the
   future route.  The useful quantity is predicted occupancy at the carrier's
   arrival time, not current obstacle position.
5. **Moving traversal-cost pulse.**  A cost/risk pulse advects across one route;
   the scout observation predicts future route cost at arrival.
6. **Stale dynamic report.**  The same dynamic information would help if timely,
   but the report arrives after commitment, so the correct action is no dispatch.

The dynamic cases are intentionally time-dependent because they test the
interaction between observation age, communication latency, and controllability.
They should be used as admission/regression cases before introducing moving
objects into the full VMAS terrain benchmark.

## Physical reference controller

`conditional_scout.py` contains:

- `ScoutTask`: one request with modality, viewpoint and latest-useful step;
- `ConditionalScoutController`: `IDLE -> DISPATCH -> ACQUIRE -> REPORT -> RETURN`;
- `scout_dispatch_teacher`: paired full-continuation IDLE/DISPATCH teacher;
- `run_carrier_primary_episode`: observation-only carrier planner plus optional
  scout reference execution.

The scout controller is deliberately deterministic at first.  The first learned
quantity should be the carrier-side dispatch value.  Only after that gate is
validated should scout route selection or carrier pH shaping be jointly trained.

## Required comparisons

Report at least:

- carrier only;
- two agents with communication disabled;
- always-active scout;
- uncertainty-triggered scout;
- downstream-value-triggered conditional scout;
- centralized/full-information diagnostic upper reference.

Separate the parallel-access value from the communication value:

`J_solo - J_two_no_comm` and `J_two_no_comm - J_two_comm`.

Do not require nonzero communication on every episode.  A good selective policy
should have low false-dispatch rate on no-need and too-late worlds while retaining
high dispatch recall on cooperation-required worlds.

## Learned dispatch and long-horizon visualization

The exact admission suite is not a training run.  For an end-to-end
training/testing/visualization diagnostic, use the two additional modules:

- `conditional_scout_learning.py`: randomizes the exact static/dynamic cases,
  trains an MLP to predict the **ex-ante scout value**
  `J_idle - J_dispatch`, and evaluates selective dispatch on held-out cases;
- `conditional_scout_long_horizon.py`: turns one admitted case into an explicit
  point-navigation rollout with carrier/scout motion and writes PNG/GIF traces.

Train the dispatch-value model:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_learning train \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_learning \
  --n-train 10000 --n-val 2000 --epochs 80
```

Held-out test:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_learning eval \
  --ckpt CoPh-sense/coph_terrain/results/conditional_scout_learning/dispatch_value_best.pt \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_learning/test \
  --n-test 4000
```

Visualize a long-horizon oracle-conditional moving-blocker rollout:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_long_horizon \
  --case moving_blocker_prediction --policy oracle \
  --max-steps 900 --frame-stride 8 \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_rollout/oracle
```

Visualize the learned policy:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_long_horizon \
  --case moving_blocker_prediction --policy learned \
  --ckpt CoPh-sense/coph_terrain/results/conditional_scout_learning/dispatch_value_best.pt \
  --max-steps 900 --frame-stride 8 \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_rollout/learned
```

Useful policies are `never`, `always`, `oracle`, and `learned`.  Useful cases
include `static_hidden_block_useful`, `static_local_known_no_need`,
`static_hidden_block_too_late`, `moving_blocker_prediction`,
`moving_cost_prediction`, and `moving_blocker_stale_report`.

Each long-horizon output folder contains:

- `metrics.json`: dispatch, timing, selected route and path costs;
- `trace.npz`: carrier/scout trajectories and dynamic hazard state;
- `trajectory.png`: complete spatial paths;
- `trajectory.gif`: animated carrier/scout/hazard motion;
- `timeseries.png`: goal distance with report and commitment times.

A convenience wrapper runs training, held-out testing, and several visual
comparisons:

```bash
bash run_conditional_scout_demo.sh
```

For a shorter smoke run:

```bash
EPOCHS=10 NTRAIN=1500 NVAL=400 NTEST=500 FRAME_STRIDE=20 \
  bash run_conditional_scout_demo.sh
```

The long-horizon visualizer is a point-navigation diagnostic.  It is deliberately
separate from the final VMAS E2 environment so that scout timing/value can be
validated before adding dynamic VMAS landmarks and pH energy reshaping.

## Geometric long-horizon revision (v3)

The long-horizon diagnostic now treats the scout as a deployable resource of
the carrier rather than an independently staged robot:

- the scout starts docked immediately next to the carrier;
- if dispatch is rejected, it remains docked and follows the carrier;
- if dispatch is accepted, it launches from the carrier, navigates to the
  remote sensing site, dwells, reports, and returns to the **moving carrier**;
- the visualization keeps running after carrier goal arrival until a dispatched
  scout is recovered (subject to `--max-steps`).

The world is no longer geometrically empty.  Both agents obey an inflated-disk
safety constraint around five fixed obstacles, a moving background patrol, and
case-specific hidden/dynamic blockers.  The controller uses a repulsive shaping
term before contact and a hard feasibility projection at the safety boundary.
The output logs minimum clearance, avoidance-active steps, and hard-safety
interventions for both carrier and scout.

To force a particular hidden dynamic mode in a GIF, use `--mode-name`.  For
example, visualize the moving blocker crossing the upper corridor:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_long_horizon \
  --case moving_blocker_prediction --mode-name crossing --policy oracle \
  --max-steps 1200 --frame-stride 8 \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_rollout/crossing
```

And the same object moving away so that the scout can release the shorter top
route:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_long_horizon \
  --case moving_blocker_prediction --mode-name clearing --policy oracle \
  --max-steps 1200 --frame-stride 8 \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_rollout/clearing
```

Each rollout also writes `world_geometry.json`.  In `metrics.json`, check
`min_carrier_clearance`, `min_scout_clearance`, `carrier_avoidance_steps`,
`scout_avoidance_steps`, `scout_recovered_at_end`, and
`final_carrier_scout_distance` in addition to dispatch/timing metrics.


## Forward generated sensing target (v4)

The orange sensing marker is no longer a fixed world coordinate.  A carrier
first moves with the scout docked.  At the request epoch the rollout computes a
new scout task from the carrier state and current intended route:

\[
q_s^{\rm target}
=\mathcal G(q_c,v_c,\rho_c,T_{\rm commit}-t),
\]

where \(\rho_c\) is the current route hypothesis.  The generator projects the
carrier onto the route polyline, looks forward by the maximum distance that the
scout can still visit while preserving dwell, communication, and a report-time
margin, offsets the target to a safe vantage side of the route, and rejects
candidates with poor obstacle clearance.  Thus the generated target must lie in
front of the carrier motion at dispatch rather than at a predeclared orange
landmark.

The visualization now records and draws:

- the carrier position at the instant it requests scouting;
- the generated forward sensing target;
- the carrier-to-target scout task geometry;
- the scout-to-carrier report link;
- the no-scout/prior route and, when the report changes the decision, the route
  selected after the report.

For the clearest causal demonstration use the moving-blocker `clearing` mode:

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.conditional_scout_long_horizon \
  --case moving_blocker_prediction --mode-name clearing --policy oracle \
  --max-steps 1200 --frame-stride 8 \
  --outdir CoPh-sense/coph_terrain/results/conditional_scout_rollout/forward_clearing
```

In `metrics.json`, inspect `request_time`, `request_carrier_position`,
`generated_scout_site`, `scout_site_is_forward`,
`route_changed_due_to_report`, `carrier_wait_time_for_scout`, and
`route_cost_improvement_from_report`.  In the clearing-mode diagnostic the
conservative no-scout route is the bottom route, while a timely scout report can
release the shorter/lower-cost top route.
