# Shallow-labyrinth navigation root-cause diagnostic

**Status:** diagnostic only. No terrain, deadline, sensor, candidate, budget,
cost, controller, or E2 training setting was changed. The E2 resource freeze
remains on hold.

## Paired protocol

Thirty already-excluded shallow-labyrinth parents (seeds 820002, 820005,
..., 820089) were replayed from identical initial states and random seeds.
This uses the same parents as the failed resource audit, so it is a
**development diagnosis**, not fresh confirmatory evaluation. Each rollout
used the 65 s provisional horizon, no active acquisition, and the same
fixed pH controller unless the run explicitly states otherwise.

| Run | Planning/control geometry | Material | Replanning | Executor | Success |
| --- | --- | --- | --- | --- | ---: |
| A | Local observations | Local observations | 20 steps (1 s) | pH | 16/30 |
| B | Full occupancy map | Local observations | 20 steps | pH | **29/30** |
| C | Local observations | Full traction map | 20 steps | pH | 22/30 |
| D | Full occupancy map | Full traction map | 20 steps | pH | **30/30** |
| E | Local observations | Local observations | Every step (0.05 s) | pH | 19/30 |
| F | Local observations | Local observations | Every step | VMAS velocity-PD diagnostic | 0/30 |
| G | Full occupancy map | Full traction map | Every step | VMAS velocity-PD diagnostic | 0/30 |

The privileged inputs in B--D and G are applied only inside diagnostic
rollouts. B reveals occupancy to the planner and pH local execution while
leaving appearance/traction locally observed. C reveals true traction but
keeps occupancy local. The map, physics, seeds, action limits, and clock
remain paired. F/G are an *uncalibrated* velocity-PD tracker, **not** a
perfect tracker or a valid structural pH comparison. Since G fails even
with full information, F/G cannot identify a pH disadvantage; they should
not enter E2 claims or the freeze decision.

Of A's 14 failures, 13 were deadlines and one was immobilization. B rescued
13 of them with **no A-success regressions**. C rescued seven but lost one
A-success map; D rescued all 14. E rescued six but lost three A-success maps.
B's only failure was a deadline; D completed that map. Thus material
information has some effect, but **revealing static geometry explains most
of the navigation gap**. More frequent replanning alone does not close it.

## Failure instrumentation

The same diagnostic logged newly revealed occupied cells, invalidation of
the current planned path by those cells, pH shield interventions, waypoint
changes, progress, and travel. In the final 10 s, A failures versus A
successes averaged:

| Quantity (both agents combined) | A failures (n=14) | A successes (n=16) |
| --- | ---: | ---: |
| Newly discovered obstacle cells | 19.36 | 7.12 |
| Current-path invalidations | 4.00 | 0.69 |
| Shield interventions | 5.93 | 0.25 |
| Forward progress, m | 2.49 | 4.10 |
| Distance travelled, m | 6.38 | 6.25 |

The failed policies still move, but make less goalward progress while
discovering and replanning around more walls. These descriptive contrasts
are not independent of episode duration or terminal location; the paired
A-to-B intervention is the stronger causal evidence. The one A
immobilization had 70 shield interventions in its final window. In the
failed resource audit, acquisition detours produced many additional
immobilized branches, so the 13/14 deadline split above does not replace
that audit's failure accounting.

As a clearance check, we computed free-cell distance to the nearest
occupied cell, minus a 0.215 m carrier-and-safety allowance. The fraction
of free cells below that margin averaged 0.1262 in A-failure maps versus
0.1238 in A-success maps. The sampled full-information path's minimum
clearance was 0.185 m in both groups. These crude whole-map/path summaries
do **not** isolate a narrow-corridor geometry defect. The current evidence
instead points to late wall discovery and invalidated global paths, with
local shield stalls as a consequence in some episodes.

## Decision

The diagnostic supports the concern that E2-v1 is substantially an
unknown-maze exploration problem. Full geometry restores 29/30 successful
pH continuations without changing the pH controller, whereas a 20-fold
increase in replanning frequency yields only 19/30. This is evidence
against treating the failure as solely a short planning cadence or an
intrinsic pH execution defect.

**Do not edit the benchmark or freeze resources from this result alone.**
The scientifically cleaner next design decision is whether E2-v1 should
expose static obstacle topology while retaining uncertain material,
traction, and local traversability, or instead commit to building and
validating an explicit unknown-geometry exploration backbone. Whichever is
chosen requires a globally applied specification change and a new audit on
fresh excluded parents before E2 training.

Raw paired records: `navigation_root_cause_labyrinth30_v1.json`.
Free-cell clearance records: `labyrinth_clearance_excluded30_v1.json`.
Reproduction: `PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph
CoPh-sense/external/envs/phmarl-py38/bin/python -m
coph_terrain.navigation_root_cause --count 30 --workers 12`.
