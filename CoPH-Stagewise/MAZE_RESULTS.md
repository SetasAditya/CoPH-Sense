# Cooperative maze results

[Cooperative animation](results/cooperative_maze_v2/figures/cooperative_maze.gif) · [First-warning still](results/cooperative_maze_v2/figures/first_warning.png) · [Implementation and reproduction](MAZE_DEMO.md)

The scene has 14 junctions, 19 corridor connections, and three initially unknown physical gates. Both agents know the surveyed corridor layout. The second agent starts five seconds later from the same point. The main animation shows one cooperative run with both agents’ local Hamiltonian landscapes, force components, plans, and discovery/report events.

**All 8/8 agent missions completed, with zero collisions and zero timeouts**, across two development seeds and communication disabled/reciprocal fixed-rate. These are development demonstrations, not held-out maze validation.

| Quantity | Communication disabled | Reciprocal fixed-rate |
|---|---:|---:|
| Scout goal time, both seeds | 72.84 s | 72.84 s |
| Carrier goal time, both seeds | 78.20 s | 72.56 s |
| Carrier enters first blocked approach | Yes | No |
| Carrier enters third blocked approach | Yes | Yes, before the warning arrives |
| Scout / carrier stage transitions | 14 / 14 | 14 / 12 |

The first warning demonstrates the requested mechanism: the scout senses G1 at **11.88 s** and the carrier receives the report at **12.56 s**. The carrier avoids the blocked branch before locally sensing that gate at **31.40 s**, from a different part of the maze. Without communication, it enters the approach at 16.28 s and senses the gate at 17.00 s.

The later warnings are less uniformly useful. For G3, the carrier has already entered the approach at **47.72 s** when the warning arrives at **48.08 s**. It still learns before its own sensing, but this is not a pre-junction avoidance success. G2 is not entered from the tested approach by either arm; its reports are not counted as a demonstrated avoidance benefit. The visualization retains all these events.

Approach entry is measured using the fixed strip `gate_x - 1.8 < x < gate_x` and `abs(y - gate_y) < 0.5`, recorded in `branch_entry_checks.json`. This describes entry into the surveyed centerline approach, not collision or entry into the obstacle itself.

The carrier finishes **5.64 s earlier** with communication in both seeds. The result demonstrates geometry-information sharing and route-target revision within the PH controller. It does not isolate an advantage from counterfactual policy learning, prove all messages useful, or establish general superiority in unknown maps.

92 tests passed. Artifact checks verify all 64 execution-source fingerprints with the separately archived rendering-only override, measured provenance for every transmitted geometric observation, exact control-time model snapshots, paired conditions, byte ceilings, and flow budgets. Maximum local-update work was **662/1024 flows**. The largest physical-energy residual was **1.69e-15**. A 4,766-entry ledger separates fixed-state target changes from model/geometry shaping changes; these are not physical work.

Team communication totals were 3,954,946 and 4,027,203 bytes for the two seeds, under the four-million-byte per-agent ceilings. Complete simulation runtimes were approximately 362–381 s without communication and 586–614 s with communication, including evaluator audits; rendering is additional. These are measured development runtimes, not matched-computation claims.

The earlier `cooperative_maze_v1` batch terminated with status 143 before saving complete episodes and remains explicitly marked incomplete. Its truncated debugging pilots are preserved. V2 changes cache repeated graph costs and bypass full projection enumeration only when the nominal control is already feasible; numerical equivalence is tested. The original scene parameters and horizons were retained.

The world panel shows unsensed true gate sections in gray and sensed sections in red. That evaluator display does not grant the learners hidden geometry. Each Hamiltonian panel uses the exact recorded model, momentum, waypoint and support; gray areas are unsupported or walls. The full applied force includes damping, clipping and safety constraints, and need not equal the conservative Hamiltonian-gradient force.
