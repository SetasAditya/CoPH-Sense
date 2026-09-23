# CoPH-Terrain E2 implementation status

The final natural-benchmark acquisition audit is governed by
`MISSION_INTEGRATED_STOP_RULE.md` and runs with
`python -m coph_terrain.mission_integrated_audit` from the package root on
`PYTHONPATH`. Its `920000--920019` maps are excluded from fitting.
`python -m coph_terrain.e2_dev_smoke` and
`python -m coph_terrain.e2_dev_replay` provide a disposable training-manifest
A3 critic plumbing check. The smoke checkpoint is not a final A3/A4/A5 model
or a confirmatory E2 result.
The fresh audit did not pass; `E2_SPLIT_DECISION.md` records the resulting
freeze of E2-Natural and the separate E2-Complementarity condition.
`natural_manifests_v3.py` writes and verifies replacement Natural splits
under the frozen current generator. The original v2 files no longer verify
and cannot be used for final training.

This package is the procedural VMAS benchmark under construction. The
[pretraining protocol](E2_PROTOCOL.md) fixes the scientific objective,
statistical decision rule, and complete-rival selection. The existing
`coph_fork` finite/two-fork checkpoints remain separate regression controls.

Implemented so far:

- Five deterministic 12 m × 12 m map families (open, bottleneck, shallow
  labyrinth, deep labyrinth, irregular) with public static occupancy, latent
  surface-support and traction fields, and exact VMAS obstacle boxes.
  Episode realizations change the terrain properties under fixed topology.
- Archived, hashed **v2** parent-map manifests for 500 train, 100 validation,
  100 calibration, 200 in-distribution test, and 200 OOD test groups. ID
  topology families have matched low/medium/high shortest-path risk strata;
  OOD has separate matched-topology and natural compound-shift slices. The
  unstratified v1 manifests are preserved separately. **v2 hashes and risk
  strata are invalid under the revised generator; do not use them for E2
  training.** A new manifest version is pending the resource freeze.
- Two-agent VMAS dynamics, passive local appearance, viewpoint-based
  surface scans and traction probes with dwell, measurement budgets, charged
  delayed packets and ACKs, persistent local evidence, separate material-risk
  and mission-cost ledgers, and deterministic replay.
  The passive sensor is cell-quantized and cached within a map cell; analytic
  disc–box collision accounting is tested against VMAS overlap queries.
- An observation-only risk costmap, A* waypoint planner, 8–16 candidate
  viewpoint/modality descriptors, and a terrain-wide split pH waypoint
  executor using public occupancy and the actor's local terrain-property
  beliefs. Hidden-terrain substitution tests verify candidate and force
  invariance at the same local history.
- Fractional-tail empirical CVaR and the paired, hierarchical bootstrap with
  the declared success gate.
- A variable-candidate pairwise set critic with shared map/memory encoders and
  a latent set-interaction decoder with separate expected nonrisk-cost and
  bounded, monotone risk-quantile outputs. Quantile Huber supervision uses
  individual rollout returns. The
  risk-sensitive deployment score is implemented; no E2 weights are trained.
- An excluded-seed generator audit for risk coverage, shortest-path exposure,
  and alternative route structure, plus offline PNG/GIF rendering from
  evaluator-only replay artifacts. The renderer writes a separate carrier
  actor-view GIF reconstructed from locally observed and delivered evidence;
  the truth-overlay GIF remains evaluator-only.
- `figures/render_maze_information_gif.py` produces a compact three-map
  animation from the saved successful labyrinth replay at parent seed 600042:
  true risk, scout belief, and carrier belief. The scout's surface scan and
  grip probe are **scripted physical acquisitions**; both charged packets are
  actually delivered before their values appear in the carrier map. In a
  paired physical continuation with the same acquisitions, sending both
  readings costs 1.9321 versus 2.4563 when they are withheld. This is an
  information-flow visualization, not evidence that buying the pair was
  optimal in E2-Natural. Regenerate with
  `PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python
  CoPh-sense/coph_terrain/figures/render_maze_information_gif.py`.
- A batched PyTorch pH step with CPU parity tests and an A100 smoke check.
  The single-branch Python/NumPy VMAS environment also supports `cuda:0/1`,
  but moving that one-branch loop to CUDA is not faster. In a kernel-only
  benchmark (100 steps), one A100 processed about 16.8k branch-steps/s at
  batch 137 and 120.8k at batch 1024. This is **not full rollout throughput**:
  candidate planning, sensing, packets, VMAS state, and ledger updates still
  need batching before those GPU gains can be used by the teacher.
- An attributed conditional-scout benchmark and a carrier-primary VMAS
  integration. The geometric layer keeps Yiwang's route geometry and analytic
  dispatch model intact, while reporting analytic decisions separately from
  executed kinematic rollouts. The material layer exposes IDLE and public
  `ScoutTask` candidates, charges the complete dispatch/recovery protocol, and
  estimates dispatch value under a replay-conditioned finite prior of complete
  terrain realizations with the frozen material-pH executor. Carrier success now requires recovery
  of every dispatched scout; reaching the goal cannot erase an outstanding
  resource.

**Open before E2 training:** the revised resource/deadline nondegeneracy audit,
controller-specific latest-actionable-switch guard, counterfactual teaching
and scalable policy fitting, complete-rival implementations, and
throughput/vectorization for the five-seed confirmatory experiment. The
archived hidden-topology model-free pilot used full-map geometry in its
*planner only* as a feasibility upper reference. It is not a
deployable-policy result under the revised task. Do not
call the current environment an E2 benchmark result or train on test groups.
The split pH integrator records storage but has no energy-tank or safety
certificate in this package.

The physical-audit prototype evaluates empty/single/pair sensing sets under
one matched scripted continuation and complete pH rollouts. Its first paired
smoke case took about 13 s on CPU and 14 s on one A100. A complete 11-set
parent took 77.85 s on CPU (11 continuations, mean 587 steps), or about
0.14 complete continuations/s serially. A subsequent SciPy shortest-path
implementation cut that parent to 48.01 s, but resolves equal-cost paths
differently and therefore defines a new planner version; the two costs must
not be compared as if only runtime changed. Neither pilot freezes resource
caps. Run the new Torch benchmark with
`python -m coph_terrain.benchmark_torch_execution --device cuda:1` in a
CUDA-visible shell.

For the original planner, all 11 full CPU and CUDA-backed VMAS continuation
returns matched exactly, while CUDA took 88.74 s versus CPU 77.85 s. With
the new planner, eight independent CPU workers finished 88 complete
continuations in 96.54 s (0.912/s); they covered different parent maps, so
that is a representative aggregate rate rather than a matched speedup
factor. At this measured rate, 2,200 audit continuations would require about
40 minutes and 10,000 teacher continuations about three hours, before
collection/storage overhead. The A100 Torch batch kernel is much faster at
large batch size, but no GPU-native full-continuation speedup is claimed.
For two matched new-planner continuations, CPU and CUDA outcomes again matched
exactly, while CPU took 7.90 s and CUDA took 9.13 s. The nine-map new-planner
feasibility rerun was 9/9 with full-map planning and 8/9 with local-only
planning.

The nine-map excluded-seed feasibility pilot completed 9/9 with a privileged
full-map *planner* and 6/9 with the observation-only, no-active-sensing
planner. This is evidence that the physical worlds are traversable and that
missing information can matter; it is not an E2 method comparison. The
subsequent environment-only rule gives a provisional 1300-step (65 s)
deadline. The fresh shared-menu physical audit failed its predeclared
coverage gate, especially in shallow labyrinths, so the deadline and
four-measurement/four-packet caps are **not frozen**. See
`results/pilot/PHYSICAL_RESOURCE_REVISION.md` before E2 training.

```bash
PYTHONPATH=CoPh-sense python -m coph_terrain.manifests verify

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m unittest discover -s CoPh-sense/coph_terrain/tests -q

PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m coph_terrain.pilot_environment --count 9 --full-map-oracle
```
