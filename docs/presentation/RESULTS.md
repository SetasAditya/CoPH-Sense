# Five-scenario presentation delivery

The gallery is built from preserved experiments; no controller, learner, scene, cost, seed, or communication rule changed. No new training or navigation episodes were run.

| Scenario | Displayed record | Interpretation |
|---|---|---|
| Unexpected dynamics | seed 87000, drag 0.25 → 0.45 | Online identification after a drag/goal intervention; fixed initially correct model and matched phMARL references |
| Heterogeneous hazards | original animation seed 79000, two mild hazards | Shared graded route evidence under risk weights 0.25 and 6; binary advice reference |
| Reciprocal discovery | original animation seed 76000 | Reciprocal versus disabled exchange; exact control-time model and packet events |
| Symmetric encounter | seed 300, four-agent swap | Programmed final protection/priority versus shared base layer; missing force/yield records explicit |
| Sparse evidence | first original acquisition group, query IDs 0–11 | Saved-state queries, original paths regenerated and all cost estimates checked; archived certification flags |

Native/encounter animation seeds were not preregistered in the original manifests; the lowest declared seeds are used without outcome search. The hazard and crossing seeds match their original animation selections. These are planar simulations shown obliquely, not 3D physical experiments.

The static site includes five MP4s, five GIF previews, five posters, and 326 per-frame records. Frame counts are 70 / 65 / 66 / 65 / 60. Playback is compressed, with mission times or query indices retained. GIF previews subsample frames; MP4 provenance is authoritative for exact event jumps.

## Verification

- 19 original input hashes verified unchanged.
- 262 exact model/frame snapshots checked; 119 nominal-force checks have maximum error 5.43e-20.
- All 12 recovered path-pair prices agree with the saved truth and representation estimates to 1e-12.
- Desktop/mobile browser checks decode all five videos, verify exact event-button seeking, keyboard navigation, no JavaScript errors, and no horizontal mobile overflow.
- Current package tests: 248 passed, 7 subtests passed, 41 explicit archive-dependent skips. The historical archived-source-hash failure is not claimed repaired.
- Native tests: 26 passed in the pinned Python 3.8 environment. Baseline admission and independent primitives: nine tests passed.

`log_audit.json` retains 212 adverse native/encounter records and complete hazard/crossing completion inventories. Missing native velocities, encounter controls/yield/model traces, and archived uncertainty sample distributions remain explicit.

## New prior-work comparisons

DistGP and UGV-conditioned exposure planning remain blocked at native admission by incomplete published experiment specifications. Official implementations/examples were not located in eight successfully consulted primary/API sources, but author code is not mandatory. Independent equation-level implementations now verify FITC/tree Gaussian belief propagation against dense conditioning (mean error 2.22e-16, covariance error 8.67e-19), log-normal moments against quadrature (error 1.59e-22), and A* against Dijkstra (equal objective). These are primitive correctness checks only, with frozen inputs and explicit assumptions. They do not reproduce the published native experiments or establish CoPH superiority. DistGP still lacks selected kernel, sampling and inference settings; exposure planning lacks the exact native map, sensing and search specification. `benchmarks/scenario_baselines/` preserves the original sealed intent, feasibility assessment and independent-source admission gate. No comparative evaluation was run.

The paper appends a scenario overview and exactly two labeled 3D placeholders; existing experiments and related work are preserved. The live gallery is https://setasaditya.github.io/CoPH-Sense/; GitHub Pages deployment and HTTP 200 were verified before the manuscript citation was added.
