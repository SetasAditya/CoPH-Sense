# Pre-registered experiment spec: decision-sensitive receiver reconstruction (for Codex to run)

## Context
The selected opt-in receiver (`receiver_fit_mode='precision_bounded'`, `receiver_solver_mode='active_set'`, `receiver_search_mode='diagnostic'`; results in `CoPH-Stagewise/results/active_set_receiver_v2`) is now numerically reliable. Solver rejections fell from 66 to 3 of 300 updates, and 60/60 missions completed. But branch choices never change.

The offline diagnosis (`results/reconstruction_quality_v1/DIAGNOSIS.md`) showed why. The fork fixture's median route margin is about 3.6 cost units, while reconstruction shifts margins by about 10⁻³. The open claim, *receiver reconstruction can change route choice*, can only be tested on a fixture whose decisions are sensitive to field changes of the size reconstruction produces.

Decision: **I write this specification; Codex implements and runs it.** This file is the deliverable, to be copied to `CoPH-Stagewise/DECISION_SENSITIVE_SPEC.md`.

## Non-negotiable protocol
1. **Freeze before treatment.** Geometry, hazard family, seeds, arms, the eligibility gate, endpoints and the analysis are written into a manifest, and hashed before any treatment arm runs. Nothing is re-tuned after a treatment outcome is seen.
2. **No outcome-based selection.** Scenes come from a declared grid. A scene is run only if it passes the acquisition-only eligibility gate (step 3), computed from communication-free acquisition snapshots before arms diverge. Every eligible scene is run and reported, never just the best.
3. **Defaults unchanged.** No controller, packet, support, objective, tolerance, box, budget or validator change. Legacy defaults and the 187-test suite (plus the known hash error) must still pass.

## Fixture family (fork from `decision_suite.NODES/EDGES`)
- **Reuse.** Start from `complementary_reconstruction.specification('complementary', seed)`, with the same starts, acquisition at 0.96 s, `risk_weights=[.25, 6.]`, `unknown_risk=.02` and corridor radius 0.85.
- **Route costs.** For the risk-averse receiver (agent 1, w=6), the bypass (1,4)-(4,5)-(5,2) is 3.4 m longer than the direct edge (1,2). At background risk that costs about 3.4 × (1 + 4·6·0.02) ≈ 5.0 extra. The receiver is therefore indifferent when the direct edge's excess exposure is E* = ∫(r − 0.02) ds ≈ 0.21.
- **Grid (declared now).** One corridor-spanning Gaussian patch on the direct edge, with:
  - centre x ∈ {2.0, 2.5, 3.0} at y = 0;
  - widths (0.6, 1.0);
  - amplitude a chosen so that E/E* ∈ {0.6, 0.8, 0.9, 1.1, 1.25, 1.6}.

  That gives 18 scenes, plus two controls: a hazard-free scene, and the `redundant` start case on the median scene.
- **Peer evidence.** The scout (agent 0) must be able to sense the patch before the receiver's branch decision. Keep the existing complementary starts; if the gate shows no eligible anchors, the scene is ineligible rather than moved.

## Acquisition-only eligibility gate (per scene and seed)
These are computed on the communication-free acquisition snapshot and on matched copies of the receiver's last update before branch entry. Treatment outcomes are never used.
- **(a) Support.** Supported query points on **both** outgoing routes.
- **(b) Evidence.** At least one delivered, rectangle-eligible peer anchor (reuse `reconstruction_diagnostics.eligibility`).
- **(c) Decision sensitivity.** The local-only route margin |C_direct − C_bypass|, from `reconstruction_diagnostics.ranking(..., details=True)` under `supported_field` scoring, is ≤ 0.5 cost units, **and** the margin's interval straddles 0. The interval is computed from the receiver's own uncertainty: propagate its stage curvature or bootstrap ensemble through the route cost, with the runtime-available quantities only.
- **(d) Relevance (evaluator-only, declared).** The truth-optimal branch for the receiver differs between at least one pair of grid amplitudes at that position. This guarantees that the family contains both correct-direct and correct-bypass scenes.

## Arms, seeds, endpoints
- **Arms** (reuse `receiver_corrections.ARMS` plus the selected modes):
  - local-only;
  - legacy receiver;
  - selected corrected fit (active-set/diagnostic);
  - selected combined (plus relative validator).
- **Seeds.** 8 noise seeds per eligible scene, declared now (79300–79307).
- **Primary endpoint.** Receiver branch correctness against the truth-optimal branch. This is an evaluator-only integral of 1 + 4·w·r_true along each route at the receiver's weight, reusing `hazard_oracle.true_risk` / `route_oracle`-style quadrature.
- **Primary comparison.** Selected corrected fit vs local-only: paired by (scene, seed), exact McNemar test on discordant pairs, with a seed-bootstrap CI on the difference in proportion correct. State that scenes share geometry, so seeds are not fully independent.
- **Secondary endpoints.** Receiver mission cost (time + weighted risk), route-query value MAE before the decision, solver and validator rejections, bytes and computation.
- **Expected-null controls.** In the hazard-free and redundant scenes no branch difference is expected; any difference is reported as a failure of the control.

## Stop and report rules
- If no scene passes the gate, stop and report the gate table. Do not move hazards or starts.
- Report every scene and seed, including ineligible ones, together with the gate values.
- A positive result is stated only for the declared primary comparison. Everything else is descriptive.

## Deliverables (Codex)
- `coph_stagewise/decision_sensitive.py`: grid, gate, runner (fresh directory, frozen manifest, resume only with a matching fingerprint), reusing `complementary_reconstruction.acquire/gate_snapshot`, `receiver_corrections` capture and audit patterns, and `staggered_demo.execute`.
- `results/decision_sensitive_v1/`: `manifest.json` (hashed before treatment), `gate.json`, episodes, `FINDINGS.md`, `REPORT.md` with GIFs for the seed-79300 median eligible scene.
- `tests/test_decision_sensitive.py`:
  - the grid and gate are deterministic;
  - the gate reads no treatment output;
  - E is computed correctly;
  - the truth-optimal branch flips across E*;
  - the manifest is frozen before treatment.

## Verification (what I will check when Codex reports)
- The manifest hash predates every treatment episode; the gate table covers all 20 × 8 scene-seeds.
- Every eligible scene was run and reported. McNemar and bootstrap numbers match the episode rows.
- The full suite passes apart from the known hash error, and legacy defaults reproduce the archived runs bit for bit.


---
# Resolved protocol (approved before execution)

This addendum resolves ambiguities; the original specification above is preserved verbatim.

- Gate at the communication-free receiver's last learner-update opportunity before outgoing-branch entry; acquisition at 0.96 s is descriptive. Missing snapshots fail. Finish all 160 screens before treatments.
- Grid scenes require all eight seeds to pass. Controls require support and valid evidence, but are exempt from margin, uncertainty-straddling and truth-flip checks. Run no controls if no grid scene qualifies.
- E*=3.4(1+24*0.02)/24 is a design reference. Amplitudes use the finite Gaussian integral along (1,2). Truth includes all bypass tails; do not force an exact flip at E*.
- Compare fixed branches (1,2) and (1,4),(4,5),(5,2), using production quadrature. Connector alternatives are not the gate margin.
- Use 128 diagnostic-only receiver-local time-block bootstrap resamples; share multiplicities across retained stages. Preserve reserved data, overlap rows, ridge/prior, geometry, support and anchor/unknown fallback. Center perturbations on the committed margin and require the 2.5–97.5 percentile range to contain zero. Fewer than two blocks or nonfinite results fails. This is an uncalibrated conditional screening range, not a confidence guarantee.
- The redundant control uses x=2.5, E/E*=0.9. Animate seed 79300 of the lower-middle eligible grid scene sorted by (x,ratio).
- All four arms retain reciprocal fixed-rate packets and remote route anchors. Local-only means no received-field fitting, not no communication. Corrected/combined use precision_bounded/active_set/diagnostic; combined retains relative_local validation. Replay is disabled.
- Isolated sender exports maintain a separate stage-rotation cursor; only isolated receiver copies validate screening packets. No packet affects communication-free screening learners.
- Preserve 500 steps, damping 0.85, the controller, support, objectives, packets, validators, 4-million-byte budgets, 300-second packet lifetime and 1,024-flow ceiling.
- Primary outcome is actual first branch traversal. Missing/unclassified entry is unsuccessful. Truth ties within 1e-10 accept either recognized branch. Controls are outside primary inference.
- Primary comparison is corrected fit versus local-only: two-sided exact McNemar, p=1 with zero discordances, and 10,000 seed-cluster bootstrap draws with RNG seed 179300. Retain all scenes within sampled seeds. A positive claim requires positive difference, 95% interval above zero and nominal p<0.05. Report dependence, eight-cluster and eligibility-conditioning limitations.
- Freeze sources/configuration/inputs/environment before screening; seal gate/selection before treatments. Reproduce the 18 archived baseline episodes before treatments. Resume requires matching fingerprints. No retuning, substitute scenes, adaptive communication, historical replay or held-out tests.

## Reproduction

From CoPH-Stagewise:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl python -m coph_stagewise.decision_sensitive suite --out results/decision_sensitive_v1 --workers 4
```

Commands also include `freeze`, `gate`, `report`, `render`. The suite runs full tests, explicitly preserving the known archived-source-hash error, repeats baseline episodes, screens and seals eligibility, then runs eligible comparisons. Zero eligible grid scenes means no treatments or treatment GIF.
