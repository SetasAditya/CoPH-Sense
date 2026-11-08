# Receiver reconstruction diagnostic

This opt-in development suite diagnoses the existing four hazard-pilot scenes. It does not change controller, fitting, support, communication or validation rules. Preserved pilot records remain in `results/hazard_pilot_v2`.

Run from `CoPH-Stagewise`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m coph_stagewise.cli receiver-reconstruction analyze --reference results/hazard_pilot_v2 --out results/receiver_legacy_analysis_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m coph_stagewise.cli receiver-reconstruction suite --reference results/hazard_pilot_v2 --out results/receiver_reconstruction_v1 --workers 4
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/coph-mpl python -m coph_stagewise.cli receiver-reconstruction render --out results/receiver_reconstruction_v1
python -m coph_stagewise.cli receiver-reconstruction report --out results/receiver_reconstruction_v1
```

The suite archives Python sources and freezes specifications, settings, source hashes and reference hashes before execution. It analyzes all 32 saved pilot episodes, runs four field-arm tracing regressions, and gates twelve comparison episodes on regression success. Resumption requires matching fingerprints and verifies saved episode hashes; failed runs remain preserved. Use the archived source checkout to reproduce a frozen run after subsequent code changes.

The three comparison arms disable policy replay: shared anchors without received field fitting, supported direct local fields without received fitting, and supported direct cooperative fields with fitting. All retain local fitting, executed-input damping identification, reconciliation and measured validation. Communication opportunities and payload protocol are identical; realized contents can differ after trajectories diverge. Risk weights (0.25/6), start delay (25 s), unknown-risk prior (0.02), seed 79000, sensing, hazards and horizons come directly from the pilot manifests.

Receiver fitting accepts anchors inside retained stage rectangles. Direct-field routing additionally requires local training measurements within 0.65 units, a committed field snapshot, and newest-supported-stage selection. These are separate conditions. Remote anchors do not certify field support. The received reconstruction store retains the latest packet per sender, while the remote routing store can retain multiple stage contributions; traces expose replacement and expiry without changing either store.

Audits run on the carrier at every delivery before its first geometric branch entry and the first delivery per subsequently visited stage, including rejected/unsupported packets. Entry is the first logged displacement greater than 0.25 units from the shared start, not a claim about when planning occurs. Both complete receiver clones receive the same packet; one fits received residuals and the other suppresses them. Both update against actual retained measurements. No future synthetic measurements are introduced. Frozen-anchor rankings isolate field reconstruction; natural rankings also include regeneration of local anchors. Truth is evaluated only after branching to compute value/gradient error at identical quadrature points. It never becomes a learner observation or route input.

Every audit records model acceptance, validation residuals, packet provenance, support masks, unsupported predictions, field and control revisions, route rankings and computation. Per-control traces use the actual agent-specific risk weight and unknown-risk prior. Margins between node routes can reflect virtual-connector changes; actual edge traversal is reported separately. Query repetitions and one noise seed are not independent generalization evidence.

Numerical regression excludes only additive diagnostic metadata and changed evaluator audit indices. Operational decisions, versions, acceptance, work and controls are compared under the existing atol=1e-12/rtol=1e-10 contract. Bitwise and packet-byte equality are separately disclosed. The historical archived-source-identity test is intentionally not applicable to newly added source files; its exclusion is explicit in verification logs, not a weakened runtime test.

Report artifacts include legacy rows/lineage, new packet-to-model lineage, matched audit errors, paired agent outcomes, support/timing plots, four synchronized three-arm GIFs, regression checks, and complete failure/incomplete inventories. Success is valid attribution, not improvement. No fusion tuning, support expansion, new scenes, store redesign or held-out evaluation is authorized by this suite.

## Completed result and reproduction correction

[Findings](results/receiver_reconstruction_v1/FINDINGS.md): the carrier's eleven deliveries per episode were remote-route-only. No reconstruction contribution existed to fit or overwrite. All three primary arms had identical completion times, risk and branch sequences; all 24 comparison missions completed safely. This pilot cannot establish the quality of active receiver fusion.

The completed run preserves its original source archive. `reporting_revision.json` records two reporting-only corrections discovered after all episodes completed, with a corrected reporter snapshot. Use `bash scripts/reproduce_receiver_reconstruction.sh` for a fresh reproduction that applies that correction to the archived checkout, or `bash scripts/verify_receiver_reconstruction.sh` to verify the existing artifacts. No episode rerun or controller/learner change was needed for the corrections. Verification checks frozen runtime/configuration, packet bytes, measurement provenance, audit selection/isolation, flow ceilings, and GIF frames. The new diagnostic command and frozen manifests remain separate from legacy commands and results.
