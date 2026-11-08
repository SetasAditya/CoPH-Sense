# Cooperative mechanism attribution

This development study separates geometry sharing, remote route scoring, supported field reconstruction, and current-state versus historical policy replay. The [implementation contract](IMPLEMENTATION_CONTRACT.md) maps the mathematical proposal to the executable mechanisms and their limitations.

The frozen development suite has completed **112 episodes and 224/224 agent goals** in [`results/attribution_v1/`](results/attribution_v1/). The [manifest](results/attribution_v1/manifest.json) records the seven arms, two scenes, eight seeds, unchanged configurations, and animation seed 75000. The 28 smoke episodes were reused under matching fingerprints. **All 224 missions reached their goals, with zero collisions and zero timeouts.** The final [artifact checks](results/attribution_v1/artifact_checks.json) passed; no failed or incomplete episodes remain.

The [generated report](results/attribution_v1/REPORT.md) contains paired comparisons and eligibility denominators. The synchronized animations show the full cooperative historical-replay run: [blocked maze](results/attribution_v1/figures/blocked_maze.gif) and [soft-risk maze](results/attribution_v1/figures/soft_risk_maze.gif). Their first-packet figures deliberately retain the first eligible opportunity, including zero effects.

The completed eight-seed comparisons establish the following:

- **Geometry sharing helps avoidance in the blocked maze.** Against local evidence, it reduces mean team blocked-branch entries from 4 to 3 and retreat from 1.131 m to 0.709 m. Both arms reach every goal. This is an avoidance benefit, not an increase in success rate.
- **Field assimilation changes receiver estimates and controls, but has not demonstrated additional navigation benefit beyond route anchors.** Common-supported probe errors improve slightly in these comparisons; completion and blocked-branch entries do not improve. In the soft-risk maze, the mean risk-cost component increases rather than decreases. Estimation improvement and navigation benefit are distinct findings.
- **Historical replay is active, but added navigation value is not established.** Nonzero policy gradients and supported starts are logged. All 56 soft-risk episodes follow the same surveyed-edge route sequence across the seven arms. The report compares historical and current-state replay directly and retains their actual work, support exclusions, and domain failures.

The reconstruction arm's matched packet audits changed field coefficients in **282/312** cases and applied controls in **259/312** cases (nonzero threshold $10^{-10}$). Small revisions count here; these counts are not evidence of a useful outcome. All 312 had a valid 32-step audit comparison, including branches that terminated at the goal.

| Replay work per team episode (mean logged flows) | Current state | Historical entries |
|---|---:|---:|
| Blocked maze | 25,689.75 | 97,508.75 |
| Soft-risk maze | 13,748.00 | 27,456.50 |

Historical replay produced 3,396 nonzero-gradient updates across the two scenes, yet added no completion or blocked-branch avoidance benefit over current-state replay. Its extra computation is real: about 3.8 times the operational replay/guard work in the blocked maze and twice in the soft-risk maze. Audit work is additional and separate. The maximum observed local update used 662 flows, below the unchanged 1,024 ceiling.

The stall diagnostic recorded 64 overlapping 16-step windows, all in the blocked maze; every affected mission still completed. The soft-risk maze recorded none. These windows are retained in the machine-readable rows, not relabeled as failures or erased by successful termination. The empty failure gallery corresponds to empty `failures.json` and `incomplete.json` records.

These results establish mechanism isolation and reproducible development evidence. They do not establish broad superiority of cooperative field learning or historical replay.

The seven arms are local evidence, local historical replay, geometry only, remote route scoring, receiver reconstruction, current-state replay, and historical replay. All communication arms transmit the same packet format and charge the whole payload; receiver switches isolate the information used. The current-state comparator retains the same measured history and models but uses one current replay start. All replay arms use the same 1,024-flow local-update ceiling; actual computation is reported. Flow counts are the primary work comparison. Wall times include heterogeneous parallel scheduling (12 workers initially, three for the final interrupted-run resume), and audit work is reported separately.

The blocked scene preserves the surveyed maze and hidden gates. The traversable soft-risk scene replaces gates with risk patches at the same centers. Both retain the five-second delayed carrier start, 1,500-step active horizons, fixed-rate communication, four-million-byte ceilings per agent, and 300-second packet lifetime. Seeds vary indexed sensor noise, not scene geometry or plant damping. This is evidence on two layouts, not eight independent maze designs.

Validation: full artifacts passed checks for 8,344 packets, 1,560 isolated audits, 14,233 updates with remote field residuals disabled, all 73 source files, both fingerprinted documents, and both animations. Maximum physical energy-accounting residual was $1.73\times10^{-15}$. Separately, 112 applicable numerical/contract tests passed, including nine attribution tests; a short integration exercised packet audits, probe logging, reporting, failure figures, and exact-model visualization. The original routed-validation source-selection test rejects the modified working tree, as intended. Its archived 51-file source inventory was checked independently and remains intact. No frozen benchmark gate was weakened.

From `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.attribution_suite suite --out results/attribution_v1 --profile all --workers 12

MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.attribution_suite report --out results/attribution_v1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.attribution_suite render --out results/attribution_v1
```

The suite resumes only under identical source/configuration fingerprints. Completed episode hashes are verified before reuse; recorded execution failures are preserved rather than silently retried. `report` can run against partial outputs and explicitly lists missing episodes. `render` uses the preselected historical-replay episode for each scene and the first eligible packet for each fixed-state comparison, including zero-effect packets.

During execution an unrelated `crossing_visuals.py` edit changed the live-tree inventory. The remaining execution uses the exact frozen archive in `/tmp/coph-attribution-frozen-v1/CoPH-Stagewise`, with the original absolute result path. `execution_context.json` records this isolation; the user's live visualization edit is preserved. Reproduction against a changed working tree requires extracting `sources.zip` into an isolated package and copying the two fingerprinted formulation/contract documents into the same relative layout. The verifier checks the archived execution source, rather than weakening the source gate.

For a self-contained reproduction from the saved source and document archives, use `bash scripts/reproduce_attribution.sh results/attribution_v1 results/attribution_reproduction`. This creates an isolated checkout and preserves the live implementation. Run `bash scripts/verify_attribution_artifacts.sh results/attribution_v1` after the complete suite and both animations finish. The verifier checks source and document archives, pairing, packet provenance, isolation, the computation ceiling, and animation metadata. Interrupted partial attempts are preserved in the execution record; completed-episode computation excludes their unavailable total work.

The generated report distinguishes navigation readiness, model/control revisions, and demonstrated outcome benefits. Immediate and 8/16/32-step true-world audits use matched receiver snapshots and indexed noise, with no future communication. Their future observations stay in hypothetical branches. Route/geometry, field/damping, and policy effects are shown in an ordered decomposition, with clipping and safety projection separately recorded. Mission cost is secondary; failed, unsupported, and zero-effect cases remain evidence.
