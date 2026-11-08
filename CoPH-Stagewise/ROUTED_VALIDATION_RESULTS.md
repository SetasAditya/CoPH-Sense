# Routed-progress: selected-controller validation

**Held-out navigation gate: PASS.** 128/128 agent goals across 64/64 completed episodes; 0 collisions and 0 timeouts.

Routed-progress was selected from development results before held-out execution. The original four-configuration development gate remains failed because protected-progress stalls in hazard slalom. Its results remain intact; this is a separate, explicitly recorded selected-controller gate.

| Communication | Episodes | Goals / agents | Collisions | Timeouts | Stagnation replans | Braking steps | Maximum stage |
|---|---|---|---|---|---|---|---|
| disabled | 32 | 64/64 | 0 | 0 | 0 | 0 | 9 |
| reciprocal_fixed | 32 | 64/64 | 0 | 0 | 0 | 0 | 9 |

All 81 tests passed. Every original source file matched the development manifest and archive; all 218 reference result files remained unchanged. Every episode passed pairing, resolved-controller, communication, cost-component and byte-budget checks. Maximum update work was 614/1,024 flow evaluations; the maximum physical energy residual was 2.07e-15. Full execution and rendering took 7.0 minutes with 32 workers.

The four static scenes used the predeclared seeds 110000–110007. Controller and learner code, parameters, horizons, sensing, costs, safety constraints and fixed-rate communication were unchanged. No held-out outcomes were used for tuning. Final-test namespaces remain untouched.

Routed-progress is validated on these static, independent-agent navigation scenarios. This does not establish inter-agent collision avoidance, changing-hazard forecasting, or adaptive communication reliability.

Navigation reliability is separate from communication benefit. Earlier isolated deliver/withhold audits verified that reciprocal evidence changes receiver field estimates and controls; those development audits remain preserved. Equal completion rates provide no demonstrated completion advantage from communication. Secondary mission-cost differences, bytes and computation are recorded for all paired cases; no superiority claim follows from navigation readiness.

[Held-out report and four synchronized GIFs](results/routed_validation_v1/REPORT.md) · [All failures](results/routed_validation_v1/figures/failures/index.html) · [Manifest and fingerprints](results/routed_validation_v1/manifest.json) · [Artifact checks](results/routed_validation_v1/artifact_checks.json) · [Reproduction](ROUTED_VALIDATION.md)
