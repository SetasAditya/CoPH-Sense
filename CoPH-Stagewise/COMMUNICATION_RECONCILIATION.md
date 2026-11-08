# Communication-accounting reconciliation

The compact-link bandwidth evaluation previously omitted each agent's declared risk weight when constructing its simulator and its cost-component log. Its learner nevertheless used the declared weight. For the carrier, risk was priced with weight 6 in planning and weight 1 in evaluation. The original `hazard_demo` / `staggered_demo` evaluation already applied the correct weight.

The compact-link runner now records the raw integral `sum(dt * risk_truth(after_position))`, its weighted contribution `4 * risk_weight * raw_risk`, time, effort and failure. `time_risk_cost` matches the original hazard endpoint. `cost` additionally includes effort and failure. Raw exposure is not an extra additive cost component.

Bandwidth records retain each agent's status, payload and acknowledgment bytes and their team sums. Each agent's sum is checked against its own ceiling. A team-total advertisement size cannot be compared directly with one agent's ceiling. Byte budgets use decimal bytes: 8,000 / 16,000 / 32,000 / 4,000,000.

`accounting_mode="legacy"` reproduces the superseded metric solely for parity checks. The corrected mode is the default. `byte_budget` is an optional keyword argument to the compact-link runner, with its previous 4,000,000-byte default. The runner's default learner factory now propagates scene task weights and the unknown-risk prior; the old bandwidth harness already supplied these through its custom factory.

## Reproduction

Run from `CoPH-Stagewise`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m coph_stagewise.communication_reconciliation all --workers 16
```

Commands `freeze`, `regression`, `suite`, `report`, and `render` are available separately. Default output is `results/communication_reconciliation_v1/`. Resume refuses changes to operational source, configuration, input hashes or the recorded numerical environment. Report-only source provenance is recorded separately.

The original-result regression reproduces four scenes, seed 79000, field and no communication: 8 episodes. Evaluator-only packet audits are omitted from these reproductions and excluded from operational comparisons. Trajectories, controls, observations, model versions, update decisions/work, packet contents and cost components are checked against the archived original results.

The corrected bandwidth comparison contains 72 episodes: eight scene/seed pairs, one no-communication reference per pair, and two selectors at four per-agent ceilings. Each corrected episode also has a matched legacy-accounting execution. The gate checks operational trajectories, controls, observations, versions, decisions/work and communication events. The legacy outcomes reproduce the archived bandwidth rows. Timers and accounting fields are excluded from operational parity; discrete decisions/work remain exact and floating comparisons use absolute tolerance 1e-10. Bitwise canonical projections and complete serialized-file differences are reported separately.

## Interpretation

The original hazard experiment uses the original channel and replay-enabled learner. The newer bandwidth comparison uses compact transport and evidence-only learning. Compare corrected versus legacy accounting within the same runner; do not attribute all differences between the two experiment families to accounting.

Report communication versus no communication separately from receiver-relevant selection versus retained-stage rotation. The two seeds vary observation noise in four fixed, designed scenes; they do not establish independent-world generalization. Existing hazard and bandwidth artifacts remain intact. Existing communication, fitting, controller, support, safety and cadence settings are preserved.
