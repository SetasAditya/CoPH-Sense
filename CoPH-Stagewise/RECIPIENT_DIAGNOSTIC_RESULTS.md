# Recipient-value prediction diagnosis

The diagnostic is implemented and evaluated. It confirms that sending Hamiltonian
context does not by itself make packet-value prediction reliable. A more faithful
receiver representation alone is not a sufficient fix: reconstructed-world error
and limited supported horizon remain material.

## Evidence

- 38 tests pass, including callback isolation, matched starts/costs, horizon-prefix
  consistency, unsupported domains, and missing-advertisement masks.
- Ten existing development episodes completed: five static scenes × two seeds.
- All 20 agent trajectories, final models, packets, mission costs, and communication
  bytes exactly reproduce the previous fixed-rate development runs.
- 59 first-eligible packet opportunities were audited. Irrelevant-evidence scenes
  had no eligible packets; they remain in the ten-episode denominator.
- All 1,416 branch/horizon checkpoints pass artifact checks. The exact receiver /
  true-world 16-step effects agree with the existing causal audits to 1e-12.

Each opportunity crosses exact versus advertised receiver state with true versus
frozen reconstructed world. Every condition shares the starting physical state,
continuation routine, control/cost conventions, and indexed sensor randomness.
The frozen world uses the incumbent receiver's current-stage field and damping;
this is a controlled diagnostic, not the legacy scheduler's sampled environment.

## Findings

1. **Receiver reconstruction introduces error.** Even when both receiver variants
   run in the true simulator, advertised-state benefit estimates have mean
   absolute error 0.00413, 0.00862, and 0.01914 at 8, 16, and 32 steps across all
   59 cases. These are comparable to or larger than the corresponding mean
   absolute true benefits (0.00272, 0.00717, 0.01406).
2. **Exact receiver state does not solve world-model error.** At 16 steps, the
   exact-receiver/frozen-world condition has common-case MAE 0.00534 versus
   0.00547 for the advertised receiver in that same world. Both remain inaccurate.
   At 32 steps the exact receiver is worse on the surviving subset. Effects
   interact; errors can cancel, and richer status is not automatically better.
3. **Longer horizons reveal different effects but lose model support.** Common
   valid opportunities fall from 43/59 at 8 steps to 27/59 at 16 and 15/59 at 32.
   In true-world branches, benefit changes sign between 8 and 32 steps in 13/53
   cases with non-negligible effects at both horizons. This supports investigating
   delayed effects, not a claim that simply lengthening replay fixes prediction.
4. **The scheduler is not repaired yet.** These are evaluator-only interventions.
   Private receiver snapshots and truth were not used for training or scheduling.
   No packet-format, policy, penalty, or rate changes were made.

Next implementation priority: supported prediction across retained stage models,
paired with a legally advertised receiver-state summary. Validate these jointly
against the diagnostic before retraining the packet-value predictor. Do not force
transmissions by lowering costs, or treat unsupported predictions as zero benefit.

## Artifacts

- [Generated report and scatter](results/recipient_diagnostics_v1/REPORT.md)
- [All comparisons](results/recipient_diagnostics_v1/rows.json)
- [Additional effect/horizon analysis](results/recipient_diagnostics_v1/analysis.json)
- [Contract verification](results/recipient_diagnostics_v1/contract_checks.json)
- [Reproduction command](COOPERATIVE.md#recipient-prediction-diagnosis)

The new CLI command is `recipient-diagnostics`. Existing commands and results are
preserved. The manifest and source ZIP record the exact evaluated code. Audit-only
work totals 8,640 simulated transitions, 58,102 replay transitions, and 145,744
identification-work units; branch wall time is 171.9 seconds, within 274.8 seconds
for the full diagnostic run. Work counts are operation proxies, not FLOPs.
