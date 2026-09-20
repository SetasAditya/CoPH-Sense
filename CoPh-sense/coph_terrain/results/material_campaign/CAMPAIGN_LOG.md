# Material-pH CoPH campaign log

This directory records the closed six-stage campaign requested after the
material-aware executor calibration. A stage is marked passed only from its
own declared artifact; later results are not used to revise an earlier gate.

## Frozen prerequisites

- Material checkpoint SHA-256: `327c744dea093a5436e02ec03328690bac71dbbdf57fd7713a13c5735cdcbda6`
- Calibration artifact SHA-256: `8432339b8a4324b2355fac2ea093cfd1a45855fca086c81baceb5be671bc5b22`
- Frozen A4 checkpoint SHA-256: `f914ec0f5418dccc94b278ada001d94409432df007149737d13ecd5953c6ef37`
- Executor: `material_ph`, force limit `4 N`

## Stage status

1. A3/A5 retraining under material-pH: **failed gate**
   - validation mean realized regret: `0.0508546`
   - validation exact action rate: `0.75`
   - pair-optimal states: `0`
   - retained/withheld optimum switches: `0`
2. Complete causal-stack gate: **failed closed**
   - diagnostic candidate selected `skip` freely, so the chain did not reach NEED
3. Single on-policy A3/A5 refresh: **not run; stage 2 prerequisite failed**
4. Six matched complete methods: **runner implemented; real policies blocked by stages 1–3**
5. Paired confirmatory ID/OOD evaluation: **not launched; prerequisites failed**
   - paired/resumable smoke completed using the explicitly synthetic adapter only
6. Tables, figures, GIFs, and results text: **smoke renderers complete; scientific results blocked by stage 5**

## Gate decision

The six stages were attempted in order. Continuing with on-policy refresh or
confirmatory evaluation would convert a failed acquisition learner into a
mislabelled final result. The campaign therefore stops at the predeclared
causal prerequisite rather than tuning on test maps or forcing a non-skip A3
action.

## Integrity notes

- Existing `manifests_v3_natural` hashes became stale after the executor and
  environment source changed. No old v3 run will be called confirmatory.
- The current Codex sandbox cannot access the NVIDIA driver (`nvidia-smi`
  fails). CPU smoke and correctness gates run here; GPU launch commands and
  manifests will be emitted for execution on the two-A100 host.
- Persistent homology, more than two agents, dynamic topology, RELLIS, and a
  structural pH-superiority claim are outside this campaign.

## Gate 2 integration scaffold — 2026-09-18

- Added `frozen_need_a4.py`, a deployment adapter for the frozen 19-feature
  history+NEED A4 checkpoint. It enforces innovation/NEED compatibility before
  SEND and exposes no receiver-private or hidden-terrain inputs.
- Added `material_full_stack_gate.py` with the fixed Gate-1 dependency
  `results/material_campaign/a3_a5_frozen.pt`. The runner logs A3, evidence,
  NEED, A4, A5, and belief-to-material-pH force intervention stages.
- The runner fails closed if the Gate-1 checkpoint is absent or incompatible.
- Targeted A4 adapter tests: 2/2 passed.
- Current full-stack status: pending Gate 1; no disposable checkpoint substituted.

## Controlled Gate 1.1 coverage rerun — 2026-09-18

- Gate 1 failed with no pair-optimal or memory-switch teacher states. The
  three-stratum rerun is specified in `train_material_a35_gate11.py` and writes
  seed-disjoint manifests before collecting any labels.
- Natural retains the frozen v3 maps. Complementarity uses the already-approved
  two-region challenge with ex-ante values averaged over four compatible
  first-region worlds. Moving-A5 teacher-forces one physical scout reading and
  packet, then compares retained and withheld carrier histories. All continuations
  use `material_ph`; no environment prices, executor parameters, A4 checkpoint,
  or critic architecture are changed.
- Four fixed terrain corners per challenge parent are included by construction,
  regardless of their teacher-optimal action. The training objective gives each
  stratum equal weight while keeping the original value/quantile losses.
- This is one development rerun. If its gate fails, the next step is a legal-input
  sufficiency audit, not another model sweep.
- A preflight run was interrupted during Natural collection before any
  challenge labels or fitting: its new collector had mistakenly targeted the
  center of the challenge region. The corrected collector exactly matches the
  approved first-region targets `(8,30)->(10,30)` for geometry and
  `(10,30)->(10,30)` for traction. The preflight manifests are archived with
  `preflight_invalid` filenames; fresh manifests hash the corrected collector.

### Gate 1.1 result

- The single corrected rerun completed with separate train/validation parents
  and CUDA critic fitting; the exact physical teacher remained CPU-bound.
- Validation: global mean realized regret `0.0203418` versus required `<0.02`,
  exact action rate `0.90` versus required `>=0.90`.
- By stratum: Natural regret `0.0508546` and `75%` exact; Complementarity regret
  `0` and `100%` exact; Moving-A5 regret `0` and `100%` exact.
- The apparent challenge accuracy is not evidence of the intended mechanisms:
  Complementarity had `0` pair-optimal validation states and Moving-A5 had `0`
  retained/withheld optimum switches on the independent validation parent.
  One Moving-A5 training corner did switch, but the fitted critic incurred high
  regret on training challenge states. Gate 1.1 therefore **failed** and its
  checkpoint is named `a3_a5_candidate_failed.pt`.
- The proposed next stage remains the legal-input sufficiency audit, not a
  second fitting run. Full-stack, refresh, and confirmatory stages remain closed.

### Targeted legal-input sufficiency audit

- Replayed the fixed `(R1 safe, R2 bad)` Moving-A5 corner under the train and
  validation parents, with no fitting or new seed selection. The audit is in
  `gate11/gate11_sufficiency_audit.json`.
- On the train parent, retained and withheld states had **different deployed
  input hashes** and different exact optima: R2 pair (index 6) versus R1
  geometry (index 1). The critic chose SKIP in both, losing `8.27` and `9.60`
  in the single-world teacher ledger. This particular memory switch is legally
  representable but was not learned by the fitted critic.
- On the validation parent, the two legal inputs also differed, but the exact
  optimum was SKIP in both; the critic chose SKIP and had zero regret. Thus the
  held-out failure is **absence of a decision contrast**, not evidence that
  the critic learned a memory switch.
- No pair-optimal state occurred on held-out Complementarity rows, so this
  audit cannot establish pair-action identifiability or competence. Gate 1.1
  remains failed; there is no frozen A3/A5 checkpoint for the full-stack gate.

## Gate 1.2 preflight — stopped before new teacher-pool collection

- Replayed only the existing Gate 1.1 training rows under the same material-pH
  continuation. No validation or test row was used for optimization.
- Two-state Moving-A5 switch micro-overfit: **passed at step 800** with the
  unchanged `VariableSetValueNet`, Adam `3e-4`, and absolute mission MSE plus
  risk-quantile Huber loss. Both actions matched the exact training actions
  (R2 pair retained; R1 geometry withheld), each with zero training regret.
  Artifact: `gate11/micro_overfit_report.json`.
- Full 32-state training micro-overfit: **failed** at the predeclared 3,000-step
  ceiling. Exact training actions were `25/32 = 78.125%` against the required
  `>=98%`; mean training regret was `0.34094`. By stratum, correct actions were
  Natural `15/20`, Complementarity `4/4`, Moving-A5 `6/8`. Artifact:
  `gate11/micro_overfit_full_report.json`.
- This is not proof that network capacity is inadequate. The current Natural
  and Moving-A5 teacher labels are one-realized-world continuations, whereas
  deployment must choose from a belief over hidden worlds. The training loss
  also fits absolute mission values and single-return risk quantiles while
  the actor selects by mission plus CVaR. These mismatches must be audited
  before blaming model size or collecting hundreds of additional labels.
- Following the predeclared hard stop, no mechanism-certified Gate 1.2 pool,
  final A3/A5 fit, full-stack rollout, or confirmatory evaluation was launched.
