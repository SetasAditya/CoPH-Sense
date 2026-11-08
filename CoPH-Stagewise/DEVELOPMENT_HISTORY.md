# CoPH-Stagewise

Execution now attracts toward the mission goal by default. Memory windows, stage advancement and replay retain their existing stage-local targets. Use `--navigation-target stage` with `rollout`, `compare`, or `suite` to reproduce the previous controller. This is an isolated control correction; replay alignment and unsupported-entry handling remain separate research issues.

The corrected evaluation completed all 630 smoke/development episodes successfully. See [MISSION_GOAL_RESULTS.md](MISSION_GOAL_RESULTS.md) for the paired results and remaining limitations; historical diagnostic findings below describe the previous stage-target controller.

Independent **single-agent mechanism prototype**. It neither imports nor alters the two reference implementations. Reciprocal communication is the subsequent milestone, not implemented here.

## Run

From this directory, with NumPy and PyTorch installed:

```bash
python -m coph_stagewise.cli check
python -m coph_stagewise.cli rollout --out results/my_rollout --seed 0
python -m coph_stagewise.cli compare --out results/my_comparison --seeds 3
python -m coph_stagewise.cli compare --out results/my_obstacles --scene obstacle --seeds 3
```

Alternatively install with `python -m pip install -e .` and use `coph-stagewise`. Outputs must be new directories. `--previous 3` selects three previous stages; zero is a no-history ablation. Comparisons run frozen, evidence-only, full, no-history, three-history, and no-reconciliation arms with identical seed-indexed sensor noise and environment realizations.

## Learning cycle

1. Fit local RBF risk fields from measured samples and identify scalar damping from recorded executed inputs and transitions. Unit mass/input map are known. Field samples with index divisible by five and transitions with ID divisible by five are validation-only.
2. Reconstruct previous stage fields jointly using supported overlap anchors. Freeze fields and damping for counterfactual replay from stored entry states.
3. Differentiate short trajectory cost through every state/control step into scalar-potential policy coefficients. Candidate domain exit rejects the batch. Hypothetical trajectories never become sensor measurements.
4. Separately reconcile damping/policy proposals by exact box/trust-region projection, backtrack replay cost, and validate reserved observations. Roll back invalid updates.
5. Commit versions and provenance, record shaping-energy jumps, then advance overlapping stages and evict old records.

Physical storage is kinetic energy. Risk/avoidance are navigation shaping energies, applied exactly once through control. The smooth softplus avoidance penalty is not IPC or a safety certificate. Swept collision terminates a mission and incurs failure cost; there is no silent physical projection.

## Components and artifacts

- `records.py`: immutable observations/transitions; bounded stage ledger and configuration.
- `model.py`: float64 exact plant, physical work/heat, field potential, differentiable bounded control.
- `learner.py`: reconstruction, replay, reconciliation, commitment, stage advancement. No simulator import or hidden-world input.
- `world.py`: private true field/damping/geometry, local sensor, contact detection and scoring.
- `cli.py`: numerical checks, paired runs, source/config manifests and JSON serialization.

Per-run artifacts include controls, measured transitions, physical energy residuals, clipping work, model versions, acceptance reasons, validation errors, replay costs/gradient norms, final ledger and counterfactual trajectories. `summary.json` separates actual mission cost, failures, field/damping error and computation. `paired.json` contains matched cost differences. Counterfactual transition counts and wall time are reported separately rather than assigned an arbitrary price in the physical mission objective.

## Limits

- Controlled x-axis corridor progression; 30% box overlap; current plus two previous stages. This is not a general route planner.
- A 4-by-4 RBF field with inverse-softplus regression. Overlap matches latent scalar field values; derivative matching is deferred.
- Radius-limited risk samples and range-visible circular obstacles; no occlusion, communication, transformers, HMC/MPPI, or entropy dynamics.
- Curvature inverses are **uncalibrated ridge curvature**, not posterior covariance.
- The evidence-only arm changes its induced potential as the risk estimate changes, but omits counterfactual policy learning. The frozen arm retains sensing and stage structure without fitting parameters; fresh obstacles may still change its avoidance context.
- Static replay is labeled `static_current_environment_replay`. No past-time inference for a changing environment is claimed.
- Short default scenes traverse at most three stages; two- versus three-history arms can coincide. A longer synthetic test checks actual eviction; longer missions are needed for a scientific memory-budget comparison.
- The no-reconciliation arm removes physical proposal bounds and reserved-data acceptance, but retains nonfinite guards, numerical damping safeguards, input limits, and replay-domain checks.
- Timeouts/collisions remain in complete reported cost. All current runs are development studies, not held-out superiority evidence.

## Attribution

Exact held-force flow and energy accounting adapt `belief_planning.port_step` and `complete_potential.energy_balance` in CoPH-Sense and `navigation.plant_step` inside the CoPH-v2 ZIP. The scalar-gradient policy follows the structural idea of `NavigationPotential`, but preserves position derivatives through replay. No historical weights/results or full-scene geometry are imported.

The research specification is `../docs/COOPERATIVE_STAGEWISE_PH_FORMULATION.md`. See `RESULTS.md` for executed checks and measured outcomes. Numerical validity and mission improvement are separate gates.

The executed validation suite passed **13 tests**. Rebuild the development summary with:

```bash
python -m coph_stagewise.report results/risk_validated results/obstacle_validated --out RESULTS.md
```

The `*_validated` directories contain the current recorded comparisons and full final ledgers. Earlier `*_development` directories preserve initial runs before ledger serialization was added. Neither set is untouched test data.

## Diagnostic evaluation suite

The newer [cooperative attribution suite](ATTRIBUTION_RESULTS.md) isolates geometry reports, remote route anchors, supported reconstruction, and replay starts on blocked and soft-risk mazes. The [implementation contract](IMPLEMENTATION_CONTRACT.md) distinguishes implemented mechanisms from approximations and deferred parts of the formulation. Run it with `python -m coph_stagewise.attribution_suite suite --out results/attribution_v1 --profile all`; `report` and `render` regenerate artifacts from its logs.

The evaluation suite adds nine scenario variants, seven paired arms, evaluator-only replay audits, compressed update snapshots, static figures and four-panel GIFs. See [EVALUATION.md](EVALUATION.md) for commands and the frozen protocol, including the current-state comparator's explicitly reported padding budget. The numerical and evaluation contract suite now contains **22 tests**. Existing learner training rules and earlier results are preserved.

Executed diagnostic findings are summarized in [DIAGNOSTIC_RESULTS.md](DIAGNOSTIC_RESULTS.md), with links to both complete suite reports. The development comparison shows improved identification but worse overall mission performance than frozen control; it does not establish an advantage from retaining more stages.

Cooperative extension: see [implementation, protocol, limitations and reproduction](COOPERATIVE.md). `cooperative-suite` runs two identical local learners with reciprocal Hamiltonian-information exchange, paired communication arms, separate calibration/training seeds, and evaluator-only audits. Existing single-agent commands remain available.

[Cooperative evaluation results and limitations](COOPERATIVE_RESULTS.md) include the complete 1,008-episode development comparison and generated artifacts.

The opt-in [completion-first navigation mode](COMPLETION_FIRST.md) adds a progress-preserving scalar potential, direct goal capture, bounded stagnation replanning, and versioned receiver contexts. Its frozen evaluation command preserves prior results and gates held-out execution on complete, collision-free development performance. See `results/completion_first_v1/` for run records and readiness gates; passing unit tests alone does not establish a working navigation system.

[Completion-first results and GIFs](COMPLETION_FIRST_RESULTS.md): routed-progress passes both development configurations; protected-progress fails through hazard-slalom braking stalls. That original all-controller gate remains failed; selected-controller validation is reported separately below.

[Selected routed-progress held-out validation](ROUTED_VALIDATION_RESULTS.md) passed all 128 agent missions with zero collisions and timeouts across 64 episodes. [Reproduce the separate validation command](ROUTED_VALIDATION.md). Navigation reliability is established for the tested static scenes; an added completion benefit from communication is not established.

[Crossing traverses](CROSSING_DEMO.md): opposite starts, swapped goals and two hidden full-width doors. Reciprocal geometry sharing saves 27.1 m of team travel (predicted 28.0 m), twice the sum of the two one-way arms; A→B alone saves nothing. Development demonstration; the two seeds differ only in sensor noise.

Focused decision diagnostics: [configuration, commands, and interpretation](DECISION_DIAGNOSTICS.md).

[Graded hazards](HAZARD_DEMO.md): in four heterogeneous-agent development scenarios, sharing a learned graded environmental field let the risk-averse receiver apply its own preferences, with lower worst-case excess cost than the tested binary summaries and sender-specific advice. Raw measurements matched it at more bytes; receiver field assimilation added no navigation benefit in this pilot.

[Complementary-coverage reconstruction diagnosis](COMPLEMENTARY_RECONSTRUCTION.md): an opt-in `receiver-reconstruction --profile complementary` experiment gates navigation comparisons on actual packet fitting and supported outgoing-edge queries. It freezes the prescribed fork, acquisition interval and development seeds, then separates immediate model changes from navigation outcomes. Existing hazard-pilot defaults and results remain unchanged.

See [Receiver fitting and validation corrections](RECEIVER_CORRECTIONS.md) for the opt-in bounded fit, additional validation gate, and frozen diagnostic commands.

See [Active-set receiver fitting](ACTIVE_SET_RECEIVER.md) for independent step/search options and the gated four-way solver experiment.

### Decision-sensitive receiver reconstruction

The opt-in [preregistered experiment](DECISION_SENSITIVE_SPEC.md) freezes a 20-scene, eight-seed screening grid, requires all eight seeds to pass each scene's pre-branch eligibility gate, and runs four fixed-rate receiver-fitting arms only if a grid scene qualifies. Run `python -m coph_stagewise.decision_sensitive suite --out results/decision_sensitive_v1 --workers 4` with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. The module also provides `freeze`, `gate`, `report`, and `render`. No controller or learner defaults change.
