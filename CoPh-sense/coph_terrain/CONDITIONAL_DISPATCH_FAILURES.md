# Conditional dispatch: current failures and their meaning

**Status: 23 September 2026.** This note records the remaining failures after
repairing the conditional-scout teacher and acceptance gate. It distinguishes
observed failures from possible explanations. The corrected experiment is a
restricted finite-terrain-prior diagnostic, not final E2.

## Intended decision

The carrier chooses between `IDLE` and legally feasible scout tasks from its
own history:

\[
d^*(h)=\arg\max_{d\in\{\mathrm{IDLE}\}\cup\mathcal D(h)}
V(h,d),\qquad
V(h,d)=\mathbb E[J_{\rm idle}-J_{\rm dispatch(d)}\mid h].
\]

The expectation is over complete terrain realizations compatible with the
carrier's observations and the public physical history. A profitable dispatch
must remain better after scout travel, exposure, dwell, sensing, packet/ACK,
carrier delay, recovery, failure, and final mission costs are charged.

## What has been repaired

The current teacher no longer creates hypotheses by inserting observed cells
into independently sampled hidden maps. It now:

- records carrier observations, delivered evidence, ACKs, timestamps, issued
  actions, and public teammate motion;
- excludes uncommunicated scout readings;
- declares complete terrain hypotheses with fixed weights and hashes;
- scores persistent appearance observations once with the simulator's
  clipped-Gaussian likelihood;
- enforces exact measurement compatibility and replays the causal prefix;
- blocks labels when posterior support is empty;
- averages paired complete IDLE/dispatch continuations with an indexed random
  packet schedule;
- records posterior weights, effective sample size, rejection causes,
  finite-prior variation, and packet Monte Carlo variation;
- fixes the IDLE prediction to exactly zero, masks infeasible tasks, and selects
  IDLE on numerical ties;
- checks delivery causally by comparing delivered and withheld branches at the
  same position, velocity, time, goal, and unrelated evidence;
- requires recovery, complete ledger reconciliation, and fully charged benefit
  before accepting the material gate.

The implementation boundary is therefore substantially healthier. The current
failure is no longer evidence of the previous hidden-map-splicing bug.

## Confirmed failures

### 1. The learned scorer collapses to IDLE

On the untouched test split, the learned policy selected IDLE in all eleven
evaluable states. The finite-prior conditional oracle dispatched in three.

| Metric | Result | Required |
| --- | ---: | ---: |
| Mean conditional regret | **0.01907** | `< 0.02` |
| Optimal-action agreement | **72.7%** | `>= 90%` |
| Useful-dispatch recall | **0/3** | `>= 90%` |
| Unnecessary-dispatch rejection | **8/8** | `>= 90%` |

Mean regret passes only because useful dispatches are rare and their average
margin is modest. It therefore hides a complete failure on the scientifically
important positive class. IDLE itself obtains the same test metrics as the
learned policy.

### 2. Useful-dispatch supervision is sparse

The frozen admission split contained one useful, thirteen unnecessary, and ten
infeasible non-IDLE candidates. The 24-state training split contained only four
useful candidates, with several unresolved candidates. This is enough to admit
training under the predeclared rule, but it is weak support for learning the
positive ranking boundary.

This is a measured coverage limitation. It does not yet prove whether a larger
predeclared training pool would fix the learner.

### 3. The complete causal gate cannot pass when the policy never dispatches

The demonstration must be chosen by the frozen learned policy. Because it chose
IDLE, no freely selected rollout could establish

\[
\text{dispatch}\rightarrow\text{measurement}\rightarrow\text{delivery}
\rightarrow\text{belief change}\rightarrow\text{pH proposal change}
\rightarrow\text{trajectory divergence}.
\]

This is a policy failure, rather than a reason to force a favorable dispatch.
The causal intervention code exists, but the learned policy did not activate
it.

### 4. Fully charged dispatch benefit and recovery were not demonstrated by the learned policy

Since the selected action was IDLE, the learned-policy evaluation could not
show that a dispatched scout was recovered and cheaper than IDLE after all
resource costs. These subgates correctly remain failed. A delivery-only benefit
would not be sufficient.

### 5. One required canonical context is absent

The frozen data contain useful and too-late contexts, but do not contain the
predeclared known-evidence canonical case under the implemented public-history
predicate. The complete gate therefore fails independently of the aggregate
metrics. This is a coverage defect in the restricted diagnostic, not evidence
that the policy would necessarily fail that behavior.

### 6. The earlier geometric objective does not match physical execution

The historical geometric scorer learned its analytic labels well, but the
physical rollout preferred IDLE in every analytically useful case. The physical
visual score excludes the analytic route-cost term, so the two objectives are
not directly comparable. Historical scores are preserved; the mismatch is now
documented rather than explained away as motion cost alone.

## What v9 does and does not fix

`CoPH-Sense-main-v9` supplies a useful dynamic-risk-field experiment: the scout
moves, sends a time-indexed spatial risk packet, changes the carrier's belief,
and returns. It is suitable as an information-assimilation and controller
diagnostic.

It does not fix the failed material dispatch learner:

- dispatch is a fixed threshold heuristic, not a learned conditional value;
- its controller factorial chooses a reference packet by inspecting the true
  hidden field and maximizing evaluator-side route improvement;
- it has no finite-prior legal-history posterior or causal replay teacher;
- it has no frozen train/validation/test dispatch gate;
- post-arrival recovery delay is simulated but excluded from `total_cost`,
  which uses carrier goal time;
- the included tests check basic mechanics, not conditional regret, leakage,
  recovery accounting, or delivery causality.

The time-indexed risk packet, dynamic scout-site allocation, and matched
controller baselines may be ported later. They must be evaluated through the
repaired teacher and ledger rather than used as evidence that dispatch learning
has passed.

## Unresolved explanations

The following are hypotheses, not established conclusions:

1. **Positive coverage may be insufficient.** Four useful training candidates
   may not support the value/ranking learner.
2. **Value scale may favor the safe solution.** Large costs for clearly bad
   tasks can dominate regression while the small decisive differences near
   zero receive little influence.
3. **The representation may omit a decisive legal statistic.** The current
   identical-input audit found no conflicting exact labels in the collected
   data, but the sample is too small to establish general sufficiency.
4. **Posterior support may be too coarse.** A four-world finite prior is exact
   for the declared diagnostic but may yield brittle conditional values.
5. **Candidate generation may provide too few useful, actionable tasks.** This
   could be a property of the fixed task distribution rather than critic
   optimization.

None of these hypotheses justifies changing held-out worlds, prices, terrain,
or acceptance thresholds after seeing the result.

## Current scientific conclusion

The repaired system can compute auditable conditional dispatch labels under a
declared finite terrain prior. The frozen diagnostic contains some profitable
dispatches, but the learned candidate scorer does not identify them and behaves
like `always IDLE`. Consequently, the material conditional-scout pipeline has
not earned progression to final E2.

The result does **not** show that scout information is never valuable. It shows
that the present learner and training coverage have not converted that value
into a locally executable dispatch policy.

## Smallest defensible next gate

If this line is continued, keep the teacher, material-pH executor, physics,
prices, task layouts, existing validation/test manifests, and thresholds
frozen. Change only development/training coverage:

1. generate a larger seed-disjoint finite-prior training pool using the same
   declared generator;
2. confirm useful and unnecessary decisions before fitting, without selecting
   rows according to model performance;
3. report the useful margin distribution and train/validation coverage;
4. fit the existing scorer once with stratified state sampling or
   decision-focused weighting declared before evaluation;
5. evaluate once on the existing untouched test manifest;
6. require the complete gate, including freely selected causality, recovery,
   known/useful/late canonicals, and fully charged benefit.

If useful coverage remains negligible, or the fixed model again collapses to
IDLE, close this restricted dispatch experiment as a negative result. Do not
launch final E2 from aggregate regret alone.

## Reproducible records

- Corrected gate:
  `results/conditional_scout_material_finite_v2_admission/gate.json`
- Frozen split manifests:
  `results/conditional_scout_material_finite_v2_admission/*_manifest.json`
- Full status and historical geometric result:
  `CONDITIONAL_SCOUT_STATUS.md`
- Main implementation:
  `conditional_scout_material.py`
- Fail-closed runner:
  `run_conditional_scout_material.py`
