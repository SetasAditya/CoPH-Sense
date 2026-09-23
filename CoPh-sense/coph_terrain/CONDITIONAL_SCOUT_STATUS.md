# Conditional-scout implementation status

## Scope

This extension separates two questions that the source GIF conflated:

1. Can a simple value model predict the source benchmark's analytic dispatch
   rule?
2. Does dispatch improve the executed physical continuation after all scout,
   communication, delay, recovery, and mission costs are charged?

The geometric benchmark is a kinematic diagnostic. The VMAS experiment is the
candidate material-pH pipeline. Neither result is presented as final E2.

## Stage 1: frozen geometric benchmark

The port preserves the supplied geometry, costs, modes, generated sensing
sites, feature representation, controller, carrier-primary option, tests, and
GIF renderer. It trains on seed-disjoint data and evaluates `never`, `always`,
`oracle`, and `learned` on paired static/dynamic cases. Outputs include:

- analytic value error, decision accuracy, recall, false dispatch, conditional
  cost, and oracle regret;
- physical success, completion time, path lengths, report timing, route
  changes, clearance/collision, recovery, and rollout score;
- an explicit analytic-versus-physical disagreement table;
- representative GIFs labelled as kinematic diagnostics.

The frozen run used 10,000/2,000/4,000 seed-disjoint analytic examples. It
achieved `0.9765` dispatch accuracy, `0.9708` useful-dispatch recall, `0.0166`
false-dispatch rate, and `0.002680` mean analytic regret. All 48 paired
physical rollouts succeeded without collision and every dispatched scout was
recovered. Physical execution nevertheless preferred IDLE in all six cases,
disagreeing with the analytic rule in its three nominally useful cases. This
is the central benchmark finding: the supplied analytic score predicts its
own labels well, but its value does not survive the supplied physical rollout
objective. The discrepancy is retained and was not used to modify the task.

## Stage 2: material-pH integration

The deployed dispatch interface contains IDLE and feasible `ScoutTask`
candidates derived only from public observations. A task records modality,
route-relative target/viewpoint, dwell and sensing metadata, latest useful
step, estimated value, and provenance/request identifier.

For each public history and task, the paired teacher estimates

\[
V(h,d)=\mathbb E[J_{\rm idle}-J_{\rm dispatch(d)}\mid h].
\]

The original v1 smoke labels spliced observed cells into independently sampled
maps; those labels are now superseded. In finite v2, each hypothesis is a
complete declared terrain realization. It is replayed through the causal
prefix and rejected if it conflicts with carrier-visible measurements or
public motion. Carrier appearance is scored once per persistent observed cell
under the simulator's clipped-Gaussian model. The record includes posterior
weights, effective sample size, rejection reasons, finite-prior variance, and
packet Monte Carlo variance.

The dispatch branch charges travel, terrain exposure, dwell, sensing,
packets/ACKs, carrier delay, recovery, failure, and final mission cost. The
carrier must reach the goal and recover a dispatched scout. Hidden-terrain
substitution tests verify that candidate features do not change at an
identical legal history.

The corrected frozen admission split contained one supported useful, thirteen
supported unnecessary, and ten infeasible non-IDLE candidates, so training was
admitted. Separate 24/8/12-state train/validation/test manifests were fixed in
advance; one test prefix terminated before a dispatch decision, leaving eleven
test decisions. Terrain priors, continuation seeds, source hashes, and the
trained checkpoint are recorded with the result.

The complete finite-prior gate failed. The learned critic selected IDLE on all
eleven test decisions. Its mean conditional regret was `0.01907`, narrowly
inside the `<0.02` threshold, but exact action agreement was `72.7%` and useful
dispatch recall was `0/3`; unnecessary-dispatch rejection was `8/8`. The
finite-prior oracle's mean relative benefit over IDLE was `0.01907`. Because
the frozen learned policy chose IDLE, the freely selected delivery intervention
could not establish belief-to-force or trajectory causality, recovery, or a
fully charged profitable dispatch. The known-evidence canonical was also not
represented by this small frozen split. These failures correctly prevent final
E2 from launching.

This is a restricted finite-prior result, not final E2. It localizes the
remaining failure to rare useful-dispatch coverage and learned ranking: the
teacher now produces both signs without hidden-world splicing, but the fitted
policy did not recover the useful decisions. The environment, prices, layouts,
and untouched test outcomes were not altered after inspection.

## Gate

Training is allowed only when the frozen development admission pool contains both useful and
unnecessary dispatches. The held-out gate requires mean regret below `0.02`,
at least `90%` exact IDLE/task selection, `90%` useful-dispatch recall,
`90%` unnecessary-dispatch rejection, canonical useful/known/late decisions,
and a free causal rollout that changes delivered belief, material-pH force,
trajectory, and fully charged team cost. Failure stops final E2 and is
reported by source: coverage, labels, ranking, or physical continuation.

The causal check clones the exact pre-delivery state and intervenes only on
delivery. It compares pH proposals at identical position, velocity, and time,
then executes paired delivered/withheld continuations. Metric thresholds alone
cannot pass the gate when causality, canonical cases, complete dispatch value,
or recovery fails.

The complete machine-readable record is
[`results/conditional_scout_material_finite_v2_admission/gate.json`](results/conditional_scout_material_finite_v2_admission/gate.json).
