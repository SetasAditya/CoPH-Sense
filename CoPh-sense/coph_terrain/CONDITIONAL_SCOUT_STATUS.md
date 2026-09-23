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

Each hypothesis keeps the same public geometry, physics state, observations,
messages, and RNG snapshot. Unobserved material is drawn from another
realization of the same parent map; observed cells retain their realized
values. IDLE and dispatch then use cloned states, paired randomness, the same
compact requested-report protocol, and the frozen material-pH executor.

The dispatch branch charges travel, terrain exposure, dwell, sensing,
packets/ACKs, carrier delay, recovery, failure, and final mission cost. The
carrier must reach the goal and recover a dispatched scout. Hidden-terrain
substitution tests verify that candidate features do not change at an
identical legal history.

The one-state smoke admission audit produced zero positive and four negative
non-IDLE candidates. It therefore stopped before training, as required by the
predeclared gate. This is a coverage observation from a smoke run, not the
final admission result.

## Gate

Training is allowed only when the fixed source pool contains both useful and
unnecessary dispatches. The held-out gate requires mean regret below `0.02`,
at least `90%` exact IDLE/task selection, `90%` useful-dispatch recall,
`90%` unnecessary-dispatch rejection, canonical useful/known/late decisions,
and a free causal rollout that changes delivered belief, material-pH force,
trajectory, and fully charged team cost. Failure stops final E2 and is
reported by source: coverage, labels, ranking, or physical continuation.
