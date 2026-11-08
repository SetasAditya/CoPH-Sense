# Additive route integration and decision-window packet sequences

`route_integral_mode="additive_v1"` is an opt-in navigation-context setting.
The missing setting retains legacy integration. Apply it to the active navigation
context and retained `entry_navigation` contexts together when enabling the mode
on an existing agent; newly created stages inherit the active context.

The common route scorer builds one nonnegative piecewise-constant risk density
on each surveyed edge, using the existing nominal 0.3-unit sampling scale.
All partial-edge costs use exact overlaps with those same cells. Off-axis
connections follow a clearance-checked projection spur and then the edge.
Execution and replay use that scored polyline. A journey split into partial
edges therefore retains the same integral; extra backtracking has positive cost.
This consistency property does not certify environmental-field support, improve
field accuracy by itself, or establish navigation stability.

`SequenceAdvertisement` supplies explicit estimated graph costs/support,
surveyed layout and current position. Its bounded, compressed serialization is
measured directly. It contains no raw readings, reserved validation history,
executed transitions or truth. The sequence predictor considers permutations of
up to three ordinary locally reconstructed stage packets at fixed opportunities.
It uses nominal centreline travel at 1 m/s with frozen evidence, and never creates
sensor observations. This graph-level forecast is not a physical pH replay.

The evaluation freezes eight existing hazard specifications, two observation-noise
seeds, all 64 preserved snapshot inputs and the earliest carrier snapshot per
scene/seed before any comparisons. It requires eight bitwise legacy operational
reproductions, runs matched instantaneous receiver updates, and compares frozen
packet orders through 32 actual simulator steps with indexed sensor noise.
Separate 16 full episodes test additive routing with rotation and repaired
coverage at the existing four-million-byte per-agent ceilings.

The real receiver continuations retain reconstruction, damping identification,
validation, reconciliation, safety, rollback and the 1,024-flow update limit.
Future sensor samples are evaluator-only branch data. Hypothetical packet headers
advance at fixed opportunities while their source evidence and absolute expiry
remain frozen. These are causal diagnostics with no other incoming packets,
not predictions of future sender sensing or full cooperative navigation runs.

Commands, run from `CoPH-Stagewise/`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/coph-mpl \
python -m coph_stagewise.decision_sequence_suite suite --workers 4

MPLCONFIGDIR=/tmp/coph-mpl \
python -m coph_stagewise.decision_sequence_suite render
```

`freeze` and `report` commands are also available. Resume refuses changed source,
input, configuration or numerical-environment fingerprints. Outputs and frozen
sources are in `results/decision_sequence_v1/`; prior artifacts are preserved.

Incurred time plus declared weighted risk is distinct from the diagnostic endpoint
that adds a remaining-route spatial cost at nominal speed. The latter is not full
mission cost. Selection budgets use a fresh four-million-byte diagnostic window,
not the unavailable remaining lifetime allowance of a saved operational snapshot.
All status, payload, acknowledgement and computation costs remain visible.

The old `decision_packet_v1` diagnostic's hand-written truth helper omitted the
world background risk 0.02. This experiment reprices those saved graph routes
using `NetworkWorld.risk_truth`, retaining the original measurements separately.
The operational world and mission-cost logger already included the background.
