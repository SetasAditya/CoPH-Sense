# Decision-based packet scoring diagnostic

The opt-in `decision_packet` module ranks ordinary retained-stage packets by an
estimated change in receiver route cost. It uses explicitly advertised route
quadrature values/support and the receiver risk weight, plus the sender's
local-only reconstructed fields. There are no operational link/default changes.

Run from `CoPH-Stagewise/`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m coph_stagewise.decision_packet_suite suite --workers 4

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. python results/decision_packet_v1/verify_and_diagnose.py
```

`freeze` and `report` subcommands are also available. Resume requires identical
source, input, configuration and numerical-environment fingerprints. Sources are
archived before baseline regeneration or packet interventions.

The eight existing rotation episodes are regenerated to capture complete sender
and receiver states; archived JSON lacks those states. Operational decisions,
controls, models, work and events must reproduce before comparisons proceed.
The experiment uses the first opportunity for each sender/receiver-stage pair
in both directions. Every candidate is evaluated on an isolated complete receiver
copy through ordinary assimilation, update, validation, rollback and routing.

The endpoint is static spatial route quadrature under evaluator-only truth,
including connectors. It is not executed mission cost or a navigation trial.
Frozen snapshots receive a clock-only advance for legal receipt; future sensing
and actual next-boundary delivery are not simulated. Advertisements are measured
as actual standalone compressed JSON, with no assumed cache savings. No arbitrary
conversion between bytes and route-cost units is introduced.

Results: [FINDINGS.md](results/decision_packet_v1/FINDINGS.md) and
[REPORT.md](results/decision_packet_v1/REPORT.md). The progression gate failed:
the proposed selector chose the same stage as rotation on 64/64 snapshots, while
the repaired coverage selector avoided two small adverse connector changes.
All 148 post-update routes were correctly predicted. Finer-sampling diagnostics
identify nonadditive edge/connector quadrature in the two benefit-sign errors.
Deferred or multi-packet information value remains unmeasured.

The score stays experimental. No new navigation treatments, held-out runs,
controller changes or additional tuning were performed.
