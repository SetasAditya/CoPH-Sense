# Shared-origin, staggered-start communication demonstration

Both agents use routed-progress and share the same start and destination. Agent 0 starts at 0.00 seconds; agent 1 starts at exactly 5.00 seconds. The second agent is inactive beforehand: no sensing, learning, transmission, reception, or buffered pre-start evidence. This is a new development demonstration, separate from the preserved held-out validation.

The simulator uses an event clock with each agent retaining its 0.12-second held-input step. A five-second offset is not rounded to a multiple of that step. Both agents receive their full active mission horizon. Each sender has a fixed communication opportunity every eight local steps (0.96 seconds); packets enter an explicit queue and are delivered at the receiver's next control boundary. Both agents sense before same-time exchanges. Sender/receiver timestamps, packet age, status/payload/acknowledgment bytes, acceptance reasons, and model versions are recorded.

Communication-off and reciprocal runs have matched, predeclared local and receive-update opportunities. This prevents additional optimization calls alone from being mistaken for an information benefit. The controller, field fitting, replay, validation, reconciliation, and rollback implementations are unchanged. The original source fingerprints are verified before the new run. Sensor observations retain global timestamps and independent measurement IDs. Physical work, safety/clipping work, route/model/stage/observation shaping jumps, and replay work are kept separate.

At the first eligible packet in each receiver stage, an evaluator clones the complete receiver state into deliver and withhold branches. Both branches perform the same local update. The audit records the difference in environmental coefficients, applied control, and shaping Hamiltonian at fixed physical state. These are immediate packet effects, not mission-benefit predictions. Audit work is separate and cannot mutate operational agents or become observations.

Two layouts are frozen before execution: a 12-unit slalom with a narrow passage and multiple risk patches (420 steps per agent), and the 20-unit long obstacle corridor (600 steps per agent). Each uses development seeds 73000 and 73001 with both communication settings: eight episodes. Animation seed 73000 is fixed before execution. Prior validation and final-test namespaces remain untouched.

Run from `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
  python -m unittest discover -s tests -p test_staggered_demo.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
  python -m coph_stagewise.staggered_demo --out results/staggered_demo_v1 --workers 8
```

Use `--render-only` to regenerate the report and GIFs. Animations compare off/on trajectories, show the second agent waiting and activating, draw packet-delivery arrows and field anchors, and display the complete communication timeline alongside packet-only causal effects. Red arrows indicate rejected evidence; ordinary local fitting is not attributed to packets. Displayed positions are recorded boundary states, not fabricated interpolation. Geometry overlays use logged model snapshots. Playback is time-compressed; global-time labels are authoritative.

A five-second head start does not guarantee useful messages. Current sender anchors can lie outside the follower's retained spatial support and be rejected. No support expansion, artificial effect magnification, adaptive communication, or inter-agent contact dynamics is added. Failures remain visible in episode records and the failure gallery.
