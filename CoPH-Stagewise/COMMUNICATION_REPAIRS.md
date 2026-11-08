# Opt-in communication mechanisms

`communication_repairs.py` adds three independent options. Defaults preserve legacy coding, retained-stage rotation, and no extra consistency check:

```python
CommunicationRepairOptions(
    wire_mode="legacy",                 # or "compact_v1"
    payload_selection="rotation",       # or "receiver_relevant"
    consistency_mode="off",             # or "local_residual"
)
```

The compact codec retains the complete packet exactly. It compresses canonical JSON into a versioned binary frame with a size declaration and SHA-256 digest. It removes no evidence, gradients, covariance, support, timestamps, or provenance. Decoding returns the ordinary `Packet`/`MazePacket`, which still passes through ordinary receiver validation.

`FixedRateRepairLink` is an opt-in transport wrapper. After ordinary sensing, call `offer` at the sender's existing update interval, then `deliver_due` at receiver control boundaries. The link accounts separately for receiver advertisements, sender wire payloads, acknowledgments, and processing. Delivery runs ordinary assimilation and one receiver update for the due batch. Repeated delivery is empty. Payload selection does not change transmission opportunity times.

Receiver advertisements contain bounded route coordinates, sampled local support coordinates, model version, and evidence acknowledgments; no local measured values or validation records. Selection scores retained stages by supported advertised route queries, missing advertised support, and unacknowledged local evidence. Missing advertised support is only an approximation to missing receiver knowledge. Advertisements can cost more than a compact payload and are charged explicitly.

The optional local-residual check quarantines messages inconsistent with a local-only field proposal at locally supported anchors. Its scale uses reserved local error plus the declared sensor-noise variance. The threshold is fixed at four. Unsupported anchors are unchecked; this heuristic is not a calibrated confidence test. It changes neither ordinary fitting nor its validator. Quarantining a packet prevents its insertion into both route and fitting stores; the ordinary local update still runs.

## Saved-state evaluation

Results are under `results/mechanism_repairs_v1/communication/`. The initial suite compares legacy and corrected receiver fitting on 160 preserved complementary snapshots, with eight packet conditions. It does not regenerate the earlier campaign's unsaved acquisition contexts. A consistency phase runs only after the corrected receiver shows residual per-case bias damage.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python -m coph_stagewise.communication_repair_suite freeze --out results/mechanism_repairs_v1/communication

# Same thread environment for the following commands.
PYTHONPATH=. python -m coph_stagewise.communication_repair_suite payloads --out results/mechanism_repairs_v1/communication
PYTHONPATH=. python -m coph_stagewise.communication_repair_suite faults --out results/mechanism_repairs_v1/communication --workers 8
```

Every resume verifies explicit runtime dependencies and saved-input hashes. The preserved run used its original frozen package for the gated continuation after an unrelated concurrent taper-width correction changed a runtime file; see `frozen_runtime_resume.json`.

To resume the preserved initial and consistency phases after later runtime edits, restore their verified sources automatically:

```bash
PYTHONPATH=. python -m coph_stagewise.communication_repairs_reproduce --out results/mechanism_repairs_v1/communication --phase initial --workers 8
PYTHONPATH=. python -m coph_stagewise.communication_repairs_reproduce --out results/mechanism_repairs_v1/communication --phase consistency --workers 8
```

The separately frozen consistency phase is `consistency_v2/`. The first attempt computed its updates but failed to serialize a NumPy diagnostic boolean; incomplete outputs remain under `aborted_serialization_v1/`. The version-two amendment converts diagnostic scalars to Python primitives and changes no arithmetic, threshold, or gate. Its saved-state results reject useful clean information as well as biased information, so the heuristic remains off by default.

The fair payload benchmark gives field, raw readings, and sampled costmap the identical four actual measurements, locations, support and provenance. The field is refitted from only that batch. This removes the earlier four-versus-twelve sampling mismatch. It measures byte/error tradeoffs and codec fidelity, not navigation superiority. Actual compact transport and receiver updates are exercised separately, as are controlled requests for older retained-stage information. No new mission-level benefit is claimed.
