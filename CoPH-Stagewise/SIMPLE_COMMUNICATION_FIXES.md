# Simple, opt-in communication repairs

The existing `FixedRateRepairLink`, packet schema and defaults are unchanged.
`simple_communication.py` adds `SimpleOptions(ack_route_used=False,
compact_status=False)` and `SimpleRepairLink`.

- **Acknowledgment/interface option:** include exact live source IDs from
  received field evidence and retained route-evidence cells. Advertise a bounded
  set of known route-anchor coordinates. Score only previously uncovered receiver
  route queries that the candidate's actual anchors and local measured support
  cover. Fully acknowledged or already-covered candidates cannot win by merely
  containing many measurements. This option keeps legacy status-before-payload
  charging.
- **Overhead option:** serialize full or delta advertisements losslessly with
  compression, digests and independently checked cache bases. Only successfully
  transmitted/decoded advertisements advance caches. Reserve each endpoint's
  normal payload before admitting advertisements. If the joint admission cannot
  fit, use normal retained-stage rotation without charging an unsent status.
  This option bundles coding and admission; the study does not separate them.

Neither option changes fixed transmission opportunities, fitting, routing,
controller, support certification or environmental packet contents. Advertised
coordinates/acknowledgments are a compact knowledge interface, not a calibrated
posterior. Receiver-local export previews are computation, not free information:
their time and work are charged separately from actual transmitted bytes.

The frozen suite contains four existing hazard scenes, seeds86100–86101, byte
ceilings8k/16k/32k/4MB, five arms (rotation, existing selector, acknowledgment-only,
overhead-only, both), and one disabled run per scene/seed: **168 episodes**. All
baselines are rerun rather than substituted. The original scene specifications,
risk weights,25-second delay and horizons are preserved.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python -m coph_stagewise.simple_communication_suite suite \
  --out results/simple_fixes_v1/communication --workers 16
PYTHONPATH=. python -m coph_stagewise.simple_communication_suite render \
  --out results/simple_fixes_v1/communication
PYTHONPATH=. python -m coph_stagewise.simple_communication_report \
  --out results/simple_fixes_v1/communication
```

Use the same single-thread environment for each command. Resume requires the
original source/configuration fingerprint; do not edit a frozen manifest.

Read `results/simple_fixes_v1/communication/FINDINGS.md` and `VERDICT.json` for
endpoint-separated paired results. Codec, provenance/expiry, reservation,
fallback, hidden-data isolation and cache tests are in
`tests/test_simple_communication.py`. No adaptive-rate or generalization claim is
established by this development campaign.
