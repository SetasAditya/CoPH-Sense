# Current-algorithm experiment campaign

This development campaign evaluates six claims separately: numerical consistency,
native online identification, message representation, receiver reconstruction and
replay, reciprocal discovery, and evidence handling under channel faults.
It does not add temporal estimation, adaptive transmission rates, inter-agent
physical coordination, or unknown-map navigation.

Each suite freezes its sources, settings, seeds, budgets, outcomes, and exclusion
rules before treatment execution. Results are written under
`results/current_campaign_v1/`, preserving earlier studies. A failed eligibility
gate stops its treatments; it does not justify changing the fixture or report a
zero treatment effect.

The coordinator provides immutable protocol recording, hash verification and an
artifact inventory:

```bash
cd CoPH-Stagewise
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=. python -B -m coph_stagewise.campaign verify
PYTHONPATH=. python -B -m coph_stagewise.campaign status
PYTHONPATH=. python -B -m coph_stagewise.campaign report
```

The same commands are available through `python -m coph_stagewise.cli campaign`.
`report` refreshes derived views only. `freeze` refuses a different protocol in
an existing directory. Suite manifests own their experiment-specific resumption
rules and acceptance criteria; the campaign inventory does not infer success
from the existence of a report.

## Hardware and numerical execution

GPU access requires running outside this environment's restricted sandbox. A
recorded external probe verified two NVIDIA A100 40 GB devices and a successful
PyTorch CUDA allocation. Both devices had substantial existing workloads at the
probe time. No other workload is stopped or modified.

Small float64 reconstruction and replay jobs use single-thread CPU workers to
preserve archived numerical conventions. Device changes require an explicit
record and parity checks; GPU availability alone does not make a CPU runner a
GPU implementation. The hardware probe is stored separately from operational
learner inputs.

## Interpretation

- Energy accounting does not establish global switched-system stability.
- Identification results in VMAS concern the native controller transfer; field
  reconstruction, packets and historical replay are inactive in that task.
- Message comparisons must distinguish identical packet envelopes from native
  compression codecs, and report actual bytes and processing.
- Reconstruction and replay require eligible, supported decisions; route-anchor
  effects do not establish additional receiver-learning benefit.
- Reciprocity is evaluated on layouts selected without predicted or observed
  communication gains. Unreachable or failed layouts remain explicit records.
- Fault audits test the particular channel and validation mechanism exercised;
  they do not establish calibrated uncertainty or universal robustness.
- Independent layouts, repeated noise seeds and quadrature samples have different
  statistical units and are reported separately.

The generated campaign report links each suite's detailed report, plots,
animations, exclusions and machine-readable records. Full-test logs retain
failures, including the previously documented archived-source-hash check.

## Results and reproduction

Read [the campaign findings](results/current_campaign_v1/FINDINGS.md) for the
scientific conclusions and [the generated index](results/current_campaign_v1/REPORT.md)
for individual artifacts. `worker_limit=3` in the coordination protocol refers
to assistant workstreams; each suite records its CPU process count separately.

The following commands run new evaluations; use fresh output directories and
single-thread numerical settings (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`). From `CoPH-Stagewise/`:

```bash
PYTHONPATH=. python -B -m coph_stagewise.physical_campaign --out <new-physical-dir>
PYTHONPATH=. python -B -m coph_stagewise.physical_execution_campaign --out <new-execution-dir>
PYTHONPATH=. python -B -m coph_stagewise.campaign_reconstruction gate --out <new-reconstruction-dir> --workers 8
PYTHONPATH=. python -B -m coph_stagewise.campaign_communication suite --suite representation --out <new-representation-dir> --workers 8
PYTHONPATH=. python -B -m coph_stagewise.campaign_communication suite --suite reciprocity --out <new-reciprocity-dir> --workers 8
PYTHONPATH=. python -B -m coph_stagewise.campaign_communication suite --suite faults --out <new-fault-dir>
```

From the workspace root, native evaluation uses its preserved environment:

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B \
  benchmarks/phmarl/identification_campaign.py --out <new-native-dir> --workers 4
```

Resumption requires the original suite fingerprint. If unrelated source additions
cause refusal, restore the suite's `source_archive/*.py` into an isolated
`coph_stagewise/` package and use its parent as `PYTHONPATH`, as recorded in
`reciprocity/RESUME_NOTES.md`; do not rewrite the manifest. Reconstruction owns
its `sources.zip` and separate verification archive. Reports and renderer source
hashes distinguish postprocessing from frozen experimental execution.

After all suites complete, verify episode counts, pairing, per-agent byte
ceilings, update budgets, snapshot hashes and unchanged operational sources:

```bash
PYTHONPATH=. python -B -m coph_stagewise.campaign_verify
PYTHONPATH=. python -B -m coph_stagewise.campaign report
```
