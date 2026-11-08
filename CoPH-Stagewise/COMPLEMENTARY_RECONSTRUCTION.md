# Complementary-coverage reconstruction diagnosis

This opt-in development profile preserves the hazard-pilot command and outputs. It tests the prescribed surveyed fork, two noise seeds (79100–79101), three coverage cases, fixed-rate reciprocal exchange, and no policy replay. It does not change the controller, learner, support, validation or safety rules.

Run from `CoPH-Stagewise`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/coph-mpl \
python -m coph_stagewise.receiver_reconstruction suite \
  --profile complementary --out results/complementary_reconstruction_v1

python -m coph_stagewise.receiver_reconstruction analyze --profile complementary --out results/complementary_reconstruction_v1
python -m coph_stagewise.receiver_reconstruction report --profile complementary --out results/complementary_reconstruction_v1
python -m coph_stagewise.receiver_reconstruction render --profile complementary --out results/complementary_reconstruction_v1
```

The suite writes a frozen manifest and source archive before acquisition. Existing outputs are resumed only when the source/configuration fingerprint matches. The runner deliberately executes sequentially; `--workers` remains a hazard-profile option. All three methods use the unchanged packet exporter with retained-stage rotation.

Both agents sense, move and fit normally from time zero. At 0.96 s, after both agents sense but before export and the next local update, the communication-free acquisition run is frozen. Exports run on copies of those actual sender states. Gate checks inspect actual packets, receiver rectangles, the current sensor disk, committed measurement-supported route quadrature on edges (1,2) and (1,4), branch traversal, validation, and nonzero message-fitting work. The informative direction is 0→1 for complementary and 1→0 for reverse. Each must pass at both seeds. Redundant coverage is recorded without requiring novelty.

On any required gate failure, all 18 comparative episodes remain explicitly not run. No alternate fixtures or seeds are tried. Matched acquisition audits and support/error plots are still produced. Three-arm navigation GIFs are generated only if navigation comparisons are authorized by a passed gate.

For a passing gate, the suite runs shared anchors, direct local field, and direct cooperative field. It verifies each initial acquisition prefix against the communication-free reference and preserves first-delivery-per-stage and last-pre-branch-delivery complete evaluator snapshots. Each audit holds packet contents and remote anchors fixed and toggles received residual fitting. Audit computation is separate from operational flow accounting. Sources, proposed and committed fields, validation outcomes, fixed-anchor scores, regenerated-anchor scores and controls remain inspectable.

Branch entry is a geometric marker: projection more than 0.25 m along either outgoing edge, within 0.25 m perpendicular distance. It is not the instant of a route-planning decision. Redundant initial coordinates are identical with independent indexed noise; distinct risk weights may produce different subsequent acquisition paths, which the records report rather than assuming they coincide.

`*.pkl.gz` files are trusted local evaluator snapshots. Never load untrusted pickle files. Runtime learners never receive these complete snapshots or evaluator truth. Prior 25-second relay results are a **non-overlap reference**, useful for routing despite being ineligible for reconstruction, not an irrelevant-information control.

Reproduce from the frozen archive by extracting `sources.zip` into a separate directory and running with that directory on `PYTHONPATH`, using a fresh output directory. The JSON manifest contains every scene, learner and channel setting and source hash. Tests: `python -m unittest discover -s tests -p 'test_complementary_reconstruction.py' -v` plus the existing reconstruction, route-field, cooperative, navigation and replay contract tests.

Artifact verification and archive reproduction:

```sh
bash scripts/verify_complementary_reconstruction.sh results/complementary_reconstruction_v1
bash scripts/reproduce_complementary_reconstruction.sh results/complementary_reconstruction_v1 results/complementary_reconstruction_reproduction
```

The verifier checks archive/source/config integrity, acquisition equality, actual first exported packets, measured-only export provenance, payload bytes, flow ceilings, exactly-once delivery, required audit selection, frozen audit snapshots, weighted quadrature sums and animation frame manifests. The reproduction script extracts the frozen archive to `/tmp`, applies the documented presentation-only missing-margin correction when present, and executes it against a fresh output directory. That fresh manifest records the corrected reporter hash; controller and learner sources match the original archive. Neither command overwrites existing episodes.

Completed evaluation: [findings and interpretation](results/complementary_reconstruction_v1/FINDINGS.md), [generated report and GIFs](results/complementary_reconstruction_v1/REPORT.md), and [artifact verification](results/complementary_reconstruction_v1/artifact_verification.json).
