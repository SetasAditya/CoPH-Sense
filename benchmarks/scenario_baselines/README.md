# Scenario baseline admission

This directory records source discovery, equation-level implementation checks, and the native-example admission gate for DistGP and UGV-conditioned exposure planning. It contains no CoPH-versus-baseline result.

| Baseline | Pinned primary source | Current status |
|---|---|---|
| DistGP | [arXiv:2603.07351v1](https://arxiv.org/html/2603.07351v1) | Independent FITC/tree-GBP primitives verified; published native experiment pending |
| UGV-conditioned exposure planning | [arXiv:2606.12306v3](https://arxiv.org/html/2606.12306v3) | Independent moment/risk/route-cost primitives verified; published native experiment pending |

## What is implemented

`primitive_checks.py` independently implements the relevant published equations on tiny declared inputs. `primitive_inputs.json` and the code hash were frozen in `primitive_manifest.json` before outputs were generated. `primitive_results.json` reports:

- FITC natural parameters against direct Gaussian conditioning, and tree Gaussian belief propagation against direct conditioning of the identical assembled factor graph (means and marginal covariances).
- Log-normal moments against deterministic Gaussian quadrature; exposure risk and the eight-connected route objective; A* against Dijkstra on the identical tiny fixed map, with an admissible heuristic and explicit objective-component sums.

The kernel, observations, map, speed, and corner convention in these checks are explicitly ours. These are **primitive correctness checks only**, not native-example reproduction, published numerical-result reproduction, full active-sensing evaluation, or evidence of CoPH superiority.

## What is still missing

Author code is not mandatory. A separately identified independent implementation can be admitted after its equations, complete native problem, and source-grounded reference checks are documented. The remaining blockers are incomplete native inputs/settings, detailed in [REIMPLEMENTATION_FEASIBILITY.md](REIMPLEMENTATION_FEASIBILITY.md):

- **DistGP:** static field kernel/noise, trajectory speed/dt/sample count, inducing resolution, initial tree, and inference/convergence schedule. The public SST and occupancy descriptions also omit particular bounds/assets or selected settings needed to reproduce their numbers.
- **Exposure planning:** exact OSM crop/preprocessing, absolute exposure scale, threat/start distributions, sensing confidence/cadence, native UGV speed/prefix, Dubins radius, and IPP budget/tree settings. Table I supplies many method coefficients; those are not the missing part.

On 2026-11-24, eight primary/API source checks returned HTTP 200. The manuscripts, author pages, public author repository catalogs, and exact-name/title searches exposed no corresponding official implementation/native runner. `source_capture.json` preserves URLs, hashes, and search metadata. This does not establish that no implementation exists. No author was contacted and no unrelated repository was substituted.

## Reproduction and comparison order

1. Complete the missing native assets/settings through released material or an explicitly declared independent protocol, with omissions and assumptions visible.
2. Pin official code or independent source plus paper version; install only into an isolated environment.
3. Execute a complete native-setting check with declared inputs, seeds, environment, reference endpoints and logs.
4. Validate the CoPH-facing adapter on identical inputs; finalize and freeze matched settings before comparative outcomes.

`matched_protocol_intent.json` and `protocol_intent_seal.json` preserve the initial comparison intent and source-search status. They are deliberately unchanged and **are not a finalized executable comparison protocol**. Their initial official-code-only wording is historical; current admission accepts either provenance route. A mathematical primitive check cannot satisfy native admission.

For DistGP, the intended first comparison uses identical four-reading batches and supported path pairs from the existing saved-state study. That isolates representation, not distributed fusion. For exposure planning, retain the original one-UGV/aerial-scout roles; a controlled-scout CoPH adapter tests belief/route use, while full acquisition comparison also requires matched aerial mobility and planning budgets. A reciprocal planar extension must be named separately.

## Commands

From the repository root:

```bash
python benchmarks/scenario_baselines/audit.py status
python benchmarks/scenario_baselines/audit.py verify
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/scenario_baselines/primitive_checks.py --out /tmp/coph-baseline-primitives.json
python -m unittest discover -s benchmarks/scenario_baselines -p 'test_*.py' -v
```

The admission tool's `freeze` command requires a passing native record, complete native specifications and reference checks. Official-source records require a pinned repository; independent-source records require the exact paper version, equation verification, and a matching own-source hash. It does not automatically certify scientific fidelity from metadata. The existing phMARL reproduction remains in `../phmarl/`.
