# Native pH-MARL checkpoint reproduction

This directory evaluates the authors' published pH-MARL checkpoint on the
native VMAS 1.2.6 `simple_spread` (Navigation) scenario. The upstream clone is
kept unchanged at `../external/phMARL` at commit
`b529c05b7ec29c26d55627657131e26ef39ff3b1`.

The isolated environment follows the versions stated by the authors: Python
3.8, PyTorch 1.13.1, VMAS 1.2.6, Gym 0.26.2, and NumPy 1.24.0.

Run from the project root:

```bash
CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/phmarl_reproduction/evaluate_native_checkpoint.py
```

The wrapper uses the upstream checkpoint without retraining. It remaps the
legacy serialized actor to CPU and renders RGB arrays headlessly. Reported
metrics include the summed-agent reward convention used in the upstream
evaluator, a team-size-normalized mean-agent return, landmark assignment/coverage, collision pair
steps, and a matched-seed uniform-random control.

## Reproduction result

Ten matched seeds with four agents and a 400-step horizon produced:

| Metric | Published pH-MARL checkpoint | Uniform random |
|---|---:|---:|
| Mean-agent return | -207.94 | -857.38 |
| Final mean assignment distance | 0.103 | 0.900 |
| Final landmark coverage within 0.15 | 75.0% | 7.5% |

The checkpoint reduces final assignment distance by 88.6% relative to the
matched random control. Per-episode measurements and configuration are stored
in `results/metrics.json`; the headless VMAS rollout is
`results/simple_spread_phmarl_seed9.gif`.

The upstream `evaluation.py` was not edited. This wrapper corrects three
evaluation-only compatibility issues: `torch.has_cuda` selects an unavailable
GPU in the current sandbox, the serialized module retains legacy CUDA device
attributes after tensor remapping, and the upstream loop increments its time
counter twice per simulator step. It also disables visible-window rendering so
VMAS can render through EGL on the server.
