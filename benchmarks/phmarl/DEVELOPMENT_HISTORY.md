# phMARL native navigation reproduction

Evaluate the published navigation checkpoint, with no retraining and no changes to CoPH-Stagewise.

## Reproduce locally

From the workspace root:

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python \
  benchmarks/phmarl/reproduce.py \
  --out benchmarks/phmarl/results/native_v2
```

The runner refuses to overwrite results. The existing upstream checkout is reused read-only at `CoPh-sense_wrong/external/phMARL`, pinned to `b529c05b7ec29c26d55627657131e26ef39ff3b1` (verified against public remote HEAD). The isolated existing environment has Python 3.8.20, torch 1.13.1, numpy 1.24.0, VMAS 1.2.6, gym 0.26.2 and scipy 1.10.1. It is separate from CoPH-Stagewise's environment. Environment and runner paths are recorded in each manifest.

The authors' source is https://github.com/EduardoSebastianRodriguez/phMARL (MIT license). The checkpoint is `data/actor_net_simple_spread_1400000_LEMURS.pth`; it is not the parser's default frame 1900000. A published whole-module checkpoint is loaded using its upstream class definitions. CPU tensor/device remapping is applied without changing weights. No dependency installation or existing environment modification is needed for this workspace.

## Protocol

Four agents, ten sequential episodes from seed 1, the native `simple_spread` scenario and deterministic actor-distribution means. Execute original `evaluation.py` with its double timestep increment intact and rendering intercepted. Verify an independent 200-step wrapper matches all operational arrays bitwise. Separately evaluate a corrected 400-step wrapper and matched uniform-random controls. Store the actual number of simulator steps, not just the requested horizon. Neither run is a retraining reproduction or a comparison against CoPH.

All four protocols retain the evaluator's constructor parameters, including ratio=2; the VMAS 1.2.6 simple_spread scenario ignores that keyword and initializes positions in [-1,1]. The native simulator has dt=0.1, spherical agent radius 0.15, and its own collision and drag dynamics. The checkpoint's preprocessing uses each agent's corresponding landmark and masks peer interactions using its serialized communication radius. It does not use our surveyed graph or environmental-risk estimator.

Results: [report](results/native_v1/REPORT.md), [animation](results/native_v1/navigation_episode0.gif). Animation uses logged positions, includes initial state, and was preselected as episode 0. The GIF is a trajectory visualization, not a native-renderer recording; playback is compressed and the title shows actual simulator steps.

## Native CoPH controller transfer

The first comparison is deliberately a **scalar-potential controller and native drag identification transfer**, not an implementation of the full cooperative stagewise algorithm in VMAS. See [comparison report](results/controller_transfer_v1/REPORT.md).

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B -m unittest discover \
  -s benchmarks/phmarl -p test_adapter.py -v
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B benchmarks/phmarl/compare.py \
  --out benchmarks/phmarl/results/controller_transfer_v2
```

No original CoPH source is edited. `adapter.py` imports the original potential, gradient command and reference flow, adapts physical radii, and converts reference velocity to native component-bounded actions. It receives observations and executed actions only. Moving peers enter instantaneous barriers within radius 0.45; this is not certified collision protection. The task supplies no risk measurements, so risk reconstruction, received-field fitting, environmental packets, historical replay and graph routing are inactive. Each agent uses the same indexed goal convention as the checkpoint.

The four arms are the published checkpoint, frozen drag estimate, online drag identification, and public-native-drag reference. The known-native arm distinguishes control transfer from online estimation. All use the original four-agent task, native reward and 400-step horizon. This is a fixed development comparison with no tuning or held-out claim; all collisions and incomplete supplementary coverage remain reported.

## Shared protection and fresh baseline comparison

The next milestone compares six arms: original phMARL, simple native attraction, protected attraction, protected frozen CoPH transfer, protected online CoPH transfer, and protected phMARL. Every protected arm uses the same observation-only action projection and near-goal offset rule. No original controller or previous result file is changed.

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B -m unittest discover \
  -s benchmarks/phmarl -p 'test_*.py' -v
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B benchmarks/phmarl/safety_suite.py \
  --out benchmarks/phmarl/results/native_safety_v3 --workers 3
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -B benchmarks/phmarl/check_safety_artifacts.py \
  benchmarks/phmarl/results/native_safety_v2
```

[Complete comparison](results/native_safety_v2/REPORT.md). Settings and source hashes are frozen before two old development cases and twenty fresh seeds (81000–81019): 132 episodes total. Animation seed 81000 is fixed in advance. The interrupted v1 development runner is retained; its worker thread-configuration error was corrected without changing controller settings, before any fresh-seed evaluation.

Protection uses only peers inside radius 0.45, estimates their velocities from consecutive observations, and flags unavailable estimates. It uses public native drag/dt for reciprocal radial constraints over one to four steps. Exact feasible 2D projection enforces native action bounds; infeasibility falls back to logged bounded braking. Near a goal, local separating halfplanes allow an offset instead of demanding exact landmark centering. This can produce stalls and is not a certified safety guarantee. New contacts, initial overlap recovery, infeasibility and missed coverage remain reported. The full cooperative reconstruction/replay algorithm is still not active in this native task.
