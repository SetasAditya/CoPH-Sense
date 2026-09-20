# CoPH-Sense

**Cooperative sensing and selective belief sharing for unknown-terrain navigation with port-Hamiltonian execution.** This repository is a research implementation, not a completed benchmark claim. The finite and scripted VMAS mechanism checks work; the learned acquisition stack has **not** passed its current gate, so the held-out E2 comparison has **not** launched.

## Research question

Two moving agents have different, incomplete views of terrain. A scout may learn something about a region before the carrier reaches it; the carrier may also discover information useful to the scout. CoPH-Sense asks **which evidence is worth acquiring and sending, given its cost, what the teammate already knows, and whether it can still change a physical decision?** Delivered evidence updates each agent's persistent terrain belief and therefore its material-risk Hamiltonian and trajectory.

The learning contribution under test is the information policy. The material-aware port-Hamiltonian (pH) executor is a frozen downstream controller in the current campaign. The original material model source and checkpoint are included so this integration is reproducible.

## Repository map

| Path | Purpose |
| --- | --- |
| [`CoPh-sense/coph_terrain/`](CoPh-sense/coph_terrain/) | Current procedural VMAS unknown-terrain environment, A3/A4/A5 code, material-pH adapter, teacher, tests, and campaign artifacts. |
| [`CoPh-sense/coph_fork/`](CoPh-sense/coph_fork/) | Finite and moving two-fork regression tasks for acquisition, sharing, memory, and actionability. |
| [`full_code/`](full_code/) | Material-aware Hamiltonian model and its force/integration dependencies. |
| [`repair_experiments/outputs/behavioral_soft_force_risk_encoder_recall_full/best.pt`](repair_experiments/outputs/behavioral_soft_force_risk_encoder_recall_full/best.pt) | Frozen material-aware model checkpoint used by the adapter. |
| [`scripts/build_dfc2018_stagewise.py`](scripts/build_dfc2018_stagewise.py) | Data-generation functions imported by the material model source. |
| [`CoPh-sense/phmarl_reproduction/`](CoPh-sense/phmarl_reproduction/) | Separate native pH-MARL reproduction. |
| [`CoPh-sense/reports/`](CoPh-sense/reports/) | Professor updates and presentation figures. |
| [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) | Problem formulation, experiment breakdown, measured results, and present roadblock. |

The custom VMAS task is implemented in `CoPh-sense/coph_terrain/environment.py`; VMAS itself is an installable dependency. The local 3.7 GB virtual environment and third-party pH-MARL checkout are deliberately excluded. Archived exploratory output can be regenerated from source or obtained from the original workspace; the reported gate artifacts and checkpoints are included here.

## Setup and smoke test

The validated local environment used Python 3.8 with the pinned [`requirements.txt`](requirements.txt). From the repository root:

```bash
python3.8 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  .venv/bin/python -m unittest discover -s CoPh-sense/coph_terrain/tests -q
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  .venv/bin/python -m unittest discover -s CoPh-sense/coph_fork/tests -q
```

The material executor loads `full_code/train_material.py` and the included checkpoint through `CoPh-sense/coph_terrain/material_executor.py`. `CoPh-sense/coph_terrain/results/material_ph_calibration_v1.json` records the frozen control calibration. GPU access is useful for fitting; the exact VMAS counterfactual teacher remains largely CPU-bound in this implementation.

For the current campaign, read [`CAMPAIGN_LOG.md`](CoPh-sense/coph_terrain/results/material_campaign/CAMPAIGN_LOG.md) before running training scripts. `material_full_stack_gate.py` intentionally fails closed without a **passed** A3/A5 checkpoint; the included `a3_a5_candidate_failed.pt` is diagnostic and must not be presented as a frozen deployable model.

## Current evidence

- The restricted moving two-fork [replay GIF](CoPh-sense/coph_fork/results/a5_moving_bridge/physical_search/frozen_two_fork_information.gif) shows delivered Region-1 evidence redirecting the carrier to Region 2. The canonical paired team cost is **2.116 versus 3.089** with reset memory. This is a scripted-VMAS mechanism demonstration, not a held-out E2 result.
- A separate frozen bidirectional NEED task demonstrates that receiver-side need information can make SEND/HOLD decisions identifiable. Its learned history+NEED A4 has mean decision regret **0.001419** on 384 controlled test cases, close to the request-match heuristic's **0.001736**.
- The present material-pH A3/A5 Gate 1.1 **failed**: held-out acquisition regret **0.02034** missed the `<0.02` threshold, and validation contained **zero** pair-optimal and **zero** memory-switch cases. A 32-state training micro-overfit reached only **25/32** exact actions. The final paired ID/OOD evaluation is therefore pending.

These results and their limits are detailed in [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md). No structural pH-versus-non-pH superiority or full E2 benefit is claimed.
