# CoPH-Sense workspace

The current implementation is [`coph_terrain/`](coph_terrain/README.md): the procedural unknown-terrain task, material-aware pH executor, learning code, tests, and experiment record. Its current decision and failed-gate evidence are in [`coph_terrain/results/material_campaign/CAMPAIGN_LOG.md`](coph_terrain/results/material_campaign/CAMPAIGN_LOG.md). Final held-out evaluation has not launched.

The conditional-scout extension has two deliberately separate layers. The
[geometric benchmark](coph_terrain/CONDITIONAL_SCOUT_PROTOCOL.md) ports
Yiwang's dock–dispatch–sense–report–recover system as an attributed kinematic
diagnostic. The material integration in
[`conditional_scout_material.py`](coph_terrain/conditional_scout_material.py)
uses the same protocol inside VMAS and labels dispatch from complete paired
material-pH continuations over compatible terrain hypotheses. The current
gate record is summarized in
[`CONDITIONAL_SCOUT_STATUS.md`](coph_terrain/CONDITIONAL_SCOUT_STATUS.md).

[`coph_fork/`](coph_fork/README.md) contains the finite and moving two-fork mechanism checks used as regression controls. [`phmarl_reproduction/`](phmarl_reproduction/README.md) contains the separate native pH-MARL reproduction. [`reports/`](reports/) contains presentation artifacts; [`6aa17bc0c8c4eaab118e7968/`](6aa17bc0c8c4eaab118e7968/) contains the manuscript checkout.

The earlier program-scoring/actor prototype has been removed from the active tree and preserved in [`archive/`](archive/README.md). It is historical evidence, not an active baseline. `external/envs/` is a local, ignored Python environment used by the VMAS run instructions; `external/phMARL/` is a third-party checkout. Neither is project source.

Run the current tests from the repository root:

```bash
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  -m unittest discover -s CoPh-sense/coph_terrain/tests -q
```

Run the two conditional-scout stages with:

```bash
cd CoPh-sense
MPLCONFIGDIR=/tmp/mpl external/envs/phmarl-py38/bin/python \
  -m coph_terrain.conditional_scout_geometric_benchmark

MPLCONFIGDIR=/tmp/mpl external/envs/phmarl-py38/bin/python \
  -m coph_terrain.run_conditional_scout_material
```
