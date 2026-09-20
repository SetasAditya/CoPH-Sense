# CoPH-Sense workspace

The current implementation is [`coph_terrain/`](coph_terrain/README.md): the procedural unknown-terrain task, material-aware pH executor, learning code, tests, and experiment record. Its current decision and failed-gate evidence are in [`coph_terrain/results/material_campaign/CAMPAIGN_LOG.md`](coph_terrain/results/material_campaign/CAMPAIGN_LOG.md). Final held-out evaluation has not launched.

[`coph_fork/`](coph_fork/README.md) contains the finite and moving two-fork mechanism checks used as regression controls. [`phmarl_reproduction/`](phmarl_reproduction/README.md) contains the separate native pH-MARL reproduction. [`reports/`](reports/) contains presentation artifacts. The professor-shared manuscript checkout is kept outside this public code repository.

The earlier program-scoring/actor prototype is historical and is not an active baseline. The local Python environment, third-party pH-MARL checkout, and historical binary archive are intentionally excluded from this publication; see the [repository README](../README.md) for setup and scope.

Run the current tests from the repository root:

```bash
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-coph \
  .venv/bin/python \
  -m unittest discover -s CoPh-sense/coph_terrain/tests -q
```
