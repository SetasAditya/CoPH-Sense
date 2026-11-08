# CoPH-Sense

Cooperative port-Hamiltonian navigation with learned environmental fields. Agents identify dynamics from executed inputs, retain evidence across overlapping stages, and share supported field summaries that each receiver interprets using its own task preferences.

## Quick start

Python 3.10+; CPU execution is supported. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './CoPH-Stagewise[dev]'
cd CoPH-Stagewise
python -m pytest tests -q
python -m coph_stagewise.cli rollout --out results/example --seed 0
```

Use a new output directory for each run. The rollout is a small single-agent example. For the two-agent crossing demonstration, including reports and animations:

```bash
python -m coph_stagewise.crossing_demo --out results/crossing --workers 2
```

## Code

| Path | Contents |
|---|---|
| [CoPH-Stagewise/](CoPH-Stagewise/) | Current learner, simulator, reciprocal communication, routing, evaluation and tests |
| [benchmarks/phmarl/](benchmarks/phmarl/) | Native VMAS checkpoint comparisons and online-identification experiments |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, optional archives, test scope and reproduction notes |
| [docs/COOPERATIVE_STAGEWISE_PH_FORMULATION.md](docs/COOPERATIVE_STAGEWISE_PH_FORMULATION.md) | Mathematical formulation |
| [docs/results/](docs/results/) | Selected recorded result summaries |

The current implementation uses static environments, bounded stage memory, float64 identification/replay, fixed-rate communication, and separate physical/shaping-energy accounting. Supported historical replay and receiver reconstruction are implemented; their additional navigation benefit is not established. Native VMAS studies isolate control and identification.

Earlier material-sensing prototypes remain in `CoPh-sense/`, `full_code/`, and `repair_experiments/`; their original instructions are preserved in [docs/LEGACY_README.md](docs/LEGACY_README.md). Their Python 3.8 requirements are separate from the current package. Large generated runs, virtual environments and caches are excluded from new source additions.
