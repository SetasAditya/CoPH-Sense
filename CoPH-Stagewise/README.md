# CoPH-Stagewise

Current CoPH implementation. Install from the repository root with `python -m pip install -e './CoPH-Stagewise[dev]'`, then run commands from this directory.

```bash
python -m pytest tests -q
python -m coph_stagewise.cli rollout --out results/example --seed 0
python -m coph_stagewise.cli compare --out results/paired --seeds 3
python -m coph_stagewise.crossing_demo --out results/crossing --workers 2
```

- `coph_stagewise/`: models, bounded stage ledger, local/received evidence, reconstruction, replay, routing, safety and evaluation.
- `tests/`: numerical and mechanism tests; preserved-archive checks skip explicitly when their inputs are unavailable.
- `paper_figures/`: figure generators; recorded experiment inputs are required.
- `scripts/`: additional reproduction and artifact-validation commands.

See [development notes](../docs/DEVELOPMENT.md), [implementation contract](IMPLEMENTATION_CONTRACT.md), [cooperative protocol](COOPERATIVE.md), and [recorded findings](../docs/results/README.md). Historical experiment notes are retained for provenance; `DEVELOPMENT_HISTORY.md` describes earlier milestones and is not the current setup guide.
