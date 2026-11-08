# Native phMARL comparisons

Controller transfer and executed-transition drag identification in native VMAS `simple_spread`. Environmental communication and historical replay are inactive in this benchmark.

Use a separate Python 3.8 environment; the serialized upstream checkpoint was evaluated with PyTorch 1.13.1 and VMAS 1.2.6. From the repository root:

```bash
python3.8 -m venv .venv-native
.venv-native/bin/pip install -r benchmarks/phmarl/requirements.txt
git clone https://github.com/EduardoSebastianRodriguez/phMARL external/phMARL
git -C external/phMARL checkout b529c05b7ec29c26d55627657131e26ef39ff3b1
.venv-native/bin/python -m unittest discover -s benchmarks/phmarl -p 'test_*.py' -v
.venv-native/bin/python benchmarks/phmarl/reproduce.py --out benchmarks/phmarl/results/reproduction
```

The upstream MIT-licensed project supplies its code and `data/actor_net_simple_spread_1400000_LEMURS.pth`; it is not vendored here. `COPH_PHMARL_ROOT` can override the default `external/phMARL` directory. Only load checkpoints from the trusted pinned upstream source.

Experiment runners include `heldout_campaign.py`, `drag_shift_campaign.py`, and `final_layer_campaign.py`. They preserve their declared scenes, cohorts, gates, and output conventions. Some diagnostic/parity runners additionally require preserved experiment archives; see [development notes](../../docs/DEVELOPMENT.md). Full campaigns can be expensive; the original worker defaults and experimental settings are unchanged.

Selected summaries are in [docs/results](../../docs/results/). The earlier workspace-specific instructions are retained in `DEVELOPMENT_HISTORY.md`. Contact persistence under a particular escape replay is a diagnostic, not a proof that collision is unavoidable.
