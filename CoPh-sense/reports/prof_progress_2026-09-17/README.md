# CoPH-Sense implementation progress report

`progress.tex` is a standalone, five-page update for Prof. Bajaj. `progress.pdf`
is the compiled version. The figures are generated from saved results and the
current terrain generator with `make_figures.py`.

From the repository root, rebuild the figures and PDF with:

```bash
PYTHONPATH=CoPh-sense MPLCONFIGDIR=/tmp/mpl-prof-progress \
  CoPh-sense/external/envs/phmarl-py38/bin/python \
  CoPh-sense/reports/prof_progress_2026-09-17/make_figures.py
cd CoPh-sense/reports/prof_progress_2026-09-17
pdflatex -interaction=nonstopmode -halt-on-error progress.tex
pdflatex -interaction=nonstopmode -halt-on-error progress.tex
```

The numerical figure uses `coph_fork/results/a3/summary.json`,
`a4_v2/summary.json`, `a5_memory_v2/comparison.json`, and
`ph_factorial/summary.json`. The maze panels use current
`coph_terrain.generator` maps at seeds 600002 and 600003. The qualitative
information-flow image is copied from
`coph_terrain/results/visualizations/maze_pair_comm_seed600042/`.

Other numerical claims are cross-checked against the milestone result JSONs
and `coph_terrain/E2_SPLIT_DECISION.md` and
`coph_terrain/E2_COMPLEMENTARITY_DEV_STATUS.md`. Cost units differ between
milestones. Neither the qualitative maze preview nor disposable E2 development
runs are held-out learned-policy evidence.
