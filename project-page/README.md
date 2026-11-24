# CoPH scenario gallery

Static GitHub Pages site: `https://setasaditya.github.io/CoPH-Sense/`.

`dist/` is the deployable site: five recorded-log MP4s, GIF previews, posters, source hashes and frame provenance. `.github/workflows/project-page.yml` publishes it after changes on `main`. The paper retains exactly two labeled 3D placeholders; the gallery supplies the extended scenario sequences.

## Regenerate media

From the repository root, with the current package and an FFmpeg installation supporting H.264:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.scenario_logs \
  --archive /path/to/original/CoPH-Stagewise/results \
  --out docs/presentation/log_audit.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=CoPH-Stagewise python -m coph_stagewise.scenario_presentation \
  --archive /path/to/original/CoPH-Stagewise/results \
  --site project-page --frames 64
```

Only trusted original snapshots may be unpickled for saved-state query reconstruction. Rendering does not train or advance operational agents. It reads logged states and models; 12 original path pairs are regenerated from frozen randomness and checked against archived prices. Gaussian uncertainty distributions are not fabricated.

Native drag and encounter cohorts did not preregister animation seeds: their lowest declared seeds are selected explicitly. Hazard/crossing preserve the original animation seeds. Missing historical signals remain unavailable. Oblique world views are planar simulations; model surfaces use energy or risk as height. Playback is compressed and displays original mission timestamps or query indices.

## Validation

`python project-page/verify_gallery.py` checks resources, media hashes, decoded frame counts, provenance, and the paper placeholders. `--archive /path/to/original/results` also rechecks source hashes, exact model versions, model/nominal-force agreement, saved-query prices, and immutable input files. Browser checks are recorded in `docs/presentation/browser_checks.json`.

New DistGP and exposure-planning comparisons remain blocked at native admission until the missing native assets/settings and reference checks are available. Independent paper-pinned implementations are allowed; their primitive correctness checks have passed. The page links checked sources and sealed comparison intents; it claims no unrun comparative result.
