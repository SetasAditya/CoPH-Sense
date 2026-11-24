# Saved-log presentation audit

The extraction helper `CoPH-Stagewise/coph_stagewise/scenario_logs.py` reads preserved experiment artifacts without advancing an operational learner or changing any archive. `log_audit.json` records source hashes, missing signals, fixed example selections, and failure/incomplete inventories.

```bash
# From the repository root; point to a trusted original result archive.
PYTHONPATH=CoPH-Stagewise OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m coph_stagewise.scenario_logs \
  --archive /path/to/CoPH-Stagewise/results \
  --out docs/presentation/log_audit.json
```

## Extraction interface

- `load_crossing(archive)` / `load_hazards(archive)` return `episodes` keyed by comparison arm, retaining complete original records. `stagewise_frame(episode, time)` returns the last actual state at or before mission time, its exact control-time model, controls, navigation, observation, and past communication events. The `maze_visuals.landscape` helper evaluates the logged shaping model and its support mask. Its surface height is energy, not terrain elevation.
- `load_dynamics(archive)` / `drag_frame(log, step)` read native position/action/reward arrays, protection traces, per-agent estimates and the actual assigned goals. State samples have times `0,.1,...40` seconds; commands span the following step interval. The intervention is at 20 s. Velocities, a full nominal-force decomposition and environmental field snapshots were not retained.
- `load_encounters(archive)` / `encounter_frame(log, sample)` prepend true initial positions to the preserved subsampled path. Samples are at **0,.1,.5,...39.7 s**, because the original recorder subsampled after the first integration. The final-layer liveness archive does not contain command, yielding-event or model-revision traces. Display those signals as unavailable rather than reconstructing them from position changes. Sampled minimum separation is not the continuous minimum.
- `load_uncertainty(archive)` provides saved decision/certification rows. `regenerate_uncertainty_geometry(archive, pairs=12)` recovers the first original group's first 12 path pairs using the frozen RNG, then checks truth and all predicted costs against saved rows to `1e-12`. It reads a **trusted local pickle snapshot**, never updates the learner, and does **not** resample the 400 Gaussian perturbations. Mark this presentation as a saved-state query sequence, never as an episode or physical-time animation. Use the saved certified/flagged labels; the sample distributions themselves are unavailable.

## Example selection and failure visibility

Crossing seed 76000 and hazard seed 79000 are the original manifest's animation seeds. Drag seed 87000 and encounter seed 300 are the lowest declared seeds; those manifests did not preregister an animation seed. The positive drag change and four-agent swap demonstrate the stated mechanisms and are explicitly named, with no favorable-seed search. Uncertainty uses the first original acquisition group and contiguous query IDs 0–11.

All available native-drag phase-two coverage/contact failures, final-layer native coverage/contact records, and encounter failures/contacts are retained in the audit inventory. The complete hazard/crossing completion files are also counted, including any incomplete episodes. An escape replay is not a proof that contact was unavoidable. Attraction is the encounter baseline; matched phMARL is the native benchmark baseline.

## Checks

`tests/test_scenario_logs.py` checks strict timestamps, exact post-step encounter sample times, initial/terminal state selection, event cutoff and input immutability. The archive audit additionally checks native record shapes, control-model version agreement, terminal-state agreement, original animation seeds, and all regenerated uncertainty query costs. Missing signals remain explicit.
