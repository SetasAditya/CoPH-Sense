# Completion-first cooperative navigation

The opt-in `shaping_mode='progress'` works with `controller_mode='protected'` or `'routed'`, retained-stage mission replay, and the existing safety layer. Defaults remain legacy. Environmental fitting, executed-input damping identification, measured validation, reconciliation, rollback, reciprocal fixed-rate exchange, and the learning cost objective remain active.

For target $w$, $e=q-w$, and nonnegative $S=\theta_r\widehat r+\theta_b B$,

$$V=\frac{k}{2}\|e\|^2+\alpha\frac{\|e\|^2}{\|e\|^2+\ell^2}S,$$

with $k=1$, $\ell=0.5$, $\eta=0.5$, and $\alpha=\min(1,2\eta k\ell/L)$ (one when $L=0$). The uniform bound uses frozen field coefficients, sensed obstacle radii, and policy upper bounds. It does not depend on position or a candidate policy. Optimization and validation use the same reconstructed field and geometry; recomputing this deterministic bound gives the same value throughout the batch. All position derivatives, including the taper derivative, remain in the graph.

Since $S\ge0$, the taper derivative has a nonnegative radial contribution. Also $r/(r^2+\ell^2)\le1/(2\ell)$, so

$$e^\top\nabla V\ge kr^2-\alpha Lr^3/(r^2+\ell^2)\ge(1-\eta)kr^2.$$

This excludes a cancellation equilibrium of the nominal potential away from a fixed target. It does not prove completion of a constrained controller with changing waypoints and partial observations.

Routing still uses observed-space A*, sensed inflated disks, and risk-aware path costs. In progress mode, a clear observed segment within one unit selects the mission goal directly. Seventeen position records detect displacement and target-distance reduction below 0.05 over sixteen steps; forced replanning is limited to once per eight-step update interval. Histories are branch-local and bounded. Unknown space remains uncertified. A route or safety failure brakes and remains a navigation failure.

The centralized potential is used by execution, replay, boundary pilots, packet continuations, Hamiltonian export, and fixed-state energy diagnostics. Route-target changes use the full potential difference. The controller logs attraction and soft-shaping forces, bound, alpha, taper, route progress, stagnation, projection and guards. Receiver summary schema 3 explicitly declares `progress_taper_v1`; schema 1/2 retain legacy semantics and cannot be used as progress contexts. Raw environmental packet anchors and receiver-specific goals are unchanged.

Run from `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl \
  python -m coph_stagewise.completion_evaluation \
  --out results/completion_first_v1 --workers 32
```

The suite freezes all scenes, source hashes, communication settings, seed namespaces, and animation seed before execution. It runs controlled learned/frozen disk and passage checks and isolated reciprocal deliver/withhold audits, then 32 smoke and 128 development episodes. Each of four development configurations must achieve 64/64 agent goals with zero collisions. Only then may it execute the frozen 64 held-out routed episodes (seeds 110000–110007); final-test seeds remain untouched. Failures are not removed, horizons and goal thresholds are not changed, and mission cost is secondary evidence.

`--resume` requires identical source hashes. `--render-only` regenerates reports and GIFs from recorded trajectories. Results contain full episodes, machine-readable rows, paired legacy/communication outcomes, complete failure galleries, gates, source archive, and synchronized animations. True-world audit branches never modify operational agents or become training evidence.

Supplementary read-only analyses (run from the package directory) are reproducible with:

```bash
bash scripts/check_completion_regressions.sh results/completion_first_v1
bash scripts/analyze_completion_results.sh results/completion_first_v1
```

The second command requires completed development artifacts and checks the source fingerprint, episode counts, failure-gallery coverage, flow ceiling, physical energy residuals and held-out gate. Both scripts record their own fingerprints. In animations, paths and controls use the recorded tick; sensed geometry and retained counterfactuals use the most recent logged post-update snapshot at or before that tick. Truth backgrounds are evaluator-only. No missing measurements are synthesized for display.
