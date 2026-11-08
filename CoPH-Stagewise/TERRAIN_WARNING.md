# Scout warning and carrier avoidance

This opt-in development demonstration adds a **surveyed two-route prior** and a bounded static terrain map to the cooperative learner. Both agents know the same route layout; neither receives terrain truth. They start at the same point, at 0 s and 5 s. Both use the same learner, sensing radius, controller, and mission goal. “Scout” and “carrier” describe their temporal roles, not different learning algorithms.

The direct route crosses hazardous terrain. A bypass branches before the carrier can sense that terrain. Geometry contains no circular obstacles in this experiment: the communication effect concerns environmental risk, not communicated walls. The route prior is new and must not be confused with autonomous discovery of a global map.

## Implementation

- `terrain_warning.py` implements the opt-in learner and surveyed-route scoring. The existing routing entrypoint dispatches only when the explicit surveyed-route context is present. Default controllers retain their previous behavior.
- Route cost remains path length plus four times estimated risk integral, with unknown risk 1. Supported value/gradient anchors define local Taylor estimates within 0.65 m. Overlapping predictions use a conservative maximum clipped to [0,20]; this is an explicit planning heuristic, not a calibrated confidence bound.
- Exported anchors favor locally measured high-risk locations, using the existing locally reconstructed field and bootstrap covariance. Packets retain H, physical state, task potential, environmental values/gradients, support, units, identity and source provenance. Static report expiry is 60 s; status, payload and acknowledgments are charged under the existing two-million-byte per-agent ceiling.
- Remote reports outside retained stages enter a separate route-evidence store. They never enter measured observations, identify receiver damping, or certify environmental replay support. Supported retained-stage messages additionally use the original cooperative reconstruction and validation loop. Remote-only route estimates are not presented as passing measured-data validation.
- Spatial memory uses 0.2 m cells: at most 256 local model-estimate cells and 128 remote cells, plus eight latest sender/stage packets. A new estimate replaces its sender/location contribution; it does not erase reports elsewhere or count as statistically independent evidence. Remote cells retain packet/source lineage and expire. This static memory was needed to prevent the hazard from disappearing as recent summaries changed coverage.
- Route targets enter the existing progress-preserving scalar potential. Execution and historical replay use the same routing dispatch, potential, constraints, and integration. Each replay update freezes the route-evidence context at retained entries. Remote route support does not expand the physical replay domain. Fixed-state target changes remain separate shaping-energy jumps.

## Reproduce

From `CoPH-Stagewise/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg \
python -m coph_stagewise.terrain_warning_demo --out results/terrain_warning_reproduction --workers 4
```

This runs seeds 74000–74001, communication off and reciprocal fixed-rate, with a 60 s active horizon per agent. The manifest and source archive are frozen before execution. Seed 74000 is the preselected animation case. `--render-only` regenerates artifacts in an existing output directory.

## Results

[Animation](results/terrain_warning_v2/figures/unsafe_shortcut.gif) · [Report](results/terrain_warning_v2/REPORT.md) · [Machine-readable metrics](results/terrain_warning_v2/rows.json)

Both seeds produced the same detection/decision ordering:

| Event | Communication off | Reciprocal communication |
|---|---:|---:|
| Scout first senses risk ≥ 1 | 5.16 s | 5.16 s |
| Carrier first receives a risk ≥ 1 warning | — | 5.84 s |
| Carrier first selects bypass | 12.68 s | 8.72 s |
| Carrier first senses risk ≥ 1 locally | 10.16 s | 18.32 s |
| Carrier reaches goal | 41.60 s | 36.44 s |

All **8/8 agent missions** completed with zero collisions or timeouts. Carrier risk exposure fell from approximately 1.026 to 0.632 risk-seconds (**38.4%**) and completion was **5.16 s earlier**. The threshold 1 labels hazard detection for evaluation; it is not a binary runtime route-switch rule.

The paired runs demonstrate an effect of accumulated communication in this scene. Isolated per-stage deliver/withhold audits measure the incremental effect of one packet given already received reports; those incremental effects can be small even when the accumulated warnings change the route. Do not interpret an arrow as proof that that particular packet caused the turn.

86 tests passed, including five new terrain tests and four staggered-driver tests. Additional checks cover bounded spatial retention, expiry, identical execution/replay controls, and three-step finite-difference gradients. The four frozen routed-validation source-selection tests were excluded from this modified-source regression run; their archived results are preserved. Maximum local-update work was 326/1024 flows. Routing and extra export/model-summary computation remain outside that local replay counter and are reflected in wall time, not a claim of equal compute.

Exploratory truncated pilots are preserved under `results/terrain_warning_v2/pilots/`; they showed route oscillation and incomplete missions before spatial retention. They are development debugging evidence without independent source fingerprints, not scored full-horizon trials. `terrain_warning_v1` records an interrupted launch using the pre-memory implementation; it is explicitly marked incomplete. Neither is hidden or used to claim success.

This is evidence for one designed static terrain-warning mechanism, under a common surveyed route prior and a new conservative spatial planning map. It does not validate obstacle-sharing, unknown-map exploration, dynamic hazards, calibrated risk uncertainty, adaptive rates, or general superiority of the previous unmodified learner. Held-out and final-test seeds remain untouched.
