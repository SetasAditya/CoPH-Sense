# Crossing traverses: a scenario where communication has to pay off

[Animation](results/crossing_traverses_v1/figures/crossing.gif) · [Summary figure](results/crossing_traverses_v1/figures/summary.png) · [Run report](results/crossing_traverses_v1/REPORT.md)

The earlier maze could not show the scout–carrier mechanism clearly. The agents started together, dead ends were only about one sensor radius deep, and the gates were disk clusters. This scenario fixes all three:

```
 y=7.5 N ═══[▮]════════════════════════╗   ▮ = hidden full-width door (slab)
       ║   x=2.5                       ║
 y=4   A ══════════════════════[▮]════ B   A: (0,4) → (12,4)    B: (12,4) → (0,4)
       ║                       x=9.5   ║
 y=0   S ══════════════════════════════╝   south row open; shortest true route 20 m
```

- **Opposite starts, swapped goals.** What one agent has already seen is what the other will need next, so warnings can have value in both directions.
- **Dead ends are deep compared with sensing.** Doors are 1.15 m from one end and 8.15 m from the other, and the sensor reaches 1.35 m to a door surface. No agent can see a door from a junction or from another row.
- **The planner's optimistic first choice is wrong.** With no knowledge, the middle row (12 m) is shortest, then north (19 m), then south (20 m).
- **Doors are real slabs**, a segment with radius 0.1 m spanning the 1.7 m corridor. They are used in sensing, swept collision, the planner's edge blocking, the shaping barrier, the CBF projection, the discrete guard, replay and packet validation (`shapes.py`). Disks keep their original formulas exactly; the earlier test suite is unchanged.

## Frozen predictions and measurements

The predictions come from route arithmetic (equal speeds, instant turns, zero delay). They were written into `manifest.json` before any episode ran. Results are for 2 seeds × 4 arms; all 16 agent missions reached their goals with zero collisions.

| Arm | Team wasted, predicted | Team wasted, measured | A finish | B finish |
|---|---:|---:|---:|---:|
| Communication off | 51.2 m | 44.9 m | 132.7 s | 134.4 s |
| One-way A → B | 51.2 m | 44.9 m | 132.7 s | 134.4 s |
| One-way B → A | 37.2 m | 31.4 m | 92.0 s | 134.4 s |
| Reciprocal | 23.2 m | 17.8 m | 92.0 s | 92.5–93.0 s |

| | Predicted | Measured |
|---|---:|---:|
| Sum of one-way savings | 14.0 m | 13.4 m |
| Reciprocal saving | 28.0 m | 27.1 m |

The reciprocal saving is twice the sum of the one-way savings. The reason is the order of events, not more sensing:

1. B sees the middle door at 4.0 s; A is told at 4.8 s and turns back 1.5 m in instead of 8.2 m.
2. Because A turned back early, A reaches the north door first (20.9 s).
3. B is told at 21.1 s, 1.2 m into the north row instead of 8.2 m.
4. With A → B only, A is always too late to help, so that arm saves exactly nothing.

Measured dead-end penetrations match the predictions to within 0.35 m (see the report). Measured waste sits about 3 m below prediction in every arm because agents cut corners relative to the centerline route.

Other checks:

- The largest physical energy residual is 1.5e-15.
- Update work peaks at 518 of the 1,024 flow evaluations allowed.
- All 192 delivered packets were accepted.

## Limitations

- The two seeds change only sensor noise. Geometry and timing are deterministic, so they are not independent replications.
- This is a development demonstration, not held-out evaluation.
- The agents are independent plants and pass through each other in the south row; there is no inter-agent collision avoidance.
- Communication range is unlimited and the rate is fixed.
- The benefit comes from sharing geometry and replanning routes. This scenario does not isolate a contribution from counterfactual policy learning.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/coph-mpl \
python -m coph_stagewise.crossing_demo --out results/crossing_reproduction --workers 8
python -m pytest tests/test_crossing.py
```

The full run takes about 20 minutes with 8 workers. Use `--render-only` to regenerate figures from existing episodes.
