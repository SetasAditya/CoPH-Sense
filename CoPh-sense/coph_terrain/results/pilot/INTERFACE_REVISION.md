# Public acquisition-interface revision

**Scope.** This revision changes only the public viewpoint proposal rule. The
layout generator, latent fields, risk formula, passive appearance, 65 s horizon,
sensing and packet prices, and 4+4 measurement/four-packet caps are unchanged.
The eight-action menu now comprises up to four reachable public regions,
with geometry and traction proposed at the same viewpoint when each remains
feasible before the deadline. Regions are ranked from the observed costmap,
alternative paths obtained by penalizing the current route, local unresolved
terrain, appearance ambiguity, and public travel cost. The generator reads
no hidden terrain or evaluator values.

The first placement check on previously audited development states reported
geometry proximity in 10/17 regions and traction proximity in 6/17, versus
9/17 and 0/17 under the old menu. **That proximity rule was too permissive
for traction:** it counted a probe one 0.2 m cell away, although the actual
reading covers only the probed cell. These are archived proximity figures,
not exact measurement coverage or evidence of a performance gain. Later
audits use the actual contact cell and LOS scan footprint.

The traction action still produces a point-local reading. Although the
generator's terrain patches are spatially smooth, the actor has no calibrated
posterior model conditional on that reading. Broadening the measurement
footprint or filling neighboring belief cells by an arbitrary kernel would
change the observation problem and could overstate sensor value. A future
correlation-aware posterior must be fitted and tested against the declared
generator before being used by every compared method; it is not part of this
revision.

Parents 850000--850039, steps 120 and 360, were evaluated by
`interface_revision_audit.py` and `interface_value_funnel.py`. The first
twelve parents, 850000--850011, had already appeared in the cap-stress audit;
we therefore report **only 850012--850039** as the disjoint fresh result.
`interface_summary.py` regenerates that subset from the unchanged raw
artifacts. All costs are realized-world, restricted scripted pH continuations;
the ideal 1 m traction reveal is evaluator-only and is not a deployed sensor.

| Fresh 28-parent subset | Result |
| --- | ---: |
| Physical states with a successful tested continuation | 56/56 |
| Evaluated physical decisions: skip / singleton / pair winners | **51 / 3 / 0** (54 total) |
| Nonempty route screens with ideal paired value >0.05 | 8/51 |
| Screens with strong ideal colocated interaction | 3/51 |
| Top ideal-value regions near public G / T candidate under old proximity rule | 5/8 / 3/8 |
| Top regions with actual point-sensor pair value >0.01 | 7/8 |
| Top regions with dwell+fees pair value >0.01 | 2/8 |
| Top regions with full physical pair value >0.01 | **0/8** |

The physical audit has 54 evaluated decisions; the value screen has 51
nonempty public-route region sets. They use different eligibility rules, so
the denominators must not be combined. Among the eight high-ideal-value
regions, median paired value falls from **+0.0793** for the ideal reveal to
**+0.0403** for actual point sensing without acquisition costs, **−0.0163**
after dwell and sensor fees, and **−0.2718** under the full physical detour
and communication protocol. This does not attribute the entire loss to a
single charge: the latter stage also changes motion, timing, and routes.

The interface correction improved candidate proximity, and the current
terrain still contains decision-relevant ideal interactions. It did **not**
make physical set acquisition competitive. Consequently the E2 resource
freeze and final-manifest teacher training remain on hold. The next
controlled investigation is the physical detour/dwell path and genuinely
modality-specific sensing poses. A calibrated spatial traction posterior is
secondary. Any posterior must be trained only on permitted training maps,
verified for calibration on separate maps, and supplied to every compared
information policy. We should not lower sensing prices or change the terrain
model on the basis of this result.
