# Physical acquisition opportunity-cost decomposition

**Historical diagnostic:** this audit preceded the later correction that
requires the robot to settle before starting sensor dwell. The synthetic
travel/dwell tiers below remain diagnostic of opportunity cost, but the
"actual sensing" and "full" rows are **pre-correction execution results**
and must not be used as current physical-value estimates. The fresh
post-correction outcome is in `POSE_TOUR_AUDIT.md`.

This evaluator-only audit holds the terrain generator, observation and risk
models, 65 s horizon, prices, and caps fixed. It uses the eight high-ideal-pair
states in disjoint parents 850012--850039 identified **before** the pose
revision. Every comparison starts at the same public decision state and
realized hidden world. The values below are \(Q(\varnothing)-Q(\cdot)\):
positive means the specified pair intervention improves realized team cost.
They are not belief-expected \(Q^*\) or independently sampled learning
replications.

| Paired acquisition tier | Median value vs skip |
| --- | ---: |
| Actual local footprint, no physical travel (prior funnel) | +0.0403 |
| Public-route travel to the old common target pose, then free immediate local reveal | **−0.0223** |
| Travel and physical dwell, then free reveal | −0.0821 |
| Travel, dwell, and declared sensor fees | −0.1421 |
| Previous tier plus priced packet and ACK, with no radio motion or delivery delay | −0.1820 |
| Actual sensing and motion, no transmission | −0.1606 |
| Actual sensing, communication, and motion | **−0.2718** |

The fee-plus-radio tier is a **price-only counterfactual**, not a delivered
message. The actual no-send and full tiers use the environment's sensor and
packet implementations. Consequently the rows are useful for localization
but are not one additive causal decomposition. Across matched states, dwell
adds median cost 0.0400 beyond travel, sensor fees add exactly 0.0600, and
the priced packet/ACK adds median 0.0395. Actual transmission and its induced
motion add median 0.0656 compared with actual no-send; this last difference
also includes any downstream effect of delivered evidence.

For the full pair relative to skip, median ledger increases were 0.1000
time, 0.0600 sensing, 0.0641 communication, 0.0324 material exposure, and
0.0118 motion. Median completion increased by 50 simulation steps. The
median extra carrier path length was only 0.172 m, so distance alone is a
poor proxy for acquisition opportunity cost: stopping, return-to-range,
route timing, and the resulting physical continuation matter.

At physical arrival, a cost-free local reveal changed the coarse planned
route in four of eight cases; after dwell, it changed that signature in
three of eight. Under the paired arrival-state continuation, the late reveal
had at most 0.01 value in four cases on immediate arrival and six cases
after dwell. This is an **actionability proxy for the scripted continuation**,
not a certified latest-feasible-switch time.

The staged result motivated changing acquisition *poses and tours* before modifying
the traction posterior or any benchmark price. Geometry can observe a target
region from a nearby public-LOS pose; traction still requires contact with
the region. The following revision now represents a common target region
with modality-specific poses and chooses a two-sensor order from public
travel costs. Its fresh 860000--860039 audit is recorded separately.

Artifact: `physical_detour_decomposition_fresh28_v1.json`.
