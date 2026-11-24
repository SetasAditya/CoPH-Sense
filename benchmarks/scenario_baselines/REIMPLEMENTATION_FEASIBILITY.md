# Independent reimplementation feasibility

This addendum distinguishes missing authors' code from missing experiment inputs. Authors' code is **not a necessary condition** for future admission: an explicitly labeled independent implementation can qualify after equation checks, disclosed numerical choices, and a reproducible native-setting check. The existing sealed records describe the completed source search; they are not changed here. Their official-source-only gate must not be treated as a general scientific requirement.

## DistGP

Primary source: [arXiv:2603.07351v1](https://arxiv.org/html/2603.07351v1), §§III-B–D, IV-B–D and V. The FITC observation model, GP conditional factors, natural-parameter Gaussian message updates, and local posterior prediction are sufficient to implement the core estimator independently.

The simplest native experiment family is specified in §IV-B: a static 2D GP field on a 4×4 robot grid over [-1,1]², random within-cell motion, target tolerance 0.02, regular inducing grids, and evaluation excluding points within 0.06 of training observations. It compares tree connections against added local edges over ten random datasets. However, the paper does not give the static field kernel parameters, observation-noise level, trajectory speed/timestep and observation count, inducing-grid resolution, exact initial tree, or convergence/message-scheduling settings. §IV-D adds communication radius 0.15 and six held-out squares but omits their coordinates. These omissions prevent reproducing the published curves numerically, even with an independent implementation.

A useful next self-check is feasible: independently implement FITC factors plus block Gaussian belief propagation; check tree posterior means **and covariances** against direct conditioning of the identical assembled Gaussian factor graph, and check converged loopy means against that graph's direct linear solve. Report loopy covariance approximation separately. All synthetic inputs and tolerances must be declared before executing. This would validate equations, not reproduce Figure 3/4, the SST benchmark, or CoPH superiority. SST reproduction additionally needs exact ocean bounds, preprocessing and selected hyperparameters; occupancy reproduction needs its scan/trajectory assets and fixed kernel lengthscale.

## UGV-conditioned exposure planning

Primary source: [arXiv:2606.12306v3](https://arxiv.org/html/2606.12306v3), §§III-A–D, IV-A and Table I. This source provides substantially more fixed settings than the source-search record alone suggests: the belief/reward coefficients, terrain speed fractions, 5 km square domain, 10 m grid, three UAVs at 30 m/s, 200 m footprint, and 100 paired trials. Equations 1–11 support independent belief-moment, route-cost and reward-map implementations.

The complete native run still requires the exact OpenStreetMap crop/raster and semantic preprocessing; absolute terrain exposure-rate scale; threat shape/size and sampling distribution; start/goal/UAV-heading distributions and seeds; nominal UGV speed and committed-prefix length; sensing-confidence function and update cadence; Dubins turn radius; IPP budget, expansion/termination and tree-recycling settings; and hierarchical Voronoi resolution/assignment details. The log-mean update is described as analogous but not specified as a complete numerical update. Choosing these freely would produce a related independent simulation, not a reproduction of Tables II–IV.

A feasible partial self-check would verify analytic log-normal moments against Monte Carlo, eight-connected A* against Dijkstra using Equations 3–5, and reward-map bounds/limiting cases using Table I and Equations 7–11. It would not exercise the unspecified acquisition loop or reproduce the printed exposure-reduction percentages.

## Decision

Implementations from the equations are feasible; a faithful **published numerical experiment** is not currently specified by the retrieved public inputs. The bounded independent primitive checks in `primitive_checks.py` have now passed on frozen tiny inputs; no native-setting or comparative run was executed. The next admissible step can be a separately preregistered equation-level implementation with its assumptions explicit, or completion of the missing native assets/settings. Neither requires claiming that official code is mandatory. Final matched comparisons remain pending native-setting and adapter validation; no missing setting should be selected from comparative outcomes.
