# Final pre-freeze acquisition audit

This rule was written before evaluating the fresh `920000--920019` parent maps.
This is the **only** mission-integrated sensing revision: each target lies on
the agent's current public mission route, surface geometry is scanned from an
earlier line-of-sight route cell, and traction is probed at the target cell.
Terrain generator, risk, prices, horizon, caps, packet semantics, and pH
executor remain unchanged. The same menu is available to every future method.

Audit all three map families on 20 fresh parents, at decision steps 120 and
360, using the public eight-candidate-per-agent menu. Compare skip, all
singletons, and eight stratified pairs under paired fixed-world physical
continuations. For each nonempty set, include both send and hold when
possible. A sensing choice counts as useful only if it completes successfully,
captures its target reading, and beats skip by more than 0.01 total mission
plus material-exposure units. A pair is optimal only if it beats skip and
every tested singleton by more than 0.01 and comes from a distinct parent map.

The audit passes only if at least 95% of scheduled states have a successful
continuation, at least 15% of evaluated states have a useful nonempty choice,
and at least two distinct parent maps have pair-optimal states. These are
restricted single-world scripted-continuation comparisons, not belief-optimal
values or learned-policy results.

If all gates pass, freeze the natural E2 interface. If any fails, stop editing
the natural benchmark and split evaluation into `E2-Natural` (the current
distribution) and `E2-Complementarity` (a separately specified controlled
challenge). In either case, `E2-dev` training weights are disposable and are
not confirmatory results.
