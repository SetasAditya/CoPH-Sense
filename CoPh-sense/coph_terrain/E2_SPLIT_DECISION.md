# E2 hard-stop decision — 2026-09-17

The one allowed mission-integrated sensing revision and the one fresh audit
are complete. **The natural benchmark did not pass the predeclared sensing
nondegeneracy gate.** The natural environment/interface is frozen here; no
further price, risk, layout, horizon, cap, or menu tuning is authorized by
this decision.

The new generator offers only target cells on the agent's public mission
route. Geometry is scanned from an earlier visible point on that route;
traction is probed on contact with the target. The physical pair executor
keeps those poses instead of optimizing back onto a detour. The audit used
20 fresh parent maps (920000--920019), three natural topology families,
steps 120/360, all available singleton actions, eight stratified pairs,
and both send/hold continuations for nonempty actions. A valid improvement
had to complete, capture the advertised target, and beat skip by >0.01.

| Predeclared check | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Successful continuation coverage | at least 95% | 40/40 | pass |
| States with useful physical sensing | at least 15% | 2/39 (5.13%) | fail |
| Distinct maps with pair-optimal states | at least 2 | 0 | fail |

The restricted best set was skip in 36 evaluated states, singleton in three,
and pair in zero. The two gains above 0.01 occurred at parent seeds 920005
(step 120, gain 0.22584) and 920010 (step 120, gain 0.08526), both with
evidence held. These are fixed-world physical continuations, not belief
optimality or learned-policy results. The full artifact is
`results/pilot/mission_integrated_fresh20_v1.json` (SHA-256
`d9739e07f6585627410a0cff1d3ad51df95476b7be1b9b06c57585a400ca49fe`).

## Evaluation split

**E2-Natural** uses this frozen procedural distribution and interface for
success, cost, risk/CVaR, and resource accounting. Skip is a legitimate,
strong baseline here. Do not use it alone to claim an advantage from
set-valued sensing or cooperative memory.

**E2-Complementarity** is a separate controlled challenge, to be specified
and generated before any confirmatory training or evaluation. It should
retain the same sensing/communication physics, prices, horizons, and pH
executor, but construct *early, physically actionable* decision regions in
which geometry and traction jointly alter a feasible route while each alone
may not. It also needs a later unresolved region so delivered scout evidence
can redirect carrier sensing. Include redundant and additive controls, and
sample unaltered test maps from a held-out challenge manifest. Report its
results separately; never pool its hand-controlled state prevalence into
E2-Natural rates.

## Reproduction boundary

The frozen primary implementation hashes are:

```
generator.py       dccf1d27c84e9535c3437016932c13e14d3c0aca815e87bb2ba7f4ec2c169bc0
scenario.py        819981bb1e7a81ea0addee480b671ad86cbeb737dd475e665065f81f0a4da8d3
environment.py     ab18bcc0df108348cb2ff7a25e7e9170a6a4ce491d36d1c1e4361adcb5c9f1d8
planning.py        224b06899d7a9d4334810552587fee168341afcdc89b0421e1d4c3dda9c8b272
physical_audit.py  0159958c5b8104b2f3839909e10c642fe0350031ee9e87869f860090ac1b2bb5
execution.py       1ca11e96ef65021e68e564654b1af8c460e6759eb9b8c1737db6861aedfa3fd2
value_model.py     54edee4fc343f6ddb5f927cc2f27b99a7767d6c1f698b4f586910533b8106913
```

The v2 train/validation/test manifests remain the original hashed files.
Disposable `results/e2_dev` weights may be used to debug software but must be
discarded before final training.
