# Historical policy learning and current route selection

The original `stage_prediction.retained_replay` already starts every trajectory
from its own retained entry and deep-copied entry context. It differentiates the
existing time/risk/effort loss through multiple integration steps to update one
shared two-coefficient policy. That implementation was not the historical
route-voting bug.

The separate `route_decision.evaluate_routes(... starts="historical")` diagnostic
used current-position route identities across older starts and averaged their
detached costs. Those route labels are not interchangeable across positions.

`SharedPolicyReplayLearner` is an opt-in wiring guard:

```python
from coph_stagewise.historical_policy_replay import SharedPolicyReplayLearner
from coph_stagewise.route_decision import RouteDecisionOptions

agent = SharedPolicyReplayLearner(
    config, initial_state, goal, agent_id="0", mode="full",
    network=(nodes, edges),
    route_options=RouteDecisionOptions(
        mode="prefix", starts="current", terminal="graph", horizon=8,
    ),
)
```

It delegates policy fitting to the original retained replay. It checks that
retained entries, copied contexts, and chosen horizons remain fixed during an
update, and logs each entry's own first planned route and trajectory. It requires
the actual discrete route choice to use the current physical state. Graph mode
is also supported. Historical route identities never vote for current execution.

The existing objective, support checks, strict/hold transition setting, gradient
steps, trust bounds, measured validation, transactional rollback, and flow
accounting remain active. A positive configured flow ceiling no larger than
1,024 is required. Current prefix evaluation uses only the budget remaining after
learning and retains the incumbent when unsupported. No unseen stage or measured
support is created.

Reproduce the saved-context engineering checks from the workspace root:

```bash
PYTHONPATH=CoPH-Stagewise OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph_simple_replay \
python -m coph_stagewise.historical_policy_replay_suite suite \
  --out CoPH-Stagewise/results/simple_fixes_v1/replay --workers 8
```

The frozen suite compares the original full policy learner, the guarded learner
with graph routing, the guarded learner with current-only prefix routing, and the
previous evidence-only historical discrete voter. The last is a different
mechanism; its absent policy gradients are not a failed policy-learning result.

On all 24 saved contexts, the guarded graph configuration exactly reproduces the
original full policy's models, controls, acceptance and work. Both have valid
policy batches and nonzero gradients in 24/24 cases and policy changes larger
than $10^{-10}$ in 23/24. All 24 finite-difference checks pass, with maximum absolute
gradient error $4.36\times10^{-11}$. The combined current-prefix configuration
remains supported in 8/24 cases; no unsupported continuation is treated as valid.

The prior `replay_hold_v1` navigation stop rule remains in force. These snapshots
are engineering regressions only. No new mission episodes, fixture search,
held-out runs, or navigation-superiority claim are included.

See [the report](results/simple_fixes_v1/replay/REPORT.md) and
[the machine-readable verdict](results/simple_fixes_v1/replay/VERDICT.json).
