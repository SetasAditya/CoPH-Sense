# Compact receiver residuals (opt-in)

`receiver_localization_mode="compact"` keeps the same-update local-only 4×4 RBF
proposal fixed and fits an additional compact field from eligible received
value/gradient residuals. The default remains `"global"`.

For local proposal weights $w_0$, the new model is

$$
\widehat r(q)=\frac14\operatorname{softplus}
\left(\phi(q)^\top w_0+\sum_j c_j\psi(\|q-z_j\|/R)\right),
\qquad
\psi(t)=(1-t)_+^4(4t+1),\quad R=0.65.
$$

The centers $z_j$ are deduplicated eligible packet anchors, capped by
`anchor_cap`. The solver enforces $|c_j|\leq0.5$ relative to the zero residual;
the bound does not accumulate across iterations. The correction and its spatial
gradient are exactly zero outside the union of the compact neighborhoods. The
softplus preserves nonnegative risk. This is a representation extension, not an
equivalent reparameterization of the original global RBF fit.

The local information matrix uses the existing frozen delta-method precision
with sensor noise $0.003$, now in compact coefficient coordinates. Joint overlap
rows, ridge regularization, and inverse marginal blocks remain present. Received
value/gradient rows retain their existing covariance and age terms. These
information matrices remain approximations rather than calibrated posteriors.

## Runtime use

```python
from coph_stagewise.cooperative import CooperativeLearner

agent = CooperativeLearner(
    config, initial_state, goal, agent_id="1", mode="evidence",
    receiver_fit_mode="precision_bounded",
    receiver_solver_mode="active_set",
    receiver_search_mode="diagnostic",
    receiver_validation_mode="relative_local",
    receiver_localization_mode="compact",
)
```

The same options pass through the existing network/maze learner constructors.
Compact fitting requires `precision_bounded`; the evaluated configuration uses
the selected active-set/diagnostic solver and relative-local validator. Its
20-iteration and $10^{-6}$ stationarity limits are unchanged. `mode="evidence"`
disables policy replay while retaining reconstruction and physical identification.

Pending residuals are separate from committed stage context. Only a successfully
validated whole-model update publishes them. Failure retains the complete prior
version, residuals, and routing snapshot. Reserved measured values govern the
relative-local gate; no synthetic observations are inserted. Without eligible
messages, the local-only proposal and controls remain identical to global mode.

Execution, frozen replay, supported-field routing, scalar potentials, gradient
bounds, and receiver summaries carry the committed residual. Summary contexts
declare the mode; legacy contexts keep their previous meaning. Environmental
exports still refit exclusively from local measured readings, so received
corrections are not recursively exported as independent evidence. Compact
neighborhoods never expand measurement, replay, or route-field support.

## Reproduction

From the repository root:

```bash
PYTHONPATH=CoPH-Stagewise OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph_localized_mpl \
python -m coph_stagewise.localized_receiver_evaluation suite \
  --out CoPH-Stagewise/results/mechanism_repairs_v1/reconstruction --workers 8

PYTHONPATH=CoPH-Stagewise OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph_localized_mpl \
python -m coph_stagewise.localized_receiver_supplement \
  --out CoPH-Stagewise/results/mechanism_repairs_v1/reconstruction
```

The first command refuses resume after a dependency, input, or numerical-setting
fingerprint changes. It compares local-only, corrected global, and corrected
compact fits on 24 current-campaign snapshots and 160 preserved complementary
audits. The second evaluates already-saved proposals and commitments on the exact
prior 24 fixed-branch query sets; it performs no fitting or mission execution.

## Evidence and limitations

- On the exact prior 24 fixed-branch query sets, compact fitting improved
  committed value MAE in 14, worsened it in 7, and was unchanged in 3. Corrected
  global fitting improved 12 and worsened 12. All 24 updates were accepted.
- On the 160 preserved full-graph audits, corrected global fitting improved
  129 and worsened 31, while compact fitting improved 96, worsened 61, and was
  unchanged in 3. Compact fitting had 18 relative-validation rejections.
- Exact outside-neighborhood value/gradient equality was checked at 146,970
  evaluator query occurrences. These are numerical checks, not independent trials.
- The evaluation demonstrates bounded locality and mixed estimation results.
  It does not demonstrate navigation superiority. Defaults remain unchanged.

See [the report](results/mechanism_repairs_v1/reconstruction/REPORT.md),
[the exact-query supplement](results/mechanism_repairs_v1/reconstruction/ORIGINAL_QUERY_REPORT.md),
and their machine-readable proposal/commitment records.
