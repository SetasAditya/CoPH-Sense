# Supported stage-potential blending

`Config(stage_blend_mode="smooth", controller_mode="protected" or "routed", replay_domain="retained", replay_objective="mission")` enables the new mode. `"legacy"` remains the default. Positive stage overlap is required.

For a fixed retained collection, let $I(q)$ contain only stages passing the unchanged rectangular and local-measurement support test. With rectangular half-widths $h$, taper width $b=\min(W,H)\min(\rho,0.5)$, and $s(z)=z^2(3-2z)$ on $z\in[0,1]$, define

$$
z_{k,a}(q)=\operatorname{clip}_{[0,1]}\frac{h_a-|q_a-c_{k,a}|}{b},\qquad
\beta_k(q)=\prod_a s(z_{k,a}(q)),\qquad
\omega_k(q)=\frac{\beta_k(q)}{\sum_{j\in I(q)}\beta_j(q)}.
$$

Every stage uses the same current target and sensed obstacle geometry. The scalar potential and conservative force are

$$
V_{\mathrm{blend}}(q)=\sum_{k\in I(q)}\omega_k(q)V_k(q),\qquad
u_{\mathrm{conservative}}=-\sum_k\left[\omega_k\nabla V_k+V_k\nabla\omega_k\right].
$$

The entire scalar is differentiated in PyTorch; partition derivatives are retained. Singleton membership has weight identically one, avoiding tiny-denominator cancellation. No supported member raises `DomainExit`; a degenerate multi-stage partition is explicitly rejected. The environmental value used for model replay cost uses the same partition. Safety projection, guards, clipping and integration retain their existing paths.

`FrozenStages.energy`, nominal control, historical replay and execution share the potential hook. Navigation logs member IDs, versions, weights, algebraic work and support-membership context jumps when the previous context can be evaluated. Model, geometry or target changes that prevent this attribution are explicitly marked unavailable. Diagnostic evaluations of a previous membership never certify support or become observations.

This removes an active-index handoff jump within a fixed certified atlas. It does **not** establish global smoothness: the hard 0.65 measurement-support boundary, stage creation/eviction, model commits and waypoint changes remain context events. Spatial partition derivatives also mean the original per-stage radial progress bound does not automatically apply. General stability, safety or passivity is unproved.

Rich control logs contain every retained blend basis and support coordinates; compact correction coefficients are included when enabled. Rendered landscapes reproduce these logged potentials, including partition and residual derivatives. Legacy log dictionaries retain their prior keys.

Run the controlled comparison with:

```bash
python -m coph_stagewise.overlap_repairs --out <new-output-directory>
```

Final evidence: `results/mechanism_repairs_v1/overlap/final_v2/REPORT.md`. In this controlled frozen-atlas crossing, smooth-mode fixed-state energy and applied-force handoff jumps are zero, multistep gradient error is $2.04\times10^{-11}$, and execution/replay controls agree exactly. This is mechanism evidence, not a navigation-benefit result.
