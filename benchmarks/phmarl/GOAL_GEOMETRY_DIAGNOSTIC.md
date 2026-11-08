# Static landmark targets and independent allocation

This opt-in experiment changes near-goal targets, retaining actual-motion protection. It follows the completed reset-assignment diagnostic. Legacy targets and constructor defaults remain unchanged.

Every agent independently reconstructs all initial agent positions and all landmarks from its own full native observation. The ordered peer list identifies peer indices. Each enumerates all 24 one-to-one assignments, minimizes total squared reset distance, and uses lexicographic exact ties. All four assignments must agree before episodes run. This requires the native full observation; it is not a method for a sensor limited to radius 0.45, and no consensus protocol is introduced.

For landmark positions $g_j$, fixed pair normals $n_{jk}=(g_j-g_k)/\|g_j-g_k\|$ define the convex target problem

$$
\min_{\delta}\frac12\sum_j\|\delta_j\|^2,\qquad
n_{jk}^{\mathsf T}(g_j+\delta_j-g_k-\delta_k)\geq0.32,\qquad
\|\delta_j\|\leq0.149999.
$$

An isolated pair splits its separation deficit symmetrically. The joint solve prevents independent pair shifts from invalidating another pair's constraints. Coincident landmarks use a deterministic index-based normal; they fail the pairwise necessary condition in any case. A feasible solution is a constructive witness that the target positions satisfy separation and remain inside the unchanged 0.15 coverage radius. A failed fixed-normal program alone does not prove global infeasibility. A landmark pair closer than 0.02 gives a genuine pairwise impossibility certificate; passing that condition is insufficient for general clusters.

Targets are solved at reset with SciPy SLSQP (`ftol=1e-12`, `maxiter=200`), frozen, and selected inside the existing 0.35 near-goal region. Beyond that region the original assigned landmark remains the nominal target. The 0.149999 bound is an explicit numerical interior margin. Action constraints, clearance 0.02, physical radius 0.15, drag, learning rules, goal threshold and 400-step horizon are unchanged.

Run from the repository root:

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -m unittest discover -s benchmarks/phmarl -p 'test_*.py'
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python benchmarks/phmarl/goal_geometry_campaign.py --out CoPH-Stagewise/results/native_goal_geometry_v1 --workers 4
```

The runner refuses existing output. It freezes sources, numerical versions, parameters and archived inputs before screening all 20 starts (82000–82019). Eighty archived baselines must reproduce before 120 new episodes execute:

- CoPH frozen/known/online damping with minimum-distance assignment and legacy targets: 60 archived baseline episodes.
- The same three modes with static targets: 60 new episodes.
- Unmodified phMARL checkpoint: 20 archived baseline episodes.
- phMARL with the same assignment, without protection: 20 episodes.
- Assigned phMARL with legacy protection/targets: 20 episodes.
- Assigned phMARL with static targets and the same protection as CoPH: 20 episodes.

The checkpoint's own `LEMURS_actor.forward` also selects the index-specific landmark slot. The matched allocation wrapper permutes landmark slots consistently in every observation; positions, velocities and peer slots remain unchanged. Protected assigned actors receive the same target treatment as CoPH. No checkpoint weights are modified and no training reproduction is claimed.

Outputs include 200 episode records, local-observation agreement and constructive-feasibility records, archive parity, paired outcomes, all incomplete/contact cases, separate reset planning work, and 51-frame comparison GIFs for previously observed cases 82004, 82017 and 82011. These are reused development starts; final-test seeds remain untouched. Receiver field reconstruction, historical replay and adaptive communication are absent from the native adapter.
