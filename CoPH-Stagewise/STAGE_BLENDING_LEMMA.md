# Passivity between events for smooth stage overlap

Draft statement for the appendix. It extends Proposition `prop:energy` (physical balance and model revisions) to `stage_blend_mode='smooth'` ([stage_prediction.py](coph_stagewise/stage_prediction.py), `blend_weights` and `potential_at`). No new runs are needed: the controlled checks in `results/mechanism_repairs_v1/overlap/final_v2` already show the zero fixed-support handoff jump and a 2.04e-11 finite-difference error.

## Statement (LaTeX)

```latex
\begin{lemma}[Passivity between events under smooth overlap]\label{lem:blend}
Let $M$ be a fixed set of retained stages, each with a $C^1$ potential $V_i$ on its rectangle $R_i$.
Let $s(z)=z^2(3-2z)$ for $z\in[0,1]$, extended by $0$ below and $1$ above. Define
$\phi_i(q)=\prod_{k=1,2}s\big((h_k-|q_k-c_{i,k}|)/\omega\big)$ with $0<\omega<\min_k h_k$ (code: $\omega=0.9$, $h=(1.5,2)$), $w_i=\phi_i/\sum_{j\in M}\phi_j$ and
$V_M=\sum_{i\in M}w_iV_i$.
On any open set where membership is $M$ and $\sum_j\phi_j>0$:
(i) $V_M$ is $C^1$, $\{w_i\}$ is a partition of unity, and $\nabla V_M=\sum_i w_i\nabla V_i+\sum_i V_i\nabla w_i$;
(ii) with $\widehat H=K+V_M$ and $u=u^{nom}+\delta u$, $u^{nom}=-\nabla V_M-\kappa v$,
$\dot{\widehat H}=-(d+\kappa)\norm v^2+v^\top\delta u$, so the map $\delta u\mapsto v$ is passive with storage $\widehat H$.
At a change of membership from $M$ to $M'$ at state $q$, $\Delta K=0$ and $\Delta\widehat H=V_{M'}(q)-V_M(q)$.
This jump is zero when every entering or leaving stage has $w_i(q)=0$ (a crossing of a rectangle edge), and it is recorded otherwise.
\end{lemma}

\begin{proof}
$s$ is $C^1$ on $\mathbb R$ with $s'(0)=s'(1)=0$, and $|\cdot|$ is smooth away from the rectangle's center lines, where $s$ equals $1$ in the overlap band.
Hence each $\phi_i$ is $C^1$, $w_i$ is $C^1$ where $\sum\phi_j>0$, and $\sum_iw_i=1$, which gives (i).
For fixed $M$, $V_M$ is a single time-invariant $C^1$ function of $q$. Proposition~\ref{prop:energy} is stated for any fixed $C^1$ potential, so it applies with $V=V_M$:
$\dot{\widehat H}=v^\top(u-dv)+\nabla V_M^\top v=-(d+\kappa)\norm v^2+v^\top\delta u$, which is (ii).
A membership change happens at a fixed physical state, so $K$ is unchanged and the jump is the difference of potentials. If $w_i(q)=0$ for the entering or leaving stage, then $V_{M'}(q)=V_M(q)$.
\end{proof}

\begin{corollary}[Bounded storage]
If $\delta u=0$ between events, then $\widehat H(t)\le\widehat H(0)+\sum_{e\le t}\Delta\widehat H_e^{+}$, where $\Delta\widehat H_e^{+}$ are the positive recorded jumps
(support-radius exits, stage creation or eviction, model revisions, target changes).
When the recorded jumps are summable, $\norm v$ stays bounded.
\end{corollary}
```

## What this does and does not cover

- **Covered:** the claim that smooth overlap removes the shaping jump at handoffs inside fixed supported membership, stated and proved. This is the same exact balance used elsewhere in the paper, applied to the blended potential.
- **Not covered:**
  - The hard 0.65 m support boundary. A stage can leave membership with $w_i>0$, and that is a recorded jump.
  - Stage creation and eviction, model revisions and target changes, which are also recorded jumps.
  - Force clipping and the protection projection. These enter as $\delta u$, and the sign of $v^\top\delta u$ is not controlled.
  - Convergence or asymptotic stability. The corollary bounds the storage only.
- **What a general stability result would need:** a bound on the positive jumps, for example through a dwell time or a cap on revision energy, together with a sign condition on $v^\top\delta u$ for the safety filter, as in the Califano condition $P_{safe}\le d_p$.
