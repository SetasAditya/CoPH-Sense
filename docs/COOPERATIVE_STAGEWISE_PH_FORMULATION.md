# CoPH-Sense: Cooperative Stagewise Port-Hamiltonian Learning

This revision organizes the method around the professor’s learning loop: reconstruct the local dynamics from a persistent ledger, learn through forward prediction and backward counterfactual reconstruction, then reconcile the proposed update with physical constraints before committing it. Cooperative messages enter that same learning loop. This is a research formulation, not an implementation report or a theorem of superiority.

The loop structure follows the discussion. The counterfactual loss, projection metric, communication-value estimator, and uncertainty model below are proposed numerical choices. Entropy and latent-context inference are included in the architecture but their stochastic dynamics remain an explicit open modeling decision.

## Method at a glance

At integration time $t_n$ in spatial stage $k$, agent $i$ has a ledger $\mathscr L_{i,n}$. It reconstructs all quantities required to evaluate its current dynamics, predicts motion, and compares prediction with actual observations. New evidence also revises its estimates of recent stages. From stored entry states it rolls forward under those revised estimates and uses those counterfactual trajectories in a learning objective. A separate constrained optimization reconciles the proposed parameters before they enter the committed ledger.

$$
\boxed{
\mathscr L_{i,n}
\xrightarrow{\text{reconstruct RHS}}
\widehat f_{i,n}
\xrightarrow{\text{forward + backward learning}}
\widetilde\Theta_{i,n+1}
\xrightarrow{\text{physical reconciliation}}
\Theta_{i,n+1}^{+}
\xrightarrow{\text{commit}}
\mathscr L_{i,n+1}}
$$

Received peer evidence enters ledger reconstruction; committed local estimates supply outgoing messages. Each agent runs the same loop. Spatial-stage advancement surrounds this process; several learning cycles may occur before a stage exit. Across episodes, an outer training process improves initialization and communication decisions. These three time scales must not be conflated.

### What the ledger contains

| Ledger component | Stored content | Update and use |
|---|---|---|
| Measured history | Timestamped $q,p,u,o$, sensor quality, source IDs | Append actual observations; never replace them with predictions |
| Stage geometry | Current and previous $B$ regions, entry states, targets, overlap anchors, frames | Create on stage entry; retain $B=2$ or $3$ previous stages |
| Dynamics estimates | $M^{-1},J,R,G$ and their parameter versions | Calibrate or learn only identifiable blocks |
| Field estimates | $V,H$, geometry/risk fields and derivatives | Update from measurements, overlaps, and usable messages |
| Stochastic context | Latent-context estimate, uncertainty, and $S$ if defined | Infer from observations; specify the entropy interpretation before using its gradient |
| Counterfactual repository | Earlier entry states, revised field versions, predicted trajectories, weights | Reconstruct backward and roll forward; feed the learning objective |
| Communication context | Peer status estimates, timestamps, ACKs, provenance | Estimate recipient usefulness; prevent duplicate evidence |

The inductive invariant is **availability and consistency of the quantities required by the right-hand side**, not that every estimate is exact. At initialization, initialize each required quantity or declare it unavailable. At each update, preserve versions and uncertainty. At stage exit and termination, commit the ledger and account for unresolved errors and obligations. A finite ledger is an approximate information state; it does not make a partially observed system exactly Markovian by assertion.

### Algorithm sketch A: One inductive learning cycle

The following equations are the main algorithm. Sections 1–8 define their variables and numerical objectives; Section 9 supplies the concrete mechanical pH step.

**A1. Reconstruct the current local model from measurements and delivered evidence.**

$$
\begin{aligned}
\mathscr L^-_{i,n}&\leftarrow
\operatorname{Assimilate}(\mathscr L^+_{i,n-1},o_{i,n},x_{i,n},\mathcal M_{i,n}^{\mathrm{valid}}),\\
\mathcal Z_{i,n}&\leftarrow
\operatorname{Reconstruct}(\mathscr L^-_{i,n})
=\{\widehat M^{-1},\widehat J,\widehat R,\widehat G,
\widehat H,\nabla\widehat H,\partial_t\widehat H,\widehat\Sigma\},\\
\widehat f_{i,n}(x,u)&\leftarrow
(\widehat J-\widehat R)\nabla_x\widehat H+\widehat G u,\\
\widehat y_{i,n}&\leftarrow\widehat G^\top\nabla_x\widehat H,\\
\widehat{\dot H}_{i,n}&\leftarrow
-\nabla\widehat H^\top\widehat R\nabla\widehat H
+u^\top\widehat y_{i,n}+\partial_t\widehat H.
\end{aligned}
$$

$\operatorname{Assimilate}$ means provenance-aware insertion and alignment, not an unspecified Kalman filter. $\operatorname{Reconstruct}$ evaluates the stored parameterization and its derivatives. If a stochastic extension is selected, extend $\mathcal Z$ with its explicitly defined entropy, diffusion, and latent-context quantities and derive the corresponding dynamics and energy law first.

**A2. Integrate, observe, and construct forward error.**

$$
\begin{aligned}
u_{i,n}&\leftarrow\kappa_i(x_{i,n};\Theta^-_{i,n}),\\
\widehat x_{i,n+1}&\leftarrow\Phi_{\Theta^-_{i,n}}^{\Delta t}(x_{i,n},u_{i,n}),\\
(x_{i,n+1},o_{i,n+1})&\leftarrow\operatorname{ExecuteAndSense}(x_{i,n},u_{i,n}),\\
r^x_{i,n}&\leftarrow x_{i,n+1}-\widehat x_{i,n+1},\\
L_{\mathrm{forward}}&\leftarrow
L_x+\mu_yL_y+\mu_EL_E+\mu_oL_o.
\end{aligned}
$$

Inputs satisfy actuator limits. Energy error uses the same frozen parameter version on both ends of the physical interval. Newly measured data are appended before the next fit.

**A3. Reconstruct recent stages and learn from counterfactual rollouts.**

$$
\begin{aligned}
\overline\Theta&\leftarrow
\operatorname{EvidenceFit}(\mathscr L^-_{i,n}\cup\{x_{i,n+1},o_{i,n+1}\}),\\
\mathcal R_i^{\mathrm{cf}}&\leftarrow
\operatorname{ReconstructAndRoll}(\overline\Theta,\{x_{i,r}^{\mathrm{entry}}\}_{r=k-B}^{k}),\\
L_{\mathrm{learn}}(\Theta)&\leftarrow
L_{\mathrm{forward}}+\mu_{\mathrm{ov}}L_{\mathrm{overlap}}
+\mu_{\mathrm{msg}}L_{\mathrm{message}}
+\mu_{\mathrm{cf}}L_{\mathrm{cf}}
+\mu_{\mathrm{prox}}\|\Theta-\Theta^-\|_P^2,\\
\widetilde\Theta&\leftarrow\Theta^-
-\eta P_{\mathrm{step}}^{-1}\nabla_\Theta L_{\mathrm{learn}}(\Theta^-).
\end{aligned}
$$

Repeat the bounded proposal steps if needed. $P_{\mathrm{step}}\succ0$ may be a damped Gauss–Newton matrix. The evidence fit and counterfactual objective are specified in Section 5; reconstructed futures are not observation labels.

**A4. Reconcile physical constraints, validate, and commit.**

$$
\begin{aligned}
\Theta^\star&\leftarrow\arg\min_{\Theta\in\mathcal C_{\mathrm{phys}}}
\tfrac12\|\Theta-\widetilde\Theta\|_W^2,\\
\Theta^+&\leftarrow
\begin{cases}
\Theta^\star,&\mathsf{Validation}(\Theta^\star)=1,\\
\Theta^-,&\text{otherwise},
\end{cases}\\
\Delta H^{\mathrm{learn}}&\leftarrow
H_{\Theta^+}(x_{i,n+1},t_{n+1})-H_{\Theta^-}(x_{i,n+1},t_{n+1}),\\
\mathscr L^+_{i,n+1}&\leftarrow
\operatorname{Commit}(\mathscr L^-_{i,n},\mathcal D_{\mathrm{new}},
\Theta^+,\Sigma^+,\mathcal R_i^{\mathrm{cf}},\Delta H^{\mathrm{learn}}).
\end{aligned}
$$

A failed or infeasible reconciliation retains the previous model and records the failure. This is a proposed constrained optimization, not a claim that clipping arbitrary gradients preserves physics. Fit constraints and reserved validation checks are defined separately below.

### Algorithm sketch B: Reciprocal cooperation and stage advancement

$$
\begin{aligned}
m_{i\to j}&\leftarrow\operatorname{Encode}(\mathscr L_i^+,\Omega_i,t_i),\\
\widehat A_{i\to j}&\leftarrow
\operatorname{RecipientValue}(m_{i\to j},\widehat{\mathscr L}_{j|i}),\\
\lambda_i^*&\leftarrow\operatorname{RateOptimize}(\widehat A_i,\mathrm{bytes},\mathrm{delay},\mathrm{budget}),\\
\mathcal M_j^{\mathrm{valid}}&\leftarrow
\operatorname{DeliverValidateAlign}(m_{i\to j}),\\
\mathscr L_j^-&\leftarrow\operatorname{Assimilate}(\mathscr L_j^+,\mathcal M_j^{\mathrm{valid}})
\quad\longrightarrow\quad\text{Algorithm A for agent }j,\\
\mathsf{exit}(\Omega_{i,k})=1&\Longrightarrow
k\leftarrow k+1,\quad
\mathscr L_i\leftarrow\operatorname{Retain}_{k-B:k}
\operatorname{InitializeNextStage}(\mathscr L_i).
\end{aligned}
$$

Apply the same steps for $j\to i$. Communication changes the receiver's reconstructed model and its learning problem; it does not overwrite its physical state. Rate and packet choices are supporting mechanisms, not the definition of cooperation itself.

## 1. Problem and notation

Consider $N$ agents moving through a partially observed, potentially changing environment. Agent $i$ has configuration $q_i\in\mathbb R^{d_i}$, momentum $p_i$, physical state $x_i=(q_i,p_i)$, actuator input $u_i$, and local observations $o_i$. Its velocity is $v_i$; $V_i$ denotes potential energy and is never used to denote velocity. Physical integration index $n$, spatial-stage index $k$, and model-update index $\ell$ are distinct.

Each agent operates on a sequence of local spatial regions $\Omega_{i,k}$. Consecutive stages have a nonempty overlap

$$
\mathcal O_{i,k}=\Omega_{i,k-1}\cap\Omega_{i,k}\ne\varnothing.
$$

The stage has an entry state, a local exit target, a local coordinate frame, and an observation-supported field model. A stage is a spatial sensing/planning region, not a training phase. Stage advancement uses a declared geometric rule, such as reaching the exit or entering the next region.

At stage $k$, retain the current stage and $B\in\{2,3\}$ previous stages:

$$
\mathcal K_i(k)=\{\max(0,k-B),\ldots,k\}.
$$

The retained record is

$$
\mathcal B_i=
\left\{\Omega_{i,r},\mathcal D_{i,r},\theta_{i,r},\Sigma_{i,r},
\mathcal A_{i,r},\mathcal P_{i,r}\right\}_{r\in\mathcal K_i(k)}.
$$

Here $\mathcal D_{i,r}$ contains timestamped local transitions and observations; $\theta_{i,r}$ contains field and pH parameters; $\Sigma_{i,r}$ summarizes estimation uncertainty; $\mathcal A_{i,r}$ contains overlap anchors; and $\mathcal P_{i,r}$ records evidence provenance and message IDs. Each record also stores stage targets and control inputs used during the transitions. A cap on samples and anchors per stage makes memory bounded.

Agent $i$ acts using its own observations and delivered messages only:

$$
\mathcal I_i(t)=\sigma\{o_i(s),x_i(s),u_i(s),m_{j\to i}
:s\le t,\ t_{\mathrm{arrival}}(m_{j\to i})\le t\}.
$$

Any estimate of another agent’s knowledge must be constructed from public protocol information and received status/ACK messages. Private simulator state is not an admissible input.

## 2. Local pH model and three consistency equations

Within a stage, the learned controlled model is

$$
\dot x_i=f_{\theta_i}(x_i,u_i,t)
=\bigl[J_i(x_i)-R_{\theta_i}(x_i)\bigr]
\nabla_x H_{\theta_i}(x_i,t)
+G_{\theta_i}(x_i)u_i,
\tag{1}
$$

$$
y_i^{\mathrm{port}}=G_{\theta_i}(x_i)^\top\nabla_x H_{\theta_i}(x_i,t),
\tag{2}
$$

$$
\frac{dH_{\theta_i}}{dt}
=-\nabla_xH_{\theta_i}^\top R_{\theta_i}\nabla_xH_{\theta_i}
+(y_i^{\mathrm{port}})^\top u_i
+\partial_tH_{\theta_i}.
\tag{3}
$$

Equation (3) is the chain-rule consequence of (1)–(2) when $J_i=-J_i^\top$, with model parameters held fixed over the physical interval. It supplies a consistency condition on observed transitions. It is not an independent source of information when evaluated only on the same model’s exact predicted trajectory.

Use the canonical interconnection for the first implementation. Enforce dissipation structurally, for example $R_\theta=L_\theta L_\theta^\top$, and use a positive-definite calibrated mass matrix. Learn only declared components of $G_\theta$ supported by action-tagged excitation; unconstrained simultaneous learning of $H,R,G$ need not be identifiable.

Define

$$
H_{i,k}(q,p,t)
=\tfrac12p^\top M_i^{-1}p
+V_i^{\mathrm{task}}(q;g_{i,k})
+V_{i,k}^{\mathrm{env}}(q,t).
\tag{4}
$$

The task potential belongs to the agent and its current target. The environment term encodes the locally observed field. One implementable representation is

$$
V_{i,k}^{\mathrm{env}}(q,t)
=\sum_{a=1}^{m_k}w_{i,k,a}(t)\,\phi_{k,a}(q)
+\sum_{b\in\mathcal C_{i,k}}\alpha_{i,k,b}b_{\mathrm{contact}}(d_b(q)),
\qquad \alpha_{i,k,b}\ge0.
\tag{5}
$$

The basis, units, activation domains, and contact-barrier convention are fixed and recorded. A neural scalar field can replace the basis, but the packet and receiver fit must still specify field evaluations and gradients. Choose a common energy reference for each connected overlap component; absolute potential values are otherwise ambiguous up to an additive constant.

For mechanical agents with momentum input, use $G_i=[0;B_i]$. The input/output pairing in (2) then defines power. A clearance/progress/speed vector used for adaptation is a separate observable $z_i^{\mathrm{task}}$, not the power-conjugate output $y_i^{\mathrm{port}}$.

### 2.1 Physical evolution versus learning updates

Online learning changes the model at discrete times. At the same physical state $x_n$, record the model-update jump

$$
\Delta H_n^{\mathrm{learn}}
=H_{\theta^{\ell+1}}(x_n,t_n)-H_{\theta^\ell}(x_n,t_n).
\tag{6}
$$

Changes of stage target or coordinate/model representation have separate jumps. They are not actuator work or dissipation. Equation (3) applies between these jumps; explicit environmental time variation belongs in $\partial_tH$. This distinction prevents a successful fit from hiding model changes as physical energy balance.

The deployed input is $u_i=\kappa_i(x_i,\theta_i,g_i)$ subject to actuator limits. If the Hamiltonian already contributes the conservative force in (1), do not add the same potential gradient again through $G_iu_i$. A physical-plant formulation with a separate energy-shaping controller is an alternative, but its storage function and supplied power must be derived consistently. Equations (1)–(5) specify a navigation model; matching that model to the physical plant remains an explicit identification requirement.

## 3. Overall objective

Let $\Xi$ denote learned initialization parameters, online-update settings, and communication-policy parameters. Optimize expected complete team cost:

$$
\min_{\Xi}\;\mathbb E[J_{\mathrm{team}}(\Xi)],
\tag{7}
$$

$$
J_{\mathrm{team}}
=\sum_i\left[
\ell_{i,T}(x_i(T))
+\int_0^T\left(\ell_i^{\mathrm{task}}(x_i,t)
+c_u\|u_i\|^2\right)dt\right]
+c_b\sum_{m\in\mathcal M}\mathrm{bytes}(m)
+c_c\sum_i N_i^{\mathrm{update}}
+c_fN^{\mathrm{failure}}.
\tag{8}
$$

Use the same declared cost coefficients and failure definitions for all comparisons. For a scout–carrier experiment, include scout travel, return/recovery obligations, and unfinished communication obligations through completion. More general team tasks can use their own fixed ledger.

Constraints include legal information, actuator bounds, structural pH conditions, finite buffer capacity, and network limits:

$$
u_i\in\mathcal U_i,\quad R_i\succeq0,\quad J_i=-J_i^\top,
\quad |\mathcal K_i|\le B+1,
\quad \sum_j\lambda_{ij}(t)b_{ij}(t)\le C_i.
\tag{9}
$$

Here $\lambda_{ij}$ is a sending rate, $b_{ij}$ a candidate packet size, and $C_i$ an expected byte-rate limit. Enforce hard byte caps separately at packet admission. Online model fitting below is a computational surrogate for (7), not an assertion that every local update minimizes complete team cost.

## 4. Local identification from realized observations

For a transition $a=(x_a,u_a,x_a^+,\Delta t_a,o_a,t_a)$, define three residuals using a fixed parameter snapshot:

$$
r_a^x(\theta)=x_a^+-\Phi_\theta^{\Delta t_a}(x_a,u_a),
\tag{10}
$$

$$
r_a^y(\theta)=\widetilde y_a^{\mathrm{port}}
-G_\theta(x_a)^\top\nabla H_\theta(x_a,t_a),
\tag{11}
$$

$$
r_a^E(\theta)
=H_\theta(x_a^+,t_a+\Delta t_a)-H_\theta(x_a,t_a)
-\mathcal Q_a\!\left[
-\nabla H_\theta^\top R_\theta\nabla H_\theta
+u^\top G_\theta^\top\nabla H_\theta
+\partial_tH_\theta\right].
\tag{12}
$$

$\Phi_\theta$ is the specified numerical integrator; $\mathcal Q_a$ is a declared quadrature rule evaluated on the observed/interpolated physical segment. Residual tolerances include sensor and integration error. Equation (11) is used only when a measured or independently calibrated port output exists. Computing both sides from the same network gives no independent supervision and must not be counted as a measured loss.

Environmental observations have an explicit observation model $\widehat o_\theta$ and residual $r_a^o=o_a-\widehat o_\theta$. The local loss is

$$
L_{\mathrm{local}}(\theta)
=\sum_{a\in\mathcal D}\omega_a
\left(\|r_a^x\|_{W_x}^2
+\mu_y\|r_a^y\|_{W_y}^2
+\mu_E|r_a^E|^2
+\mu_o\|r_a^o\|_{W_o}^2\right),
\tag{13}
$$

where $\|r\|_W^2=r^\top Wr$. The matrices scale units and uncertainty; $\omega_a$ accounts for observation quality and temporal relevance. Missing measurements omit their terms.

This implements the requested use of state, output, and energy relationships in learning. Structural parameterization enforces algebraic properties; observed residuals check whether the learned model explains the actual motion.

## 5. Correcting the retained stages

### 5.1 Overlap consistency

Select anchors $q_a$ in the observed part of each overlap. Transform them into a common coordinate frame and evaluate both field estimates at the same time $t_a$. Define

$$
L_{\mathrm{overlap}}
=\sum_{r=k-B+1}^{k}\sum_{a\in\mathcal A_{i,r}}
\left[
\left|V_{i,r}^{\mathrm{env}}(q_a,t_a)
-V_{i,r-1}^{\mathrm{env}}(q_a,t_a)-c_{i,r}\right|_{w_V}^2
+\left\|\nabla V_{i,r}^{\mathrm{env}}(q_a,t_a)
-\nabla V_{i,r-1}^{\mathrm{env}}(q_a,t_a)\right\|_{W_F}^2
\right].
\tag{14}
$$

Indices outside the retained set are omitted. $c_{i,r}$ aligns additive energy offsets, with one reference fixed. Match the shared environmental component: distinct stage goals need not have equal task potentials. Dynamic observations from different times are not forced to agree without temporal prediction.

### 5.2 Evidence-conditioned reconstruction of previous stages

Let $\Theta_i=\{\theta_{i,r}:r\in\mathcal K_i\}$. First reconstruct a reference model using measured evidence and usable messages only:

$$
\overline\Theta_i\approx\arg\min_{\Theta_i}
\left[\sum_rL_{\mathrm{local}}(\theta_{i,r})
+\mu_{\mathrm{ov}}L_{\mathrm{overlap}}
+\mu_{\mathrm{msg}}L_{\mathrm{message}}
+\mu_{\mathrm{prox}}\|\Theta_i-\Theta_i^-\|_P^2\right].
\tag{15}
$$

Apply the physical checks below to this reference too. Freeze its parameters when constructing a counterfactual batch. Use a declared ensemble $\{\overline\Theta_i^{(h)},\omega_h\}_{h=1}^{K_h}$ if uncertainty requires multiple reconstructions, with $\omega_h\ge0$ and $\sum_h\omega_h=1$.

For each retained entry state, roll a candidate policy forward in each frozen reconstructed environment:

$$
\begin{aligned}
\widetilde x_{r,0}^{(h)}&=x_{r,\mathrm{entry}}^{\mathrm{observed}},\\
\widetilde u_{r,s}^{(h)}(\Theta)&=
\kappa_\Theta(\widetilde x_{r,s}^{(h)},g_r),\\
\widetilde x_{r,s+1}^{(h)}(\Theta)&=
\Phi_{\overline\Theta^{(h)}}^{\Delta t}
\bigl(\widetilde x_{r,s}^{(h)}(\Theta),\widetilde u_{r,s}^{(h)}(\Theta)\bigr),\\
C_{r,h}^{\mathrm{cf}}(\Theta)&=
\sum_{s=0}^{T_r-1}\Delta t\,\ell_r
(\widetilde x_{r,s}^{(h)},\widetilde u_{r,s}^{(h)})
+\ell_{r,T}(\widetilde x_{r,T_r}^{(h)}).
\end{aligned}
$$

The integrator here must represent the reconstructed environment under a **candidate control**, with the shaping force applied exactly once. If (1) represents an already closed-loop navigation field, separate its physical environment model and policy before applying this construction. This separation is a required modeling choice, not a license to optimize both the evaluation world and policy together.

Define one explicit proposed counterfactual learning objective:

$$
L_{\mathrm{cf}}(\Theta)
=\sum_{r\in\mathcal K_i}\nu_r\sum_h\omega_h
\left[C_{r,h}^{\mathrm{cf}}(\Theta)
-C_{r,h}^{\mathrm{cf}}(\Theta_i^-)\right],
\qquad\nu_r\ge0,\quad\sum_r\nu_r=1.
$$

The reference term is constant during differentiation. The gradient of the candidate cost flows through the entire reconstructed segment, so it can change the approach to an overlap, not only the model at the overlap. Weighted aggregation across retained starts and hypotheses gives a precise interpretation of repository averaging. It does not average incompatible physical states into a new actual state.

This is a model-based learning choice proposed here, not a loss specified verbatim by the professor. The reference environment is held fixed to prevent the candidate from reducing cost merely by changing the world against which it is evaluated. Restrict rollouts to supported regions, cap horizons, and penalize or reject trajectories leaving the trusted model domain. Use actual closed-loop validation because model exploitation remains possible.

In a changing environment, distinguish two questions: (i) a past-time reconstruction, which uses a temporal smoothing model to infer what was true then; and (ii) a current-environment replay from an earlier entry state, which evaluates a hypothetical current traversal. Label every repository entry with its temporal interpretation. Fresh observations cannot simply be asserted to have been true at an earlier time.

### 5.3 Error-driven learning proposal

$$
\begin{aligned}
L_{\mathrm{learn}}(\Theta)&=
\sum_rL_{\mathrm{local}}(\theta_r)
+\mu_{\mathrm{ov}}L_{\mathrm{overlap}}
+\mu_{\mathrm{msg}}L_{\mathrm{message}}
+\mu_{\mathrm{cf}}L_{\mathrm{cf}}
+\mu_{\mathrm{prox}}\|\Theta-\Theta^-\|_P^2,\\
P_\ell\delta\Theta_\ell&=-\nabla_\Theta L_{\mathrm{learn}}(\Theta^{(\ell)}),
\qquad P_\ell\succ0,\\
\widetilde\Theta^{(\ell+1)}&=\Theta^{(\ell)}+\eta_\ell\delta\Theta_\ell.
\end{aligned}
\tag{16}
$$

Backtrack $\eta_\ell$ on the frozen-batch objective. A Gauss–Newton approximation may precondition the residual terms; include the counterfactual cost gradient explicitly, since it is not generally a squared residual. This is where backward reconstructed trajectories now contribute to learning.

### 5.4 Physical reconciliation before commitment

For the same fixed context and parameter version, define a finite-data admissible set

$$
\mathcal C_{\mathrm{phys}}=
\left\{\Theta:
M\succeq m_{\min}I,\ J=-J^\top,\ R\succeq0,
\alpha\ge0,\ G\in\mathcal G_{\mathrm{cal}},
E_x^{\mathrm{fit}}\le\epsilon_x,
E_y^{\mathrm{fit}}\le\epsilon_y,
E_E^{\mathrm{fit}}\le\epsilon_E
\right\}.
$$

$\mathcal G_{\mathrm{cal}}$ is the calibrated/identifiable input-map class. $E_a^{\mathrm{fit}}$ are scaled empirical residual statistics; omit unavailable measurements. Positive mass, skew interconnection, and nonnegative dissipation are structural conditions. Empirical residual limits are finite-data consistency checks, not global physical guarantees.

$$
\Theta^\star=\arg\min_{\Theta\in\mathcal C_{\mathrm{phys}}}
\tfrac12\|\Theta-\widetilde\Theta\|_W^2,
\qquad W\succ0.
$$

Solve approximately with bounded iterations, check feasibility, and validate on reserved measured transitions. Permit commitment only when held-out residual limits and the declared learning-objective acceptance test pass. Otherwise retain the previous version. Reconciliation can undo some proposed objective improvement; do not claim monotonic mission improvement.

Update uncertainty and provenance on commitment, record the energy jump (6), and write the accepted model back to every affected stage record. The measured repository remains immutable; counterfactual trajectories have their own storage and labels. Information unsupported by observations, overlap structure, or valid temporal inference retains uncertainty.

## 6. Communicated Hamiltonian information

For sender $i$ and receiver $j$, a packet is

$$
m_{i\to j}=
\bigl(i,\mathrm{id},t_s,\Omega,\mathcal T,
q_s,p_s,v_s,\widehat H_s,\widehat V_s^{\mathrm{task}},
\mathcal Z_s,\Sigma_s,\mathcal P_s\bigr).
\tag{17}
$$

$\mathcal T$ identifies the coordinate transform; $\Omega$ is the supported region; $\mathcal P_s$ identifies source evidence, model version, and any evidence already received from others. State quantities use declared units and mass conventions. Velocity is redundant when momentum and mass are known, so a wire format may omit it. Here $V^{\mathrm{task}}$ is explicitly potential energy.

For the first implementation, transmit sparse local model anchors

$$
\mathcal Z_s=\{q_a,t_a,\widehat V_i^{\mathrm{env}}(q_a,t_a),
\widehat{\nabla V_i^{\mathrm{env}}}(q_a,t_a)\}_{a=1}^{K_m}.
\tag{18}
$$

These evaluations make the transmitted Hamiltonian information actionable. A scalar $H$ at a single state does not determine the surrounding gradient field. A compatible basis-coefficient packet is an alternative codec, evaluated under the same byte accounting.

The decomposition is necessary: agents can have different masses, targets, and task potentials. The receiver transfers the supported environmental information rather than copying the sender’s entire $H$, momentum, or controller. Position/momentum in a packet describe the estimation context; they do not replace the receiver’s physical state.

## 7. Receiver update under delay and correlated evidence

On arrival at $t_r$, set $\delta=t_r-t_s$. Transform anchors and gradients into the receiver’s frame. Predict the field to the intended use time with a declared temporal model:

$$
\bar z_m(t_r)=F_\delta\widehat z_m(t_s),
\qquad
\bar\Sigma_m(t_r)=F_\delta\Sigma_mF_\delta^\top+Q_\delta.
\tag{19}
$$

Here $z_m$ stacks transferable field values/gradients. A simple initial choice is a random-walk model with $F_\delta=I$ and $Q_\delta=\delta Q$. This increases uncertainty with age; it does not claim to predict a newly appearing obstacle. Messages outside their validity region or time horizon are rejected or held as historical evidence.

For independent evidence, the receiver’s message residual is

$$
L_{\mathrm{message}}
=\sum_{m\in\mathcal M_j^{\mathrm{usable}}}
\|h_m(\Theta_j)-\bar z_m\|_{(\bar\Sigma_m+\Sigma_{\mathrm{repr}})^{-1}}^2,
\tag{20}
$$

where $h_m$ evaluates the receiver’s corresponding field and $\Sigma_{\mathrm{repr}}$ accounts for representation mismatch. The receiver then solves (15), records (6), and rolls out its updated pH model.

Reciprocal communication creates correlations. Deduplicate packet IDs and source evidence, and never treat a returned copy of one’s own evidence as independent. If cross-correlation cannot be tracked, conservatively combine estimates in a common field representation, for example

$$
\Sigma_{\mathrm{fused}}^{-1}
=\omega\Sigma_j^{-1}+(1-\omega)\bar\Sigma_m^{-1},
$$

$$
\mu_{\mathrm{fused}}
=\Sigma_{\mathrm{fused}}
\left[\omega\Sigma_j^{-1}\mu_j
+(1-\omega)\bar\Sigma_m^{-1}\bar z_m\right],
\qquad 0\le\omega\le1.
\tag{21}
$$

Use this fused target once in place of the corresponding prior/message terms; do not also add (20) as independent evidence. The weights and covariance approximation are declared implementation choices. Conservative fusion addresses unknown correlation only insofar as the input uncertainty descriptions are themselves credible.

## 8. Recipient-dependent information value and communication rate

### 8.1 What the sender estimates

Each agent advertises compact status: timestamped pose, active stage/support, model version, and coarse uncertainty or acknowledged field summaries. These status messages also incur bytes and delay. Sender $i$ maintains an estimate $\widehat{\mathcal B}_{j|i}$ of receiver $j$’s state of knowledge, with uncertainty that grows between updates.

Define a task-relevant field vector $Z_j$ on the receiver’s predicted stage corridor. A model-based information-gain proxy is

$$
\widehat{\mathrm{IG}}_{i\to j}(m)
=\tfrac12\log\frac{\det(\widehat\Sigma_{j|i}^{Z}+\epsilon I)}
{\det(\widehat\Sigma_{j|i}^{Z,+m}+\epsilon I)}.
\tag{22}
$$

Use the same field variables and scaling in numerator and denominator. This is a Gaussian uncertainty proxy under the stated estimator, not automatically calibrated entropy. Receiver overlap, duplicate evidence, and message age enter the covariance update. A lower covariance need not imply better navigation.

The primary decision quantity is expected recipient/team cost reduction:

$$
A_{i\to j}(m\mid\mathcal I_i)
=\mathbb E\left[
C_{t:T}^{\mathrm{withhold}}-C_{t:T}^{\mathrm{deliver\ and\ update}}
\mid\mathcal I_i\right].
\tag{23}
$$

Both branches start from the same latent scenario, include delay, and use the same future exogenous randomness. The delivery branch actually executes (19)–(21), the buffered update (15), and subsequent closed-loop motion. Costs in (23) exclude the candidate message’s direct byte/processing charge, which is subtracted below; later consequences remain included.

During development, estimate (23) with paired simulated continuations over plausible receiver states and environments conditioned on the sender’s legal information. Fit a predictor

$$
\widehat A_\psi
=a_\psi\bigl(\mathcal B_i,\widehat{\mathcal B}_{j|i},m,
\widehat{\mathrm{IG}},\delta,\mathrm{channel}\bigr),
\quad
L_{\mathrm{value}}=\sum_s w_s(\widehat A_\psi(s)-\widehat A_s^{\mathrm{paired}})^2.
\tag{24}
$$

Counterfactual evaluation is used here to train communication value. It does not substitute for backward correction of stage models. A small analytical uncertainty proxy can initialize scheduling, but should not be presented as a learned mission-value estimator.

### 8.2 Rate selection

At communication interval $\Delta_c$, let $\lambda_{ij}\in[0,\lambda_{\max}]$ determine the chance of sending one eligible packet:

$$
p_{ij}=1-e^{-\lambda_{ij}\Delta_c}.
\tag{25}
$$

Choose rates by the local constrained optimization

$$
\max_{\lambda_i}
\sum_j p_{ij}\left[
\widehat A_{i\to j}-\beta_A s_{i\to j}
-c_b b_{ij}-c_{\mathrm{proc}}d_{ij}\right]
-\tfrac\rho2\sum_j\lambda_{ij}^2,
\tag{26}
$$

$$
\text{subject to}\quad
\sum_j p_{ij}b_{ij}\le C_i\Delta_c,
\qquad 0\le\lambda_{ij}\le\lambda_{\max}.
$$

$s_{ij}$ is value-estimation uncertainty, $d_{ij}$ predicted receiver processing work, and $\rho$ a declared rate regularizer. The uncertainty penalty is a scheduling heuristic unless calibrated into a confidence bound. For fixed positive net utilities this is a small concave maximization; nonpositive candidates optimally receive zero rate. Hard budgets, packet expiry, and duplicate suppression are checked before actual transmission.

Recompute value and rate after ACKs, new observations, or stage changes. Equation (26) assumes additivity of candidate marginal values within one interval; overlapping simultaneous packets require sequential admission with updated marginal values or a joint value model. A practical first implementation admits at most one model packet per sender per interval.

The same rules apply to $i\to j$ and $j\to i$. A follower can have positive value to a leader because its evidence is newer or covers a different part of a retained stage. Leadership alone does not determine communication direction or usefulness.

## 9. Numerical realization of the algorithm sketches

### 9.1 Explicit pH parameter and state updates

For a mechanical specialization, use $\theta=(w,\alpha,\rho,\gamma)$ for potential coefficients, barrier weights, damping parameters, and input-map parameters. The last symbol is not the older implementation's scalar damping coefficient. Fixed or unidentified blocks have zero increment.

$$
\begin{aligned}
\widetilde\theta&=\theta^-+(\Delta w,\Delta\alpha,\Delta\rho,\Delta\gamma),\\
\theta^+&=\operatorname{ReconcileValidate}(\widetilde\theta),\\
H^+(q,p,t)&=\tfrac12p^\top M^{-1}p+V^{\mathrm{task}}(q)
+\sum_aw_a^+\phi_a(q)+\sum_b\alpha_b^+b_{\mathrm{contact}}(d_b(q)),\\
D^+&=L_{\rho^+}L_{\rho^+}^\top,\qquad
R^+=\begin{bmatrix}0&0\\0&D^+\end{bmatrix},\qquad
G^+=\begin{bmatrix}0\\B_{\gamma^+}\end{bmatrix}.
\end{aligned}
$$

For constant calibrated $M$, freeze coefficients at the start of the integration interval. A concrete semi-implicit prediction is

$$
\begin{aligned}
\left(I+\Delta t D_n^+M^{-1}\right)\widehat p_{n+1}
&=p_n-\Delta t\nabla_qV^+(q_n,t_n)+\Delta t B_n^+u_n,\\
\widehat q_{n+1}&=q_n+\Delta t M^{-1}\widehat p_{n+1},\\
y_n^{\mathrm{port},+}&=(B_n^+)^\top M^{-1}p_n,\\
P_n^+&=-(M^{-1}p_n)^\top D_n^+(M^{-1}p_n)
+u_n^\top y_n^{\mathrm{port},+}+\partial_tV^+(q_n,t_n),\\
r_n^{E,+}&=H^+(x_{n+1},t_{n+1})-H^+(x_n,t_n)-\Delta t P_n^+.
\end{aligned}
$$

$x_{n+1}$ is measured; $\widehat x_{n+1}$ is predicted. The energy expression uses first-order quadrature and requires integration-aware tolerances. It is not an exact discrete conservation claim. Every update is followed by reconstruction of $H,R,G,y,\dot H$ before the next integration step.

For a learnable block $b$, energy consistency contributes

$$
\nabla_b(\mu_E L_E)=2\mu_E\sum_a\omega_a r_a^E\frac{\partial r_a^E}{\partial b},
\qquad
\nabla_b L_{\mathrm{learn}}=
\nabla_b(L_{\mathrm{local}}+\mu_{\mathrm{ov}}L_{\mathrm{overlap}}
+\mu_{\mathrm{msg}}L_{\mathrm{message}}
+\mu_{\mathrm{cf}}L_{\mathrm{cf}}+L_{\mathrm{prox}}).
$$

Thus the energy equation participates in model learning and physical reconciliation; it is not only a final audit.

### 9.2 Entropy, latent context, and dynamic clustering

The professor's stochastic extension calls for reconstructed $H$ and $S$ and the relevant gradients. The deterministic pH equations above do not specify an entropy-driven stochastic model. Do not add $\nabla S$ to them without defining the stochastic state, diffusion/mobility, boundary conditions, and resulting balance laws.

As a distinct candidate representation, maintain a local latent-context distribution

$$
\pi_{i,k,n}(c)=P(c\mid\mathcal D_{i,k,\le n},\mathcal M_{i,\le n}),
\qquad S^{\mathrm{context}}_{i,k,n}=-\sum_c\pi_{i,k,n}(c)\log\pi_{i,k,n}(c).
$$

This is context uncertainty, not automatically thermodynamic entropy or an entropy potential generating motion. Low/medium/high risk bins or soft clusters can summarize local estimates. Cluster stability may indicate stable inferred context, but can also reflect uninformative sensing or a stuck estimator; it does not prove the environment is static.

A simple proposed temporal Bayesian update is

$$
\pi_n(c)\propto p(o_n\mid c)\sum_{c'}P(c\mid c')\pi_{n-1}(c').
$$

Its likelihoods, transition model, and relationship to $H$ require specification and calibration. A spatiotemporal transformer is an optional estimator for these quantities, not a replacement for their definitions or the physical reconciliation step. No ELBO, exact filtering claim, or fully specified stochastic pH model is claimed here.

### 9.3 Across-episode learning

Local learning is observation-driven: measured transitions and sensor likelihoods supervise consistency without externally annotated optimal actions or Hamiltonians. Counterfactual predictions are internal model evaluations, not ground truth.

Let $\vartheta$ initialize the local estimator and $\psi$ predict message value. Across development episodes, optimize observation consistency and complete closed-loop cost, and train $\psi$ on the paired model-based estimates in (23)–(24). Count all counterfactual work. Select hyperparameters using separate validation worlds, permit only a predeclared bounded refresh, and freeze before test evaluation. The paired message-value targets are internally generated supervision; that component should not be described as entirely label-free.

## 10. Quantities to demonstrate

The primary questions are learnability, generalization, and added cooperative value. A selected successful animation is a mechanism illustration, not evidence of average superiority.

For learning round $m$, report

$$
\mathcal F_m=P(\mathrm{failure}\mid\pi_m,\mathcal E_{\mathrm{repeat}}),
\quad\mathcal J_m=\mathbb E_{e\sim\mathcal E_{\mathrm{heldout}}}[J_e(\pi_m)],
\quad\mathcal E_m^{\mathrm{model}}=\mathbb E[\|x^+-\widehat x^+_{\pi_m}\|^2].
$$

Use a fixed repeated-situation evaluation set to measure whether earlier failure modes disappear, and a separate held-out environment set to measure generalization. Plot these quantities against both observed transitions and total simulation/compute work; extra counterfactual computation is not free. Keep final test worlds untouched by model selection.

Two required scenario families follow the discussion: (i) a moving/changing hazard seen first by a leader, with follower benefit measured against communication delay; and (ii) complementary left/right coverage revealing a useful gap. Include reverse-direction fresh evidence, stale/duplicate packets, and no-benefit controls.

Compare the same two agents with communication disabled, enabled, and restricted to one direction. Also compare a single agent under declared matched sensing/compute budgets where feasible. An improvement from doubling sensing alone does not establish cooperative synergy. Define any “more than the sum of parts” statistic and its resource-matched reference before evaluation; do not infer it from team success alone.


Use matched worlds, initial states, dynamics, goals, sensing, and indexed exogenous randomness. Compare:

| Arm | Current-stage adaptation | Previous-stage correction | Model communication |
|---|---|---|---|
| Local | Yes | No | None |
| Buffered | Yes | Yes | None |
| Forward | Yes | Yes | Leader to follower |
| Reciprocal | Yes | Yes | Both directions |
| Fixed-rate | Yes | Yes | Both directions, fixed rate |
| Occupancy-sharing control | Same declared local learner | Same buffer setting | Occupancy observations rather than model packets |

Equalize resources where appropriate, and also plot cost versus bytes and processing work. Packet representations have different sizes; matching packet counts alone is insufficient.

Measure the incremental effects

$$
\Delta J_{\mathrm{buffer}}=\mathbb E[J_{\mathrm{Local}}-J_{\mathrm{Buffered}}],
\qquad
\Delta J_{\mathrm{reciprocal}}=\mathbb E[J_{\mathrm{Forward}}-J_{\mathrm{Reciprocal}}],
\tag{27}
$$

and the same-state delivery effect in (23), separately from the full communication-policy comparison. Add an ablation that keeps buffered evidence fitting but sets $\mu_{\mathrm{cf}}=0$, isolating the effect of counterfactual learning from memory alone. Report field-gradient prediction error on independently observed overlap probes, held-out transition error, energy residuals excluding learning jumps, bytes, update work, latency, and failures.

Required mechanisms include a newly observed overlap correction, useful forward warning, useful follower-to-leader correction in a changing region, and duplicate/stale/irrelevant packet controls. Report unsuccessful trials as well as successful demonstrations.

## 11. Analytical scope and unresolved choices

The algebraic energy identity follows from the structural model on fixed-parameter intervals. Backtracking can establish descent of the frozen-batch proposal objective; the separate reconciliation and validation step determines whether that proposal is committed. Neither fact proves decreasing mission cost, global safety, or convergence to an optimal path.

A complete analysis should derive, under stated observability, smoothness, uncertainty, and delay assumptions: (i) how overlap errors propagate through the retained stage chain; (ii) how model and gradient errors perturb finite-horizon rollouts; and (iii) when expected delivery benefit exceeds communication/processing cost. These are proof obligations, not established theorems in this proposal.

Before implementation, lock the following choices: the physical interpretation of the pH model and its controller; which components of $H,R,G$ are identifiable and learned; the sensor-to-field observation model; the temporal evolution model; the uncertainty/provenance representation; and whether the professor’s spoken “V” denotes velocity, potential, or another quantity. This document uses $v$ for velocity and $V$ for potential to avoid that ambiguity.

The essential computational chain is now explicit:

$$
\boxed{
\text{observed motion and local field evidence}
\longrightarrow\text{constrained local pH fit}
\longrightarrow\text{overlap-linked stage correction}
\longrightarrow\text{recipient-valued model communication}
\longrightarrow\text{receiver pH correction and new rollout}.
}
$$
