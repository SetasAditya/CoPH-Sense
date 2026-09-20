# CoPH-Lookahead v1: static terrain information exchange

This is a separate, small **development mechanism experiment**, not a new
E2-Natural generator or a confirmatory result. It retains the existing VMAS
plant, pH executor, sensor dwell and operating costs, radio range/delay,
packet budget, and cost ledger. Static topology is public: two lanes are
separated by a known wall. The background material is also public. Only two
marked lane regions contain hidden surface/traction states. No topology or
persistent-homology inference is attempted.

Each agent has its own region belief. For a region and modality, the local
Gaussian natural parameters are

\[
 (\Lambda_i,\eta_i)=(\sigma_i^{-2},\mu_i\sigma_i^{-2}).
\]

A physical measurement yields one region-level factor
\((\Delta\Lambda,\Delta\eta)\). The sender's compact message contains the
region ID, modality, that factor, posterior mean/variance, acquisition time,
and provenance. The receiver adds the factor to **its own** belief after
delivery, updates its risk potential, and then takes its next pH step. The
wire payload is charged as 48 bytes; its internal terrain-evidence object is
only an accounting placeholder and never acts as a raw sensor reading. Raw
footprint broadcasting is evaluated separately through the same radio.
The 48-byte payload has an explicit fixed binary encoding, and receiver
updates use the transmitted float32 factors rather than uncharged precision.
Duplicate provenance is fused once. The constant-region prior makes the
summary sufficient *within this diagnostic*; a natural, heterogeneous map
would need spatial covariance or a different representation.

The evaluator records a receiver-conditioned Gaussian KL proxy for novelty,
\(D_{\rm KL}(b_j^+\Vert b_j^-)\), and separately computes the complete
physical decision value \(J_{\rm HOLD}-J_{\rm SEND}\). The sender's A4
features use its own packet, local history, teammate position, public costs,
and a public-prior novelty estimate. **True receiver KL and hidden terrain
never enter deployed actor features.** A small ridge value predictor was fit
to paired development labels as a first locally executable A4 rule.

For this straight-goal static task, \(\tau_i\) is the agent's x-progress in
meters and \(\Delta\tau=\tau_{\rm scout}-\tau_{\rm carrier}\). Local passive
and active sensing support are tracked separately from legitimately received
region support. Effective look-ahead is the farthest supported point ahead
of the agent; it is a geometric diagnostic, not a guarantee that every such
cell changes a decision.

## Executed checks

- A scout physically scans the lower lane, creates an innovation packet, and
  transmits it through the declared radio. Only after delivery does the
  carrier's region belief and pH risk grid change.
- In a matched retained/withheld intervention at identical physical state
  and sunk cost, the carrier chooses a lower-lane scan without the message
  and an upper-lane scan with it. It then physically completes the upper
  scan. The acquisition rule here is a transparent **local heuristic**, not
  a learned A3/A5 critic.
- The carrier can also send its upper-lane summary back through the same
  protocol. Delivery works in both directions; useful reverse-direction
  task value is not established by that check.
- Replay, provenance deduplication, sender-only A4 inputs, physical delivery,
  receiver belief change, and the matched A5 intervention have regression
  tests.

## Development results and limits

On the fixed ten-seed static-corridor slice (safe and risky first-region
realizations, 20 paired cases), the 48-byte innovation replaces a mean
636-byte raw footprint. It adds 3.0 m of mapped effective look-ahead to the
carrier and avoids about 0.0294 cost relative to raw broadcasting. Yet SEND
beats HOLD in only 1/20 cases; mean SEND value is -0.0230. On the separate
three-seed, three-progress-gap sweep (18 cases), SEND wins 3/18 and mean
value is -0.0193. These are descriptive development cases, not independent
large-sample evidence.

The five-seed train/five-seed test A4 ridge pilot selected HOLD for every
test case, matching the restricted exact message oracle on those cases.
Mean test team cost was 0.99143 for both; novelty-only summary broadcasting
cost 1.01639 and raw broadcasting cost 1.04579. This validates selective
withholding under the tested budget but **does not establish a learned
communication advantage** on cases where sending is worthwhile.

The progress-gap development sweep also exposes a generalization failure.
Fitting A4 on the one seed with positive risky-message value and testing on
two other seeds yields 1.00429 mean team cost, versus 0.99150 for HOLD and
the restricted exact message oracle. The simple sender feature vector does
not yet predict the narrow physical value boundary reliably. This result is
preserved, not tuned away.

The [development figure](results/lookahead_dev/lookahead_gap_sweep3.png)
shows novelty versus physical value, payload size, explored overlap,
look-ahead gain, progress-gap sensitivity, and cost/risk versus radio cost.
The Gaussian belief is deliberately simple and does not fully incorporate
passive appearance or correlations between different agents' overlapping
measurements. Its KL is a diagnostic proxy, not a calibrated conditional
mutual-information estimate.

## Next gate

Keep this static task fixed. Collect a larger, seed-disjoint source pool and
report the prevalence and margins of genuinely positive SEND opportunities.
If positive cases remain rare, the correct v1 conclusion is that compressed
look-ahead works causally but the current physical task rarely repays radio
cost. Do not train on selected positive cases as though they represented the
natural distribution. A learned A3/A5 policy and complete freely chosen
rollouts remain separate work after the A4 value question is resolved.
