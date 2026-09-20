# Recipient-conditioned A4 development audit

This audit keeps CoPH-Lookahead v1's map, terrain values, pH executor, sensor,
48-byte innovation, radio protocol, prices, and continuation unchanged. Scout
geometry acquisition is forced **only to create a post-reading A4 state**.
Deployment is still free to choose HOLD. These data are disposable development
data, not an E2 test result.

`lookahead_recipient_a4.py` constructs a seed-disjoint pool from two fixed
region-risk settings, three scout starting positions, three message ages, and
occasional physically acquired private carrier overlap. Cases are not selected
by whether SEND wins. For each state it evaluates paired complete pH
continuations for HOLD, SEND at normal delay, and SEND at zero network delay.
The diagnostic decomposition closes exactly:

`J(HOLD)-J(SEND) = (J(HOLD)-J(SEND_0, nonradio))`
`- (J(SEND_d, nonradio)-J(SEND_0, nonradio)) - C_radio`.

The radio ledger is split into payload-byte charge and remaining fixed attempt,
header, and ACK charge. Zero-delay changes the network timing in an evaluator
clone; it is not a deployable alternative action. The two physical terms are
therefore descriptive paired counterfactuals, not a causal attribution to a
single isolated mechanism.

The sender-only and recipient-conditioned A4 rules use the **same ridge
estimator** and train on the same seed group. The new inputs include public
receiver pose/progress, wall distance, packet age, public-prior uncertainty,
ACKed same-region evidence, a simple route-relevance proxy, delivery slack,
and charged radio cost. They never use receiver private terrain observations.
The receiver's actual belief variance, support overlap, and route/sensing
changes are saved as evaluator-only diagnostics. In particular, unreported
private receiver measurements cannot be known perfectly to the sender; a
public-position estimate is not ground truth recipient knowledge.

The restricted oracle is `min(J(HOLD), J(SEND))` for the realized world and
scripted continuation. It is not an optimal decentralized policy over all
possible sensing and communication histories. Positive SEND prevalence and
seed-disjoint decision regret determine whether this A4 representation has
earned further development. Conditional information gain is only a novelty
diagnostic, never the training target or SEND criterion.

Run with:

`PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python -m coph_terrain.lookahead_recipient_a4 --seeds 12 --tag dev12`

The output JSON and Δτ/value plot are written under
`results/lookahead_recipient_a4/`.

## `dev12` result (development only)

The fixed contiguous seeds 981000–981011 produced 80 post-reading states:
72 natural age/risk cases and 8 physical receiver-overlap cases. Acquisition
and branch generation had no failures. SEND was worse in all 80, with mean
`J(HOLD)-J(SEND)=-0.02851` and best margin `-0.01101`. The seed-disjoint
test half likewise has no positive SEND case. Both linear rules selected HOLD
for every test state and have zero restricted decision regret. The
recipient-conditioned rule reduced value-prediction MSE from `3.104e-5` to
`2.648e-5`, but this does **not** establish better selective sharing because
the test contains no beneficial SEND decisions.

Mean decomposed value is `-0.00203` immediate-delivery nonradio physical
gain, minus `0.00048` additional delay-related nonradio cost, minus `0.02600`
radio. Of radio, `0.00240` is the 48-byte payload and `0.02360` is attempt,
header, and ACK overhead. Further compression therefore has little leverage
in this fixed task. The private-overlap cases occur about 93 steps later than
their source readings, so their larger negative values must **not** be
attributed to overlap alone; age, pose, and route progress also changed.

As an implementation positive control, rerunning the previously known
seed-980010 risky case gives `J(HOLD)-J(SEND)=+0.02186` with the same code.
It is not included in training or test, because it was already inspected.
The audit can reproduce a valuable SEND, but this new contiguous source pool
does not contain one. A larger prospective pool or a continuation that
actually executes the recipient's revised sensing decision is needed to test
positive-SEND recognition. The current scripted continuation can change the
recipient's route; it does not execute the diagnostic next-sensing heuristic,
even when that heuristic changes after delivery.

## Corrected downstream A4 continuation

`lookahead_a4_downstream.py` evaluates the **same source-state grid** with a
different A4 teacher continuation. After HOLD or after actual SEND delivery,
the carrier applies the same transparent region-priority rule to its own
current observation. The selected geometry scan is then physically executed
through `run_continuation` with the frozen pH executor; newly acquired
evidence is not automatically rebroadcast. The rule is a restricted local
continuation, not an exact decentralized optimizer or a trained A3/A5 policy.

The SEND branch still pays reconnect motion, packet/ACK cost, and delay before
the receiver's choice. The HOLD branch chooses immediately. Thus differences
in feasibility and timing are part of the actual action consequences. Each
result records whether the chosen scan completed, duplicate Region-1 sensing
was avoided, Region-2 sensing was gained, and how many new carrier support
cells lie outside the scout's already explored support. The outcome cost is
the same full mission-plus-risk ledger used in the routing-only audit.

Run the corrected audit with:

`PYTHONPATH=CoPh-sense CoPh-sense/external/envs/phmarl-py38/bin/python -m coph_terrain.lookahead_a4_downstream --seeds 12 --tag dev12`

### Corrected `dev12` result

The same 12 seeds and 80 source conditions now yield **16/80 positive SEND
values**, including 10/40 on seed-disjoint test. Both branches succeeded and
the selected carrier scan physically completed in every case. The actor
feature arrays are bitwise identical to the earlier routing-only audit; only
the teacher continuation changed. In 72 natural conditions, HOLD selected
Region 0 and delivered SEND selected Region 1. In
the 8 late private-overlap conditions both selected Region 1. The 72 switches
each replace a duplicate Region-0 scan with a Region-1 scan, gaining 10 cells
of carrier support outside the entire scout-summarized Region 0. This support
gain occurs in both profitable and unprofitable SEND cases, so it is **not** a
sufficient value criterion.

Across all conditions, mean `J(HOLD)-J(SEND)=-0.07033`: the mean nonradio
physical benefit is `-0.04433` and incremental radio cost is `0.02600`.
For the 16 beneficial cases, mean nonradio benefit is `+0.03456`, exceeding
radio cost by `0.00856`. These physical effects are dominated by mission time
and path consequences in this small task; mean material-risk exposure change
is near zero. Fixed/header/ACK radio remains `0.02360`; payload remains
`0.00240`. The private-overlap cases occur much later and incur large send
reconnect/delay cost, so overlap and timing are confounded in that stratum.

The unchanged sender-only ridge selected HOLD for every held-out case, with
mean regret `0.00203`. The recipient-conditioned ridge selected SEND four
times, caught 2/10 valuable sends, made two false sends, and achieved mean
regret `0.00184`. Its value MSE was worse than sender-only on this held-out
set. Thus the corrected teacher supplies the previously missing positive
decision labels, but this first recipient-conditioned A4 learner has **not**
earned a strong learning claim. Its next revision should use this corrected
target and held-out decision regret, not tune the environment or radio price.
