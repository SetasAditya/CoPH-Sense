# Prospective recipient-conditioned A4 gate

The environment, pH continuation, radio prices, 48-byte innovation, and
HOLD/SEND teacher are unchanged. The teacher physically executes the
receiver's next sensing decision in both branches. The learning question is
whether a recipient-aware sender policy can improve on Never SEND and the
sender-only ridge policy under the same paired continuation labels.

`large_v3/manifest.json` fixes disjoint seed blocks before any validation or
test labels: 400 training seeds, 120 validation seeds, and 600 test seeds
starting at 983000. Every seed supplies safe/risky region conditions and four
message ages (0, 8, 16, 32 steps); public scout position cycles through three
positions. Every third seed also supplies a physically obtained private
receiver-overlap case, and the next third supplies a prior delivered-and-ACKed
duplicate-message case. These conditions are selected by seed and public
history, never by teacher SEND value. The manifest records hashes of the task,
teacher, and collector sources. Shards are resumable and checked for complete
seed coverage.

The first four-shard `large_v1` and 16-shard `large_v2` collections were
throughput and prevalence pilots on training-design seeds only; neither
contains validation or held-out labels. The final `large_v3` protocol was
fixed after adding ACK histories. Those pilots must not be pooled into the
final training or evaluation splits.

The structured candidate predicts

\[
\widehat V_{\rm SEND}
=\widehat V_{\rm avoided\ repeat}
 +\widehat V_{\rm new\ support}
 -\widehat C_{\rm delay/reconnect}
 -C_{\rm radio}.
\]

The first term is nonnegative and active only when public recipient-response
features predict an avoided repeat; the second may be signed and is active
when they predict new support; the third is nonnegative; radio is the exact
declared charge. The heads are fitted from total paired SEND value, so their
separate outputs are **not identified causal effects**. The comparison tests
the decision quality of this structured inductive bias. Causal decomposition
still comes from the physical paired teacher ledger.

All model checkpoints and the novelty threshold use validation regret only.
The final test compares Never SEND, always compact SEND, novelty threshold,
sender ridge, the same sender/recipient MLP controls from the earlier run,
recipient/response ridge controls, the structured value model, and the
per-world paired hindsight oracle. The last uses realized SEND/HOLD costs and
is not a deployable decentralized optimum. Report useful
SEND recall, false-SEND rate, regret, team cost, bytes, duplicate scanning,
new-region scanning, and seed-bootstrap paired differences. If the frozen
test happens to contain fewer than hundreds of useful SENDs, report the
actual count; do not add or choose cases after viewing their labels.
The report also gives predeclared risk × source-condition × message-age
strata, so any aggregate gain can be checked against safe, risky, overlap,
ACKed, early, and late states.

The first full compact comparison found no learned A4 gain: structured value
and the value-trained MLPs matched Never SEND at mean test regret 0.000921;
sender ridge and recipient ridge had higher regret. A post hoc exact-feature
audit found zero positive-mean groups for all three deployed feature maps on
train, validation, and test. This only characterizes the empirical pools and
current feature maps; richer legal local histories could still expose a
profitable conditional SEND rule. The per-world hindsight oracle is not a
valid learned-policy comparator for that question.

The full 5,600-state raw replay confirms the resource mechanism. Always
compact SEND averages 120 charged bytes, 0.02786 radio cost, and 1.26573 team
cost. Always raw broadcast averages 711.43 bytes, 0.05743 radio cost, and
1.29531 team cost. The realized-world hindsight compact selector averages
11.94 bytes and 1.18862 team cost, but remains non-executable from the current
features. Compact representation is therefore supported; learned selective
communication is not.

The final hard-stop audit conditions SEND value on the complete legal history
under the finite Lookahead-v1 generative model. In this model, the Gaussian
innovation is a sufficient statistic for the sender's local region
measurement; public start/progress and message age summarize the scripted
trajectory; and prior sends and ACKs summarize sender-known recipient overlap.
Receiver-private scans, the evaluator's terrain label, continuation randomness,
and the seed identifier are excluded. For each legal context, hidden safe and
risky worlds and independent continuation seeds are averaged before making a
decision.

No context has positive conditional SEND value replicated on validation and
test. The train-derived Bayes estimate selects SEND in 0/5,600 test states.
The closest case is the risky, most-advanced scout at zero message age: its
test mean is -0.00265 (one-sided 95% upper bound -0.00091), despite 43% of its
individual worlds being hindsight-positive. Thus Natural A4 is closed with
HOLD as the supported decentralized decision; a history encoder is not trained.
This is a finite-task information-structure result, not a universal
impossibility statement. A selective-communication claim requires a separately
declared task where recipient benefit is identifiable from legal sender context.

Run the stopping audit with:

```bash
python -m coph_terrain.lookahead_a4_bayes_audit \
  coph_terrain/results/lookahead_recipient_a4/large_v3
```

Collection command (run from `CoPh-sense`):

```bash
python -m coph_terrain.lookahead_a4_collect \
  --out-dir coph_terrain/results/lookahead_recipient_a4/large_v3 \
  --start 983000 --train-seeds 400 --validation-seeds 120 --test-seeds 600 \
  --ages 0 8 16 32 --include-acked-duplicate --workers 32
```

After collection:

```bash
external/envs/phmarl-py38/bin/python -m coph_terrain.lookahead_a4_raw_collect \
  coph_terrain/results/lookahead_recipient_a4/large_v3 --workers 16

external/envs/phmarl-py38/bin/python -m coph_terrain.lookahead_a4_large_learning \
  coph_terrain/results/lookahead_recipient_a4/large_v3
```

Then render the same A4 figures, with the structured model's held-out
decisions overlaid on the value-versus-progress plot:

```bash
MPLCONFIGDIR=/tmp/mpl-coph external/envs/phmarl-py38/bin/python \
  -m coph_terrain.lookahead_a4_figures \
  coph_terrain/results/lookahead_recipient_a4/large_v3/learning_large_v1.json \
  --development coph_terrain/results/lookahead_recipient_a4/large_v3/train.json \
  --prospective coph_terrain/results/lookahead_recipient_a4/large_v3/validation.json \
  --selected-method structured_value \
  --out coph_terrain/results/lookahead_recipient_a4/large_v3/figures
```
