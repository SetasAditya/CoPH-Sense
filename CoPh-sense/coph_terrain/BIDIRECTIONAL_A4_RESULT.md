# Frozen bidirectional A4 result

Natural Lookahead-v1 remains closed: realized SEND benefits are not
predictable from the sender's legal history. The separate NEED-conditioned
task changes that information structure with a charged 32-byte receiver need
summary followed by an optional charged 48-byte belief innovation.

The admission gate passed in both directions before learning. One fixed
two-layer value MLP was then trained once on 288 rows, selected by value MSE
on 192 validation rows, and evaluated on the frozen 384-row confirmatory
block. The primary result is paired decision regret.

| Method | Regret | Useful recall | False SEND | Team cost | Charged bytes |
|---|---:|---:|---:|---:|---:|
| Never SEND | 0.04464 | 0% | 0% | 1.15681 | 56.0 |
| Always compact | 0.01969 | 100% | 100% | 1.13186 | 128.0 |
| Request match | 0.00174 | 100% | 0% | 1.11390 | 80.0 |
| Sender-only A4 | 0.01113 | 91.4% | 45.7% | 1.12330 | 99.9 |
| History + NEED A4 | **0.00142** | **100%** | 6.25% | **1.11358** | 83.0 |
| Conditional Bayes lookup | 0.00142 | 100% | 6.25% | 1.11358 | 83.0 |
| Realized-world oracle | 0 | 100% | 4.30% | 1.11216 | 82.1 |

The NEED-aware learner reaches the finite conditional Bayes reference and
avoids all 128 useful duplicate scans. Its advantage over request matching is
small: mean regret difference -0.000317 with a seed-bootstrap 95% interval of
[-0.001999, 0.001288]. The interval includes zero. Therefore this experiment
supports the information-structure and local-executability claim, not a claim
that a neural value model outperforms the transparent heuristic.

Directionally, history + NEED has zero regret and zero false SENDs for
scout-to-carrier. Carrier-to-scout regret is 0.00284 with a 12.5% false-SEND
rate. Those extra sends match the conditional Bayes lookup: in some legally
identifiable reverse-direction regimes, transmission has positive expected
physical value even when the coarse request label is `irrelevant`.

No architecture or loss sweep was performed. The task, packet formats, radio
prices, pH continuation, context generator, and seed blocks remain frozen.
The next experiment integrates this policy with A3/A5 and pH execution.
