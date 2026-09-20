# Recipient-response A4 development comparison

The fixed CoPH-Lookahead v1 simulator, 48-byte innovation codec, packet/ACK
charges, pH executor, and corrected one-scan receiver continuation are reused
without change. The acquisition is forced only to create A4 source states.

The previously inspected seeds 981000–981011 are training data. A prospective
contiguous block 981012–981023 was collected before fitting; seeds 981012–17
are validation and 981018–23 are reserved test. Whole parent seeds, not
individual age/risk variants, determine the split. The new pool has 80 states
and 13 beneficial compact SEND cases; the held-out test half has only four
beneficial cases, so positive-SEND recall is a small-sample diagnostic.

The actor receives local packet data and public teammate pose/progress. Its
knowledge estimate incorporates ACKed packet factors and the public prior,
never the carrier's private observations. A sender-side counterfactual applies
the same transparent sensing heuristic to that estimated receiver state with
and without the candidate innovation. It supplies predicted HOLD/SEND scan,
duplicate avoidance, disjoint region area, travel change, and score margins.
These are estimates, not oracle receiver actions. The private-belief mutation
test checks the information boundary.

All MLP comparisons use the same 24-input, 32–32 SiLU network and 250-epoch
budget. Smaller feature sets are zero-padded. We compare sender-only features,
basic recipient context, response-conditioned features, and response features
with a jointly weighted sign loss. There are three predeclared model seeds.
Checkpoints are chosen by validation decision regret every 10 epochs; test is
read only after that selection. Never SEND, always compact SEND, novelty
threshold (selected on validation), sender ridge, raw broadcast, and the
restricted compact SEND/HOLD oracle are included.

Primary evaluation is paired decision regret for the compact action set.
The table also reports useful-SEND recall, false-SEND rate, precision, team
cost, success, communication bytes/cost, duplicate and new-region scans,
disjoint support, and communicated look-ahead. Raw broadcast is a third
action and is compared on team cost/bytes plus its cost gap to the compact
oracle; its value is not mislabelled as two-action regret.

The train/test package is a **development diagnostic**, not an ICLR
confirmatory result. A larger independent benchmark and learned A3/A5
continuation would be required for the eventual full-stack claim.

## Prospective v1 result

The new 12-seed block contained 13/80 useful compact sends. The reserved
six-seed test half contained only 4/40, so differences in recall are noisy.
On that test set:

| Method | Mean paired regret | Useful SEND recall | False SEND rate | Mean team cost | Mean charged bytes |
|---|---:|---:|---:|---:|---:|
| Never SEND | 0.000800 | 0/4 | 0/36 | 1.18868 | 0 |
| Always compact SEND | 0.079223 | 4/4 | 36/36 | 1.26711 | 120 |
| Sender ridge | **0.000348** | 2/4 | 0/36 | **1.18823** | 6 |
| Sender MLP | 0.001980 | 2/4 | 4/36 | 1.18986 | 18 |
| Basic recipient MLP | 0.005438 | 3/4 | 9/36 | 1.19332 | 36 |
| Receiver-response MLP | 0.001980 | 2/4 | 4/36 | 1.18986 | 18 |
| Receiver-response MLP + sign | 0.004290 | 3/4 | 7/36 | 1.19217 | 30 |
| Restricted compact oracle | 0 | 4/4 | 0/36 | 1.18788 | 12 |

Validation-tuned novelty chose HOLD everywhere and tied Never SEND. Raw
broadcast always sent, cost `1.29669` per state on average, and charged
`711.6` bytes (including headers and ACK) versus 120 for a compact send.
Its average cost gap to the compact HOLD/SEND oracle was `0.10880`; that is
not the two-action regret in the table.

The response MLP has lower test value MSE than the sender MLP (`0.000336`
versus `0.000363`) but selects the *same six messages*. The sign loss catches
one additional useful message while making three additional false sends.
Hence neither the response representation nor the auxiliary sign loss has
earned a positive learning claim on this split. The sender ridge's small
cost gain over Never SEND is also uncertain: a seed-block bootstrap interval
for its per-seed regret difference includes zero (approximately
`[-0.00146, 0]`). The pilot supports the physical mechanism and corrected
teacher, while A4 decision learning remains unresolved.
