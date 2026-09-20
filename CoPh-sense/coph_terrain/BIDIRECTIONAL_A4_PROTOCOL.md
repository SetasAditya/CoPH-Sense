# Bidirectional A4 admission protocol

This is a separate controlled communication task. It does not revise or
replace Natural Lookahead-v1, whose Bayes-optimal supported action is HOLD.

The receiver may send a charged 32-byte `NeedPacket` containing only:

- relevant region and modality;
- its current uncertainty;
- declared relevance to its next task decision;
- latest physically useful receipt time;
- receiver progress; and
- a request identifier.

The sender combines this packet with its legal local history and its existing
48-byte belief innovation. Neither packet contains hidden terrain, evaluator
cost, future randomness, or the receiver's complete private belief.

The fixed protocol is:

1. receiver emits `NEED` and pays header, payload, attempt, and ACK charges;
2. only after delivery may the sender condition on the need;
3. sender chooses `SEND`, `HOLD`, or an explicit charged `DECLINE`;
4. evidence is usable only after its own delivery and before the declared
   actionability deadline; and
5. the receiver updates its belief, chooses its next acquisition, and executes
   the same frozen pH continuation used in both SEND and HOLD branches.

Before learning, construct seed-disjoint useful, irrelevant, and redundant
contexts by public task/need variables, never by CoPH performance. The same
sender innovation must occur in paired contexts where the receiver declares:

- **useful:** matching unresolved region, relevant decision, timely delivery;
- **irrelevant:** region does not affect the pending decision; and
- **redundant:** equivalent evidence is already ACKed.

For each legal context, resample compatible hidden worlds and continuation
randomness and estimate

\[
V_{\rm SEND}(\mathcal H_i,q_j)
=\mathbb E[J_{\rm HOLD}-J_{\rm SEND}\mid\mathcal H_i,q_j].
\]

The final numerical admission rule is fixed before the confirmatory seed block:

- in **each direction**, the one-sided 95% lower bound for useful NEEDs must
  exceed `+0.01` team-cost units; and
- in **each direction**, the one-sided 95% upper bounds for irrelevant and
  redundant NEEDs must be below `-0.005`.

The earlier sign-check pools are development evidence because a numerical
margin had not yet been written down. The gate is evaluated once on a fresh
confirmatory block. If admission fails, close the controlled task without
model fitting. If
it passes, freeze task code and manifests, then compare Never SEND, always
broadcast, request-matched heuristic, sender-only A4, and history-plus-need A4
under identical charged continuations.
