# Disposable E2-dev pipeline

The first executed engineering pass uses only groups from
`results/manifests_v2/train.json`. `e2_dev_smoke.py` generates physical
skip/single/pair teacher continuations for a small carrier-local menu, fits
the existing `VariableSetValueNet`, and saves a disposable checkpoint.
`e2_dev_replay.py` reloads it and executes the selected action on a separate
training-manifest development map. The run proves only that the data,
candidate features, checkpoint, and physical replay connect.

It is intentionally **not** a final CoPH-Sense training run: the teacher uses
one realized world per action, so it does not estimate belief-conditional
expected cost or CVaR; the sampled maps and labels are tiny; A4 is still the
scripted send-all continuation; and A5 has no post-delivery training states.
Its weights must be discarded before any confirmatory E2 run.

The subsequent `e2_dev_stack.py` adds a post-reading A4 SEND/HOLD teacher,
paired retained/withheld A5 histories, separate learned sharing parameters,
and an A3/A5 critic shared across pre- and post-delivery observations.
`e2_dev_refresh.py` collects one round of counterfactual labels from histories
visited by the learned stack. These are still disposable engineering runs.
The old v2 manifests currently fail `python -m coph_terrain.manifests verify`
because `generator.py` changed. Their group IDs are used solely as a
development seed list; their archived map hashes are not treated as valid.
New current-generator splits and hashes are required before final training.
Those replacement `results/manifests_v3_natural` files have now been written
and verified across 1,100 disjoint parent groups. The disposable v2-seed
engineering checkpoints are not promoted into that split; final training
must start with newly initialized weights and read the verified v3 manifests.

The next training implementation must collect four distinct records without
leaking the evaluator's latent map into an actor input:

1. **A3 pre-acquisition:** local evidence/memory and public candidates,
   paired latent-world continuations for skip/single/pair choices, and the
   actual frozen A4/A5 continuation.
2. **A4 post-observation:** realized local reading and owned evidence,
   delivered-message alternatives including silence, with transport/timing
   charged and no request revealing a feature by itself.
3. **A5 post-delivery:** recipient-local memory after a packet arrives,
   including matched retained/reset interventions, then subsequent
   acquisition choices and physical outcomes.
4. **Closed-loop rollout:** both agents execute only decisions available at
   their local histories; evaluate complete-policy cost and success on
   separate development groups before locking the final training recipe.

For final fitting, use multiple independent latent realizations per public
state and paired randomness across actions. The pilot one-world best action
is a debugging reference, not a target for claimed expected-value or
tail-risk performance.

The separately implemented Complementarity development generator has not
passed its physical admission rule; see `E2_COMPLEMENTARITY_DEV_STATUS.md`.
Its rejected maps must not be used as pair-positive training cases or as a
confirmatory test set. The A4/A5 teacher-forcing interfaces can be exercised
on disposable examples, but final mixed-distribution fitting waits for a
frozen, admissible challenge manifest.

Following the professor's look-ahead clarification, the next separate
mechanism track is documented in `LOOKAHEAD_V1.md`: two moving pH agents,
region-level innovation messages, sender-only task-value features, and
matched local-memory interventions on public static topology. It is kept
distinct from E2-Natural and from the unfinished pair-positive
Complementarity admission effort.
