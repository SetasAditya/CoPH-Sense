# E2-Complementarity development gate (not frozen)

The separate challenge generator, evaluator-only belief-expected acquisition
oracle, source-pool audit entry point, and matched memory intervention are
implemented. The E2-Natural generator, physical prices, risk formula, sensor
and packet semantics, and pH executor were not retuned for this work.

The challenge is **not yet admissible**. No Complementarity train/validation/
test manifest has been frozen and no result from this prototype should be
reported as a learned-policy or confirmatory E2 result.

Development parent 970000, four equiprobable hidden Region-1 material states.
The first diagnostic row was run before Region 2's safe status was exposed;
the latter two rows make Region 2 publicly safe:

| Restricted continuation | skip | geometry | traction | pair | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Carrier moves immediately, wider band | 1.2232 | 1.2900 | 1.2975 | 1.3773 | Pair loses; delivery after region entry |
| Scout ahead, carrier farther back, carrier moves | 1.2814 | 1.3534 | 1.3511 | 1.4543 | Pair loses; delivery after region entry |
| Scout ahead, carrier waits for selected readings | 1.2814 | 1.4159 | 1.4408 | 4.2692 | Pair branch fails in one hidden state |

The last row is a restricted team continuation that fully charges waiting. In
the high-surface/high-traction-risk state its pair branch costs 12.6835 and
does not deliver both readings. Hence it fails the all-branches-successful
admission rule even before considering the negative pair-value margin.

These failures are useful: they show that simply widening a material band or
making the carrier wait does not create a sound complementary physical task.
Do not train a pair-positive critic on these maps or relabel the failed
branches. The next design decision is to specify a physically feasible
challenge in which (i) both modalities can be acquired and delivered before
the carrier's route decision in every latent state, (ii) singles do not
already induce the pair's maneuver, and (iii) pair value remains positive
after all motion, dwell, packet, and waiting costs. Freeze a new source-pool
and its admission rate before learned-policy evaluation.

The evaluator checks initial observation equivalence across latent states,
all-branch physical success, actual capture and delivery, a conservative
before-region delivery proxy, and a belief-expected pair margin above 0.02.
It does not use any CoPH checkpoint for case admission. These checks do not
make this provisional generator a completed benchmark.

The disposable teacher-forcing smoke path does execute: a physical geometry
reading is acquired, SEND and HOLD are evaluated after that realized reading,
the SEND branch delivers to the carrier, and a matched retained/withheld A5
memory pair is produced. On the tested safe world HOLD costs 1.04393 and
SEND costs 1.10641, so this example correctly supplies a **hold** label,
not evidence that sharing is beneficial. This smoke result is not a held-out
decision-regret or complete-policy result.
