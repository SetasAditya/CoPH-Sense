# Conditional-scout geometric benchmark

This is a kinematic geometric diagnostic. It is not a VMAS/material-pH result.

Analytic held-out count: 4000
Analytic dispatch accuracy: 0.9765
Analytic regret to oracle: 0.002680
Physical paired rollouts: 48
Physical success rate: 1.0000
Physical collision rate: 0.0000

| Case | Analytic dispatch | Physical dispatch preferred | Disagree |
|---|---:|---:|---:|
| static_hidden_block_useful | True | False | True |
| static_local_known_no_need | False | False | False |
| static_hidden_block_too_late | False | False | False |
| moving_blocker_prediction | True | False | True |
| moving_cost_prediction | True | False | True |
| moving_blocker_stale_report | False | False | False |
