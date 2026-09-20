# Confirmatory campaign summary

Paired blocks: 2

## ood_test

| Method | Success | Team cost | Risk CVaR.95 | Bytes | Duplicate | Disjoint | Look-ahead |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent | 1.0000 | 1.2181 | 0.3098 | 0.0 | 3.0000 | 10.0000 | 0.2000 |
| raw_broadcast | 1.0000 | 1.2064 | 0.3085 | 900.0 | 2.5500 | 11.2000 | 0.3200 |
| compact_broadcast | 1.0000 | 1.1515 | 0.2864 | 180.0 | 2.1000 | 12.4000 | 0.4400 |
| novelty_mi | 1.0000 | 1.1569 | 0.2886 | 100.0 | 1.6500 | 13.6000 | 0.5600 |
| need_request_match | 1.0000 | 1.1104 | 0.2682 | 80.0 | 1.2000 | 14.8000 | 0.6800 |
| full_coph | 1.0000 | 1.1043 | 0.2696 | 70.0 | 0.7500 | 16.0000 | 0.8000 |

## test

| Method | Success | Team cost | Risk CVaR.95 | Bytes | Duplicate | Disjoint | Look-ahead |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent | 0.0000 | 1.2276 | 0.3188 | 0.0 | 3.0000 | 10.0000 | 0.2000 |
| raw_broadcast | 1.0000 | 1.2113 | 0.2957 | 900.0 | 2.5500 | 11.2000 | 0.3200 |
| compact_broadcast | 1.0000 | 1.1510 | 0.2893 | 180.0 | 2.1000 | 12.4000 | 0.4400 |
| novelty_mi | 1.0000 | 1.1294 | 0.2798 | 100.0 | 1.6500 | 13.6000 | 0.5600 |
| need_request_match | 0.0000 | 1.1018 | 0.2853 | 80.0 | 1.2000 | 14.8000 | 0.6800 |
| full_coph | 1.0000 | 1.0844 | 0.2753 | 70.0 | 0.7500 | 16.0000 | 0.8000 |

