- Baseline selected option: **Stabilize Core**.
- This artifact separates convergence, frontier stability, and directional driver stress.

Convergence by world count:

| Worlds | Runs | Recommendation consistency | Selected EV std (EUR) | Selected P05 std (EUR) |
| --- | --- | --- | --- | --- |
| 5000 | 6 | 1 | 4103.626321 | 7582.662322 |
| 10000 | 6 | 1 | 4697.819849 | 5069.319992 |
| 20000 | 6 | 1 | 4357.899094 | 3822.326607 |
| 40000 | 6 | 1 | 3052.347796 | 3442.31377 |

Frontier stability across the published seed/world-count grid:

| Threshold | Switching option(s) | Min boundary | Max boundary | Observed switch type(s) |
| --- | --- | --- | --- | --- |
| EV tolerance | none |  |  | no_switch_observed |
| Regret cap | Do Nothing | 594458.396419 | 609360.573822 | exact_match |
| Downside floor | none |  |  | no_switch_observed |

Monte Carlo error bands by option and world count:

| Worlds | Option | EV range (EUR) | P05 range (EUR) |
| --- | --- | --- | --- |
| 5000 | Do Nothing | 574.991169 | 1187.732788 |
| 5000 | Feature Extension | 11078.193779 | 17387.170141 |
| 5000 | New Capability | 5043.62274 | 16131.56764 |
| 5000 | Stabilize Core | 12313.39738 | 24033.870629 |
| 10000 | Do Nothing | 687.231818 | 1804.175681 |
| 10000 | Feature Extension | 13586.083964 | 17278.492876 |
| 10000 | New Capability | 6127.906007 | 13430.470092 |
| 10000 | Stabilize Core | 15350.017276 | 16211.794346 |
| 20000 | Do Nothing | 317.532848 | 1080.829976 |
| 20000 | Feature Extension | 9738.126869 | 7996.616018 |
| 20000 | New Capability | 5750.057252 | 15984.609209 |
| 20000 | Stabilize Core | 12789.854273 | 11015.138784 |
| 40000 | Do Nothing | 439.909523 | 614.46537 |
| 40000 | Feature Extension | 5988.993105 | 7512.740908 |
| 40000 | New Capability | 2045.21212 | 8046.824338 |
| 40000 | Stabilize Core | 10230.50699 | 10654.470878 |

Directional stress tests on the strongest material drivers:

| Parameter | Stress level | Tested value | Selected option | Recommendation changed? |
| --- | --- | --- | --- | --- |
| baseline_failure_rate | low | 0.04 | Do Nothing | yes |
| baseline_failure_rate | base | 0.1 | Stabilize Core | no |
| baseline_failure_rate | high | 0.15 | Stabilize Core | no |
| stabilize_failure_reduction | low | 0.4 | Do Nothing | yes |
| stabilize_failure_reduction | base | 0.6 | Stabilize Core | no |
| stabilize_failure_reduction | high | 0.8 | Stabilize Core | no |
| baseline_failure_rate + stabilize_failure_reduction | paired_all_adverse | paired override | Do Nothing | yes |
| baseline_failure_rate + stabilize_failure_reduction | paired_all_supportive | paired override | Stabilize Core | no |
| baseline_failure_rate + stabilize_failure_reduction | paired_opposing_challenge | paired override | Do Nothing | yes |
| baseline_failure_rate + stabilize_failure_reduction | paired_opposing_relief | paired override | Feature Extension | yes |