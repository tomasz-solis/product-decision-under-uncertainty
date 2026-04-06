- Baseline selected option: **Stabilize Core**.
- This artifact separates convergence, frontier stability, and directional driver stress.

Convergence by world count:

| Worlds | Runs | Recommendation consistency | Selected EV std (EUR) | Selected P05 std (EUR) |
| --- | --- | --- | --- | --- |
| 5000.0 | 6.0 | 1.0 | 4103.626321 | 7582.662322 |
| 10000.0 | 6.0 | 1.0 | 4697.819849 | 5069.319992 |
| 20000.0 | 6.0 | 1.0 | 4357.899094 | 3822.326607 |
| 40000.0 | 6.0 | 1.0 | 3052.347796 | 3442.31377 |

Frontier stability across the published seed/world-count grid:

| Threshold | Switching option(s) | Min boundary | Max boundary | Observed switch type(s) |
| --- | --- | --- | --- | --- |
| EV tolerance | none | nan | nan | no_switch_observed |
| Regret cap | Do Nothing | 594458.396419 | 609360.573822 | exact_match |
| Downside floor | none | nan | nan | no_switch_observed |

Monte Carlo error bands by option and world count:

| Worlds | Option | EV range (EUR) | P05 range (EUR) |
| --- | --- | --- | --- |
| 5000 | Do Nothing | 574.9911694832117 | 1187.7327882494428 |
| 5000 | Feature Extension | 11078.193779130612 | 17387.17014123255 |
| 5000 | New Capability | 5043.622740239371 | 16131.567640122725 |
| 5000 | Stabilize Core | 12313.397379597183 | 24033.87062856555 |
| 10000 | Do Nothing | 687.2318183672614 | 1804.1756813888496 |
| 10000 | Feature Extension | 13586.08396369958 | 17278.492875617696 |
| 10000 | New Capability | 6127.906006940408 | 13430.470091939205 |
| 10000 | Stabilize Core | 15350.017275900522 | 16211.794345931965 |
| 20000 | Do Nothing | 317.53284805905423 | 1080.8299755372282 |
| 20000 | Feature Extension | 9738.126868683845 | 7996.616018267174 |
| 20000 | New Capability | 5750.057252384839 | 15984.609208585229 |
| 20000 | Stabilize Core | 12789.854273485136 | 11015.138783787726 |
| 40000 | Do Nothing | 439.90952261201164 | 614.4653702901851 |
| 40000 | Feature Extension | 5988.993104582827 | 7512.7409081957885 |
| 40000 | New Capability | 2045.2121203844436 | 8046.824337571394 |
| 40000 | Stabilize Core | 10230.50698982447 | 10654.470877728309 |

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