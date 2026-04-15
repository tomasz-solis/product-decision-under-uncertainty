- Baseline selected option: **Stabilize Core**.
- This artifact separates convergence, frontier stability, and directional driver stress.
- Dependency ablation: the correlated run selects **Stabilize Core**, while the independence rerun selects **Stabilize Core**.
- For the correlated selected option, downside P05 moves from **-306541.279637** under independence to **-360761.416674** with dependencies.

Dependency ablation by option:

| Option | Correlated EV (EUR) | Independent EV (EUR) | Correlated P05 (EUR) | Independent P05 (EUR) |
| --- | --- | --- | --- | --- |
| Do Nothing | -104670.596932 | -104635.377626 | -145373.574751 | -145589.021658 |
| Stabilize Core | 373233.468058 | 341973.625789 | -360761.416674 | -306541.279637 |
| Feature Extension | 197059.823415 | 169177.136693 | -420803.720046 | -379416.724567 |
| New Capability | -991538.254173 | -986144.078217 | -1408500.775471 | -1392532.791016 |

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