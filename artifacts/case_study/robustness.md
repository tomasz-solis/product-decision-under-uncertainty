- Baseline selected option: **Do Nothing**.
- This artifact separates convergence, frontier stability, and directional driver stress.

Convergence by world count:

| Worlds | Runs | Recommendation consistency | Selected EV std (EUR) | Selected P05 std (EUR) |
| --- | --- | --- | --- | --- |
| 5000.0 | 5.0 | 1.0 | 213.13737 | 406.082053 |
| 10000.0 | 5.0 | 1.0 | 248.798547 | 621.840731 |
| 20000.0 | 5.0 | 1.0 | 110.399593 | 394.520007 |

Frontier stability across the published seed/world-count grid:

| Threshold | Switching option(s) | Min boundary | Max boundary | Observed switch type(s) |
| --- | --- | --- | --- | --- |
| EV tolerance | none | nan | nan | no_switch_observed |
| Regret cap | Feature Extension | 195734.910254 | 204280.897869 | grid_bracket |
| Downside floor | Feature Extension | -147133.728698 | -144580.119938 | grid_bracket |

Directional stress tests on the strongest material drivers:

| Parameter | Stress level | Tested value | Selected option | Recommendation changed? |
| --- | --- | --- | --- | --- |
| do_nothing_drift_cost_eur | low | 30000.0 | Do Nothing | no |
| do_nothing_drift_cost_eur | base | 50000.0 | Do Nothing | no |
| do_nothing_drift_cost_eur | high | 90000.0 | Do Nothing | no |