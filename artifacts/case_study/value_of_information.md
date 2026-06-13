- This view answers which uncertainty is worth resolving before deciding. It is computed against the expected-value-optimal action, not the policy pick.
- Expected-value-optimal action under full uncertainty: **Stabilize Core**.
- EVPI (value of resolving every uncertainty before deciding): €131,457.
- EVPPI below is the value of resolving one parameter on its own. Each EVPPI is bounded by the EVPI, and the values are not additive across parameters.

| Parameter | Unit | EVPPI | Share of EVPI | What it measures |
| --- | --- | --- | --- | --- |
| extension_uptake | adopter_share | €72,118 | 55% | Share of users who adopt the optional express path. |
| extension_value_per_uptake_eur | eur_per_adopting_unit | €35,757 | 27% | Additional value created by one adopting unit before decay or disablement. |
| baseline_failure_rate | share_of_volume | €28,312 | 22% | Share of annual volume that fails on the legacy path before any intervention. |
| stabilize_failure_reduction | share_of_failure_removed | €19,422 | 15% | Share of baseline failure exposure removed by the core-stability option. |
| cost_per_failure_eur | eur_per_failure | €11,328 | 9% | Direct operational cost created by one failed unit. |
| failure_to_churn_rel | revenue_penalty_share | €5,000 | 4% | Relative downstream value loss that follows when one failure damages trust, retention, or future conversion for the affected unit. |
| stabilize_core_launch_delay_months | months | €1,190 | 1% | Time before the stabilize-core investment begins producing production benefits. |