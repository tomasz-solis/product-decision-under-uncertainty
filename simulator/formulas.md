# Formula Appendix

This appendix gives the compact equation view of the current model.

## Shared terms

- `V`: annual volume in units from `simulation.annual_volume`
- `T`: time horizon in years from `simulation.time_horizon_years`
- `r`: annual discount rate from `simulation.discount_rate_annual`
- `f`: `baseline_failure_rate`
- `v`: `value_per_success_eur`
- `c_f`: `cost_per_failure_eur`
- `c_churn`: `failure_to_churn_rel`
- `p_reg`: `regression_event_prob`
- `L_reg`: `regression_event_cost_eur`
- `D`: `do_nothing_drift_cost_eur` for internal operating drag only

## Timing terms

For each intervention option `o`:

- `delay_o`: `*_launch_delay_months`
- `ramp_o`: `*_benefit_ramp_months`
- `k_o`: `*_cost_overrun_multiplier`
- `B_o`: discounted equivalent benefit-years after applying launch delay and ramp
- `A_o`: discounted equivalent active-years after launch
- `R_o`: discounted equivalent residual-drift years before full effect

Those timing weights are computed on a monthly grid.

The cost buckets are intentionally separate:

- `c_f` is direct failure cost
- `c_churn * v` is customer-side value loss caused by a failure
- `D` is background internal drift cost and excludes customer harm

## Do Nothing

```text
Value_do_nothing = -(D * discounted_horizon_years)
```

## Stabilize Core

```text
improvement = f * stabilize_failure_reduction

Value_stabilize =
  (V * B_stabilize * improvement * v)
  + (V * B_stabilize * improvement * c_f)
  + (V * B_stabilize * improvement * c_churn * v)
  - (stabilize_core_upfront_cost * k_stabilize)
  - (stabilize_core_annual_maintenance_cost * A_stabilize)
  - (D * R_stabilize)
  - regression_loss_stabilize
```

## Feature Extension

```text
extension_improvement = f * extension_uptake * extension_exposure_reduction
extension_created_value =
  V * B_extension * extension_uptake * extension_value_per_uptake_eur
extension_realized_value = extension_created_value * (1 - extension_loss_rate)

Value_extension =
  (V * B_extension * extension_improvement * v)
  + (V * B_extension * extension_improvement * c_f)
  + (V * B_extension * extension_improvement * c_churn * v)
  + extension_realized_value
  - (feature_extension_upfront_cost * k_extension)
  - (feature_extension_annual_maintenance_cost * A_extension)
  - (D * discounted_horizon_years)
  - regression_loss_extension
```

## New Capability

```text
successful_units = V * B_new_capability * (1 - f)
uplift_value = successful_units * v * new_capability_uplift

Value_new_capability =
  uplift_value
  - (new_capability_upfront_cost * k_new_capability)
  - (new_capability_annual_maintenance_cost * A_new_capability)
  - (D * discounted_horizon_years)
  - regression_loss_new_capability
```

## Regression loss

```text
regression_loss_option = 1[event occurs] * L_reg

event occurs ~ Bernoulli(p_reg * option_multiplier)
L_reg ~ Lognormal(median, p95)
```
