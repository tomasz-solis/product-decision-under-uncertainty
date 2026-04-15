# Assumptions

This file defines the main assumption names used in the case study. For provenance, see [parameter_provenance.md](parameter_provenance.md). For the dependency-modeling choice, see [../docs/adr/adr-001-gaussian-copula.md](../docs/adr/adr-001-gaussian-copula.md).

## Core economics

- `baseline_failure_rate`
  Share of annual volume that fails on the legacy path before intervention.
- `value_per_success_eur`
  Total value created when one unit succeeds.
- `cost_per_failure_eur`
  Direct operational cost created by one failed unit.
- `failure_to_churn_rel`
  Relative value penalty that follows when failures damage trust or retention.

## Option effectiveness

- `stabilize_failure_reduction`
  Share of baseline failure exposure removed by the core-stability option.
- `extension_uptake`
  Share of users who adopt the express path.
- `extension_exposure_reduction`
  Failure-exposure reduction for adopters of the express path.
- `extension_value_per_uptake_eur`
  Additional value created per adopting unit before leakage.
- `extension_loss_rate`
  Share of extension value lost through decay or disablement.
- `new_capability_uplift`
  Relative value uplift created by the new capability on successful volume.

## Release risk

- `regression_event_prob`
  Baseline probability that a shipped option triggers a regression event.
- `regression_event_cost_eur`
  Heavy-tailed severity of that event.
- `*_regression_prob_multiplier`
  Option-specific multiplier applied to the baseline release-event probability.

## Costs and timing

- `*_upfront_cost_eur`
  One-time investment cost for each option.
- `*_annual_maintenance_cost_eur`
  Annual ownership cost for each option after launch.
- `do_nothing_drift_cost_eur`
  Annual cost of carrying the current path without stabilization.
- `*_launch_delay_months`
  Months before the option starts creating production value.
- `*_benefit_ramp_months`
  Months needed to move from launch to full value realization.
- `*_cost_overrun_multiplier`
  Delivery-cost multiplier applied to the option's upfront cost.
