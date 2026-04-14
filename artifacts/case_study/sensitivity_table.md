This section is the decision-support view. It uses partial rank correlation with bootstrap intervals. The descriptive Spearman output still exists in `artifacts/case_study/sensitivity.json` for quick inspection.

### Do Nothing
| Parameter | Partial rank corr | 95% CI |
| --- | --- | --- |
| do_nothing_drift_cost_eur | +1.00 | +1.00 to +1.00 |

### Feature Extension
| Parameter | Partial rank corr | 95% CI |
| --- | --- | --- |
| extension_uptake | +0.94 | +0.93 to +0.94 |
| extension_value_per_uptake_eur | +0.70 | +0.69 to +0.71 |
| baseline_failure_rate | +0.61 | +0.60 to +0.62 |

### New Capability
| Parameter | Partial rank corr | 95% CI |
| --- | --- | --- |
| new_capability_uplift | +0.86 | +0.86 to +0.86 |
| new_capability_cost_overrun_multiplier | -0.76 | -0.77 to -0.76 |
| value_per_success_eur | +0.58 | +0.57 to +0.59 |

### Stabilize Core
| Parameter | Partial rank corr | 95% CI |
| --- | --- | --- |
| baseline_failure_rate | +0.86 | +0.85 to +0.86 |
| stabilize_failure_reduction | +0.85 | +0.85 to +0.86 |
| stabilize_core_launch_delay_months | -0.62 | -0.64 to -0.62 |