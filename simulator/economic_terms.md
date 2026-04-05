# Economic Terms

This page defines the monetized terms in the model so the value stack can be challenged without guessing what each bucket contains.

## Direct unit economics

- `value_per_success_eur`
  Value created when one unit succeeds. This is the base value pool that can be recovered, retained, or uplifted.
- `cost_per_failure_eur`
  Direct operational cost of one failed unit. This covers support, retries, and manual recovery tied to that failure.

## Customer-harm term

- `failure_to_churn_rel`
  Downstream value loss caused by the affected failure itself. It stands for trust damage, delayed conversion, or retention loss on the exposed unit.
  It does not include internal engineering time, on-call load, or recurring platform drag.

## Internal-drag term

- `do_nothing_drift_cost_eur`
  Ongoing internal operating drag from keeping the legacy path in place. It covers firefighting, coordination overhead, and workaround cost that accrue even outside a specific failed unit.
  It does not include customer-side revenue loss. That is why it can sit beside `failure_to_churn_rel` without double counting the same harm.

## Option-specific value terms

- `extension_value_per_uptake_eur`
  Extra adopter-side value created by the feature extension. This is additive convenience value, not recovered reliability value.
- `new_capability_uplift`
  Relative uplift on successful volume from the new capability. This is upside created after a success, not failure avoided.

## Worked example

If one checkout fails:

- `cost_per_failure_eur` captures the direct recovery cost of that failed checkout.
- `failure_to_churn_rel * value_per_success_eur` captures the customer-side value loss if that failure damages trust for the affected user.
- `do_nothing_drift_cost_eur` does nothing in that single-event calculation. It is the background operating drag of living with the unstable platform over time.

That is the intended split throughout the simulator and formula appendix.
