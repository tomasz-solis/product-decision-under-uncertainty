# Model Spec

This file is the prose companion to the formula appendix in [formulas.md](formulas.md).

## Shared structure

- The simulator samples one joint world of uncertain parameters.
- That world is evaluated under four mutually exclusive options.
- Cashflows are modeled on a monthly grid and discounted over a fixed 24-month horizon.

## Option mechanics

- `Do Nothing`
  Pays the drift cost across the whole horizon.
- `Stabilize Core`
  Carries launch delay, rollout ramp, and cost-overrun risk, then earns recovered value plus avoided failure and churn costs.
- `Feature Extension`
  Carries launch delay, adoption ramp, and cost-overrun risk, then earns adopter-only reliability gains plus extension-created value while legacy drift continues.
- `New Capability`
  Carries the longest delivery path in the base case, then earns uplift on successful volume while legacy drift continues.

## Robustness views

- Summary and regret metrics are descriptive outputs of the sampled worlds.
- Sensitivity uses per-option Spearman rank association.
- The selected-vs-runner-up payoff diagnostic is descriptive. It shows which sampled parameters move with that payoff gap.
- The full-option policy frontier is separate. It re-runs the policy across all options and records the first threshold change that flips the recommendation.
- The runner-up threshold table is secondary context only. It shows when the current runner-up clears its own blocking threshold, not the first full policy switch.
