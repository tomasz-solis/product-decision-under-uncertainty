# Methodology

This workflow is narrow on purpose. It follows one decision from assumptions to recommendation and keeps each step visible enough to challenge.

## 1. Define the decision and the horizon

- One decision question
- Four mutually exclusive options
- One fixed 24-month horizon
- Discounted monthly cashflows rather than a one-shot full-horizon payout

The published case models incremental net value in euros over that horizon.

## 2. Register the assumptions before running the model

Every modeled input has provenance attached to it.

- Parameters live in [simulator/parameter_registry.yaml](simulator/parameter_registry.yaml)
- Policy, dependency, scenario, and reporting thresholds live in [simulator/assumption_registry.yaml](simulator/assumption_registry.yaml)
- The generated output exports a unified `assumption_manifest.json` so the source of truth is machine-readable

## 3. Sample uncertain worlds

The model keeps marginal distributions in config, then applies a small Gaussian-copula dependency layer to the assumptions that are most likely to move together:

- worse reliability worlds also raise failure cost and failure-linked churn
- stronger extension uptake tends to come with stronger extension value
- higher release-event probability tends to come with higher release severity

The dependency map is small on purpose. It is there to avoid the weaker assumption that every uncertain variable moves independently.

## 4. Model timing, delivery, and value realization

Each intervention option now has:

- launch delay
- benefit ramp
- delivery-cost overrun uncertainty

That means the options do not receive full-horizon benefits from month one. Drift continues while work is still in flight, and delayed options earn less discounted value.

Equation-level detail lives in [simulator/formulas.md](simulator/formulas.md).

## 5. Apply the recommendation policy

The final recommendation is encoded in [simulator/config.yaml](simulator/config.yaml). The current policy works like this:

- keep options that clear a downside floor and a mean-regret cap
- within that feasible set, maximize expected value
- if the EV gap is inside a configured tolerance band, prefer the lower-regret option

The published artifacts now split two different questions:

- a descriptive selected-vs-runner-up payoff diagnostic
- a full-option policy frontier showing which threshold change flips the recommendation first

## 6. Publish descriptive robustness views carefully

The sensitivity and payoff-delta views are descriptive, not causal.

- Sensitivity is rank association inside the sampled joint distribution.
- Payoff-delta rows show which sampled parameters move with the selected-minus-runner-up gap.
- Policy-frontier rows, separately, re-run the full policy across all options and record the first threshold change that flips selection.

That distinction matters most in the current published case, because the selected option is not the EV leader.

## 7. Generate artifacts

The generator script writes JSON, CSV, and markdown fragments into [artifacts/case_study](artifacts/case_study). Metadata fingerprints the config, registries, code path, generator script, and lockfile so the app can tell when published governance artifacts are stale.

## 8. Prepare the public-evidence seam honestly

There is still no real public calibration slice in the published case. What is ready now is the evidence-to-parameter contract:

- raw files belong in [data/public/README.md](data/public/README.md)
- each raw file needs a manifest entry in [data/public/sources.yaml](data/public/sources.yaml)
- profiling starts in [scripts/profile_public_evidence.py](scripts/profile_public_evidence.py)
- calibration targets live in [simulator/calibration_contract.yaml](simulator/calibration_contract.yaml)
- candidate metrics are emitted by [scripts/build_parameter_candidates.py](scripts/build_parameter_candidates.py)
- derived outputs live in [artifacts/evidence/public_data_profile.md](artifacts/evidence/public_data_profile.md) and [artifacts/evidence/parameter_candidates.md](artifacts/evidence/parameter_candidates.md)

Worked example:

- If one failure causes direct support cost, that sits in `cost_per_failure_eur`.
- If that same failure damages customer trust for the affected unit, that sits in `failure_to_churn_rel * value_per_success_eur`.
- If the team keeps living with a brittle platform and burns engineering time every month, that sits in `do_nothing_drift_cost_eur`.

That keeps the workflow concrete without inventing evidence that is not there.
