# Case Study: Platform Reliability Investment

This is the published case-study view of the current model. For the repo map and evidence workflow, start with [README.md](README.md).

## Decision frame

- Decision: where to place one major platform investment over the next 24 months
- Context: checkout reliability is under pressure, the product still needs forward motion, and only one primary path can be funded
- Outcome definition: discounted 24-month net value in euros

## Options

1. `Do Nothing`
   Keep the current platform path and absorb ongoing drift.
2. `Stabilize Core`
   Refactor the legacy checkout flow to remove failure exposure.
3. `Feature Extension`
   Add an optional express path that helps only for adopters.
4. `New Capability`
   Build a new fraud-prevention capability that adds upside while leaving the legacy drift in place.

## Analytical scaffolding

- Assumption provenance: [simulator/parameter_provenance.md](simulator/parameter_provenance.md)
- Formula appendix: [simulator/formulas.md](simulator/formulas.md)
- Config and policy: [simulator/config.yaml](simulator/config.yaml)

## Published recommendation

<!-- GENERATED:CASE_STUDY_RECOMMENDATION:START -->
- Recommendation: **Do Nothing**.
- Policy: `guardrailed_expected_value`.
- Why it wins: Do Nothing is the only option that passes both guardrails.
- Best excluded alternative: **Feature Extension** has the strongest excluded EV case, but it misses the downside floor by about €247,304.
- Expected-value comparison: the selected option trails **Feature Extension** by €44,651.
- Published run: `20,000` worlds, seed `42`, annual volume `250,000`, horizon `2` years, discount rate `8%`, declared model version `4.0.0`.
<!-- GENERATED:CASE_STUDY_RECOMMENDATION:END -->

## Default scenario summary

<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:START -->
| Option | Expected Value | P05 | Median | P95 |
| --- | --- | --- | --- | --- |
| Feature Extension | €-60,019 | €-547,304 | €-91,349 | €530,971 |
| Do Nothing | €-104,671 | €-145,374 | €-102,378 | €-69,716 |
| Stabilize Core | €-207,577 | €-693,811 | €-249,949 | €431,055 |
| New Capability | €-956,398 | €-1,382,691 | €-967,359 | €-492,711 |
<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:END -->

## Regret and win-rate view

<!-- GENERATED:CASE_STUDY_REGRET:START -->
| Option | Win Rate | Mean Regret | P95 Regret |
| --- | --- | --- | --- |
| Feature Extension | 42% | €159,584 | €548,931 |
| Do Nothing | 37% | €204,236 | €714,254 |
| Stabilize Core | 21% | €307,142 | €778,389 |
| New Capability | 0% | €1,055,963 | €1,693,140 |
<!-- GENERATED:CASE_STUDY_REGRET:END -->

## Scenario comparison

<!-- GENERATED:CASE_STUDY_SCENARIOS:START -->
Scenario descriptions:
- `Mid-range pressure` (`mid_range_pressure`): Reliability pressure is meaningful, but the business is not yet in a crisis state.
- `Reliability crisis` (`reliability_crisis`): Reliability pain intensifies, drift gets costlier, and delivery gets harder.
- `Growth-friendly recovery` (`growth_friendly_recovery`): Economics strengthen, reliability pressure eases, and delivery conditions improve.

| Scenario | Selected Option | Option | Expected Value | P05 | Mean Regret | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| Mid-range pressure | Do Nothing | Feature Extension | €-60,019 | €-547,304 | €159,584 | no |
| Mid-range pressure | Do Nothing | Do Nothing | €-104,671 | €-145,374 | €204,236 | yes |
| Mid-range pressure | Do Nothing | Stabilize Core | €-207,577 | €-693,811 | €307,142 | no |
| Mid-range pressure | Do Nothing | New Capability | €-956,398 | €-1,382,691 | €1,055,963 | no |
| Reliability crisis | Do Nothing | Do Nothing | €-153,927 | €-199,129 | €56,948 | yes |
| Reliability crisis | Do Nothing | Stabilize Core | €-348,817 | €-811,564 | €251,838 | no |
| Reliability crisis | Do Nothing | Feature Extension | €-497,904 | €-813,561 | €400,925 | no |
| Reliability crisis | Do Nothing | New Capability | €-1,683,258 | €-2,082,311 | €1,586,279 | no |
| Growth-friendly recovery | Feature Extension | Feature Extension | €399,332 | €-138,798 | €117,811 | yes |
| Growth-friendly recovery | Feature Extension | New Capability | €102,449 | €-458,526 | €414,694 | no |
| Growth-friendly recovery | Feature Extension | Stabilize Core | €17,164 | €-551,826 | €499,979 | no |
| Growth-friendly recovery | Feature Extension | Do Nothing | €-80,035 | €-112,801 | €597,178 | no |
<!-- GENERATED:CASE_STUDY_SCENARIOS:END -->

## Guardrail eligibility

<!-- GENERATED:CASE_STUDY_ELIGIBILITY:START -->
- This is the policy-defining table for the current run.
- An option must pass both the downside floor and the regret cap to stay eligible.

| Option | Expected Value | P05 | Downside Slack | Mean Regret | Regret Slack | Eligible | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feature Extension | €-60,019 | €-547,304 | €-247,304 | €159,584 | €290,416 | no | fails the downside floor |
| Do Nothing | €-104,671 | €-145,374 | €154,626 | €204,236 | €245,764 | yes | passes both guardrails |
| Stabilize Core | €-207,577 | €-693,811 | €-393,811 | €307,142 | €142,858 | no | fails the downside floor |
| New Capability | €-956,398 | €-1,382,691 | €-1,082,691 | €1,055,963 | €-605,963 | no | fails both guardrails |
<!-- GENERATED:CASE_STUDY_ELIGIBILITY:END -->

## Selected-vs-runner-up payoff diagnostic

<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:START -->
- Selected option: **Do Nothing**.
- Runner-up: **Feature Extension**.
- Mean payoff delta: €-44,651 (selected option trails the runner-up).
- P05 payoff delta: €-635,230.
- Win rate vs runner-up: 48%.
- This section is descriptive. It ranks parameters by association with the selected-minus-runner-up payoff delta inside the sampled worlds.

| Parameter | Unit | Delta rho | Sampled range | Interpretation |
| --- | --- | --- | --- | --- |
| extension_uptake | adopter_share | -0.82 | 0.101 to 0.499 | Descriptive rank association with the selected-minus-runner-up payoff delta inside the sampled worlds. |
| extension_value_per_uptake_eur | eur_per_adopting_unit | -0.61 | 2.032 to 6.967 | Descriptive rank association with the selected-minus-runner-up payoff delta inside the sampled worlds. |
| baseline_failure_rate | share_of_volume | -0.36 | 0.021 to 0.100 | Descriptive rank association with the selected-minus-runner-up payoff delta inside the sampled worlds. |
<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:END -->

## Policy frontier

<!-- GENERATED:CASE_STUDY_FRONTIER:START -->
- The first table is the full-option frontier. It re-runs the whole policy and records the first threshold change that flips the recommendation. That is the actual decision question.
- The second table is secondary. It shows when the current runner-up clears its own blocking threshold, which is useful context but not the same as the first switch.

| Threshold | Current value | Raw switching value | Display switching value | First switching option | Switch type | Direction | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | -145,373.6 | €-145,373.6 | Feature Extension | grid bracket | more restrictive | Full-option sweep: Do Nothing switches to Feature Extension once the downside floor moves up past -145,373.57 EUR. |
| Regret cap | €450,000 | 204,235.7 | €204,235.7 | Feature Extension | grid bracket | more restrictive | Full-option sweep: Do Nothing switches to Feature Extension once the regret cap moves down past 204,235.67 EUR. |
| EV tolerance | €100,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |

| Threshold | Current value | Runner-up threshold | Status | Interpretation |
| --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | €-547,304.0 | runner-up threshold reached | Feature Extension becomes eligible on downside once the floor is relaxed to -547,304.01 EUR. |
| Regret cap | €450,000 | not needed | already non-binding | Feature Extension already passes the regret cap. |
| EV tolerance | €100,000 | not needed | not binding | EV tolerance is not the binding threshold because both options are not yet eligible. |

| Threshold | Tested range | Selection switched? | Switching option(s) |
| --- | --- | --- | --- |
| EV tolerance | €75,000 to €125,000 | no | none |
| Regret cap | €154,235.7 to €504,235.7 | yes | Feature Extension |
| Downside floor | €-350,000 to €-100,000 | yes | Feature Extension |
<!-- GENERATED:CASE_STUDY_FRONTIER:END -->

## Published-case stability

<!-- GENERATED:CASE_STUDY_STABILITY:START -->
- Stability runs: `15` published-case reruns across multiple seeds and world counts.
- Selected-option P05 range: €2,554.
- Runner-up P05 range: €23,946.

Recommendation frequency:

| Option | Runs | Share |
| --- | --- | --- |
| Do Nothing | 15 | 100% |

EV leader frequency:

| EV leader | Runs | Share |
| --- | --- | --- |
| Feature Extension | 15 | 100% |
<!-- GENERATED:CASE_STUDY_STABILITY:END -->

## Material sensitivity

<!-- GENERATED:CASE_STUDY_SENSITIVITY:START -->
### Do Nothing
| Parameter | Spearman |
| --- | --- |
| do_nothing_drift_cost_eur | -1.00 |
Only `do_nothing_drift_cost_eur` cleared the materiality threshold of |rho| >= 0.10.

### Feature Extension
| Parameter | Spearman |
| --- | --- |
| extension_uptake | +0.82 |
| extension_value_per_uptake_eur | +0.60 |
| baseline_failure_rate | +0.36 |

### New Capability
| Parameter | Spearman |
| --- | --- |
| new_capability_uplift | +0.68 |
| new_capability_cost_overrun_multiplier | -0.46 |
| value_per_success_eur | +0.29 |

### Stabilize Core
| Parameter | Spearman |
| --- | --- |
| baseline_failure_rate | +0.85 |
| cost_per_failure_eur | +0.61 |
| failure_to_churn_rel | +0.46 |
<!-- GENERATED:CASE_STUDY_SENSITIVITY:END -->

## How to read the result

- `Feature Extension` still looks best on expected value in the base case, but it misses the downside floor badly enough to fall out of policy scope.
- `Do Nothing` is not a victory lap result. It survives because it is the only option that clears the current guardrails.
- The scenario table is a worldview exercise, not just an option-effectiveness toggle. It lets exogenous business conditions move too.
- The payoff-delta section is descriptive. The policy-frontier section is the one that answers what would need to change for the recommendation to flip.
