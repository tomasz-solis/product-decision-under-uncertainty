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
- Best excluded alternative: **Feature Extension** has the strongest excluded EV case, but it misses the downside floor by about €244,675.
- Expected-value comparison: the selected option trails **Feature Extension** by €39,771.
- Published run: `20,000` worlds, seed `42`, annual volume `250,000`, horizon `2` years, discount rate `8%`, declared model version `4.0.0`.
<!-- GENERATED:CASE_STUDY_RECOMMENDATION:END -->

## Default scenario summary

<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:START -->
| Option | Expected Value | P05 | Median | P95 |
| --- | --- | --- | --- | --- |
| Feature Extension | €-64,899 | €-544,675 | €-100,730 | €534,888 |
| Do Nothing | €-104,671 | €-145,374 | €-102,378 | €-69,716 |
| Stabilize Core | €-218,673 | €-691,700 | €-261,208 | €403,660 |
| New Capability | €-954,763 | €-1,376,447 | €-965,188 | €-490,072 |
<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:END -->

## Regret and win-rate view

<!-- GENERATED:CASE_STUDY_REGRET:START -->
| Option | Win Rate | Mean Regret | P95 Regret |
| --- | --- | --- | --- |
| Feature Extension | 42% | €157,845 | €545,646 |
| Do Nothing | 38% | €197,616 | €714,530 |
| Stabilize Core | 20% | €311,619 | €777,772 |
| New Capability | 0% | €1,047,708 | €1,677,387 |
<!-- GENERATED:CASE_STUDY_REGRET:END -->

## Scenario comparison

<!-- GENERATED:CASE_STUDY_SCENARIOS:START -->
Scenario descriptions:
- `Mid-range pressure` (`mid_range_pressure`): Reliability pressure is meaningful, but the business is not yet in a crisis state.
- `Reliability crisis` (`reliability_crisis`): Reliability pain intensifies, drift gets costlier, and delivery gets harder.
- `Growth-friendly recovery` (`growth_friendly_recovery`): Economics strengthen, reliability pressure eases, and delivery conditions improve.

| Scenario | Selected Option | Option | Expected Value | P05 | Mean Regret | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| Mid-range pressure | Do Nothing | Feature Extension | €-64,899 | €-544,675 | €157,845 | no |
| Mid-range pressure | Do Nothing | Do Nothing | €-104,671 | €-145,374 | €197,616 | yes |
| Mid-range pressure | Do Nothing | Stabilize Core | €-218,673 | €-691,700 | €311,619 | no |
| Mid-range pressure | Do Nothing | New Capability | €-954,763 | €-1,376,447 | €1,047,708 | no |
| Reliability crisis | Do Nothing | Do Nothing | €-153,927 | €-199,129 | €52,897 | yes |
| Reliability crisis | Do Nothing | Stabilize Core | €-357,867 | €-808,713 | €256,836 | no |
| Reliability crisis | Do Nothing | Feature Extension | €-500,764 | €-812,667 | €399,733 | no |
| Reliability crisis | Do Nothing | New Capability | €-1,681,957 | €-2,073,900 | €1,580,926 | no |
| Growth-friendly recovery | Feature Extension | Feature Extension | €392,528 | €-139,294 | €117,817 | yes |
| Growth-friendly recovery | Feature Extension | New Capability | €103,807 | €-460,528 | €406,539 | no |
| Growth-friendly recovery | Feature Extension | Stabilize Core | €4,022 | €-555,513 | €506,324 | no |
| Growth-friendly recovery | Feature Extension | Do Nothing | €-80,035 | €-112,801 | €590,381 | no |
<!-- GENERATED:CASE_STUDY_SCENARIOS:END -->

## Guardrail eligibility

<!-- GENERATED:CASE_STUDY_ELIGIBILITY:START -->
- This is the policy-defining table for the current run.
- An option must pass both the downside floor and the regret cap to stay eligible.

| Option | Expected Value | P05 | Downside Slack | Mean Regret | Regret Slack | Eligible | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feature Extension | €-64,899 | €-544,675 | €-244,675 | €157,845 | €292,155 | no | fails the downside floor |
| Do Nothing | €-104,671 | €-145,374 | €154,626 | €197,616 | €252,384 | yes | passes both guardrails |
| Stabilize Core | €-218,673 | €-691,700 | €-391,700 | €311,619 | €138,381 | no | fails the downside floor |
| New Capability | €-954,763 | €-1,376,447 | €-1,076,447 | €1,047,708 | €-597,708 | no | fails both guardrails |
<!-- GENERATED:CASE_STUDY_ELIGIBILITY:END -->

## Selected-vs-runner-up payoff diagnostic

<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:START -->
- Selected option: **Do Nothing**.
- Runner-up: **Feature Extension**.
- Mean payoff delta: €-39,771 (selected option trails the runner-up).
- P05 payoff delta: €-641,333.
- Win rate vs runner-up: 49%.
- This section is descriptive. It ranks parameters by association with the selected-minus-runner-up payoff delta inside the sampled worlds.

| Parameter | Unit | Delta rho | Sampled range | Interpretation |
| --- | --- | --- | --- | --- |
| extension_uptake | adopter_share | -0.82 | 0.101 to 0.499 | Descriptive rank association with the selected-minus-runner-up payoff delta inside the sampled worlds. |
| extension_value_per_uptake_eur | eur_per_adopting_unit | -0.61 | 2.010 to 6.979 | Descriptive rank association with the selected-minus-runner-up payoff delta inside the sampled worlds. |
| baseline_failure_rate | share_of_volume | -0.36 | 0.020 to 0.099 | Descriptive rank association with the selected-minus-runner-up payoff delta inside the sampled worlds. |
<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:END -->

## Policy frontier

<!-- GENERATED:CASE_STUDY_FRONTIER:START -->
- The first table is the full-option frontier. It re-runs the whole policy and records the first threshold change that flips the recommendation. That is the actual decision question.
- The second table is secondary. It shows when the current runner-up clears its own blocking threshold, which is useful context but not the same as the first switch.

| Threshold | Current value | Raw switching value | Display switching value | First switching option | Switch type | Direction | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | -145,373.6 | €-145,373.6 | Feature Extension | grid bracket | more restrictive | Full-option sweep: Do Nothing switches to Feature Extension once the downside floor moves up past -145,373.57 EUR. |
| Regret cap | €450,000 | 197,615.8 | €197,615.8 | Feature Extension | grid bracket | more restrictive | Full-option sweep: Do Nothing switches to Feature Extension once the regret cap moves down past 197,615.84 EUR. |
| EV tolerance | €100,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |

| Threshold | Current value | Runner-up threshold | Status | Interpretation |
| --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | €-544,674.9 | runner-up threshold reached | Feature Extension becomes eligible on downside once the floor is relaxed to -544,674.89 EUR. |
| Regret cap | €450,000 | not needed | already non-binding | Feature Extension already passes the regret cap. |
| EV tolerance | €100,000 | not needed | not binding | EV tolerance is not the binding threshold because both options are not yet eligible. |

| Threshold | Tested range | Selection switched? | Switching option(s) |
| --- | --- | --- | --- |
| EV tolerance | €75,000 to €125,000 | no | none |
| Regret cap | €147,615.8 to €497,615.8 | yes | Feature Extension |
| Downside floor | €-350,000 to €-100,000 | yes | Feature Extension |
<!-- GENERATED:CASE_STUDY_FRONTIER:END -->

## Published-case stability

<!-- GENERATED:CASE_STUDY_STABILITY:START -->
- Stability runs: `15` published-case reruns across multiple seeds and world counts.
- Selected-option P05 range: €2,554.
- Runner-up P05 range: €14,585.

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
| extension_value_per_uptake_eur | +0.61 |
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
| cost_per_failure_eur | +0.60 |
| failure_to_churn_rel | +0.46 |
<!-- GENERATED:CASE_STUDY_SENSITIVITY:END -->

## How to read the result

- `Feature Extension` still looks best on expected value in the base case, but it misses the downside floor badly enough to fall out of policy scope.
- `Do Nothing` is not a victory lap result. It survives because it is the only option that clears the current guardrails.
- The scenario table is a worldview exercise, not just an option-effectiveness toggle. It lets exogenous business conditions move too.
- The payoff-delta section is descriptive. The policy-frontier section is the one that answers what would need to change for the recommendation to flip.
