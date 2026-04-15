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

## Validation seam

- One public proxy dataset currently feeds the evidence workflow for `baseline_failure_rate`.
- That evidence produces a candidate metric in [artifacts/evidence/parameter_candidates.json](artifacts/evidence/parameter_candidates.json).
- It does not automatically rewrite the model config yet because the checked-in source is still a proxy benchmark, not a like-for-like SaaS checkout dataset.

## Published recommendation

<!-- GENERATED:CASE_STUDY_RECOMMENDATION:START -->
- Recommendation: **Stabilize Core**.
- Policy: `guardrailed_expected_value`.
- Why it wins: No option clears both guardrails, so the policy falls back to expected value.
- Guardrail reality: no option passes both guardrails, so **Stabilize Core** wins on expected value.
- Best remaining excluded alternative: **Feature Extension**.
- Expected-value comparison: the selected option leads **Feature Extension** by €176,174.
- Published run: `20,000` worlds, seed `42`, annual volume `250,000`, horizon `2` years, discount rate `8%`, declared model version `5.0.0`.
<!-- GENERATED:CASE_STUDY_RECOMMENDATION:END -->

## Default scenario summary

<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:START -->
| Option | Expected Value | P05 | Median | P95 |
| --- | --- | --- | --- | --- |
| Stabilize Core | €373,233 | €-360,761 | €326,965 | €1,268,354 |
| Feature Extension | €197,060 | €-420,804 | €156,314 | €944,158 |
| Do Nothing | €-104,671 | €-145,374 | €-102,378 | €-69,716 |
| New Capability | €-991,538 | €-1,408,501 | €-1,001,211 | €-539,891 |
<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:END -->

## Regret and win-rate view

<!-- GENERATED:CASE_STUDY_REGRET:START -->
| Option | Win Rate | Mean Regret | P95 Regret |
| --- | --- | --- | --- |
| Stabilize Core | 58% | €131,457 | €613,541 |
| Feature Extension | 34% | €307,630 | €1,045,854 |
| Do Nothing | 8% | €609,361 | €1,417,443 |
| New Capability | 0% | €1,496,228 | €2,389,647 |
<!-- GENERATED:CASE_STUDY_REGRET:END -->

## Scenario comparison

<!-- GENERATED:CASE_STUDY_SCENARIOS:START -->
Scenario descriptions:
- `Mid-range pressure` (`mid_range_pressure`): Reliability pressure is meaningful, but the business is not yet in a crisis state.
- `Reliability crisis` (`reliability_crisis`): Reliability pain intensifies, drift gets costlier, and delivery gets harder.
- `Growth-friendly recovery` (`growth_friendly_recovery`): Economics strengthen, reliability pressure eases, and delivery conditions improve.

| Scenario | Selected Option | Option | Expected Value | P05 | Mean Regret | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| Mid-range pressure | Stabilize Core | Stabilize Core | €373,233 | €-360,761 | €131,457 | no |
| Mid-range pressure | Stabilize Core | Feature Extension | €197,060 | €-420,804 | €307,630 | no |
| Mid-range pressure | Stabilize Core | Do Nothing | €-104,671 | €-145,374 | €609,361 | no |
| Mid-range pressure | Stabilize Core | New Capability | €-991,538 | €-1,408,501 | €1,496,228 | no |
| Reliability crisis | Do Nothing | Stabilize Core | €207,664 | €-440,239 | €44,389 | no |
| Reliability crisis | Do Nothing | Do Nothing | €-153,927 | €-199,129 | €405,981 | yes |
| Reliability crisis | Do Nothing | Feature Extension | €-325,899 | €-701,657 | €577,952 | no |
| Reliability crisis | Do Nothing | New Capability | €-1,707,050 | €-2,102,293 | €1,959,103 | no |
| Growth-friendly recovery | Feature Extension | Feature Extension | €587,875 | €-11,597 | €126,761 | yes |
| Growth-friendly recovery | Feature Extension | Stabilize Core | €377,831 | €-315,644 | €336,804 | no |
| Growth-friendly recovery | Feature Extension | New Capability | €72,159 | €-480,130 | €642,476 | no |
| Growth-friendly recovery | Feature Extension | Do Nothing | €-80,035 | €-112,801 | €794,671 | no |
<!-- GENERATED:CASE_STUDY_SCENARIOS:END -->

## Guardrail eligibility

<!-- GENERATED:CASE_STUDY_ELIGIBILITY:START -->
- This is the policy-defining table for the current run.
- An option must pass both the downside floor and the regret cap to stay eligible.

| Option | Expected Value | P05 | Downside Slack | Mean Regret | Regret Slack | Eligible | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stabilize Core | €373,233 | €-360,761 | €-60,761 | €131,457 | €318,543 | no | fails the downside floor |
| Feature Extension | €197,060 | €-420,804 | €-120,804 | €307,630 | €142,370 | no | fails the downside floor |
| Do Nothing | €-104,671 | €-145,374 | €154,626 | €609,361 | €-159,361 | no | fails the regret cap |
| New Capability | €-991,538 | €-1,408,501 | €-1,108,501 | €1,496,228 | €-1,046,228 | no | fails both guardrails |
<!-- GENERATED:CASE_STUDY_ELIGIBILITY:END -->

## Selected-vs-runner-up payoff diagnostic

<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:START -->
- Selected option: **Stabilize Core**.
- Best excluded alternative: **Feature Extension**.
- Mean payoff delta: €176,174 (selected option leads the comparison option).
- P05 payoff delta: €-611,191.
- Win rate vs comparison: 63%.
- This section is descriptive. It ranks parameters by association with the selected-minus-comparison payoff delta inside the sampled worlds.

| Parameter | Unit | Delta rho | Sampled range | Interpretation |
| --- | --- | --- | --- | --- |
| extension_uptake | adopter_share | -0.68 | 0.101 to 0.499 | Descriptive rank association with the selected-minus-comparison payoff delta inside the sampled worlds. |
| extension_value_per_uptake_eur | eur_per_adopting_unit | -0.46 | 2.032 to 6.967 | Descriptive rank association with the selected-minus-comparison payoff delta inside the sampled worlds. |
| baseline_failure_rate | share_of_volume | +0.45 | 0.041 to 0.150 | Descriptive rank association with the selected-minus-comparison payoff delta inside the sampled worlds. |
<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:END -->

## Policy frontier

<!-- GENERATED:CASE_STUDY_FRONTIER:START -->
- The first table is the full-option frontier. It re-runs the whole policy and records the first threshold change that flips the recommendation.

| Threshold | Current value | Raw switching value | Display switching value | First switching option | Switch type | Direction | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |
| Regret cap | €450,000 | 609,360.6 | €609,360.6 | Do Nothing | exact match | more permissive | Full-option sweep: Stabilize Core switches to Do Nothing when the regret cap moves up to 609,360.57 EUR. |
| EV tolerance | €100,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |
- The second table is secondary context. It follows the main comparison option, which can be the policy runner-up or the best excluded alternative depending on the branch.

| Threshold | Current value | Comparison threshold | Status | Interpretation |
| --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | €-420,803.7 | runner-up threshold reached | Feature Extension becomes eligible on downside once the floor is relaxed to -420,803.72 EUR. |
| Regret cap | €450,000 | not needed | already non-binding | Feature Extension already passes the regret cap. |
| EV tolerance | €100,000 | not needed | not binding | EV tolerance is not the binding threshold because both options are not yet eligible. |

| Threshold | Tested range | Selection switched? | Switching option(s) |
| --- | --- | --- | --- |
| EV tolerance | €75,000 to €125,000 | no | none |
| Regret cap | €400,000 to €650,000 | yes | Do Nothing |
| Downside floor | €-350,000 to €-250,000 | no | none |
<!-- GENERATED:CASE_STUDY_FRONTIER:END -->

## Published-case stability

<!-- GENERATED:CASE_STUDY_STABILITY:START -->
- Stability runs: `24` published-case reruns across multiple seeds and world counts.
- Selected-option P05 range: €24,034.
- Comparison-option P05 range: €25,234.

Recommendation frequency:

| Option | Runs | Share |
| --- | --- | --- |
| Stabilize Core | 24 | 100% |

EV leader frequency:

| EV leader | Runs | Share |
| --- | --- | --- |
| Stabilize Core | 24 | 100% |
<!-- GENERATED:CASE_STUDY_STABILITY:END -->

## Material sensitivity

<!-- GENERATED:CASE_STUDY_SENSITIVITY:START -->
This section is the decision-support view. It uses partial rank correlation with bootstrap intervals. The descriptive Spearman output still exists in `artifacts/case_study/sensitivity.json` for quick inspection.

### Do Nothing
| Parameter | Partial rank corr | 95% CI |
| --- | --- | --- |
| do_nothing_drift_cost_eur | -1.00 | -1.00 to -1.00 |

### Feature Extension
No decision-support driver cleared the current materiality threshold of |partial rho| >= 0.10.

### New Capability
No decision-support driver cleared the current materiality threshold of |partial rho| >= 0.10.

### Stabilize Core
No decision-support driver cleared the current materiality threshold of |partial rho| >= 0.10.
<!-- GENERATED:CASE_STUDY_SENSITIVITY:END -->

## How to read the result

- `Feature Extension` still looks best on expected value in the base case, but it misses the downside floor badly enough to fall out of policy scope.
- `Do Nothing` is not a victory lap result. It survives because it is the only option that clears the current guardrails.
- The scenario table is a worldview exercise, not just an option-effectiveness toggle. It lets exogenous business conditions move too.
- The payoff-delta section is descriptive. The policy-frontier section is the one that answers what would need to change for the recommendation to flip.
- The robustness artifact now includes a dependency ablation that reruns the model with all configured correlations set to zero. Use it to check whether the copula changes the recommendation or mainly changes the downside shape.
