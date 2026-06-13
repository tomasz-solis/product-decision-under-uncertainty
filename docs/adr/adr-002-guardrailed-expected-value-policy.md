# ADR-002: Guardrailed Expected-Value Decision Policy

## Decision

Encode the recommendation as a two-stage **guardrailed expected-value** policy
rather than picking the highest expected value outright. An option must clear a
downside floor (P05 >= threshold) and a regret cap (mean regret <= threshold) to
be eligible. Within the eligible set, highest expected value wins; within a
configurable EV tolerance band, the lower-regret option wins. If no option
clears both guardrails, the policy falls back to expected value and labels the
result as operating outside safe bounds.

## Why

Expected value alone treats a large negative tail and an equal-sized positive
tail as interchangeable, as long as the mean is unchanged. Most platform
investment decisions are not made by a risk-neutral actor: a P05 of -EUR 420k on
a 24-month bet is an outcome a leadership team has to live with, not a
statistical artefact to average away.

Splitting "is this option safe enough to consider?" (the guardrails) from "which
safe option is best?" (expected value within the eligible set) keeps both
questions explicit and configurable. Stakeholders can then argue against a
specific threshold instead of against the model as a whole, and the published
policy frontier makes the sensitivity of the recommendation to each threshold
legible.

## Limits

- The thresholds are elicited risk preferences, not derived from a utility
  function. They encode a leadership team's stated tolerance, not a proof.
- The fallback-to-EV branch is deliberately blunt: when nothing is safe it
  returns the least-bad option and relies on the narrative to flag that the
  decision is being made outside the policy's design envelope.
- Mean regret as the regret measure is one choice among several (max regret,
  CVaR of regret). Mean regret was chosen for interpretability.

## Alternatives Considered

- **Pure expected value**: rejected because it hides asymmetric downside, which
  is the whole reason this decision is hard.
- **Single utility function**: rejected because eliciting a defensible
  risk-aversion coefficient is harder to challenge in review than two explicit
  thresholds in business units.
- **Hard veto with no fallback**: rejected because returning "no recommendation"
  when all options fail the guardrails is less useful than returning the
  least-bad option with an explicit warning.

## References

- `simulator/policy.py` for `select_recommendation` and
  `build_policy_eligibility_table`
- `simulator/config.yaml` (`decision_policy` section) for the thresholds
- `NARRATIVE.md` for the stakeholder framing of the fallback case
- `METHODS.md` ("Guardrailed expected value") for the method summary
