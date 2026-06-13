# ADR-005: Value of Information (EVPI / EVPPI)

## Decision

Publish a value-of-information view alongside the recommendation: EVPI (the
expected gain from resolving every uncertainty before deciding) and a
per-parameter EVPPI ranking (the expected gain from resolving one parameter on
its own). EVPPI is estimated nonparametrically by partitioning the sampled
worlds into equal-count quantile bins of each parameter and taking the best
option by conditional mean within each bin.

## Why

The narrative already argued that the most useful thing a sensitivity analysis
can produce is guidance on *what to measure next* - for example, getting better
telemetry on do-nothing drift cost before committing. That advice was prose.
EVPI/EVPPI makes it a euro figure, which is exactly the form a stakeholder needs
to prioritise a measurement or discovery spend.

EVPI also has a clean interpretation in this model: it equals the mean regret of
the expected-value-optimal option, so it places a hard ceiling on what any data
collection could be worth. That ties the new view directly to the regret
machinery the decision policy already relies on.

A binned estimator was chosen over a regression- or Gaussian-process-based
EVPPI for the same reason the rest of the project avoids heavy machinery: it is
transparent, has no extra dependency, is deterministic for a fixed sample, and
is bounded in `[0, EVPI]` by construction so it cannot produce a nonsensical
result.

## Limits

- The binned EVPPI estimator is a coarse approximation for continuous
  parameters. Bin count trades bias (too few bins) against variance (too many).
- EVPPI values are not additive across parameters: resolving one parameter
  changes the information value of the others.
- The view is computed against the expected-value-optimal action, which is the
  risk-neutral baseline, not the guardrailed policy pick. It answers "what is
  the uncertainty worth?", not "what does the policy choose?".

## Alternatives Considered

- **Regression / Gaussian-process EVPPI** (Strong & Oakley): more accurate for
  continuous inputs but adds dependencies and opacity that are not justified for
  a transparent case study.
- **Reporting EVPI only**: rejected because the single EVPI number does not tell
  a stakeholder *which* uncertainty to resolve, which is the actionable part.
- **Leaving the guidance as prose**: rejected because an unquantified "measure
  this first" is weaker than a ranked euro figure a budget owner can act on.

## References

- `simulator/analytics.py` for `value_of_information` and `_evppi_for_parameter`
- `simulator/reporting.py` for `build_value_of_information_markdown`
- `tests/test_value_of_information.py` for the bounding and identity invariants
- `METHODS.md` ("Value of information (EVPI and EVPPI)")
- `NARRATIVE.md` ("Use the stress tests to decide what evidence to gather next")
