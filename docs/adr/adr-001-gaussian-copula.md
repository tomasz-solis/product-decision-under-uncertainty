# ADR-001: Gaussian Copula for Parameter Dependencies

## Decision

Use a Gaussian copula with Spearman-to-Pearson conversion to model dependency
between uncertain parameters, rather than assuming full independence or forcing
the model into one joint parametric distribution.

## Why

Full independence is the weaker assumption here. In worse reliability worlds,
failure rate, failure cost, and failure-linked churn are likely to move
together. Ignoring that relationship would make downside scenarios look cleaner
than they really are.

A single joint parametric distribution would ask for more tail-detail than this
case can support. It is harder to elicit from domain experts and harder to
defend without a comparable historical dataset.

The Gaussian copula keeps each marginal distribution in its natural form
(`tri`, `uniform`, `lognormal`, or `constant`) while still letting the model
represent a small dependency map that a reviewer can read directly from config.

## Limits

The Gaussian copula has symmetric tails. That means it can understate
simultaneous extreme moves compared with heavier-tailed alternatives such as a
t-copula. For this project, that is an acceptable simplification because the
goal is a transparent portfolio-style case study, not a production risk engine.

## Alternatives Considered

- Full independence: rejected because it understates correlated downside.
- Empirical bootstrapping: rejected because there is not yet a comparable joint
  historical dataset for the modeled parameter set.
- t-copula: deferred because the extra tail realism is not yet matched by
  equally defensible elicitation inputs.

## References

- `simulator/config.yaml` for the declared dependency map
- `simulator/simulation.py` for `_sample_dependent_uniforms()`
- `simulator/assumption_registry.yaml` for the documented assumption layer
