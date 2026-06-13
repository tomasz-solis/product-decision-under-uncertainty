# ADR-003: Shared Latent Draw for Regression Events

## Decision

Generate a single vector of uniform draws before any option is evaluated, and
have every option reuse that same vector to decide whether a release-regression
event occurs in each sampled world (comparing the shared draw against that
option's event probability, after its option-specific multiplier).

## Why

The four options are compared world by world: regret, win rate, and the
selected-minus-comparison payoff delta all depend on the per-world difference
between options. If each option drew its own independent regression realisation,
two options facing the same uncertain world could diverge purely because of
unrelated RNG luck on the regression event. That would inject noise into exactly
the cross-option comparisons the decision rests on.

A shared latent draw means options that share a world either all face the
regression event (conditional on their own probability multiplier) or none do.
Differences between options then reflect their modelled economics, not the
sampler's bookkeeping. The draw is spawned from a dedicated child of the run's
`SeedSequence`, so it is reproducible and independent of the parameter draws.

## Limits

- The shared draw induces a common-shock structure across options. That is the
  intended behaviour for a like-for-like comparison, but it means the model does
  not represent the case where one option's regression is genuinely independent
  of another's.
- The option-specific multiplier is applied to the same underlying draw, so an
  option with a higher multiplier always experiences a superset of the events a
  lower-multiplier option sees in the same worlds. This monotone coupling is a
  modelling choice, not a measured property.

## Alternatives Considered

- **Independent per-option draws**: rejected because it adds RNG noise to the
  cross-option comparison without adding modelling realism.
- **Fully deterministic expected cost** (`p * severity`): rejected because it
  removes the tail of correlated bad-luck worlds that the guardrails are meant
  to catch. The Bernoulli sampling is verified to converge to `p * severity` in
  the mean by a property test.

## References

- `simulator/simulation.py` for the `shared_risk_draws` generation in
  `_run_simulation_from_config` and its use in `_sample_regression_cost`
- `tests/test_simulation_properties.py` for the convergence-to-`p*severity` test
- `METHODS.md` ("Shared latent draw for regression events")
