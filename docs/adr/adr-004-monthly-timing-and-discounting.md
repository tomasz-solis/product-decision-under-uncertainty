# ADR-004: Monthly Timing Grid with Launch Delay, Ramp, and Discounting

## Decision

Model benefits and costs on a monthly grid over the fixed horizon rather than
applying a single discount factor to a full-horizon aggregate. Each option
carries a launch delay, a linear benefit ramp, and a cost-overrun multiplier.
The monthly active indicator, ramp weight, and discount weight combine into
discounted benefit-years, active-years, and residual-drift-years per sampled
world.

## Why

The options differ in how quickly they deliver value. A full-horizon
simplification would credit a slow, late-launching option with the same
discounted benefit as a fast one, which is wrong: delayed options should earn
less present value, and drift should keep accruing while the work is still in
flight.

The monthly grid makes three effects first-class:

- **Launch delay** postpones the first euro of benefit, so later options earn
  fewer discounted benefit-years.
- **Benefit ramp** phases value in linearly from launch to full effect rather
  than switching on instantly.
- **Residual drift** continues to cost money for the months an option has not
  yet delivered, so the do-nothing baseline is charged correctly against each
  intervention.

The cost-overrun multiplier hits the upfront cost in the correct relationship to
benefit timing, so a cost blow-out and a slow ramp compound the way they would
in practice.

## Limits

- The ramp is linear. Real adoption and reliability curves are often S-shaped;
  the linear ramp is a deliberate, legible simplification.
- Timing parameters (delay, ramp length) are elicited triangular ranges, not
  fitted to delivery data.
- Monthly granularity is a modelling grain choice. Finer (weekly) granularity
  would change nothing material at this horizon and would cost clarity.

## Alternatives Considered

- **Single full-horizon discount factor**: rejected because it erases the
  delivery-speed differences that distinguish the options.
- **Continuous-time discounting**: rejected as unnecessary precision; the
  monthly grid is accurate enough at a 24-month horizon and far easier to read.

## References

- `simulator/simulation.py` for `_option_timing_summary`,
  `_monthly_timing_profiles`, `_discount_weights`, and
  `_full_horizon_discounted_years`
- `simulator/formulas.md` for the equation-level detail
- `tests/test_simulation_properties.py` for the discount-rate monotonicity test
- `METHODS.md` ("Monthly discounting grid with launch delay and ramp")
