# Methods

This file catalogues the statistical and decision-analysis techniques used
in this project. It exists so reviewers can see the methodological choices
without reading code, and so the choices can be evaluated on their own terms.

Each entry says what the technique does, what it buys the project, where it
lives in the code, and where the limits are.

---

## Joint-world sampling

### Gaussian copula with Spearman-to-Pearson conversion

**What it does**: samples a joint distribution over uncertain parameters while
keeping each marginal distribution in its natural form (`tri`, `uniform`,
`lognormal`, or `constant`).

**What it buys**: in worse reliability worlds, failure rate, failure cost, and
failure-linked churn are likely to move together. Full independence understates
this. A single joint parametric distribution would demand more tail-detail
than the assumptions can support. The copula keeps each marginal elicitable
independently while still representing a small dependency map that a reviewer
can read directly from config.

**Where it lives**:
- `simulator/simulation.py::_sample_dependent_uniforms` — the copula sampling step
- `simulator/config.yaml` (`dependencies.rank_correlations`) — the declared dependency map
- `docs/adr/adr-001-gaussian-copula.md` — the design decision and its limits

**Limits**: Gaussian copulas have symmetric tails. Simultaneous extreme moves
are understated relative to heavier-tailed alternatives (e.g. a t-copula).
Acceptable for a case study; not for a production tail-risk engine. The ADR
documents this explicitly.

---

### Spearman-to-Pearson conversion (`2·sin(π·ρ/6)`)

**What it does**: converts a requested Spearman rank correlation into the
corresponding Pearson correlation for the Gaussian copula's normal space.

**What it buys**: rank correlations are more natural to elicit from domain
experts ("how often do these two things move together on a rank scale?") than
Pearson correlations. The conversion means the input is interpretable and the
copula math stays correct.

**Where it lives**: `simulator/simulation.py::_sample_dependent_uniforms`,
line that computes `gaussian_corr`.

---

### Nearest-correlation-matrix projection

**What it does**: if the requested correlation matrix is not positive
semi-definite (can happen with multiple elicited correlations that are
individually plausible but jointly inconsistent), projects it onto the nearest
valid correlation matrix via eigenvalue clipping.

**What it buys**: the sampler does not silently produce invalid results or
crash. A `RuntimeWarning` is emitted if the deviation exceeds 0.05, which
flags elicitation inconsistency rather than hiding it.

**Where it lives**: `simulator/simulation.py::_nearest_correlation_matrix`.

---

### Explicit Cholesky with per-marginal inverse CDF

**What it does**: the Cholesky decomposition of the correlation matrix is
computed once per run and used to transform independent standard normals into
correlated normals. Each correlated normal is then mapped to a uniform via the
normal CDF, and the uniform is mapped to the target marginal via an
analytically implemented inverse CDF (triangular, uniform, or lognormal).

**What it buys**: avoids depending on `scipy` for the copula. More importantly,
the explicit Cholesky + inverse-CDF chain makes the transform deterministic
and platform-stable for a fixed seed — the same 42-seed run produces
identical output on different machines.

**Where it lives**: `simulator/simulation.py::_sample_dependent_uniforms`,
`simulator/simulation.py::_inverse_cdf_param`.

---

### Shared latent draw for regression events

**What it does**: a single vector of uniform draws is generated before any
option is evaluated. Each option uses this same vector to determine whether
a regression event occurs (comparing the draw against that option's event
probability).

**What it buys**: options that face the same sampled world either all draw
a regression event or none do (conditional on their option-specific
probability multiplier). Without shared draws, options could silently benefit
from different RNG realisations of the same event, making the comparison
across options in one world less coherent.

**Where it lives**: `simulator/simulation.py::_run_simulation_from_config`
(where `shared_risk_draws` is generated), `simulator/simulation.py::_sample_regression_cost`.

---

## Timing and cashflow model

### Monthly discounting grid with launch delay and ramp

**What it does**: instead of applying a single discount factor to a
full-horizon aggregate, benefits are modelled on a monthly grid. Each month
has an active indicator (post-launch), a ramp weight (linear from 0 at launch
to 1 at full benefit), and a discount weight. These three profiles combine to
give discounted benefit-years, active-years, and residual-drift-years per
sampled world.

**What it buys**: options with longer launch delays and slower ramps earn
genuinely less discounted value than options with faster delivery. A
full-horizon simplification would miss this. It also means the cost-overrun
multiplier hits the upfront cost in the right relationship to the benefit
realisation timing.

**Where it lives**: `simulator/simulation.py::_option_timing_summary`,
`simulator/simulation.py::_monthly_timing_profiles`.

---

## Decision policy

### Guardrailed expected value

**What it does**: an option must pass a downside floor (P05 ≥ threshold) and
a regret cap (mean regret ≤ threshold) to be eligible. Within the eligible
set, highest EV wins. Within a configurable EV tolerance band, lowest mean
regret wins. If nothing clears both guardrails, the policy falls back to EV.

**What it buys**: separates the "is this option safe enough to consider?" gate
from the "which safe option is best?" choice. The thresholds are explicit and
configurable, so stakeholders can argue against the thresholds rather than the
model.

**Where it lives**: `simulator/policy.py::select_recommendation`,
`simulator/config.yaml` (`decision_policy` section).

---

### Policy frontier (full-option threshold sweep)

**What it does**: for each policy threshold (downside floor, regret cap, EV
tolerance), sweeps the value across a grid and reruns the full policy at each
point. Records the first threshold value that changes the selected option
across the entire option set.

**What it buys**: a practical answer to "how far would we have to be wrong
about this threshold for the recommendation to flip?" The frontier result is
more useful in a stakeholder review than the recommendation itself — it tells
you the sensitivity of the decision to the policy encoding.

**Where it lives**: `simulator/policy.py::policy_frontier_analysis`,
`simulator/policy.py::policy_frontier_grid`.

---

### Descriptive-vs-causal separation

**What it does**: the payoff-delta diagnostic (which parameters move with
selected-minus-runner-up payoff) and the policy-frontier analysis (which
threshold move flips the recommendation) are kept strictly separate in code,
documentation, and the published artifacts.

**What it buys**: avoids the common reporting error of presenting a
descriptive correlation as if it answers a causal question. A parameter that
correlates with the payoff gap is not the same as a threshold that controls
the recommendation. METHODOLOGY.md and CASE_STUDY.md both name this
distinction explicitly.

**Where it lives**: `simulator/policy.py::payoff_delta_diagnostic` vs.
`simulator/policy.py::policy_frontier_analysis`.

---

## Diagnostics and robustness

### Partial rank correlation with bootstrap CI

**What it does**: for each (option, parameter) pair, computes the partial rank
correlation of that parameter with the option payoff while controlling for all
other parameters. A bootstrap CI is computed across 40 resamples. Rows whose
CI crosses zero are dropped before the result is shown.

**What it buys**: avoids showing individual Spearman correlations that are
spurious due to inter-parameter correlation (which is real in this model, by
design). The bootstrap CI gate means only robustly material drivers appear in
the driver analysis and the recommendation markdown.

**Limits**: partial rank correlation is still a linear approximation in rank
space. Non-linear interactions between parameters are not captured.

**Where it lives**: `simulator/analytics.py::_partial_rank_correlations`,
`simulator/analytics.py::_bootstrap_partial_rank_correlations`,
`simulator/analytics.py::driver_analysis`.

---

### Independence ablation (dependency-model check)

**What it does**: reruns the full simulation with all configured correlations
set to zero (full independence) and compares recommendation, EV, and P05 to
the correlated run.

**What it buys**: a direct answer to "does the copula matter for this result?"
If the recommendation and key metrics are stable under independence, the
dependency model is not load-bearing. If they differ materially, the elicited
correlations are consequential and deserve scrutiny.

**Where it lives**: `simulator/analytics.py::independence_ablation`,
`simulator/robustness.py::build_robustness_report`.

---

### Dependency-value frontier

**What it does**: for each configured dependency pair, sweeps the rank
correlation across a grid (0.0, 0.2, 0.4, 0.6, 0.8) while holding others
fixed, reruns the full policy, and records whether the recommendation changes
at any value.

**What it buys**: independence ablation shows the correlations matter in
aggregate. The dependency-value frontier shows which individual correlations
drive the sensitivity and how far each would have to move before the
recommendation flips. This is the complement to independence ablation for
validating elicited values.

**Where it lives**: `simulator/robustness.py::_dependency_value_frontier`.

---

### Stability sweep (seed × world-count grid)

**What it does**: reruns the simulation across 6 seeds and 4 world counts
(5k, 10k, 20k, 40k), recording the recommendation, EV, and P05 for each.

**What it buys**: separates Monte Carlo noise (seed sensitivity) from
convergence (world-count sensitivity). A recommendation that is stable across
seeds at 5k worlds is different from one that converges to stability at 20k
worlds.

**Where it lives**: `simulator/analytics.py::stability_analysis`,
`simulator/robustness.py::_convergence_rows`.

---

### Materiality filter (bootstrap CI gate)

**What it does**: a sensitivity driver is only published in the report if its
absolute partial rank correlation clears a configured threshold (default 0.10)
and its bootstrap CI does not cross zero.

**What it buys**: avoids filling the driver analysis with small, noisy
associations. The published drivers are the ones that a model maintainer
would need to monitor if the recommendation were to change.

**Where it lives**: `simulator/output_utils.py::material_driver_rows`,
`simulator/config.yaml` (`analysis.sensitivity_materiality_threshold_abs_spearman`).

---

## Governance

### Assumption provenance registry

**What it does**: every modeled input — parameters, policy thresholds,
simulation settings, dependencies, scenarios — has a structured registry
entry with `source_type`, `source_reference`, `reason_for_range` (or choice),
`owner`, and `last_updated`. A machine-readable `assumption_manifest.json`
is generated and checked into the artifact directory.

**What it buys**: a reviewer can challenge a specific parameter rather than
the model as a whole. The registry makes the "why is this number here?" answer
findable without reading the code.

**Where it lives**: `simulator/parameter_registry.yaml`,
`simulator/assumption_registry.yaml`,
`simulator/provenance.py::build_assumption_manifest`.

---

### Calibration contract (evidence seam)

**What it does**: a YAML contract defines, for each evidence-backed target,
the expected raw data grain, required columns, transformation rule, quality
checks, and expected output artifact. An evidence profiler validates raw files
against the contract before candidate metrics are derived.

**What it buys**: the path from a raw public data file to a candidate
parameter value is explicit and machine-checkable, rather than implicit in a
notebook or a README paragraph. When real calibration data is available, the
seam is ready for it.

**Where it lives**: `simulator/calibration_contract.yaml`,
`simulator/calibration.py`,
`simulator/evidence.py::profile_public_evidence`.

---

### Artifact freshness check

**What it does**: the CI pipeline regenerates the published artifacts
(markdown fragments, JSON, CSV, HTML) on every push and fails if the output
differs from what is checked in. A SHA-based metadata fingerprint on config
inputs drives the stability-run cache so reruns only happen when something
changes.

**What it buys**: the published case study is always in sync with the model
code. Drift between the two is a CI failure, not a documentation problem.

**Where it lives**: `simulator/artifact_freshness.py`,
`.github/workflows/ci.yml` (the "Fail if generated outputs are stale" step).
