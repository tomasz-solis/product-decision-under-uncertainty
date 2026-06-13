# Changelog

All notable changes to the decision model and its published case study are
recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The `model_version` field in [simulator/config.yaml](simulator/config.yaml) is the
machine-readable counterpart to the model entries below and is fingerprinted into
every generated artifact.

## [Unreleased]

### Added
- **Value-of-information analysis**: EVPI and per-parameter EVPPI, published as
  `artifacts/case_study/value_of_information.{json,md}` and a new CASE_STUDY.md
  section. Quantifies which uncertainty is worth resolving before deciding
  (see ADR-005 and METHODS.md).
- `LICENSE` (MIT) so the work can be evaluated and reused without ambiguity.
- `Makefile` with the common command set (`install`, `app`, `artifacts`,
  `evidence`, `test`, `lint`, `format`, `type`, `check`, `ci`).
- `Dockerfile` for a one-command, lockfile-pinned run of the Streamlit app.
- `.pre-commit-config.yaml` mirroring the CI lint and hygiene gates locally.
- Architecture decision records: ADR-002 (guardrailed expected-value policy),
  ADR-003 (shared latent regression draw), ADR-004 (monthly timing and
  discounting), ADR-005 (value of information).
- README hero section: headline result, run/demo links, and status badges.

## [5.0.0] - Current published model

### Added
- Full-option **policy frontier**: re-runs the whole policy across a threshold
  grid and records the first change that flips the recommendation, kept
  strictly separate from the descriptive payoff-delta diagnostic.
- **Robustness artifact** with an independence ablation (all correlations set to
  zero) and a per-pair dependency-value frontier.
- **Calibration contract** and evidence-profiling seam for one public proxy
  dataset (HM Land Registry completion benchmark) feeding `baseline_failure_rate`.
- **Assumption provenance** registries (parameters and non-parameter
  assumptions) exported to a machine-readable `assumption_manifest.json`.
- **Artifact-freshness** governance: CI regenerates the published artifacts and
  fails on any drift from what is checked in.

### Changed
- Sensitivity reporting moved to partial rank correlation with bootstrap
  confidence intervals and a materiality gate; descriptive Spearman output
  retained for inspection.
- Cashflows modelled on a monthly discounting grid with launch delay, benefit
  ramp, and cost-overrun multiplier per option (see ADR-004).

### Notes
- History prior to the `5.0.0` published model predates this changelog. From
  this point forward, any change that moves the published recommendation, a
  guardrail threshold, or a modelled distribution should bump `model_version`
  and add an entry here.
