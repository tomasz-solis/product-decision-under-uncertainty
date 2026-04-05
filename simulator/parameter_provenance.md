# Parameter Provenance

This page explains how the repo handles assumption lineage.

## Source files

- Parameter registry: [parameter_registry.yaml](parameter_registry.yaml)
- Non-parameter assumption registry: [assumption_registry.yaml](assumption_registry.yaml)
- Generated manifest: [../artifacts/case_study/assumption_manifest.json](../artifacts/case_study/assumption_manifest.json)

## What is checked

- Parameter distribution types and numeric values must match [config.yaml](config.yaml).
- Policy thresholds, analysis thresholds, dependency targets, and scenario overrides are provenance-tracked too.
- Dates must be valid ISO dates.
- `source_type` must come from an allowed vocabulary.
- Empty rationale fields fail validation.

## Source types used here

- `illustrative_benchmark`
  A synthetic anchor chosen to be plausible for the domain.
- `elicited_range`
  A range chosen as if it came from structured analyst or stakeholder judgment.
- `elicited_relationship`
  A stated dependency between assumptions inside one sampled world.
- `synthetic_case_assumption`
  A modeling choice made to keep the case coherent.
- `portfolio_governance_assumption`
  A threshold or framing choice for how the decision is judged.
- `placeholder_for_real_telemetry`
  A slot that is expected to move to observed product data later.

## Worked example

One example is `stabilize_core_launch_delay_months`.

1. The raw business question is not "what is the perfect delivery date?" It is "what range of launch timing should this option face in a two-year planning case?"
2. The modeled range in [parameter_registry.yaml](parameter_registry.yaml) is `3 / 5 / 8` months.
3. That range is then used directly in [config.yaml](config.yaml), so the provenance file and the simulator cannot silently drift apart.
4. When a real public benchmark arrives, the workflow starts with [../data/public/README.md](../data/public/README.md), runs through [../scripts/profile_public_evidence.py](../scripts/profile_public_evidence.py), maps the source against [calibration_contract.yaml](calibration_contract.yaml), and writes derived notes into [../artifacts/evidence/public_data_profile.md](../artifacts/evidence/public_data_profile.md) and [../artifacts/evidence/parameter_candidates.md](../artifacts/evidence/parameter_candidates.md).

The repo still needs a real public-data slice to show raw evidence transformation in a non-synthetic way. The difference now is that the path from raw file to candidate parameter is explicit instead of implied.
