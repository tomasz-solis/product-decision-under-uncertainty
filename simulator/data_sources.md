# Evidence Workflow

The published case still runs without a real public calibration dataset. What matters now is that the repo knows what it would do with that data on day one.

## Current state

- Structured assumption registries
- Validation that registries match config values
- Generated artifacts that export the unified assumption manifest

## Planned raw-to-assumption flow

1. Add one or more raw source files under [../data/public/README.md](../data/public/README.md).
2. Register each file in [../data/public/sources.yaml](../data/public/sources.yaml) with source metadata, license, grain, expected schema, and target assumption families. Use [../data/public/sources.template.yaml](../data/public/sources.template.yaml) as the shape reference.
3. Run [../scripts/profile_public_evidence.py](../scripts/profile_public_evidence.py) to validate the manifest, check the schema, and profile missingness, duplicates, dtypes, date ranges, and numeric summaries into [../artifacts/evidence/public_data_profile.md](../artifacts/evidence/public_data_profile.md).
4. Map the profiled source to calibration targets in [calibration_contract.yaml](calibration_contract.yaml).
5. Run [../scripts/build_parameter_candidates.py](../scripts/build_parameter_candidates.py) to emit candidate metrics into [../artifacts/evidence/parameter_candidates.json](../artifacts/evidence/parameter_candidates.json) and [../artifacts/evidence/parameter_candidates.md](../artifacts/evidence/parameter_candidates.md).
6. Use source ids from the manifest as `evidence_ids` once a real source is linked into the registries.

## What is still missing

The repo still does not claim a real public calibration slice. No raw public benchmark is checked in yet, and no modeled range in the published case is currently derived from public data.
