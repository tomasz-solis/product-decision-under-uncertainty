# Public Data Staging

Put raw public source files here when the first calibration slice arrives.

## What belongs here

- raw CSV, parquet, or JSON files from public sources
- a matching manifest entry in [sources.yaml](sources.yaml)
- enough metadata to reproduce the fetch or manual extract step

## Minimum expectation

Each dataset added here should include:

- source URL or publication name
- extraction date
- license or usage note
- a short explanation of which assumption family it informs
- an `expected_schema` entry in [sources.yaml](sources.yaml)

## Next step after adding data

Run [../../scripts/profile_public_evidence.py](../../scripts/profile_public_evidence.py), then [../../scripts/build_parameter_candidates.py](../../scripts/build_parameter_candidates.py). Those scripts validate the manifest against the files, profile the raw data, and write derived artifacts into [../../artifacts/evidence/public_data_profile.md](../../artifacts/evidence/public_data_profile.md) and [../../artifacts/evidence/parameter_candidates.md](../../artifacts/evidence/parameter_candidates.md).
