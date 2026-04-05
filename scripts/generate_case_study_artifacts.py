"""Generate the checked-in case study artifacts and markdown tables."""

from __future__ import annotations

import argparse

from simulator.calibration import (
    build_parameter_candidates,
    load_calibration_contract,
    write_parameter_candidates_outputs,
)
from simulator.evidence import write_public_evidence_outputs
from simulator.project_paths import (
    CALIBRATION_CONTRACT_PATH,
    CASE_STUDY_PATH,
    EXECUTIVE_SUMMARY_PATH,
    PARAMETER_CANDIDATES_JSON,
    PARAMETER_CANDIDATES_MARKDOWN,
    PUBLIC_EVIDENCE_PROFILE_JSON,
    PUBLIC_EVIDENCE_PROFILE_MARKDOWN,
)
from simulator.reporting import (
    build_case_study_artifacts,
    update_case_study_docs,
    write_case_study_artifacts,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Generate JSON artifacts and markdown fragments for the synthetic case study."
    )
    parser.add_argument(
        "--config",
        default="simulator/config.yaml",
        help="Path to the simulator config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/case_study",
        help="Directory where generated artifacts should be written.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Write artifacts only and leave markdown docs untouched.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate artifacts and optionally update the public docs."""

    args = parse_args()
    artifacts = build_case_study_artifacts(args.config)
    write_public_evidence_outputs(
        artifacts.evidence_profile,
        json_output_path=PUBLIC_EVIDENCE_PROFILE_JSON,
        markdown_output_path=PUBLIC_EVIDENCE_PROFILE_MARKDOWN,
    )
    parameter_candidates = build_parameter_candidates(
        artifacts.evidence_profile,
        load_calibration_contract(CALIBRATION_CONTRACT_PATH),
    )
    write_parameter_candidates_outputs(
        parameter_candidates,
        json_output_path=PARAMETER_CANDIDATES_JSON,
        markdown_output_path=PARAMETER_CANDIDATES_MARKDOWN,
    )
    fragments = write_case_study_artifacts(artifacts, args.output_dir)

    if not args.skip_docs:
        update_case_study_docs(
            case_study_path=CASE_STUDY_PATH,
            executive_summary_path=EXECUTIVE_SUMMARY_PATH,
            fragments=fragments,
        )

    print("Updated public evidence artifacts in artifacts/evidence.")
    print(f"Wrote generated case-study artifacts to {args.output_dir}")
    if args.skip_docs:
        print("Skipped markdown document updates.")
    else:
        print("Updated CASE_STUDY.md and EXECUTIVE_SUMMARY.md.")


if __name__ == "__main__":
    main()
