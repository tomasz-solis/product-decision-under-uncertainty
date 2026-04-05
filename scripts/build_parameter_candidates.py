"""Build evidence-to-parameter candidate metrics from the calibration contract."""

from __future__ import annotations

import argparse

from simulator.calibration import (
    build_parameter_candidates,
    load_calibration_contract,
    write_parameter_candidates_outputs,
)
from simulator.evidence import load_public_evidence_profile, profile_public_evidence
from simulator.project_paths import (
    CALIBRATION_CONTRACT_PATH,
    PARAMETER_CANDIDATES_JSON,
    PARAMETER_CANDIDATES_MARKDOWN,
    PUBLIC_DATA_DIR,
    PUBLIC_EVIDENCE_PROFILE_JSON,
    SOURCE_MANIFEST_PATH,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the candidate-builder entrypoint."""

    parser = argparse.ArgumentParser(
        description=(
            "Build parameter candidate metrics from profiled public evidence and the "
            "checked-in calibration contract."
        )
    )
    parser.add_argument(
        "--profile",
        default=str(PUBLIC_EVIDENCE_PROFILE_JSON),
        help="Path to the profiled evidence JSON artifact.",
    )
    parser.add_argument(
        "--contract",
        default=str(CALIBRATION_CONTRACT_PATH),
        help="Path to the calibration contract YAML file.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(PUBLIC_DATA_DIR),
        help="Raw public-data directory used if the profile artifact is missing.",
    )
    parser.add_argument(
        "--manifest",
        default=str(SOURCE_MANIFEST_PATH),
        help="Public-source manifest used if the profile artifact is missing.",
    )
    parser.add_argument(
        "--output",
        default=str(PARAMETER_CANDIDATES_JSON),
        help="Path for the generated candidate JSON artifact.",
    )
    parser.add_argument(
        "--markdown-output",
        default=str(PARAMETER_CANDIDATES_MARKDOWN),
        help="Path for the generated candidate markdown artifact.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the checked-in evidence-candidate artifacts."""

    args = parse_args()
    try:
        profile = load_public_evidence_profile(args.profile)
    except FileNotFoundError:
        profile = profile_public_evidence(args.input_dir, args.manifest)
    contract = load_calibration_contract(args.contract)
    payload = build_parameter_candidates(profile, contract)
    write_parameter_candidates_outputs(
        payload,
        json_output_path=args.output,
        markdown_output_path=args.markdown_output,
    )
    print(f"Wrote parameter candidates to {args.output}")
    print(f"Wrote parameter candidate note to {args.markdown_output}")
