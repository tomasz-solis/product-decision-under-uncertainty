"""Profile manifest-declared public evidence files and write strict outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulator.evidence import profile_public_evidence, write_public_evidence_outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evidence-profiling entrypoint."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile the manifest-declared public evidence files and write strict JSON "
            "plus a short markdown note."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/public",
        help="Directory that contains raw public evidence files.",
    )
    parser.add_argument(
        "--manifest",
        default="data/public/sources.yaml",
        help="Path to the public-source manifest.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evidence/public_data_profile.json",
        help="Path for the generated JSON profiling artifact.",
    )
    parser.add_argument(
        "--markdown-output",
        default="artifacts/evidence/public_data_profile.md",
        help="Path for the generated markdown profiling note.",
    )
    return parser.parse_args()


def main() -> None:
    """Profile the current public-data staging folder and write the derived outputs."""

    args = parse_args()
    profile = profile_public_evidence(args.input_dir, args.manifest)
    write_public_evidence_outputs(
        profile,
        json_output_path=Path(args.output),
        markdown_output_path=Path(args.markdown_output),
    )
    print(f"Wrote public evidence profile to {args.output}")
    print(f"Wrote public evidence note to {args.markdown_output}")


if __name__ == "__main__":
    main()
