#!/usr/bin/env python3
"""
Data analyzer for decision quality framework.

Takes historical data (CSV) and suggests parameter ranges for config.yaml.
The PM reviews suggestions and manually updates the config as needed.

Usage:
    python3 analyze_data.py data.csv --output suggested_ranges.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import yaml


def analyze_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Analyze a single column and suggest parameter ranges.

    Returns dict with:
    - low: 5th percentile
    - mode: 50th percentile (median)
    - high: 95th percentile
    - mean: arithmetic mean
    - std: standard deviation
    - dist: suggested distribution type
    """
    clean = series.dropna()

    if len(clean) == 0:
        return {"error": "No valid data"}

    if len(clean) < 10:
        return {"error": f"Insufficient data (only {len(clean)} points)"}

    p05 = float(np.percentile(clean, 5))
    p50 = float(np.percentile(clean, 50))
    p95 = float(np.percentile(clean, 95))
    mean = float(clean.mean())
    std = float(clean.std())

    # Suggest distribution type based on shape
    skewness = (mean - p50) / (std + 1e-9)

    if abs(skewness) < 0.3:
        dist = "tri"  # Relatively symmetric, triangular is reasonable
    else:
        dist = "tri"  # Default to triangular for interpretability

    return {
        "low": p05,
        "mode": p50,
        "high": p95,
        "mean": mean,
        "std": std,
        "dist": dist,
        "n_samples": len(clean),
    }


def generate_yaml_snippet(analysis: Dict[str, Dict[str, Any]], comment_stats: bool = True) -> str:
    """
    Generate YAML snippet ready to paste into config.yaml.

    If comment_stats=True, includes mean/std/n_samples as comments for reference.
    """
    lines = ["params:"]

    for param_name, stats in analysis.items():
        if "error" in stats:
            lines.append(f"  # {param_name}: {stats['error']}")
            continue

        lines.append(f"  {param_name}:")
        lines.append(f"    dist: {stats['dist']}")
        lines.append(f"    low: {stats['low']:.6f}")
        lines.append(f"    mode: {stats['mode']:.6f}")
        lines.append(f"    high: {stats['high']:.6f}")

        if comment_stats:
            lines.append(f"    # Historical: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n_samples']}")

    return "\n".join(lines)


def generate_summary_table(analysis: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Generate a summary table for quick review."""
    rows = []
    for param_name, stats in analysis.items():
        if "error" in stats:
            rows.append({
                "parameter": param_name,
                "status": stats["error"],
                "low": None,
                "mode": None,
                "high": None,
                "n_samples": None,
            })
        else:
            rows.append({
                "parameter": param_name,
                "status": "OK",
                "low": stats["low"],
                "mode": stats["mode"],
                "high": stats["high"],
                "n_samples": stats["n_samples"],
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze historical data and suggest parameter ranges for config.yaml"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to CSV file with historical data"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for suggested YAML (default: print to stdout)"
    )
    parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        nargs="*",
        default=[],
        help="Columns to exclude from analysis (e.g., date, id, category)"
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Print summary table before YAML output"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("Error: CSV file is empty", file=sys.stderr)
        sys.exit(1)

    # Analyze each numeric column
    analysis = {}
    exclude_set = set(args.exclude)

    for col in df.columns:
        if col in exclude_set:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            analysis[col] = analyze_column(df[col], col)
        else:
            # Try to convert to numeric
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.notna().sum() > 0:
                    analysis[col] = analyze_column(numeric_col, col)
            except:
                pass

    if not analysis:
        print("Error: No numeric columns found in CSV", file=sys.stderr)
        sys.exit(1)

    # Generate summary table
    if args.summary:
        summary = generate_summary_table(analysis)
        print("\n=== Summary ===")
        print(summary.to_string(index=False))
        print()

    # Generate YAML
    yaml_output = generate_yaml_snippet(analysis)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(yaml_output)
        print(f"Suggested ranges written to: {output_path}")
    else:
        print("\n=== Suggested YAML ===")
        print(yaml_output)

    print("\nNext steps:")
    print("1. Review the suggested ranges above")
    print("2. Adjust based on your judgment (future may differ from past)")
    print("3. Copy relevant parameters into simulator/config.yaml")
    print("4. Consider scenario variations for optimistic/pessimistic futures")


if __name__ == "__main__":
    main()
