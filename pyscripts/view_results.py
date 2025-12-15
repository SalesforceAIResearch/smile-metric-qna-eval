#!/usr/bin/env python3
"""Script to view contents of pickle result files."""

import argparse
import pickle as pkl
import json
import numpy as np


def format_value(val):
    """Format a single value to 2 decimal places if numeric."""
    if isinstance(val, (float, np.floating)):
        return f"{val:.3f}"
    elif isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (list, np.ndarray)):
        formatted = [format_value(v) for v in val]
        return "[" + ", ".join(formatted) + "]"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="View contents of a pickle file")
    parser.add_argument("--input_file", "-i", type=str, default="test_smile.pkl",
                        help="Path to the pickle file (default: test_smile.pkl)")
    parser.add_argument("--format", "-f", choices=["pretty", "json"], default="pretty",
                        help="Output format: pretty (default) or json")
    args = parser.parse_args()

    # Load the pickle file
    print(f"Loading: {args.input_file}\n")
    with open(args.input_file, "rb") as f:
        results = pkl.load(f)

    # Display results
    if args.format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        if isinstance(results, dict):
            for key, value in results.items():
                print(f"\n📊 {key}:")
                print("-" * 40)
                if isinstance(value, np.ndarray):
                    for i, item in enumerate(value):
                        print(f"  [{i}] {format_value(item)}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        print(f"  [{i}] {format_value(item)}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        print(f"  {k}: {format_value(v)}")
                else:
                    print(f"  {format_value(value)}")
        else:
            print(results)
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

