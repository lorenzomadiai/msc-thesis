#!/usr/bin/env python3
"""
This script is mainly used to isolate the high-budget regime before computing
risk-sensitive metrics such as CVaR. Therefore, this
script creates the filtered CSV used by the downstream analysis scripts to compute
CVaR on that specific subset of episodes. The CVaR analysis is performed in policy_cvar_analysis.py. 
The script assumes that the input CSV already contains only episodes belonging to the high-budget
regime, obtained through this filtering step. Multiple CSV files can be provided simultaneously, 
allowing the script to aggregate results across different seeds and compute the corresponding mean and uncertainty estimates (standard deviation or standard error).

The filtering is based on quantiles of the `budget` column. For example:

    --q_low 0.75 --q_high 1.0

keeps only the upper quartile of the sampled budget distribution, which
corresponds to the high-budget regime used for the risk-bound evaluation.

Input:
    A CSV produced by evaluate_policies.py, containing one row per episode and
    at least a `budget` column.

Output:
    A filtered CSV containing only the rows whose budget falls inside the
    selected quantile interval.
"""
import argparse
import csv
import os
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an evaluation CSV so that only "
            "episodes whose sampled budget falls inside a quantile-defined interval "
            "are kept. Quantiles are computed over the `budget` column present in the "
            "input file."
        )
    )
    parser.add_argument("input_csv", type=str, help="Path to the source evaluation CSV.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional output path. Defaults to appending the quantile tag to the input filename.",
    )
    parser.add_argument(
        "--q_low",
        type=float,
        default=0.0,
        help="Lower quantile bound in [0, 1]. Example: 0.75 keeps the top quartile budgets.",
    )
    parser.add_argument(
        "--q_high",
        type=float,
        default=1.0,
        help="Upper quantile bound in [0, 1]. Must be >= q_low.",
    )
    return parser.parse_args()


def _nearest_quantile(values: np.ndarray, q: float) -> float:
    """Quantile helper compatible with older numpy versions."""
    try:
        return float(np.quantile(values, q, method="nearest"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(values, q, interpolation="nearest"))


def compute_budget_bounds(budgets: List[float], q_low: float, q_high: float) -> Tuple[float, float]:
    if not budgets:
        raise ValueError("Input CSV contains no rows with a numeric 'budget' column.")

    arr = np.asarray(budgets, dtype=np.float64)
    lo = _nearest_quantile(arr, q_low)
    hi = _nearest_quantile(arr, q_high)
    lo, hi = min(lo, hi), max(lo, hi)
    return lo, hi


def derive_output_path(input_path: str, q_low: float, q_high: float) -> str:
    base, ext = os.path.splitext(os.path.abspath(input_path))
    q_tag_low = int(round(q_low * 100))
    q_tag_high = int(round(q_high * 100))
    tag = f"q{q_tag_low}to{q_tag_high}" if q_low != q_high else f"q{q_tag_low}"
    return f"{base}_{tag}{ext or '.csv'}"


def main():
    args = parse_args()

    if not (0.0 <= args.q_low <= 1.0):
        raise SystemExit("--q_low must be within [0, 1].")
    if not (0.0 <= args.q_high <= 1.0):
        raise SystemExit("--q_high must be within [0, 1].")
    if args.q_low > args.q_high:
        raise SystemExit("--q_low must be <= --q_high.")

    if not os.path.isfile(args.input_csv):
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise SystemExit("Input CSV is empty.")
    if fieldnames is None or "budget" not in fieldnames:
        raise SystemExit("Input CSV must contain a 'budget' column header.")

    budgets = []
    for row in rows:
        try:
            budgets.append(float(row["budget"]))
        except (KeyError, TypeError, ValueError):
            continue

    low_bound, high_bound = compute_budget_bounds(budgets, args.q_low, args.q_high)

    filtered_rows = [row for row in rows if low_bound <= float(row["budget"]) <= high_bound]

    if not filtered_rows:
        raise SystemExit(
            "Quantile interval produced zero rows. Try widening the range or check the input file."
        )

    out_path = args.output_csv or derive_output_path(args.input_csv, args.q_low, args.q_high)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(
        "Saved filtered CSV:\n"
        f"  input            : {args.input_csv}\n"
        f"  output           : {out_path}\n"
        f"  q_low / q_high   : {args.q_low:.3f} / {args.q_high:.3f}\n"
        f"  budget bounds    : [{low_bound:.2f}, {high_bound:.2f}]\n"
        f"  rows kept        : {len(filtered_rows)} / {len(rows)}"
    )


if __name__ == "__main__":
    main()
