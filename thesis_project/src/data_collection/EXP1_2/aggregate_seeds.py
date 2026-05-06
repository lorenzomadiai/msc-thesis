#!/usr/bin/env python3
"""
aggregate_seeds.py

Computes between-seed statistics from multiple per-seed summary tables
produced by table_exp1and2.py.

For each agent and each metric:
  - reads the *_mean value from each per-seed summary
  - computes mean and std across those seed-level means  (ddof=1)
  - computes SEM = std / sqrt(n_seeds)

This gives a robust estimate of how stable the agent's performance is
across different episode sets (seeds).

Usage:
  # Step 1 — produce one summary per seed
  python table_exp1and2.py --csvs results/EXP1/*seed0*.csv --out_csv results/EXP1/table_seed0.csv
  python table_exp1and2.py --csvs results/EXP1/*seed1*.csv --out_csv results/EXP1/table_seed1.csv
  python table_exp1and2.py --csvs results/EXP1/*seed2*.csv --out_csv results/EXP1/table_seed2.csv

  # Step 2 — aggregate between seeds
  python aggregate_seeds.py \\
      --tables results/EXP1/table_seed0.csv results/EXP1/table_seed1.csv results/EXP1/table_seed2.csv \\
      --out_csv results/EXP1/table_between_seeds.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser(
        description="Aggregate between-seed statistics from multiple per-seed summary tables."
    )
    p.add_argument("--tables", nargs="+", required=True,
                   help="Per-seed summary CSVs produced by table_exp1and2.py.")
    p.add_argument("--agents", nargs="+", default=None,
                   help="Subset of agent names to include (default: all).")
    p.add_argument("--agent_labels", nargs="+", default=None, metavar="NAME=LABEL",
                   help="Rename agents. E.g. 'Agent Aggressive=Aggressive baseline'.")
    p.add_argument("--out_csv", default=None,
                   help="If given, save the aggregated table to this CSV path.")
    p.add_argument("--show_all", action="store_true",
                   help="Print all columns, not just *_mean.")
    args = p.parse_args()

    # Build label remap
    label_map = {}
    if args.agent_labels:
        for entry in args.agent_labels:
            eq = entry.rindex("=")
            label_map[entry[:eq]] = entry[eq + 1:]

    # Load per-seed tables
    seed_dfs = []
    for i, path in enumerate(args.tables):
        try:
            df = pd.read_csv(path)
            df["_seed_idx"] = i
            seed_dfs.append(df)
            print(f"Loaded seed {i}: {path}  ({len(df)} agents)")
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")

    if not seed_dfs:
        raise SystemExit("No tables loaded.")

    n_seeds = len(seed_dfs)
    print(f"\n{n_seeds} seeds loaded.\n")

    # Identify agents across all seeds
    all_agents = set()
    for df in seed_dfs:
        all_agents.update(df["agent"].astype(str).unique())
    agents = sorted(all_agents)
    if args.agents:
        agents = [a for a in args.agents if a in agents]
    if not agents:
        raise SystemExit("No agents found.")

    # Identify metric columns: columns ending in _mean present in all tables
    mean_cols = None
    for df in seed_dfs:
        cols = {c for c in df.columns if c.endswith("_mean")}
        mean_cols = cols if mean_cols is None else mean_cols & cols
    mean_cols = sorted(mean_cols)

    # Also propagate n_episodes (sum across seeds)
    has_n_episodes = all("n_episodes" in df.columns for df in seed_dfs)
    has_n_success  = all("n_success_episodes" in df.columns for df in seed_dfs)

    rows = []
    for agent in agents:
        label = label_map.get(agent, agent)
        row = {"agent": label}

        if has_n_episodes:
            total_ep = sum(
                int(df.loc[df["agent"].astype(str) == agent, "n_episodes"].values[0])
                for df in seed_dfs
                if agent in df["agent"].astype(str).values
            )
            row["n_episodes_total"] = total_ep
            row["n_seeds"] = n_seeds

        if has_n_success:
            total_succ = sum(
                int(df.loc[df["agent"].astype(str) == agent, "n_success_episodes"].values[0])
                for df in seed_dfs
                if agent in df["agent"].astype(str).values
            )
            row["n_success_episodes_total"] = total_succ

        for col in mean_cols:
            # Collect the per-seed mean values for this agent
            seed_means = []
            for df in seed_dfs:
                mask = df["agent"].astype(str) == agent
                if mask.any():
                    val = pd.to_numeric(df.loc[mask, col], errors="coerce").values[0]
                    if np.isfinite(val):
                        seed_means.append(val)

            seed_means = np.array(seed_means, dtype=np.float64)
            n = len(seed_means)

            base = col[:-len("_mean")]  # strip trailing _mean
            if n == 0:
                row[f"{base}_between_mean"] = float("nan")
                row[f"{base}_between_std"]  = float("nan")
                row[f"{base}_between_sem"]  = float("nan")
            else:
                std = float(np.std(seed_means, ddof=1)) if n > 1 else 0.0
                row[f"{base}_between_mean"] = float(np.mean(seed_means))
                row[f"{base}_between_std"]  = std
                row[f"{base}_between_sem"]  = std / np.sqrt(n) if n > 1 else 0.0

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Print
    if args.show_all:
        print_df = summary
    else:
        mean_out_cols = ["agent", "n_episodes_total", "n_seeds"] + \
                        [c for c in summary.columns if c.endswith("_between_mean")]
        mean_out_cols = [c for c in mean_out_cols if c in summary.columns]
        print_df = summary[mean_out_cols]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(print_df.to_string(index=False))

    if args.out_csv:
        ensure_dir(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else None)
        summary.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
