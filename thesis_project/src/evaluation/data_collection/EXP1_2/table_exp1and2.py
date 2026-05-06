#!/usr/bin/env python3
"""
table_random_settings.py

Produces a summary table from CSV files produced by eval_with_random_settings.py.

For each agent: computes the mean (and std) of every numeric column across
all episodes. Also adds a derived 'success_rate' column based on
goal_first_step vs budget.

Output: printed table + optional CSV file.

Example:
  python src/analysis/plot_random_settings.py \\
      --csvs results/randomic_settings/traindist_timeaware_seed2208_eps300_*.csv \\
      --agent_labels "Agent Aggressive=Aggressive" "Agent Conservative=Conservative" \\
      --out_csv results/table_randomic.csv

  # Keep only episodes where the conservative agent reached the goal:
  python src/analysis/plot_random_settings.py \\
      --csvs results/EXP1/5seeds/all_runs_per_seed/*.csv \\
      --conservative_filter \\
      --out_csv results/table_conserv_filtered.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def apply_conservative_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the episodes (identified by episode_idx + budget + _file_idx)
    where the *conservative* agent reached the goal (success == 1 or
    goal_first_step <= budget).  All other agents' rows for those same
    episodes are retained; the rest are dropped.

    The column '_file_idx' must already be present in *df* (added in main
    before concatenation so that episode indices from different CSV files
    don't collide).
    """
    if "conservative" not in df["agent"].astype(str).values:
        print("WARNING: 'conservative' agent not found – conservative filter skipped.")
        return df

    # Determine success for each row: prefer the explicit 'success' column,
    # fall back to goal_first_step <= budget.
    if "success" in df.columns:
        cons_ok = (
            (df["agent"].astype(str) == "conservative")
            & (pd.to_numeric(df["success"], errors="coerce") == 1)
        )
    elif "goal_first_step" in df.columns and "budget" in df.columns:
        gf  = pd.to_numeric(df["goal_first_step"], errors="coerce")
        bud = pd.to_numeric(df["budget"],          errors="coerce")
        cons_ok = (
            (df["agent"].astype(str) == "conservative")
            & (gf != -1) & (gf <= bud)
        )
    else:
        print("WARNING: cannot determine success (no 'success' or 'goal_first_step'/'budget' columns) "
              "– conservative filter skipped.")
        return df

    key_cols = ["_file_idx", "episode_idx", "budget"]
    success_keys = set(
        map(tuple, df.loc[cons_ok, key_cols].drop_duplicates().values.tolist())
    )
    n_total_cons = int((df["agent"].astype(str) == "conservative").sum())
    print(
        f"Conservative filter: {len(success_keys)} / {n_total_cons} episodes kept "
        f"({100 * len(success_keys) / n_total_cons:.1f}% of conservative episodes)."
    )

    mask = df[key_cols].apply(lambda r: tuple(r) in success_keys, axis=1)
    filtered = df[mask].copy()
    print(f"  Rows before filter: {len(df)} → after filter: {len(filtered)}")
    return filtered


def compute_summary(df: pd.DataFrame, agents: list, label_map: dict) -> pd.DataFrame:
    """
    For each agent compute mean of every numeric column across all episodes.
    Adds a derived 'success_rate' column.
    Skips non-numeric columns (except 'agent').
    """
    # Derive success_rate if possible
    if "goal_first_step" in df.columns and "budget" in df.columns:
        gf = pd.to_numeric(df["goal_first_step"], errors="coerce")
        bud = pd.to_numeric(df["budget"], errors="coerce")
        df = df.copy()
        df["success_rate"] = ((gf != -1) & (gf <= bud)).astype(float)

    # Numeric columns (excluding identifiers and cost_cum_* columns)
    skip_cols = {"agent", "episode_idx", "seed", "run_id"}
    numeric_cols = [
        c for c in df.columns
        if c not in skip_cols
        and not c.startswith("cost_cum_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for agent in agents:
        sub = df[df["agent"].astype(str) == str(agent)]
        label = label_map.get(agent, agent)
        row = {"agent": label, "n_episodes": len(sub)}
        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").astype(np.float64)
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            row[f"{col}_mean"] = float(np.mean(vals)) if n else float("nan")
            row[f"{col}_std"]  = std
            row[f"{col}_sem"]  = std / np.sqrt(n) if n > 1 else 0.0

        # mean_dist_hazard on successful episodes only
        if "mean_dist_hazard" in sub.columns and "success_rate" in sub.columns:
            succ_mask = pd.to_numeric(sub["success_rate"], errors="coerce") == 1.0
            succ_dists = pd.to_numeric(sub.loc[succ_mask, "mean_dist_hazard"], errors="coerce").astype(np.float64)
            succ_dists = succ_dists[np.isfinite(succ_dists)]
            n_s = len(succ_dists)
            std_s = float(np.std(succ_dists, ddof=1)) if n_s > 1 else 0.0
            row["mean_dist_hazard_success_mean"] = float(np.mean(succ_dists)) if n_s else float("nan")
            row["mean_dist_hazard_success_std"]  = std_s
            row["mean_dist_hazard_success_sem"]  = std_s / np.sqrt(n_s) if n_s > 1 else 0.0
            row["n_success_episodes"] = int(succ_mask.sum())

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="Compute per-agent summary table from eval CSV files."
    )
    p.add_argument("--csvs", nargs="+", required=True,
                   help="One or more CSV files.")
    p.add_argument("--agents", nargs="+", default=None,
                   help="Subset of agent names to include (default: all).")
    p.add_argument("--agent_labels", nargs="+", default=None, metavar="NAME=LABEL",
                   help="Rename agents. E.g. 'Agent Aggressive=Aggressive baseline'.")
    p.add_argument("--out_csv", default=None,
                   help="If given, save the summary table to this CSV path.")
    p.add_argument("--show_std", action="store_true",
                   help="Include std columns in the printed output (always saved if --out_csv).")
    p.add_argument("--conservative_filter", action="store_true",
                   help="Keep only episodes where the 'conservative' agent reached the goal. "
                        "All other agents are then evaluated only on those shared episodes.")
    args = p.parse_args()

    # Build label remap
    label_map = {}
    if args.agent_labels:
        for entry in args.agent_labels:
            eq = entry.rindex("=")
            label_map[entry[:eq]] = entry[eq + 1:]

    # Load CSVs — tag each file with an integer index so episode_idx + budget
    # keys remain unique across files even if they overlap numerically.
    dfs = []
    for file_idx, path in enumerate(args.csvs):
        try:
            tmp = pd.read_csv(path)
            tmp["_file_idx"] = file_idx
            dfs.append(tmp)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")
    if not dfs:
        raise SystemExit("No CSV files loaded.")
    df = pd.concat(dfs, ignore_index=True)

    required = {"agent"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    all_agents = list(df["agent"].astype(str).unique())
    agents = args.agents if args.agents else all_agents
    agents = [a for a in agents if a in all_agents]
    if not agents:
        raise SystemExit("No agents found.")

    print(f"Loaded {len(df)} episodes | Agents: {agents}\n")

    if args.conservative_filter:
        df = apply_conservative_filter(df)
        if df.empty:
            raise SystemExit("No episodes remain after conservative filter.")
        print()

    summary = compute_summary(df, agents, label_map)

    # Print: mean columns only by default, all if --show_std
    if args.show_std:
        print_df = summary
    else:
        mean_cols = ["agent", "n_episodes"] + [c for c in summary.columns if c.endswith("_mean")]
        print_df = summary[mean_cols]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(print_df.to_string(index=False))

    if args.out_csv:
        ensure_dir(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else None)
        summary.to_csv(args.out_csv, index=False)
        print(f"\nTable saved to: {args.out_csv}")


if __name__ == "__main__":
    main()

