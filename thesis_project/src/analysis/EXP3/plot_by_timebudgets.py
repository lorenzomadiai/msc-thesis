#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def plot_metric_with_band(budgets, mean, band, ylabel, title, out_path):
    budgets = np.asarray(budgets, dtype=np.int64)
    mean = np.asarray(mean, dtype=np.float64)
    band = np.asarray(band, dtype=np.float64)

    plt.figure()
    plt.plot(budgets, mean, linewidth=2)
    plt.fill_between(budgets, mean - band, mean + band, alpha=0.2)
    plt.xlabel("Time budget")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_comparison(budgets, agents_data, ylabel, title, out_path, hline=None, hline_label=None):
    """Plot multiple agents on the same figure.

    agents_data: list of (mean, std, label) tuples, one per agent.
    hline: optional float – if given, a horizontal dashed reference line is drawn at that value.
    hline_label: optional label for the hline in the legend.
    """
    budgets = np.asarray(budgets, dtype=np.int64)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure()
    for i, (mean, std, label) in enumerate(agents_data):
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        color = colors[i % len(colors)]
        plt.plot(budgets, mean, linewidth=2, label=label, color=color)
        plt.fill_between(budgets, mean - std, mean + std, alpha=0.2, color=color)
    if hline is not None:
        lbl = hline_label if hline_label is not None else "Safety budget ({:.4g})".format(hline)
        plt.axhline(hline, color="black", linestyle="--", linewidth=1.2, label=lbl)
    plt.xlabel("Time budget")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def infer_budget_col(df, user_budget_col):
    if user_budget_col and (user_budget_col in df.columns):
        return user_budget_col
    for c in ("budget", "time_budget", "B", "timeBudget"):
        if c in df.columns:
            return c
    raise KeyError(
        "Could not find a budget column. Provide --budget_col explicitly "
        "or ensure the CSV has one of: budget, time_budget, B, timeBudget."
    )


def available_cost_cum_steps(df):
    steps = set()
    pat = re.compile(r"^cost_cum_(\d+)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            steps.add(int(m.group(1)))
    return steps


def _as_float_array(series):
    # robust for pandas versions on py3.7
    return series.astype(np.float64).values


def cost_at_budget(dfB, B):
    col = "cost_cum_{}".format(int(B))
    if col in dfB.columns:
        return _as_float_array(dfB[col])

    if "cost_total" in dfB.columns:
        return _as_float_array(dfB["cost_total"])

    steps = sorted([t for t in available_cost_cum_steps(dfB) if t <= int(B)])
    if steps:
        return _as_float_array(dfB["cost_cum_{}".format(steps[-1])])

    raise KeyError("Missing {} (and no fallback cost_total / cost_cum_* available).".format(col))


def compute_series_for_run(df, budgets, budget_col):
    required = set(["ep_len", "goal_first_step", budget_col])
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise KeyError("CSV missing required columns: {}".format(missing))

    out = {
        "success_rate": [],
        "safety_efficiency": [],
        "mean_success_len": [],
        "mean_cost": [],
    }

    for B in budgets:
        dfB = df[df[budget_col].astype(int) == int(B)]
        if len(dfB) == 0:
            out["success_rate"].append(np.nan)
            out["safety_efficiency"].append(np.nan)
            out["mean_success_len"].append(np.nan)
            out["mean_cost"].append(np.nan)
            continue

        goal_first = dfB["goal_first_step"].astype(np.int64).values
        is_succ = (goal_first != -1) & (goal_first <= int(B))
        out["success_rate"].append(float(is_succ.mean()))

        ep_len = dfB["ep_len"].astype(np.float64).values
        steps_sum = np.minimum(ep_len, float(B)).sum()

        cost_vals_B = cost_at_budget(dfB, int(B))
        cost_sum = float(np.sum(cost_vals_B))
        eff = (steps_sum / cost_sum) if cost_sum > 0 else np.inf
        out["safety_efficiency"].append(float(eff))

        if np.any(is_succ):
            mean_succ_len = float(dfB.loc[is_succ, "goal_first_step"].astype(np.float64).mean())
        else:
            mean_succ_len = np.nan
        out["mean_success_len"].append(mean_succ_len)

        out["mean_cost"].append(float(np.mean(cost_vals_B)))

    return out


def interpolate_nans(arr):
    """Linearly interpolate NaN values in a 1-D array using neighbouring valid points."""
    arr = arr.copy()
    nans = np.isnan(arr)
    if not np.any(nans) or np.all(nans):
        return arr
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def merge_runs_mean_std(run_series_list):
    keys = list(run_series_list[0].keys())
    merged = {}
    for k in keys:
        arr = np.vstack([np.asarray(r[k], dtype=np.float64) for r in run_series_list])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        # Fill any budget points missing from the data via linear interpolation
        mean = interpolate_nans(mean)
        std = interpolate_nans(std)
        merged[k] = (mean, std)
    return merged


def compute_series_flat(df, budgets):
    """Like compute_series_for_run but without budget-column grouping.

    All rows in df are used for every budget point B.
    Metrics are evaluated at each B using cost_cum_B and goal_first_step.
    """
    required = {"ep_len", "goal_first_step"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise KeyError("CSV missing required columns: {}".format(missing))

    out = {
        "success_rate": [],
        "safety_efficiency": [],
        "mean_success_len": [],
        "mean_cost": [],
    }

    goal_first_all = df["goal_first_step"].astype(np.int64).values
    ep_len_all = df["ep_len"].astype(np.float64).values

    for B in budgets:
        is_succ = (goal_first_all != -1) & (goal_first_all <= int(B))
        out["success_rate"].append(float(is_succ.mean()))

        steps_sum = np.minimum(ep_len_all, float(B)).sum()
        cost_vals_B = cost_at_budget(df, int(B))
        cost_sum = float(np.sum(cost_vals_B))
        eff = (steps_sum / cost_sum) if cost_sum > 0 else np.inf
        out["safety_efficiency"].append(float(eff))

        if np.any(is_succ):
            mean_succ_len = float(df.loc[is_succ, "goal_first_step"].astype(np.float64).mean())
        else:
            mean_succ_len = np.nan
        out["mean_success_len"].append(mean_succ_len)
        out["mean_cost"].append(float(np.mean(cost_vals_B)))

    return out


def load_flat_agent_runs(csv_paths, budgets, agent_filter):
    """Load and merge runs for a flat baseline (no budget column in CSV).

    Uses compute_series_flat so that metrics are evaluated at each budget
    point B from the fixed-horizon episode data.
    """
    run_series_list = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if agent_filter != "":
            if "agent" not in df.columns:
                raise KeyError(
                    "agent_filter set but CSV '{}' has no 'agent' column.".format(path)
                )
            df = df[df["agent"].astype(str) == str(agent_filter)]
        run_series_list.append(compute_series_flat(df, budgets))
    return merge_runs_mean_std(run_series_list)


def load_agent_runs(csv_paths, budgets, budget_col, agent_filter):
    """Load and merge multiple runs for a single agent.

    csv_paths: list of CSV file paths for this agent's runs.
    Returns (mean, std) dict from merge_runs_mean_std.
    """
    run_series_list = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if budget_col not in df.columns:
            raise KeyError("CSV '{}' does not contain budget column '{}'.".format(path, budget_col))
        if agent_filter != "":
            if "agent" not in df.columns:
                raise KeyError("agent_filter set but CSV '{}' has no 'agent' column.".format(path))
            df = df[df["agent"].astype(str) == str(agent_filter)]
        run_series_list.append(compute_series_for_run(df, budgets, budget_col))
    return merge_runs_mean_std(run_series_list)


def main():
    p = argparse.ArgumentParser()

    # Agent 1
    p.add_argument("--agent1_csvs", nargs="+", required=True,
                   help="One or more CSV files for agent 1 (treated as separate runs).")
    p.add_argument("--agent1_label", default="Agent 1")

    # Agent 2
    p.add_argument("--agent2_csvs", nargs="+", required=True,
                   help="One or more CSV files for agent 2 (treated as separate runs).")
    p.add_argument("--agent2_label", default="Agent 2")

    # Agent 3
    p.add_argument("--agent3_csvs", nargs="+", required=True,
                   help="One or more CSV files for agent 3 (treated as separate runs).")
    p.add_argument("--agent3_label", default="Agent 3")

    # Agent 4 (optional)
    p.add_argument("--agent4_csvs", nargs="+", default=None,
                   help="One or more CSV files for agent 4 (optional).")
    p.add_argument("--agent4_label", default="Agent 4")

    # Agent 5 (optional)
    p.add_argument("--agent5_csvs", nargs="+", default=None,
                   help="One or more CSV files for agent 5 (optional).")
    p.add_argument("--agent5_label", default="Agent 5")

    # Aggressive baseline (optional, flat / fixed-horizon CSV)
    p.add_argument("--baseline_aggressive_csvs", nargs="+", default=None,
                   help="One or more CSV files for the aggressive baseline (optional). "
                        "These CSVs are expected to NOT have a budget column.")
    p.add_argument("--baseline_aggressive_label", default="Aggressive baseline")
    p.add_argument("--baseline_aggressive_agent_filter", default="agent2",
                   help="Value of the 'agent' column to keep from the aggressive-baseline CSVs "
                        "(default: agent2).")

    p.add_argument("--budget_min", type=int, default=130)
    p.add_argument("--budget_max", type=int, default=200)
    p.add_argument("--budget_step", type=int, default=5)

    p.add_argument("--budget_col", default=None)
    p.add_argument("--out_dir", default="plots_timeaware")
    p.add_argument("--out_prefix", default="timeaware")

    # Keep this for compatibility, but default to no filtering since your CSV seems time-aware only.
    p.add_argument("--agent_filter", default="",
                   help="If non-empty, keep only rows where agent == this value.")
    p.add_argument("--agent_labels", nargs="+", default=None, metavar="NAME=LABEL",
                   help="Rename agents in the legend. Pass pairs like: "
                        "'Agent 1=Conservative baseline' 'Agent 2=Penalty 500'.")

    args = p.parse_args()

    budgets = list(range(args.budget_min, args.budget_max + 1, args.budget_step))
    ensure_dir(args.out_dir)

    # Use the first CSV of agent 1 to infer the budget column
    df0 = pd.read_csv(args.agent1_csvs[0])
    budget_col = infer_budget_col(df0, args.budget_col)

    agents = [
        (args.agent1_csvs, args.agent1_label),
        (args.agent2_csvs, args.agent2_label),
        (args.agent3_csvs, args.agent3_label),
    ]
    if args.agent4_csvs:
        agents.append((args.agent4_csvs, args.agent4_label))
    if args.agent5_csvs:
        agents.append((args.agent5_csvs, args.agent5_label))

    merged_per_agent = []
    for csv_paths, label in agents:
        merged = load_agent_runs(csv_paths, budgets, budget_col, args.agent_filter)
        merged_per_agent.append((merged, label))

    if args.baseline_aggressive_csvs:
        merged_agg = load_agent_runs(
            args.baseline_aggressive_csvs, budgets, budget_col, args.agent_filter,
        )
        merged_per_agent.append((merged_agg, args.baseline_aggressive_label))

    # Build label remap dict from --agent_labels NAME=LABEL pairs
    label_map = {}
    if args.agent_labels:
        for entry in args.agent_labels:
            eq = entry.rindex("=")
            key, val = entry[:eq], entry[eq + 1:]
            label_map[key] = val

    base = os.path.join(args.out_dir, args.out_prefix)

    metrics = [
        ("success_rate",     "Success rate",               "Success rate vs time budget",               None, None),
        # ("safety_efficiency","Timesteps / Cost",            "Safety efficiency vs time budget",          None, None),
        ("mean_success_len", "Mean success episode length", "Mean success episode length vs time budget", None, None),
        ("mean_cost",        "Mean cost at budget",         "Mean cost vs time budget",                  15,   "Safety budget (15)"),
    ]

    suffix_map = {
        "success_rate":      "success_rate",
        "safety_efficiency": "timesteps_per_cost",
        "mean_success_len":  "mean_success_len",
        "mean_cost":         "mean_cost",
    }

    for key, ylabel, title, hline, hline_label in metrics:
        agents_data = [(m[key][0], m[key][1], label_map.get(lbl, lbl)) for m, lbl in merged_per_agent]
        plot_metric_comparison(
            budgets, agents_data,
            ylabel=ylabel,
            title=title,
            out_path="{}_{}.png".format(base, suffix_map[key]),
            hline=hline,
            hline_label=hline_label,
        )

    print("Saved plots to:")
    print("  {}/".format(args.out_dir))


if __name__ == "__main__":
    main()
