#!/usr/bin/env python3
"""
cvar_exp1and2.py

Compute CVaR per agent from evaluation CSV files.

For each agent:
    - n_episodes
    - mean_cost
    - cvar_cost
    - success_rate

If multiple CSVs are passed, they are treated as independent runs and
aggregated with mean ± SE or mean ± std.

This version generates only the CVaR plot.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _penalty_sort_key(name: str):
    m = re.search(r"p(\d+)", str(name))
    if m:
        return (1, int(m.group(1)), str(name))
    return (0, 0, str(name))


def apply_conservative_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "conservative" not in df["agent"].astype(str).values:
        print("WARNING: agent 'conservative' not found - conservative filter skipped.")
        return df

    if "success" in df.columns:
        cons_ok = (
            (df["agent"].astype(str) == "conservative")
            & (pd.to_numeric(df["success"], errors="coerce") == 1)
        )
    elif "goal_first_step" in df.columns and "budget" in df.columns:
        gf = pd.to_numeric(df["goal_first_step"], errors="coerce")
        bud = pd.to_numeric(df["budget"], errors="coerce")
        cons_ok = (
            (df["agent"].astype(str) == "conservative")
            & (gf != -1)
            & (gf <= bud)
        )
    else:
        print("WARNING: unable to determine success - conservative filter skipped.")
        return df

    key_cols = ["_file_idx", "episode_idx", "budget"]
    success_keys = set(
        map(tuple, df.loc[cons_ok, key_cols].drop_duplicates().values.tolist())
    )

    n_total_cons = int((df["agent"].astype(str) == "conservative").sum())
    print(
        f"Conservative filter: {len(success_keys)} / {n_total_cons} conservative episodes kept "
        f"({100 * len(success_keys) / n_total_cons:.1f}% of conservative episodes)."
    )

    mask = df[key_cols].apply(lambda r: tuple(r) in success_keys, axis=1)
    filtered = df[mask].copy()
    print(f"  Rows before filter: {len(df)} → after: {len(filtered)}")
    return filtered


def cvar_worst_mean(costs: np.ndarray, alpha: float) -> float:
    costs = np.asarray(costs, dtype=np.float64)
    costs = costs[np.isfinite(costs)]

    n = costs.size
    if n == 0:
        return float("nan")

    k = max(int(np.ceil(alpha * n)), 1)
    print(f"  CVaR: N={n}, alpha={alpha}, k={k}")

    return float(np.mean(np.sort(costs)[-k:]))


def compute_per_agent(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if "agent" not in df.columns:
        raise KeyError("CSV is missing the 'agent' column.")

    rows = []

    for agent, g in df.groupby("agent", sort=True):
        if "cost_total" not in g.columns:
            raise KeyError(
                f"Agent '{agent}': 'cost_total' column not found. "
                f"Available columns: {list(g.columns)}"
            )

        costs = pd.to_numeric(g["cost_total"], errors="coerce").astype(np.float64).values
        costs = costs[np.isfinite(costs)]

        if "goal_first_step" in g.columns:
            gf = pd.to_numeric(g["goal_first_step"], errors="coerce").astype(np.float64).values
            success_rate = float(np.mean(gf != -1)) if gf.size else float("nan")
        elif "success" in g.columns:
            success_rate = float(pd.to_numeric(g["success"], errors="coerce").mean())
        else:
            success_rate = float("nan")

        rows.append(
            {
                "agent": agent,
                "n_episodes": int(costs.size),
                "mean_cost": float(np.mean(costs)) if costs.size else float("nan"),
                "cvar_cost": cvar_worst_mean(costs, alpha),
                "success_rate": success_rate,
            }
        )

    return pd.DataFrame(rows)


def aggregate_runs(run_dfs: list, band_mode: str) -> pd.DataFrame:
    all_df = pd.concat(
        [df.assign(run_id=i) for i, df in enumerate(run_dfs)],
        ignore_index=True,
    )

    def _band(arr):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if arr.size <= 1:
            return 0.0

        std = float(np.std(arr, ddof=1))
        return std if band_mode == "std" else std / float(np.sqrt(arr.size))

    out_rows = []

    for agent, g in all_df.groupby("agent", sort=True):
        mc = pd.to_numeric(g["mean_cost"], errors="coerce").astype(np.float64).values
        cv = pd.to_numeric(g["cvar_cost"], errors="coerce").astype(np.float64).values
        sr = pd.to_numeric(g["success_rate"], errors="coerce").astype(np.float64).values
        ne = pd.to_numeric(g["n_episodes"], errors="coerce").astype(np.float64).values

        out_rows.append(
            {
                "agent": agent,
                "n_runs": int(g["run_id"].nunique()),
                "n_episodes_mean": float(np.nanmean(ne)),
                "mean_cost_mean": float(np.nanmean(mc)),
                "mean_cost_band": _band(mc),
                "cvar_cost_mean": float(np.nanmean(cv)),
                "cvar_cost_band": _band(cv),
                "success_rate_mean": float(np.nanmean(sr)),
                "success_rate_band": _band(sr),
            }
        )

    result = pd.DataFrame(out_rows)
    result = result.iloc[
        sorted(range(len(result)), key=lambda i: _penalty_sort_key(result.iloc[i]["agent"]))
    ]
    return result.reset_index(drop=True)


def plot_bar(
    agg_df: pd.DataFrame,
    metric_mean: str,
    metric_band: str,
    ylabel: str,
    title: str,
    out_path: str,
    label_map: dict = None,
    hline: float = None,
    hline_label: str = None,
):
    import matplotlib.ticker as mticker

    agents = agg_df["agent"].tolist()
    labels = [label_map.get(a, a) if label_map else a for a in agents]

    means = agg_df[metric_mean].astype(float).values
    bands = agg_df[metric_band].astype(float).values

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    agent_colors = [colors[i % len(colors)] for i in range(len(agents))]

    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 0.9), 5))
    x = np.arange(len(agents))

    ax.bar(
        x,
        means,
        0.65,
        yerr=bands,
        color=agent_colors,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
        edgecolor="white",
        linewidth=0.5,
    )

    if hline is not None:
        lbl = hline_label if hline_label else f"Cost Limit = {hline:.4g}"
        ax.axhline(
            hline,
            color="red",
            linestyle="--",
            linewidth=1.4,
            label=lbl,
            zorder=5,
        )
        ax.legend(fontsize=13, loc="best")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved plot: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Compute CVaR per agent and generate only the CVaR plot."
    )

    p.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="One or more CSV files. Each CSV is treated as an independent run/seed.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Tail probability for CVaR. Default: 0.1 = worst 10%%.",
    )
    p.add_argument(
        "--band",
        choices=["se", "std"],
        default="se",
        help="Uncertainty band: standard error or standard deviation.",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Subset of agents to include.",
    )
    p.add_argument(
        "--agent_labels",
        nargs="+",
        default=None,
        metavar="NAME=LABEL",
        help="Rename agents in plot. Example: conservative=Conservative",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="results/cvar_exp.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If specified, the CVaR plot is saved here.",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="cvar",
        help="Prefix for the CVaR plot file.",
    )
    p.add_argument(
        "--cost_hline",
        type=float,
        default=None,
        help="Optional horizontal safety-cost limit line.",
    )
    p.add_argument(
        "--cost_hline_label",
        type=str,
        default=None,
        help="Label for the horizontal safety-cost line.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title.",
    )
    p.add_argument(
        "--conservative_filter",
        action="store_true",
        help=(
            "Keep only episodes where the conservative agent reached the goal. "
            "All other agents are evaluated only on those shared episodes."
        ),
    )

    args = p.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")

    label_map = {}
    if args.agent_labels:
        for entry in args.agent_labels:
            eq = entry.rindex("=")
            label_map[entry[:eq]] = entry[eq + 1:]

    print(f"Loading {len(args.csvs)} CSV(s)...")

    run_dfs_per_agent = {}

    for file_idx, path in enumerate(args.csvs):
        try:
            df = pd.read_csv(path)
            df["_file_idx"] = file_idx
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            continue

        if args.conservative_filter:
            df = apply_conservative_filter(df)
            if df.empty:
                print(f"  Skipping {path}: no episodes left after conservative filter.")
                continue

        if args.agents:
            df = df[df["agent"].astype(str).isin(args.agents)]

        per_agent_df = compute_per_agent(df, args.alpha)

        for _, row in per_agent_df.iterrows():
            run_dfs_per_agent.setdefault(row["agent"], []).append(pd.DataFrame([row]))

    if not run_dfs_per_agent:
        raise SystemExit("No data loaded. Check --csvs and --agents.")

    n_runs = max(len(v) for v in run_dfs_per_agent.values())
    run_list = []

    for i in range(n_runs):
        rows = []
        for agent, dfs in run_dfs_per_agent.items():
            if i < len(dfs):
                rows.append(dfs[i])

        if rows:
            run_list.append(pd.concat(rows, ignore_index=True))

    agg = aggregate_runs(run_list, band_mode=args.band)

    if args.agents:
        available_agents = agg["agent"].astype(str).tolist()
        desired_order = [a for a in args.agents if a in available_agents]
        missing = [a for a in args.agents if a not in available_agents]

        if missing:
            print(f"WARNING: requested agents not found: {missing}")

        remaining = [a for a in available_agents if a not in desired_order]
        agg = agg.set_index("agent").loc[desired_order + remaining].reset_index()

    print(f"\nResults (alpha={args.alpha}, band={args.band}):\n")
    print(agg.to_string(index=False))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    agg.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV: {args.out_csv}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.join(args.out_dir, args.out_prefix)

        plot_df = agg

        plot_bar(plot_df, "mean_cost_mean", "mean_cost_band",
                 ylabel="Mean cost",
                 title=args.title,
                 out_path=f"{base}_mean_cost.png",
                 label_map=label_map,
                 hline=args.cost_hline, hline_label=args.cost_hline_label)

        plot_bar(plot_df, "cvar_cost_mean", "cvar_cost_band",
                 ylabel=f"CVaR_{args.alpha} (cost)",
                 title=args.title,
                 out_path=f"{base}_cvar_cost.png",
                 label_map=label_map,
                 hline=args.cost_hline, hline_label=args.cost_hline_label)

        plot_bar(plot_df, "success_rate_mean", "success_rate_band",
                 ylabel="Success rate",
                 title=args.title,
                 out_path=f"{base}_success_rate.png",
                 label_map=label_map)


        print(f"\nPlots saved in: {args.out_dir}/")


if __name__ == "__main__":
    main()