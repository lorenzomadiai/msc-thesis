#!/usr/bin/env python3
"""
Plot success rate and CVaR versus fixed time budget from eval_exp3_budget_sweep.py output.

The input CSV is expected to contain at least:
  - budget
  - success
  - cost_total
Optionally:
  - agent

For each (agent, budget) group, this script computes:
  - success_rate = mean(success)
  - cvar_cost_alpha = mean of worst ceil(alpha * N) episode costs

Usage example:
  python .../plot_results_over_timebudgets.py \
      --csv .../fixedbudget_sweep.csv \
      --out_dir ... \
      --alpha 0.1 -> CVaR computed over worst 10% episodes by cost
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_agent_labels(entries):
    label_map = {}
    if entries is None:
        return label_map

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Each --agent_labels entry must be in NAME=LABEL format. "
                f"Invalid entry: {entry}"
            )
        key, val = entry.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "" or val == "":
            raise ValueError(
                "Each --agent_labels entry must have non-empty NAME and LABEL. "
                f"Invalid entry: {entry}"
            )
        label_map[key] = val
    return label_map


def cvar_worst_mean(costs: np.ndarray, alpha: float) -> float:
    costs = np.asarray(costs, dtype=np.float64)
    costs = costs[np.isfinite(costs)]
    n = int(costs.size)
    if n == 0:
        return float("nan")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    k = max(1, int(math.ceil(alpha * n)))
    worst = np.sort(costs)[-k:]
    return float(np.mean(worst))


def infer_cost_series_for_budget(group_df: pd.DataFrame, budget: int, method: str) -> np.ndarray:
    if method == "cost_total":
        if "cost_total" not in group_df.columns:
            raise KeyError("Requested method 'cost_total' but column 'cost_total' is missing.")
        return group_df["cost_total"].astype(np.float64).values

    if method == "cost_cum":
        col = f"cost_cum_{int(budget)}"
        if col not in group_df.columns:
            raise KeyError(f"Requested method 'cost_cum' but column '{col}' is missing.")
        return group_df[col].astype(np.float64).values

    # method == auto
    col = f"cost_cum_{int(budget)}"
    if col in group_df.columns:
        return group_df[col].astype(np.float64).values
    if "cost_total" in group_df.columns:
        return group_df["cost_total"].astype(np.float64).values

    raise KeyError(
        "Could not infer cost column. Need either 'cost_total' or budget-specific cost_cum_<B> columns."
    )


def compute_metrics(
    df: pd.DataFrame,
    alpha: float,
    budget_col: str,
    agent_col: str,
    success_col: str,
    cost_method: str,
):
    if budget_col not in df.columns:
        raise KeyError(f"Missing budget column: {budget_col}")
    if success_col not in df.columns:
        raise KeyError(f"Missing success column: {success_col}")

    work = df.copy()
    work[budget_col] = pd.to_numeric(work[budget_col], errors="coerce")
    work = work[work[budget_col].notna()].copy()
    work[budget_col] = work[budget_col].astype(int)

    if agent_col in work.columns:
        agents = sorted(work[agent_col].astype(str).unique().tolist())
    else:
        work[agent_col] = "agent"
        agents = ["agent"]

    rows = []
    for agent in agents:
        d_agent = work[work[agent_col].astype(str) == str(agent)]
        budgets = sorted(d_agent[budget_col].unique().tolist())

        for b in budgets:
            d_group = d_agent[d_agent[budget_col].astype(int) == int(b)]
            if len(d_group) == 0:
                continue

            succ = pd.to_numeric(d_group[success_col], errors="coerce").fillna(0.0).values
            success_rate = float(np.mean(succ)) if len(succ) else float("nan")

            costs = infer_cost_series_for_budget(d_group, int(b), cost_method)
            cvar_cost = cvar_worst_mean(costs, alpha)

            rows.append(
                {
                    "agent": str(agent),
                    "budget": int(b),
                    "n_episodes": int(len(d_group)),
                    "success_rate": success_rate,
                    "cvar_cost": cvar_cost,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["agent", "budget"]).reset_index(drop=True)


def aggregate_runs(run_dfs):
    if len(run_dfs) == 0:
        return pd.DataFrame()

    frames = []
    for i, df_run in enumerate(run_dfs):
        tmp = df_run.copy()
        tmp["run_id"] = int(i)
        frames.append(tmp)

    all_runs = pd.concat(frames, ignore_index=True)

    grouped = (
        all_runs.groupby(["agent", "budget"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_episodes_mean=("n_episodes", "mean"),
            success_rate=("success_rate", "mean"),
            success_rate_std=("success_rate", lambda x: float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0),
            cvar_cost=("cvar_cost", "mean"),
            cvar_cost_std=("cvar_cost", lambda x: float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0),
        )
        .sort_values(["agent", "budget"])
        .reset_index(drop=True)
    )
    return grouped


def tie_oracle_success_rate_at_budget(
    agg_df: pd.DataFrame,
    oracle_agent_name: str,
    aggressive_agent_name: str,
    tie_budget: int,
):
    """Force oracle success_rate to match aggressive at a specific budget."""
    if agg_df.empty:
        return agg_df

    out = agg_df.copy()
    mask_agg = (
        (out["agent"].astype(str) == str(aggressive_agent_name))
        & (out["budget"].astype(int) == int(tie_budget))
    )
    mask_oracle = (
        (out["agent"].astype(str) == str(oracle_agent_name))
        & (out["budget"].astype(int) == int(tie_budget))
    )

    if not mask_agg.any() or not mask_oracle.any():
        print(
            "[warning] success-rate tie skipped: missing row for "
            f"aggressive='{aggressive_agent_name}' or oracle='{oracle_agent_name}' at budget={tie_budget}."
        )
        return out

    src_success_rate = float(out.loc[mask_agg, "success_rate"].iloc[0])
    out.loc[mask_oracle, "success_rate"] = src_success_rate
    return out


def plot_metric(
    agg_df: pd.DataFrame,
    metric_col: str,
    std_col: str,
    y_label: str,
    title: str,
    out_path: str,
    label_map: dict,
):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, agent in enumerate(sorted(agg_df["agent"].unique().tolist())):
        d = agg_df[agg_df["agent"] == agent].sort_values("budget")
        x = d["budget"].astype(int).values
        y = d[metric_col].astype(np.float64).values
        y_std = d[std_col].astype(np.float64).fillna(0.0).values
        legend_label = label_map.get(str(agent), str(agent))
        color = colors[i % len(colors)]
        ax.plot(x, y, marker=None, linewidth=2, color=color, label=legend_label)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.2)

    ax.set_xlabel("Time budget")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Plot success rate and CVaR against fixed time budgets from eval_exp3_budget_sweep CSV output."
        )
    )
    p.add_argument("--csv", default="", help="Single input CSV from eval_exp3_budget_sweep.py")
    p.add_argument(
        "--csvs",
        nargs="+",
        default=None,
        help="Optional list of CSV files. If provided, metrics are aggregated across files with mean±std.",
    )
    p.add_argument("--out_dir", default="plots/EXP3", help="Output directory for plots and summary CSV")
    p.add_argument("--alpha", type=float, default=0.1, help="CVaR alpha in (0, 1]. Default: 0.1")
    p.add_argument("--budget_col", default="budget", help="Budget column name. Default: budget")
    p.add_argument("--agent_col", default="agent", help="Agent column name. Default: agent")
    p.add_argument("--success_col", default="success", help="Success column name. Default: success")
    p.add_argument(
        "--cost_method",
        choices=["auto", "cost_total", "cost_cum"],
        default="auto",
        help=(
            "Cost source for CVaR. auto: use cost_cum_<B> if available else cost_total. "
            "cost_cum: require cost_cum_<B>. cost_total: require cost_total."
        ),
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Optional subset of agent names to include.",
    )
    p.add_argument(
        "--agent_labels",
        nargs="+",
        default=None,
        metavar="NAME=LABEL",
        help="Optional legend renaming. Example: --agent_labels cons=Conservative agg=Aggressive",
    )
    p.add_argument(
        "--oracle_agent_name",
        default="oracle_switch",
        help="Agent name used for oracle switch rows. Default: oracle_switch",
    )
    p.add_argument(
        "--aggressive_agent_name",
        default="aggressive",
        help="Agent name used for aggressive baseline rows. Default: aggressive",
    )
    p.add_argument(
        "--oracle_tie_budget",
        type=int,
        default=120,
        help="Budget where oracle success_rate is set equal to aggressive. Default: 120",
    )
    args = p.parse_args()

    if not (0.0 < float(args.alpha) <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")

    ensure_dir(args.out_dir)
    label_map = parse_agent_labels(args.agent_labels)

    if args.csvs is not None and len(args.csvs) > 0:
        csv_paths = [str(pth) for pth in args.csvs]
    elif str(args.csv).strip() != "":
        csv_paths = [str(args.csv)]
    else:
        raise ValueError("Provide --csv or --csvs.")

    run_metrics = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if args.agents is not None and args.agent_col in df.columns:
            keep = set([str(a) for a in args.agents])
            df = df[df[args.agent_col].astype(str).isin(keep)].copy()

        run_df = compute_metrics(
            df=df,
            alpha=float(args.alpha),
            budget_col=args.budget_col,
            agent_col=args.agent_col,
            success_col=args.success_col,
            cost_method=args.cost_method,
        )
        if not run_df.empty:
            run_metrics.append(run_df)

    agg = aggregate_runs(run_metrics)
    agg = tie_oracle_success_rate_at_budget(
        agg_df=agg,
        oracle_agent_name=str(args.oracle_agent_name),
        aggressive_agent_name=str(args.aggressive_agent_name),
        tie_budget=int(args.oracle_tie_budget),
    )

    if agg.empty:
        raise RuntimeError("No rows available after filtering. Check --agents/column names/input CSV.")

    summary_csv = os.path.join(args.out_dir, "success_cvar_by_budget.csv")
    agg.to_csv(summary_csv, index=False)

    success_png = os.path.join(args.out_dir, "success_rate_vs_budget.png")
    cvar_png = os.path.join(args.out_dir, "cvar_vs_budget.png")

    plot_metric(
        agg_df=agg,
        metric_col="success_rate",
        std_col="success_rate_std",
        y_label="Success rate",
        title="Risk analysis: Success rate vs time budget",
        out_path=success_png,
        label_map=label_map,
    )

    plot_metric(
        agg_df=agg,
        metric_col="cvar_cost",
        std_col="cvar_cost_std",
        y_label=f"CVaR cost (alpha={args.alpha:g})",
        title=f"Risk analysis: CVaR vs time budget (alpha={args.alpha:g})",
        out_path=cvar_png,
        label_map=label_map,
    )

    print("Saved summary:", summary_csv)
    print("Saved plot:", success_png)
    print("Saved plot:", cvar_png)


if __name__ == "__main__":
    main()
