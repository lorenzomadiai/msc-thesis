#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np


def cvar_worst_mean(costs, alpha: float) -> float:
    """
    Empirical CVaR:
      CVaR_alpha = mean of the worst ceil(N*alpha) episode costs.
    """
    costs = np.asarray(costs, dtype=np.float64)
    costs = costs[np.isfinite(costs)]
    N = costs.size
    if N == 0:
        return float("nan")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    k = int(np.ceil(alpha * N))
    k = max(k, 1)
    worst = np.sort(costs)[-k:]  # largest costs
    return float(np.mean(worst))


def require_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}' in CSV.")


def _budgets_list(bmin: int, bmax: int, bstep: int):
    return list(range(int(bmin), int(bmax) + 1, int(bstep)))


def compute_rows_static(dfS: pd.DataFrame, budgets, alpha: float):
    """
    Static/non-time-aware:
      one row per episode, columns: agent, ep_len, cost_cum_B (for each B).
    Returns per-run rows: agent,budget,n_episodes,mean_cost,cvar_cost
    """
    require_col(dfS, "agent")
    require_col(dfS, "ep_len")  # kept as a sanity requirement

    rows = []
    for agent in sorted(dfS["agent"].unique()):
        dA = dfS[dfS["agent"] == agent]

        for B in budgets:
            col = f"cost_cum_{B}"
            require_col(dA, col)

            costs = dA[col].astype(np.float64).values
            rows.append({
                "agent": str(agent),
                "budget": int(B),
                "n_episodes": int(len(dA)),
                "mean_cost": float(np.mean(costs)) if len(costs) else float("nan"),
                "cvar_cost": cvar_worst_mean(costs, alpha),
            })
    return rows


def compute_rows_timeaware(dfT: pd.DataFrame, budgets, alpha: float, budget_col: str, agent_filter: str = ""):
    """
    Time-aware:
      filter by budget_col==B, compute CVaR on cost_cum_B.
    Returns per-run rows: agent,budget,n_episodes,mean_cost,cvar_cost
    """
    require_col(dfT, "agent")
    require_col(dfT, budget_col)

    if agent_filter != "":
        dfT = dfT[dfT["agent"].astype(str) == str(agent_filter)]

    rows = []
    for agent in sorted(dfT["agent"].unique()):
        dA = dfT[dfT["agent"] == agent]

        for B in budgets:
            dB = dA[dA[budget_col].astype(int) == int(B)]
            if len(dB) == 0:
                rows.append({
                    "agent": str(agent),
                    "budget": int(B),
                    "n_episodes": 0,
                    "mean_cost": float("nan"),
                    "cvar_cost": float("nan"),
                })
                continue

            col = f"cost_cum_{B}"
            require_col(dB, col)

            costs = dB[col].astype(np.float64).values
            rows.append({
                "agent": str(agent),
                "budget": int(B),
                "n_episodes": int(len(dB)),
                "mean_cost": float(np.mean(costs)),
                "cvar_cost": cvar_worst_mean(costs, alpha),
            })
    return rows


def merge_uncertainty(per_run_rows_list, band_mode: str):
    """
    Input: list of DataFrames (one per run) with columns:
      agent,budget,n_episodes,mean_cost,cvar_cost

    Output: DataFrame with aggregated stats per agent,budget:
      n_runs_present, n_episodes_mean,
      mean_cost_mean, mean_cost_band,
      cvar_cost_mean, cvar_cost_band
    """
    if len(per_run_rows_list) == 0:
        return pd.DataFrame()

    # concat with run id
    frames = []
    for i, df in enumerate(per_run_rows_list):
        tmp = df.copy()
        tmp["run_id"] = i
        frames.append(tmp)
    all_df = pd.concat(frames, ignore_index=True)

    # group
    out_rows = []
    for (agent, budget), g in all_df.groupby(["agent", "budget"], sort=True):
        # values per run (some runs may have NaN)
        mean_cost_vals = g["mean_cost"].astype(np.float64).values
        cvar_cost_vals = g["cvar_cost"].astype(np.float64).values
        n_ep_vals = g["n_episodes"].astype(np.float64).values

        # consider only finite entries for stats
        mc = mean_cost_vals[np.isfinite(mean_cost_vals)]
        cv = cvar_cost_vals[np.isfinite(cvar_cost_vals)]

        n_runs_mc = int(mc.size)
        n_runs_cv = int(cv.size)

        mean_cost_mean = float(np.nanmean(mean_cost_vals)) if n_runs_mc > 0 else float("nan")
        cvar_cost_mean = float(np.nanmean(cvar_cost_vals)) if n_runs_cv > 0 else float("nan")

        # ddof=1 only if at least 2 runs
        def _band(arr):
            arr = np.asarray(arr, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 1:
                return 0.0
            std = float(np.std(arr, ddof=1))
            if band_mode == "std":
                return std
            # se
            return std / float(np.sqrt(arr.size))

        mean_cost_band = _band(mean_cost_vals)
        cvar_cost_band = _band(cvar_cost_vals)

        n_ep_mean = float(np.nanmean(n_ep_vals)) if np.isfinite(n_ep_vals).any() else float("nan")

        out_rows.append({
            "agent": agent,
            "budget": int(budget),
            "n_runs_present_mean_cost": n_runs_mc,
            "n_runs_present_cvar_cost": n_runs_cv,
            "n_episodes_mean": n_ep_mean,
            "mean_cost_mean": mean_cost_mean,
            "mean_cost_band": mean_cost_band,
            "cvar_cost_mean": cvar_cost_mean,
            "cvar_cost_band": cvar_cost_band,
        })

    return pd.DataFrame(out_rows).sort_values(["agent", "budget"]).reset_index(drop=True)

def add_budget_summary(out_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one summary row per (source, agent) aggregating across all budgets.

    Summary columns are computed as:
      - *_mean columns: mean across budgets (ignoring NaNs)
      - *_band columns: mean across budgets (ignoring NaNs)
      - n_episodes_mean: mean across budgets
      - n_runs_present_*: minimum across budgets (conservative; ensures "present for all")
    The summary row has budget='ALL' (string).
    """
    if out_df.empty:
        return out_df

    required = ["source", "agent", "budget",
                "mean_cost_mean", "mean_cost_band",
                "cvar_cost_mean", "cvar_cost_band",
                "n_episodes_mean",
                "n_runs_present_mean_cost", "n_runs_present_cvar_cost"]
    for c in required:
        if c not in out_df.columns:
            raise KeyError(f"Missing column '{c}' needed for summary.")

    # Work on a copy, ensure budget is not used as numeric in summary
    df = out_df.copy()

    summary_rows = []
    for (src, agent), g in df.groupby(["source", "agent"], sort=True):
        # Convert to numeric where relevant
        mc_mean = pd.to_numeric(g["mean_cost_mean"], errors="coerce")
        mc_band = pd.to_numeric(g["mean_cost_band"], errors="coerce")
        cv_mean = pd.to_numeric(g["cvar_cost_mean"], errors="coerce")
        cv_band = pd.to_numeric(g["cvar_cost_band"], errors="coerce")
        ne_mean = pd.to_numeric(g["n_episodes_mean"], errors="coerce")

        # Conservative run counts across budgets (must be available everywhere)
        r_mc = pd.to_numeric(g["n_runs_present_mean_cost"], errors="coerce")
        r_cv = pd.to_numeric(g["n_runs_present_cvar_cost"], errors="coerce")

        summary_rows.append({
            "source": src,
            "agent": agent,
            "budget": "ALL",
            "n_runs_present_mean_cost": int(np.nanmin(r_mc.values)) if np.isfinite(r_mc.values).any() else 0,
            "n_runs_present_cvar_cost": int(np.nanmin(r_cv.values)) if np.isfinite(r_cv.values).any() else 0,
            "n_episodes_mean": float(np.nanmean(ne_mean.values)) if np.isfinite(ne_mean.values).any() else float("nan"),
            "mean_cost_mean": float(np.nanmean(mc_mean.values)) if np.isfinite(mc_mean.values).any() else float("nan"),
            "mean_cost_band": float(np.nanmean(mc_band.values)) if np.isfinite(mc_band.values).any() else float("nan"),
            "cvar_cost_mean": float(np.nanmean(cv_mean.values)) if np.isfinite(cv_mean.values).any() else float("nan"),
            "cvar_cost_band": float(np.nanmean(cv_band.values)) if np.isfinite(cv_band.values).any() else float("nan"),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Put summary at the end (or you can sort differently)
    combined = pd.concat([df, summary_df], ignore_index=True)

    # Sorting: keep numeric budgets ordered, then ALL at the end
    def _budget_sort_key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, 10**9)

    combined["_budget_sort"] = combined["budget"].map(_budget_sort_key)
    combined = combined.sort_values(["source", "agent", "_budget_sort"]).drop(columns=["_budget_sort"]).reset_index(drop=True)

    return combined


import os

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_worst_episodes_static(csv_paths, budgets, outdir, worst_k: int):
    """
    For each agent and budget B:
      take top-K rows by cost_cum_B across ALL provided static CSVs
      and save to: outdir/static/worstK_static_<agent>_B<B>.csv
    """
    _ensure_dir(outdir)
    out_static_dir = os.path.join(outdir, "static")
    _ensure_dir(out_static_dir)

    frames = []
    for run_id, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        if "agent" not in df.columns:
            raise KeyError(f"[static] Missing 'agent' in {path}")
        df = df.copy()
        df["run_id"] = run_id
        df["src_file"] = os.path.basename(path)
        df["row_id"] = np.arange(len(df), dtype=int)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    for agent in sorted(all_df["agent"].astype(str).unique()):
        dA = all_df[all_df["agent"].astype(str) == str(agent)]

        for B in budgets:
            col = f"cost_cum_{int(B)}"
            if col not in dA.columns:
                raise KeyError(f"[static] Missing column '{col}' (needed for budget {B}).")

            dB = dA.copy()
            dB[col] = pd.to_numeric(dB[col], errors="coerce")
            dB = dB[np.isfinite(dB[col].values)]

            if len(dB) == 0:
                continue

            worst = dB.sort_values(col, ascending=False).head(int(worst_k))
            # Add rank (1..K)
            worst = worst.copy()
            worst["worst_rank"] = np.arange(1, len(worst) + 1, dtype=int)

            safe_agent = str(agent).replace("/", "_")
            out_path = os.path.join(out_static_dir, f"worst{worst_k}_static_{safe_agent}_B{int(B)}.csv")
            worst.to_csv(out_path, index=False)


def save_worst_episodes_timeaware(csv_paths, budgets, budget_col, outdir, worst_k: int, agent_filter: str = ""):
    """
    For each agent and budget B:
      filter rows where budget_col==B across ALL provided time-aware CSVs,
      take top-K by cost_cum_B,
      save to: outdir/timeaware/worstK_timeaware_<agent>_B<B>.csv
    """
    _ensure_dir(outdir)
    out_ta_dir = os.path.join(outdir, "timeaware")
    _ensure_dir(out_ta_dir)

    frames = []
    for run_id, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        if "agent" not in df.columns:
            raise KeyError(f"[timeaware] Missing 'agent' in {path}")
        if budget_col not in df.columns:
            raise KeyError(f"[timeaware] Missing budget_col '{budget_col}' in {path}")

        df = df.copy()
        df["run_id"] = run_id
        df["src_file"] = os.path.basename(path)
        df["row_id"] = np.arange(len(df), dtype=int)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    if agent_filter != "":
        all_df = all_df[all_df["agent"].astype(str) == str(agent_filter)]

    for agent in sorted(all_df["agent"].astype(str).unique()):
        dA = all_df[all_df["agent"].astype(str) == str(agent)].copy()

        for B in budgets:
            dBud = dA[pd.to_numeric(dA[budget_col], errors="coerce").astype("Int64") == int(B)].copy()
            if len(dBud) == 0:
                continue

            col = f"cost_cum_{int(B)}"
            if col not in dBud.columns:
                raise KeyError(f"[timeaware] Missing column '{col}' (needed for budget {B}).")

            dBud[col] = pd.to_numeric(dBud[col], errors="coerce")
            dBud = dBud[np.isfinite(dBud[col].values)]
            if len(dBud) == 0:
                continue

            worst = dBud.sort_values(col, ascending=False).head(int(worst_k))
            worst = worst.copy()
            worst["worst_rank"] = np.arange(1, len(worst) + 1, dtype=int)

            safe_agent = str(agent).replace("/", "_")
            out_path = os.path.join(out_ta_dir, f"worst{worst_k}_timeaware_{safe_agent}_B{int(B)}.csv")
            worst.to_csv(out_path, index=False)


def compute_cvar_for_run(df: pd.DataFrame, budgets, budget_col: str, alpha: float, agent_label: str):
    """
    Compute mean_cost and cvar_cost per budget for a single run CSV,
    where the agent identity is given externally (no 'agent' column required).
    Returns a list of dicts: agent, budget, n_episodes, mean_cost, cvar_cost.
    """
    require_col(df, budget_col)

    rows = []
    for B in budgets:
        dB = df[pd.to_numeric(df[budget_col], errors="coerce").astype("Int64") == int(B)]
        if len(dB) == 0:
            rows.append({
                "agent": agent_label,
                "budget": int(B),
                "n_episodes": 0,
                "mean_cost": float("nan"),
                "cvar_cost": float("nan"),
            })
            continue

        col = f"cost_cum_{int(B)}"
        require_col(dB, col)
        costs = pd.to_numeric(dB[col], errors="coerce").astype(np.float64).values
        costs = costs[np.isfinite(costs)]

        rows.append({
            "agent": agent_label,
            "budget": int(B),
            "n_episodes": int(len(costs)),
            "mean_cost": float(np.mean(costs)) if len(costs) else float("nan"),
            "cvar_cost": cvar_worst_mean(costs, alpha),
        })
    return rows


def main():
    p = argparse.ArgumentParser()

    # ---- 4-agent mode (one CSV group per agent, label provided via CLI) ----
    p.add_argument("--agent1_csvs", nargs="+", default=None,
                   help="CSV files for agent 1 (one per seed/run).")
    p.add_argument("--agent1_label", default="Agent 1")

    p.add_argument("--agent2_csvs", nargs="+", default=None,
                   help="CSV files for agent 2 (one per seed/run).")
    p.add_argument("--agent2_label", default="Agent 2")

    p.add_argument("--agent3_csvs", nargs="+", default=None,
                   help="CSV files for agent 3 (one per seed/run).")
    p.add_argument("--agent3_label", default="Agent 3")

    p.add_argument("--agent4_csvs", nargs="+", default=None,
                   help="CSV files for agent 4 (one per seed/run).")
    p.add_argument("--agent4_label", default="Agent 4")

    # 5th agent (optional)
    p.add_argument("--agent5_csvs", nargs="+", default=None,
                   help="CSV files for agent 5 (one per seed/run).")
    p.add_argument("--agent5_label", default="Agent 5")
    # ---- legacy mode (static + time-aware, agent label from CSV column) ----
    p.add_argument("--static_csvs", type=str, nargs="+", default=None,
                   help="CSV files for non-time-aware agents (agent label read from 'agent' column).")
    p.add_argument("--timeaware_csvs", type=str, nargs="+", default=None,
                   help="CSV files for time-aware agents (agent label read from 'agent' column).")
    p.add_argument("--timeaware_budget_col", type=str, default="budget")
    p.add_argument("--timeaware_agent_filter", type=str, default="")

    # ---- shared options ----
    p.add_argument("--budget_col", type=str, default=None,
                   help="Budget column name in CSV (auto-detected if omitted).")
    p.add_argument("--budget_min", type=int, default=130)
    p.add_argument("--budget_max", type=int, default=200)
    p.add_argument("--budget_step", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--band", type=str, choices=["se", "std"], default="se",
                   help="Uncertainty band: standard error (se) or std deviation (std).")
    p.add_argument("--out_csv", type=str, default="cvar_results_with_uncertainty.csv")
    p.add_argument("--worst_k", type=int, default=30)
    p.add_argument("--worst_outdir", type=str, default="results/worst_episodes")

    args = p.parse_args()

    budgets = _budgets_list(args.budget_min, args.budget_max, args.budget_step)

    # ------------------------------------------------------------------ #
    # Determine mode
    # ------------------------------------------------------------------ #
    use_multiagent = any(x is not None for x in [
        args.agent1_csvs, args.agent2_csvs, args.agent3_csvs, args.agent4_csvs
    ])
    use_legacy = (args.static_csvs is not None) or (args.timeaware_csvs is not None)

    if not use_multiagent and not use_legacy:
        p.error("Provide either --agent{1-4}_csvs or --static_csvs/--timeaware_csvs.")

    all_agg_frames = []

    # ------------------------------------------------------------------ #
    # Multi-agent mode
    # ------------------------------------------------------------------ #
    if use_multiagent:
        # Infer budget column from the first available CSV
        first_csv = next(
            csv for grp in [args.agent1_csvs, args.agent2_csvs, args.agent3_csvs, args.agent4_csvs]
            if grp for csv in grp
        )
        df0 = pd.read_csv(first_csv)
        bud_col = args.budget_col
        if bud_col is None or bud_col not in df0.columns:
            for c in ("budget", "time_budget", "B", "timeBudget"):
                if c in df0.columns:
                    bud_col = c
                    break
            else:
                p.error("Could not auto-detect budget column. Use --budget_col.")

        agents = [
            (args.agent1_csvs, args.agent1_label),
            (args.agent2_csvs, args.agent2_label),
            (args.agent3_csvs, args.agent3_label),
            (args.agent4_csvs, args.agent4_label),
                (args.agent5_csvs, args.agent5_label),
        ]

        for csv_paths, label in agents:
            if not csv_paths:
                continue
            run_dfs = []
            for path in csv_paths:
                df = pd.read_csv(path)
                rows = compute_cvar_for_run(df, budgets, bud_col, args.alpha, label)
                run_dfs.append(pd.DataFrame(rows))
            agg = merge_uncertainty(run_dfs, band_mode=args.band)
            agg["source"] = "timeaware"
            all_agg_frames.append(agg)

    # ------------------------------------------------------------------ #
    # Legacy mode
    # ------------------------------------------------------------------ #
    if use_legacy:
        if args.static_csvs:
            static_run_dfs = []
            for path in args.static_csvs:
                dfS = pd.read_csv(path)
                rowsS = compute_rows_static(dfS, budgets, alpha=args.alpha)
                static_run_dfs.append(pd.DataFrame(rowsS))
            static_agg = merge_uncertainty(static_run_dfs, band_mode=args.band)
            static_agg["source"] = "static"
            all_agg_frames.append(static_agg)

        if args.timeaware_csvs:
            ta_run_dfs = []
            for path in args.timeaware_csvs:
                dfT = pd.read_csv(path)
                rowsT = compute_rows_timeaware(
                    dfT, budgets, alpha=args.alpha,
                    budget_col=args.timeaware_budget_col,
                    agent_filter=args.timeaware_agent_filter,
                )
                ta_run_dfs.append(pd.DataFrame(rowsT))
            ta_agg = merge_uncertainty(ta_run_dfs, band_mode=args.band)
            ta_agg["source"] = "timeaware"
            all_agg_frames.append(ta_agg)

    # ------------------------------------------------------------------ #
    # Combine & save
    # ------------------------------------------------------------------ #
    out = pd.concat(all_agg_frames, ignore_index=True)
    out = out.sort_values(["source", "agent", "budget"]).reset_index(drop=True)
    out = add_budget_summary(out)

    out.to_csv(args.out_csv, index=False)
    print(f"Saved aggregated CVaR results to: {args.out_csv}")
    print(out.to_string(index=False))

    # Save worst episodes
    if use_legacy and args.static_csvs:
        save_worst_episodes_static(
            csv_paths=args.static_csvs,
            budgets=budgets,
            outdir=args.worst_outdir,
            worst_k=args.worst_k,
        )
    if use_legacy and args.timeaware_csvs:
        save_worst_episodes_timeaware(
            csv_paths=args.timeaware_csvs,
            budgets=budgets,
            budget_col=args.timeaware_budget_col,
            outdir=args.worst_outdir,
            worst_k=args.worst_k,
            agent_filter=args.timeaware_agent_filter,
        )

    print(f"Saved worst-{args.worst_k} episodes per agent&budget to: {args.worst_outdir}/")


if __name__ == "__main__":
    main()
