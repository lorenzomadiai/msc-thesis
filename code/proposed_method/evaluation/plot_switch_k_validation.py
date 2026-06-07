#!/usr/bin/env python3
"""
plot_switch_k_validation.py

Create clear validation plots for switching timing against oracle best_k.

Input expected:
- per-episode CSV from evaluate_switch_k_from_pool.py
  required columns: threshold, best_k, pred_k

Optional:
- summary CSV with task metrics by threshold (e.g. success_rate, mean_cost)
  to drive the trade-off plot y-axis.

Produced plots:
1) Scatter k_pred vs k_opt (one panel per threshold)
2) CDF of |k_pred - k_opt| (penalized for no-switch)
3) Boxplot of |k_pred - k_opt| (penalized)
4) Trade-off: timing MAE vs selected performance metric
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def load_per_episode(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Per-episode CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"threshold", "best_k", "pred_k"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(
            "Per-episode CSV missing columns: "
            f"{sorted(missing)}. Required: {sorted(required)}"
        )

    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["best_k"] = pd.to_numeric(df["best_k"], errors="coerce")
    df["pred_k"] = pd.to_numeric(df["pred_k"], errors="coerce")
    df = df.dropna(subset=["threshold", "best_k", "pred_k"])

    if df.empty:
        raise ValueError("No valid rows left in per-episode CSV after numeric parsing.")

    return df


def load_summary(summary_csv: str) -> pd.DataFrame:
    if not summary_csv:
        return pd.DataFrame()
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if "threshold" not in df.columns:
        raise ValueError("Summary CSV must contain 'threshold' column.")
    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df = df.dropna(subset=["threshold"])
    return df


def add_error_columns(df: pd.DataFrame, no_switch_penalty_k: int) -> pd.DataFrame:
    out = df.copy()
    out["switched"] = (out["pred_k"] >= 0).astype(int)
    out["pred_k_pen"] = np.where(out["pred_k"] >= 0, out["pred_k"], no_switch_penalty_k)
    out["delta_k"] = out["pred_k"] - out["best_k"]
    out["abs_err"] = np.abs(out["delta_k"])
    out["delta_k_pen"] = out["pred_k_pen"] - out["best_k"]
    out["abs_err_pen"] = np.abs(out["delta_k_pen"])
    return out


def _threshold_label(t: float) -> str:
    return f"thr={t:.3f}".rstrip("0").rstrip(".")


def plot_scatter(df: pd.DataFrame, out_path: str, max_points: int, seed: int):
    thresholds = sorted(df["threshold"].unique())
    n = len(thresholds)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    rng = np.random.RandomState(seed)
    k_min = float(min(df["best_k"].min(), df["pred_k"].where(df["pred_k"] >= 0, np.nan).min(skipna=True)))
    if not np.isfinite(k_min):
        k_min = float(df["best_k"].min())
    k_max = float(max(df["best_k"].max(), df["pred_k"].max()))

    for i, thr in enumerate(thresholds):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["threshold"] == thr]

        if len(sub) > max_points:
            idx = rng.permutation(len(sub))[:max_points]
            sub = sub.iloc[idx]

        sw = sub[sub["pred_k"] >= 0]
        nsw = sub[sub["pred_k"] < 0]

        if not sw.empty:
            ax.scatter(sw["best_k"], sw["pred_k"], s=16, alpha=0.45, label="switched")
        if not nsw.empty:
            # show no-switch rows at y=-1 for visibility
            ax.scatter(nsw["best_k"], np.full(len(nsw), -1), s=20, alpha=0.8, marker="x", label="no-switch")

        ax.plot([k_min, k_max], [k_min, k_max], "k--", linewidth=1.2, label="y=x")
        ax.set_title(_threshold_label(float(thr)))
        ax.set_xlabel("k_opt")
        ax.set_ylabel("k_pred")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Switch Timing: k_pred vs k_opt", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cdf_abs_error(df: pd.DataFrame, out_path: str):
    thresholds = sorted(df["threshold"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for thr in thresholds:
        sub = df[df["threshold"] == thr]
        errs = np.sort(sub["abs_err_pen"].astype(float).values)
        y = np.arange(1, len(errs) + 1, dtype=float) / float(len(errs))
        ax.plot(errs, y, linewidth=2.0, label=_threshold_label(float(thr)))

    ax.set_xlabel("|k_pred - k_opt|  (penalized no-switch)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Timing Error")
    ax.grid(alpha=0.25)
    ax.legend(title="Threshold", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_box_abs_error(df: pd.DataFrame, out_path: str):
    thresholds = sorted(df["threshold"].unique())
    data = [df[df["threshold"] == thr]["abs_err_pen"].astype(float).values for thr in thresholds]
    labels = [_threshold_label(float(thr)) for thr in thresholds]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(thresholds)), 5.0))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)

    for patch in bp["boxes"]:
        patch.set_alpha(0.5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("|k_pred - k_opt|  (penalized no-switch)")
    ax.set_title("Timing Error Distribution by Threshold")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_tradeoff_df(df_ep: pd.DataFrame, df_summary: pd.DataFrame, perf_metric: str) -> pd.DataFrame:
    rows = []
    thresholds = sorted(df_ep["threshold"].unique())

    for thr in thresholds:
        sub = df_ep[df_ep["threshold"] == thr]
        mae = float(np.mean(sub["abs_err_pen"].astype(float).values))

        if not df_summary.empty and perf_metric in df_summary.columns:
            sub_s = df_summary[df_summary["threshold"] == thr]
            if len(sub_s) > 0:
                y = float(sub_s.iloc[0][perf_metric])
            else:
                y = np.nan
        else:
            # Fallback when external performance CSV is not available.
            # Use exact match rate as an interpretable quality score.
            y = float(np.mean((sub["pred_k"] == sub["best_k"]).astype(float).values))

        rows.append({"threshold": float(thr), "timing_mae": mae, "y_metric": y})

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["timing_mae", "y_metric"])
    return out


def plot_tradeoff(df_trade: pd.DataFrame, out_path: str, y_label: str):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    x = df_trade["timing_mae"].astype(float).values
    y = df_trade["y_metric"].astype(float).values

    ax.scatter(x, y, s=65, alpha=0.85)
    for _, r in df_trade.iterrows():
        ax.annotate(
            _threshold_label(float(r["threshold"])),
            (float(r["timing_mae"]), float(r["y_metric"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    ax.set_xlabel("Timing MAE  E[|k_pred - k_opt|] (penalized)")
    ax.set_ylabel(y_label)
    ax.set_title("Trade-off Across Thresholds")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Plot switch timing validation (k_pred vs k_opt) for multiple thresholds."
    )
    p.add_argument("--per_episode_csv", type=str, required=True,
                   help="Per-episode CSV from evaluate_switch_k_from_pool.py")
    p.add_argument("--summary_csv", type=str, default="",
                   help="Optional summary CSV with threshold + performance columns.")
    p.add_argument("--performance_metric", type=str, default="success_rate",
                   help="Column name in --summary_csv used as trade-off y-axis.")
    p.add_argument("--no_switch_penalty_k", type=int, default=220,
                   help="Penalty k value used when pred_k==-1 (no switch).")
    p.add_argument("--max_scatter_points", type=int, default=3000,
                   help="Max points per threshold in scatter (for readability).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for scatter subsampling.")
    p.add_argument("--out_dir", type=str, default="plots/switch_k_validation",
                   help="Output directory for plots.")
    p.add_argument("--out_prefix", type=str, default="switch_k",
                   help="Filename prefix for generated plots.")
    args = p.parse_args()

    ensure_dir(args.out_dir)

    df_ep = load_per_episode(args.per_episode_csv)
    df_ep = add_error_columns(df_ep, no_switch_penalty_k=int(args.no_switch_penalty_k))
    df_summary = load_summary(args.summary_csv)

    trade_df = build_tradeoff_df(df_ep, df_summary, perf_metric=args.performance_metric)

    scatter_path = os.path.join(args.out_dir, f"{args.out_prefix}_scatter_kpred_vs_kopt.png")
    cdf_path = os.path.join(args.out_dir, f"{args.out_prefix}_cdf_abs_error.png")
    box_path = os.path.join(args.out_dir, f"{args.out_prefix}_box_abs_error.png")
    tradeoff_path = os.path.join(args.out_dir, f"{args.out_prefix}_tradeoff.png")
    tradeoff_csv = os.path.join(args.out_dir, f"{args.out_prefix}_tradeoff_points.csv")

    plot_scatter(
        df_ep,
        scatter_path,
        max_points=int(max(100, args.max_scatter_points)),
        seed=int(args.seed),
    )
    plot_cdf_abs_error(df_ep, cdf_path)
    plot_box_abs_error(df_ep, box_path)

    if not trade_df.empty:
        if not df_summary.empty and args.performance_metric in df_summary.columns:
            y_label = args.performance_metric
        else:
            y_label = "exact_match_rate"
        plot_tradeoff(trade_df, tradeoff_path, y_label=y_label)
        trade_df.to_csv(tradeoff_csv, index=False)
    else:
        print("Skipping trade-off plot: no valid points available.")

    print("Saved plots:")
    print(f"  {scatter_path}")
    print(f"  {cdf_path}")
    print(f"  {box_path}")
    if not trade_df.empty:
        print(f"  {tradeoff_path}")
        print(f"  {tradeoff_csv}")


if __name__ == "__main__":
    main()
