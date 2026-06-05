#!/usr/bin/env python3
"""
Plot success rate vs switch time (% of horizon) for rule-based hybrid agents.

X axis : switch_frac * 100  (0 = fully aggressive, 100 = fully conservative)
Y axis : success rate at a fixed budget B
Curves : one line per budget (--budgets), with mean ± band across seeds.

switch_frac=0.0  →  aggressive baseline (never switches)
switch_frac=1.0  →  conservative baseline (switches immediately)
"""
import argparse
import os
import glob

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def _frac_to_dirname(frac: float) -> str:
    s = f"{frac:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _load_single_csv_from_dir(pattern: str) -> pd.DataFrame:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No CSV matched: {pattern}")
    if len(hits) > 1:
        hits = sorted(hits, key=os.path.getmtime, reverse=True)
    return pd.read_csv(hits[0])


def load_hybrid_df_for_run(hyb_root: str, base_seed: int, switch_frac: float) -> pd.DataFrame:
    frac_name = _frac_to_dirname(switch_frac)
    folder = os.path.join(hyb_root, f"hybrid_swfrac{frac_name}")
    df = _load_single_csv_from_dir(os.path.join(folder, "*.csv"))
    if "agent" not in df.columns:
        df["agent"] = "hybrid"
    return df


# ── metric helpers ────────────────────────────────────────────────────────────

def success_rate_at_budget(df: pd.DataFrame, B: int) -> float:
    N = len(df)
    if N == 0:
        return np.nan
    succ = ((df["goal_first_step"] != -1) & (df["goal_first_step"] <= B)).sum()
    return float(succ) / float(N)


def mean_cost_at_budget(df: pd.DataFrame, B: int) -> float:
    N = len(df)
    if N == 0:
        return np.nan
    col = f"cost_cum_{int(B)}"
    if col not in df.columns:
        raise KeyError(f"Missing column {col}")
    return float(pd.to_numeric(df[col], errors="coerce").sum()) / float(N)


def hybrid_metric_vs_switch_frac(hyb_roots, seeds, switch_fracs, B: int, band_mode: str, metric_fn):
    """
    Generic version: for each switch_frac compute metric_fn(df, B) per seed,
    then aggregate mean ± band across seeds.
    Returns x (frac*100), means, bands.
    """
    x, means, bands = [], [], []

    for sf in switch_fracs:
        per_seed = []
        for root, seed in zip(hyb_roots, seeds):
            try:
                hdf = load_hybrid_df_for_run(root, seed, sf)
                val = metric_fn(hdf, B)
            except Exception as e:
                print(f"  [warn] frac={sf}, seed={seed}: {e}")
                val = np.nan
            per_seed.append(val)

        arr = np.array(per_seed, dtype=np.float64)
        mean = float(np.nanmean(arr))
        R = int(np.sum(~np.isnan(arr)))
        std = float(np.nanstd(arr, ddof=1)) if R > 1 else 0.0
        band = std if band_mode == "std" else (std / np.sqrt(R) if R > 0 else 0.0)

        x.append(float(sf) * 100.0)
        means.append(mean)
        bands.append(band)

    return x, means, bands


def hybrid_success_vs_switch_frac(hyb_roots, seeds, switch_fracs, B: int, band_mode: str):
    return hybrid_metric_vs_switch_frac(hyb_roots, seeds, switch_fracs, B, band_mode, success_rate_at_budget)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_metric_vs_switch_time(switch_times, curves, out_path, band_mode, ylabel, title, ylim=None, palette=None):
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    default_palette = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    colors = palette if palette else default_palette

    for i, (blabel, (means, bands)) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        means_arr = np.array(means)
        bands_arr = np.array(bands)

        ax.plot(switch_times, means_arr, linewidth=2, label=blabel if blabel else None,
                color=color, marker="o", markersize=4)
        if np.any(np.nan_to_num(bands_arr) > 0):
            ax.fill_between(
                switch_times,
                means_arr - bands_arr,
                means_arr + bands_arr,
                color=color, alpha=0.2, linewidth=0,
            )

    ax.set_xlabel("Switch time (% of horizon)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_xticks(switch_times)
    ax.set_xticklabels([f"{int(t)}%" if t == int(t) else f"{t}%" for t in switch_times],
                       rotation=45, ha="right", fontsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if any(blabel for blabel in curves):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_success_vs_switch_time(switch_times, hybrid_curves, out_path, band_mode):
    plot_metric_vs_switch_time(
        switch_times, hybrid_curves, out_path, band_mode,
        ylabel="Success rate",
        title=f"Success rate vs switch time (mean ± {band_mode})",
        ylim=(0, 1.05),
    )


def plot_tradeoff_vs_switch_time(switch_times, curves_succ, curves_cost, out_path, band_mode):
    """
    Dual-axis plot: left y = success rate, right y = hazard interactions.
    Success rate: solid blue line. Hazard interactions: dashed red line.
    If multiple budgets, each budget gets its own shade.
    """
    _ensure_dir(out_path)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # Color palette per budget (blue shades for success, red shades for cost)
    succ_colors = ["#2196F3", "#0D47A1", "#03A9F4", "#1565C0"]
    cost_colors  = ["#FF5722", "#B71C1C", "#FF9800", "#E64A19"]
    handles = []

    budgets_list = list(curves_succ.keys())
    for i, blabel in enumerate(budgets_list):
        sc = succ_colors[i % len(succ_colors)]
        cc = cost_colors[i % len(cost_colors)]
        x = np.array(switch_times)
        suffix = f" ({blabel})" if blabel else ""

        # --- success rate (left axis, solid) ---
        m_s, b_s = curves_succ[blabel]
        m_s, b_s = np.array(m_s), np.array(b_s)
        l1, = ax1.plot(x, m_s, linewidth=2.5, color=sc, linestyle="-",
                       marker="o", markersize=4, label=f"Success rate{suffix}")
        if np.any(np.nan_to_num(b_s) > 0):
            ax1.fill_between(x, m_s - b_s, m_s + b_s, color=sc, alpha=0.15, linewidth=0)
        handles.append(l1)

        # --- hazard interactions (right axis, dashed) ---
        m_c, b_c = curves_cost[blabel]
        m_c, b_c = np.array(m_c), np.array(b_c)
        l2, = ax2.plot(x, m_c, linewidth=2.5, color=cc, linestyle="-",
                       marker="s", markersize=4, label=f"Hazard interactions{suffix}")
        if np.any(np.nan_to_num(b_c) > 0):
            ax2.fill_between(x, m_c - b_c, m_c + b_c, color=cc, alpha=0.10, linewidth=0)
        handles.append(l2)

    ax1.set_xlabel("Switch time (% of horizon)")
    ax1.set_ylabel("Success rate", color=succ_colors[0])
    ax1.tick_params(axis="y", labelcolor=succ_colors[0])
    ax2.set_ylabel("Interactions with hazards (≤ B)", color=cost_colors[0])
    ax2.tick_params(axis="y", labelcolor=cost_colors[0])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)
    x_arr = np.array(switch_times)
    ax1.set_xticks(x_arr)
    ax1.set_xticklabels([f"{int(t)}%" if t == int(t) else f"{t}%" for t in x_arr],
                        rotation=45, ha="right", fontsize=8)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.set_title(f"Success rate & hazard interactions vs switch time (mean ± {band_mode})")
    ax1.legend(handles=handles, loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot success rate vs switch time (%) for hybrid rule-based agents."
    )

    # Hybrid roots (3 seeds)
    p.add_argument("--hyb_root1", type=str, required=True)
    p.add_argument("--hyb_root2", type=str, required=True)
    p.add_argument("--hyb_root3", type=str, required=True)

    # Seeds (used to build hybrid folder names)
    p.add_argument("--seed1", type=int, default=2208)
    p.add_argument("--seed2", type=int, default=2306)
    p.add_argument("--seed3", type=int, default=3101)

    # Switch fractions to sweep (X axis).
    # Include 0.0 for aggressive baseline and 1.0 for conservative baseline.
    p.add_argument("--switch_fracs", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                   help="Switch fractions. X axis = frac * 100. "
                        "0.0 = aggressive baseline, 1.0 = conservative baseline.")

    # One or more fixed budgets (one curve per budget)
    p.add_argument("--budgets", type=int, nargs="+", required=True,
                   help="Fixed budget(s) at which to evaluate success rate. One curve per budget.")

    p.add_argument("--out_prefix", type=str, default="plots/switch_time")
    p.add_argument("--band", type=str, choices=["se", "std"], default="se")

    args = p.parse_args()
    os.makedirs(args.out_prefix, exist_ok=True)

    hyb_roots = [args.hyb_root1, args.hyb_root2, args.hyb_root3]
    seeds = [args.seed1, args.seed2, args.seed3]

    hybrid_curves_succ = {}  # budget_label -> (means, bands)
    hybrid_curves_cost = {}  # budget_label -> (means, bands)
    common_x = None
    multi_budget = len(args.budgets) > 1

    for B in args.budgets:
        print(f"\nProcessing budget {B}...")
        label_key = f"B={B}" if multi_budget else ""
        x, means, bands = hybrid_success_vs_switch_frac(
            hyb_roots, seeds, args.switch_fracs, B, args.band
        )
        hybrid_curves_succ[label_key] = (means, bands)

        _, means_c, bands_c = hybrid_metric_vs_switch_frac(
            hyb_roots, seeds, args.switch_fracs, B, args.band, mean_cost_at_budget
        )
        hybrid_curves_cost[label_key] = (means_c, bands_c)
        common_x = x

    plot_success_vs_switch_time(
        switch_times=common_x,
        hybrid_curves=hybrid_curves_succ,
        out_path=os.path.join(args.out_prefix, "success_rate_vs_switch_time.png"),
        band_mode=args.band,
    )

    plot_metric_vs_switch_time(
        switch_times=common_x,
        curves=hybrid_curves_cost,
        out_path=os.path.join(args.out_prefix, "hazard_interactions_vs_switch_time.png"),
        band_mode=args.band,
        ylabel="Interactions with hazards (≤ B)",
        title=f"Hazard interactions vs switch time (mean ± {args.band})",
    )

    plot_tradeoff_vs_switch_time(
        switch_times=common_x,
        curves_succ=hybrid_curves_succ,
        curves_cost=hybrid_curves_cost,
        out_path=os.path.join(args.out_prefix, "tradeoff_vs_switch_time.png"),
        band_mode=args.band,
    )


if __name__ == "__main__":
    main()
