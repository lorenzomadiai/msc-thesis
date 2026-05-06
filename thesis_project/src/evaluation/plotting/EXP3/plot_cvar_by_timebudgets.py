#!/usr/bin/env python3
"""
Plot pre-aggregated CVaR results produced by cvar_analysis.py.

Reads a CSV with columns:
  agent, budget, mean_cost_mean, mean_cost_band,
  cvar_cost_mean, cvar_cost_band, source

Usage:
  python src/analysis/plot_cvar_results.py \
    --csv results/cvar_4agents_alpha0p1.csv \
    --out_dir plots/cvar_4agents \
    --alpha 0.1
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN values in a 1-D array."""
    arr = arr.copy()
    nans = np.isnan(arr)
    if not np.any(nans) or np.all(nans):
        return arr
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def plot_metric(budgets, agents_data, ylabel, title, out_path, hline=None, hline_label=None):
    """
    agents_data: list of (mean, band, label)
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (mean, band, label) in enumerate(agents_data):
        mean = np.asarray(mean, dtype=np.float64)
        band = np.asarray(band, dtype=np.float64)
        color = colors[i % len(colors)]
        ax.plot(budgets, mean, linewidth=2, label=label, color=color)
        if np.any(np.nan_to_num(band) > 0):
            ax.fill_between(budgets, mean - band, mean + band, alpha=0.2, color=color)

    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.2, label=hline_label)

    ax.set_xlabel("Time budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="Aggregated CVaR CSV produced by cvar_analysis.py.")
    p.add_argument("--out_dir", default="plots/cvar",
                   help="Output directory for PNG plots.")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Alpha value for CVaR (used in plot titles, default: 0.1).")
    p.add_argument("--cost_budget", type=float, default=15,
                   help="Draw a horizontal reference line at this cost level (default: 15).")
    p.add_argument("--agents", nargs="+", default=None,
                   help="If set, only plot these agents (subset of values in the 'agent' column). "
                        "Order determines legend order.")
    p.add_argument("--agent_labels", nargs="+", default=None, metavar="NAME=LABEL",
                   help="Rename agents in the legend. Pass pairs like: "
                        "'Penalty=0=Conservative baseline' 'Penalty=1000=Aggressive'.")
    args = p.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)

    # Drop summary rows (budget == 'ALL') and keep only numeric budgets
    df = df[df["budget"] != "ALL"].copy()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df = df[df["budget"].notna()].copy()
    df["budget"] = df["budget"].astype(int)

    budgets = sorted(df["budget"].unique().tolist())
    agents = df["agent"].unique().tolist()
    if args.agents is not None:
        agents = [a for a in args.agents if a in agents]

    # Build label remap dict from --agent_labels NAME=LABEL pairs
    label_map = {}
    if args.agent_labels:
        for entry in args.agent_labels:
            eq = entry.rindex("=")
            key, val = entry[:eq], entry[eq + 1:]
            label_map[key] = val

    alpha_tag = f" (α={args.alpha:g})" if args.alpha else ""

    metrics = [
        ("cvar_cost_mean",  "cvar_cost_band",  f"CVaR cost{alpha_tag}", f"CVaR cost vs time budget{alpha_tag}",  "cvar_cost"),
        ("mean_cost_mean",  "mean_cost_band",  "Mean cost",             "Mean cost vs time budget",               "mean_cost"),
    ]

    for mean_col, band_col, ylabel, title, fname in metrics:
        agents_data = []
        for agent in agents:
            dA = df[df["agent"] == agent].sort_values("budget")
            # Align to the full budget list (insert NaN for missing)
            bud_to_val = dict(zip(dA["budget"].tolist(), dA[mean_col].tolist()))
            bud_to_band = dict(zip(dA["budget"].tolist(), dA[band_col].tolist()))

            mean_vals = np.array([bud_to_val.get(b, np.nan) for b in budgets], dtype=np.float64)
            band_vals = np.array([bud_to_band.get(b, np.nan) for b in budgets], dtype=np.float64)

            # Interpolate missing budget points (e.g. budget=210 not evaluated)
            mean_vals = interpolate_nans(mean_vals)
            band_vals = interpolate_nans(band_vals)

            agents_data.append((mean_vals, band_vals, label_map.get(agent, agent)))

        plot_metric(
            budgets,
            agents_data,
            ylabel=ylabel,
            title=title,
            out_path=os.path.join(args.out_dir, f"{fname}.png"),
            hline=args.cost_budget if args.cost_budget is not None else None,
            hline_label=f"Cost budget ({args.cost_budget:g})" if args.cost_budget is not None else None,
        )

    print(f"\nDone. Plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
