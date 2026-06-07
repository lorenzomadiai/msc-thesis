#!/usr/bin/env python3
from __future__ import annotations

"""
Plots agents trajectories from the CSV produced by collect_policies_trajectories.py.

Three plot types are available (choose with --mode):

  overlay   - All trajectories of every agent overlaid on the same map.
              One subplot per agent, each trajectory is a thin line, colored
              by whether the episode was successful (goal reached) or not.

  heatmap   - 2-D density map of visited (x, y) positions per agent.
              Shows statistically where each agent spends its time.

  dist      - Distribution of min distance to hazard per episode, per agent.
              Box/violin plot — the higher the better (further from hazard).

  single    - Plot N random single episodes side by side for all agents
              (useful for qualitative inspection).

    highregime - One comparison plot for selected agents, keeping only episodes
                             in the high-budget regime (budget >= chosen quantile, default q=0.75).

Environment constants (hazard and goal) are drawn on every map.

Usage examples:
  # Overlay all trajectories
  python plot_trajectories.py --csv results/trajectories_*.csv --mode overlay

  # Heatmap comparison
  python plot_trajectories.py --csv results/trajectories_*.csv --mode heatmap

  # Min-dist-to-hazard distribution
  python plot_trajectories.py --csv results/trajectories_*.csv --mode dist

  # 5 random single episodes
  python plot_trajectories.py --csv results/trajectories_*.csv --mode single --n_single 5

  # Save to file instead of showing
  python plot_trajectories.py --csv results/trajectories_*.csv --mode overlay --save plots/traj.pdf

    # High-budget regime comparison for 3 specific methods
    python plot_trajectories.py --csv results/trajectories_*.csv --mode highregime \
            --compare_agents aggressive_baseline reward_shaped_baseline my_method --high_q 0.75
"""
import argparse
import glob
import os
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # change to "TkAgg" / "Qt5Agg" if you want an interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Environment constants  (must match the training config)
# ---------------------------------------------------------------------------
HAZARD_CENTER  = np.array([0.0, 0.0])   # hazards_locations: [(0, 0)]
HAZARD_RADIUS  = 0.7                     # hazards_size: 0.7
GOAL_CENTER    = np.array([1.1, 1.1])   # goal_locations: [(1.1, 1.1)]
GOAL_RADIUS    = 0.3                     # goal_size: 0.3
ARENA_EXTENT   = 1.5                     # placements_extents: [-1.5,-1.5,1.5,1.5]


# ---------------------------------------------------------------------------
# Decade grouping  (mirrors plot_table_comparison.py)
# ---------------------------------------------------------------------------

# Decade buckets: (label, low_inclusive, high_inclusive)
DECADES = [
    ("p10–p100",   10,    99),
    ("p100–p1k",   100,   999),
    ("p1k–p10k",   1000,  10000),
]


def _get_penalty(name: str):
    """Extract integer penalty from names like p10, p1000, p30_s29. None if absent."""
    m = re.search(r"p(\d+)", str(name))
    return int(m.group(1)) if m else None


def _is_policy_switching_agent(name: str) -> bool:
    s = str(name).lower()
    return ("policy_switch" in s) or ("switch" in s)


def _sanitize_filename_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _agent_save_path(save: Optional[str], agent: str, idx: int, total: int) -> Optional[str]:
    if not save:
        return None
    root, ext = os.path.splitext(save)
    ext = ext if ext else ".png"
    token = _sanitize_filename_token(agent) or f"agent_{idx + 1}"
    if total == 1:
        return f"{root}{ext}"
    return f"{root}__{token}{ext}"


def collapse_to_decades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the 'agent' column with decade labels.  For each
    (decade_label, episode_idx, step) the per-step values are averaged
    across all agents that belong to the decade.  Baseline agents
    (those without a p<N> pattern in their name) are kept as-is.

    success/goal_met are taken as max (1 if any agent reached the goal).
    """
    agents = df["agent"].unique().tolist()
    baselines      = [a for a in agents if _get_penalty(a) is None]
    penalty_agents = [a for a in agents if _get_penalty(a) is not None]

    parts = []

    # Baselines unchanged
    for a in baselines:
        sub = df[df["agent"] == a].copy()
        parts.append(sub)

    # Aggregate penalty agents per decade
    for label, lo, hi in DECADES:
        in_decade = [a for a in penalty_agents if lo <= _get_penalty(a) <= hi]
        if not in_decade:
            continue
        sub = df[df["agent"].isin(in_decade)].copy()

        # Per (episode_idx, step): average numeric cols, max for binary flags
        agg = (
            sub.groupby(["episode_idx", "step"], sort=False)
            .agg(
                budget=("budget", "mean"),
                seed=("seed", "first"),
                robot_x=("robot_x", "mean"),
                robot_y=("robot_y", "mean"),
                dist_to_hazard=("dist_to_hazard", "mean"),
                cost_step=("cost_step", "mean"),
                cost_cumulative=("cost_cumulative", "mean"),
                goal_met=("goal_met", "max"),
                switched=("switched", "max") if "switched" in sub.columns else ("goal_met", "min"),
                switch_event=("switch_event", "max") if "switch_event" in sub.columns else ("goal_met", "min"),
                switch_step=("switch_step", "max") if "switch_step" in sub.columns else ("goal_met", "min"),
            )
            .reset_index()
        )
        agg["agent"] = label
        parts.append(agg)

    result = pd.concat(parts, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_arena(ax, **kw):
    rect = plt.Rectangle(
        (-ARENA_EXTENT, -ARENA_EXTENT),
        2 * ARENA_EXTENT, 2 * ARENA_EXTENT,
        fill=False, edgecolor="black", linewidth=1.2, **kw
    )
    ax.add_patch(rect)


def draw_hazard(ax, alpha=0.25, dark=False):
    """Filled red circle: center=(0,0), radius=0.7 (hazards_size)."""
    edge_col = "white" if dark else "red"
    circle = plt.Circle(HAZARD_CENTER, HAZARD_RADIUS,
                        color="red", alpha=alpha, zorder=5)
    ax.add_patch(circle)
    border = plt.Circle(HAZARD_CENTER, HAZARD_RADIUS,
                        fill=False, edgecolor=edge_col, linewidth=2.0, zorder=6)
    ax.add_patch(border)
    ax.text(*HAZARD_CENTER, "hazard", ha="center", va="center",
            fontsize=7.5, color=edge_col, fontweight="bold", zorder=7)


def draw_goal(ax, alpha=0.3, dark=False):
    """Filled green circle: center=(1.1,1.1), radius=0.3 (goal_size)."""
    edge_col = "white" if dark else "green"
    circle = plt.Circle(GOAL_CENTER, GOAL_RADIUS,
                        color="lime", alpha=alpha, zorder=5)
    ax.add_patch(circle)
    border = plt.Circle(GOAL_CENTER, GOAL_RADIUS,
                        fill=False, edgecolor=edge_col, linewidth=2.0, zorder=6)
    ax.add_patch(border)
    ax.text(*GOAL_CENTER, "goal", ha="center", va="center",
            fontsize=7.5, color=edge_col, fontweight="bold", zorder=7)


def setup_map_ax(ax, title="", dark=False):
    ax.set_xlim(-ARENA_EXTENT - 0.05, ARENA_EXTENT + 0.05)
    ax.set_ylim(-ARENA_EXTENT - 0.05, ARENA_EXTENT + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="white" if dark else "black")
    ax.set_ylabel("y", color="white" if dark else "black")
    if title:
        ax.set_title(title, fontsize=10, color="white" if dark else "black")
    if dark:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
    draw_arena(ax)         # black border of the full arena
    draw_hazard(ax, dark=dark)   # red circle at (0,0) r=0.7
    draw_goal(ax, dark=dark)     # green circle at (1.1,1.1) r=0.3


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_csv(paths) -> pd.DataFrame:
    dfs = []
    for p in paths:
        for match in glob.glob(p):
            dfs.append(pd.read_csv(match))
    if not dfs:
        raise FileNotFoundError(f"No CSV files matched: {paths}")
    df = pd.concat(dfs, ignore_index=True)

    # Backward compatibility: older CSVs may not contain switch-tracking columns.
    if "switched" not in df.columns:
        df["switched"] = 0
    if "switch_event" not in df.columns:
        df["switch_event"] = 0
    if "switch_step" not in df.columns:
        df["switch_step"] = -1

    print(f"Loaded {len(df):,} step-rows  |  agents: {sorted(df['agent'].unique())}")
    return df


def episode_success(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean 'success' column at episode level, then merge back."""
    suc = (
        df.groupby(["agent", "episode_idx"])["goal_met"]
        .max()
        .rename("success")
        .reset_index()
    )
    return df.merge(suc, on=["agent", "episode_idx"])


def filter_episodes_by_agent_final_cost(
    df: pd.DataFrame,
    ref_agent: str,
    cost_gt: float,
) -> pd.DataFrame:
    """
    Keep only episode_idx values where ref_agent has final cost_cumulative > cost_gt.

    The selected episode indices are then applied to all agents, so trajectories
    remain directly comparable on the same episodes.
    """
    if "cost_cumulative" not in df.columns:
        raise ValueError("Column 'cost_cumulative' is required for --episode_filter_agent.")

    if ref_agent not in set(df["agent"].unique()):
        raise ValueError(
            f"Agent '{ref_agent}' not found. Available: {sorted(df['agent'].unique())}"
        )

    ref = df[df["agent"] == ref_agent].copy()
    final_cost = (
        ref.sort_values("step")
        .groupby("episode_idx", as_index=False)
        .tail(1)[["episode_idx", "cost_cumulative"]]
    )

    keep_eps = set(final_cost[final_cost["cost_cumulative"] > float(cost_gt)]["episode_idx"].tolist())
    out = df[df["episode_idx"].isin(keep_eps)].copy()

    print(
        f"Episode filter on agent='{ref_agent}': cost_cumulative > {cost_gt}  "
        f"-> kept {len(keep_eps)} episodes, {len(out):,} step-rows"
    )
    return out


def filter_episodes_failed_by_agent(
    df: pd.DataFrame,
    ref_agent: str,
) -> pd.DataFrame:
    """
    Keep only episode_idx values where ref_agent failed (goal never reached).

    The selected episode indices are then applied to all agents, so trajectories
    remain directly comparable on the same episodes.
    """
    if "goal_met" not in df.columns:
        raise ValueError("Column 'goal_met' is required for --failed_filter_agent.")

    if ref_agent not in set(df["agent"].unique()):
        raise ValueError(
            f"Agent '{ref_agent}' not found. Available: {sorted(df['agent'].unique())}"
        )

    ref = df[df["agent"] == ref_agent].copy()
    per_episode_success = (
        ref.groupby("episode_idx", as_index=False)["goal_met"]
        .max()
        .rename(columns={"goal_met": "success"})
    )

    keep_eps = set(per_episode_success[per_episode_success["success"] <= 0]["episode_idx"].tolist())
    out = df[df["episode_idx"].isin(keep_eps)].copy()

    print(
        f"Failed-episode filter on agent='{ref_agent}': "
        f"kept {len(keep_eps)} failed episodes, {len(out):,} step-rows"
    )
    return out


# ---------------------------------------------------------------------------
# Plot modes
# ---------------------------------------------------------------------------

def plot_overlay(df: pd.DataFrame, save: Optional[str], max_episodes: int):
    """One figure per agent, all trajectories overlaid."""
    df = episode_success(df)
    agents = sorted(df["agent"].unique())

    has_switch_cols = {"switch_event", "switched", "switch_step"}.issubset(df.columns)

    for idx, agent in enumerate(agents):
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2), constrained_layout=True)
        setup_map_ax(ax, title=agent)
        sub = df[df["agent"] == agent]
        sub_has_switch = bool(np.any(sub["switch_event"].astype(float).values > 0.5)) if has_switch_cols else False
        episodes = sub["episode_idx"].unique()
        if max_episodes and len(episodes) > max_episodes:
            rng = np.random.default_rng(0)
            episodes = rng.choice(episodes, max_episodes, replace=False)

        for ep_idx in episodes:
            ep = sub[sub["episode_idx"] == ep_idx].sort_values("step")
            success = bool(ep["success"].iloc[0])

            switched_now = bool(np.any(ep["switch_event"].astype(float).values > 0.5)) if has_switch_cols else False
            if switched_now:
                sw_row = ep[ep["switch_event"].astype(float) > 0.5].sort_values("step").iloc[0]
                sw_step = int(sw_row["step"])
                pre = ep[ep["step"] <= sw_step]
                post = ep[ep["step"] >= sw_step]
                ax.plot(pre["robot_x"].values, pre["robot_y"].values,
                        color="#1f77b4", alpha=0.55, linewidth=1.0, zorder=5)
                ax.plot(post["robot_x"].values, post["robot_y"].values,
                        color="#ff7f0e", alpha=0.85, linewidth=1.2, zorder=6)
                ax.scatter(float(sw_row["robot_x"]), float(sw_row["robot_y"]),
                           color="black", s=50, marker="*", zorder=8)
            else:
                color = "steelblue" if success else "salmon"
                ax.plot(ep["robot_x"].values, ep["robot_y"].values,
                        color=color, alpha=0.3, linewidth=0.8, zorder=5)

            # start dot
            ax.scatter(ep["robot_x"].iloc[0], ep["robot_y"].iloc[0],
                       color="black", s=10, alpha=0.4, zorder=7)
            # failure end marker
            if not success:
                ax.scatter(ep["robot_x"].iloc[-1], ep["robot_y"].iloc[-1],
                           color="salmon", s=20, marker="x", linewidths=0.8,
                           alpha=0.8, zorder=8)

        if sub_has_switch and _is_policy_switching_agent(agent):
            legend_handles = [
                mlines.Line2D([], [], color="#1f77b4", linewidth=2.0, label="risk-aware phase"),
                mlines.Line2D([], [], color="#ff7f0e", linewidth=2.0, label="aggressive phase"),
                mlines.Line2D([], [], color="black", marker="*", linestyle="None", markersize=8, label="switch step"),
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

        _save_or_show(fig, _agent_save_path(save, agent, idx, len(agents)))


def plot_heatmap(df: pd.DataFrame, save: Optional[str], bins: int):
    """
    2-D visit-density heatmap per agent.

    Improvements over the plain YlOrRd version:
    - Log-normalized colour scale so low-density paths are still visible
      while hot spots don't wash out everything else.
    - 'inferno' colormap: dark (black/purple) = unvisited, bright (yellow) = often visited.
      High contrast on both the structure and the overlaid hazard/goal circles.
    - Contour lines drawn over the heatmap to show iso-density levels.
    - Hazard and goal labels in white so they are legible on the dark background.
    """
    agents = sorted(df["agent"].unique())
    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    edges = np.linspace(-ARENA_EXTENT, ARENA_EXTENT, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    X, Y = np.meshgrid(centres, centres)

    for ax, agent in zip(axes, agents):
        sub = df[df["agent"] == agent].dropna(subset=["robot_x", "robot_y"])
        H, _, _ = np.histogram2d(sub["robot_x"], sub["robot_y"],
                                  bins=[edges, edges])
        H = H.T  # put x on horizontal axis

        # --- dark background for the axes ---
        ax.set_facecolor("black")
        fig.patch.set_facecolor("#1a1a1a")

        # Log-normalised image; vmin=0.5 so empty cells stay black
        vmax = max(H.max(), 1)
        norm = LogNorm(vmin=0.5, vmax=vmax)
        im = ax.imshow(
            H, origin="lower", aspect="equal",
            extent=[-ARENA_EXTENT, ARENA_EXTENT, -ARENA_EXTENT, ARENA_EXTENT],
            cmap="inferno", norm=norm, interpolation="gaussian", zorder=1
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("visit count (log scale)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Contour lines at a few density levels
        H_smooth = np.where(H > 0, H, np.nan)
        levels = np.logspace(0, np.log10(max(vmax, 2)), 5)
        try:
            ax.contour(X, Y, H_smooth, levels=levels,
                       colors="white", linewidths=0.4, alpha=0.35, zorder=3)
        except Exception:
            pass

        setup_map_ax(ax, title=agent, dark=True)

    fig.suptitle("Position density heatmap", fontsize=13, weight="bold", color="white")
    _save_or_show(fig, save)


def plot_dist(df: pd.DataFrame, save: Optional[str]):
    """
    Per-episode minimum distance to hazard, split by agent.
    A violin + strip plot gives both the distribution and individual values.
    """
    min_dist = (
        df.groupby(["agent", "episode_idx"])["dist_to_hazard"]
        .min()
        .reset_index()
        .rename(columns={"dist_to_hazard": "min_dist_to_hazard"})
    )

    agents  = sorted(min_dist["agent"].unique())
    n       = len(agents)
    colors  = list(mcolors.TABLEAU_COLORS.values())

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * n), 5), constrained_layout=True)

    parts = ax.violinplot(
        [min_dist[min_dist["agent"] == a]["min_dist_to_hazard"].values
         for a in agents],
        positions=range(n),
        showmedians=True,
        widths=0.6,
    )
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)

    # Strip plot (jitter)
    rng = np.random.default_rng(42)
    for i, agent in enumerate(agents):
        vals = min_dist[min_dist["agent"] == agent]["min_dist_to_hazard"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=colors[i % len(colors)], s=12, alpha=0.5, zorder=3)

    # Hazard boundary line
    ax.axhline(0, color="red", linewidth=1.2, linestyle="--",
               label="hazard boundary (dist=0)")

    ax.set_xticks(range(n))
    ax.set_xticklabels(agents, rotation=20, ha="right")
    ax.set_ylabel("min distance to hazard (per episode)")
    ax.set_title("Min distance to hazard — per agent")
    ax.legend(fontsize=8)

    _save_or_show(fig, save)


def plot_single(df: pd.DataFrame, save: Optional[str], n_single: int, seed: int):
    """Plot n_single specific episodes side by side for all agents."""
    df = episode_success(df)
    agents  = sorted(df["agent"].unique())
    n_agents = len(agents)

    # Pick shared episode indices
    common_eps = None
    for agent in agents:
        eps = set(df[df["agent"] == agent]["episode_idx"].unique())
        common_eps = eps if common_eps is None else common_eps & eps
    common_eps = sorted(common_eps)

    rng      = np.random.default_rng(seed)
    chosen   = rng.choice(common_eps,
                           min(n_single, len(common_eps)),
                           replace=False)

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, n_agents,
                              figsize=(5 * n_agents, 4.5 * n_rows),
                              squeeze=False,
                              constrained_layout=True)

    has_switch_cols = {"switch_event", "switched", "switch_step"}.issubset(df.columns)

    for row_i, ep_idx in enumerate(chosen):
        for col_j, agent in enumerate(agents):
            ax = axes[row_i][col_j]
            title = f"{agent} — ep {ep_idx}"
            setup_map_ax(ax, title=title)

            ep  = df[(df["agent"] == agent) &
                     (df["episode_idx"] == ep_idx)].sort_values("step")
            success = bool(ep["success"].iloc[0])

            switched_now = bool(np.any(ep["switch_event"].astype(float).values > 0.5)) if has_switch_cols else False
            if switched_now:
                sw_row = ep[ep["switch_event"].astype(float) > 0.5].sort_values("step").iloc[0]
                sw_step = int(sw_row["step"])
                pre = ep[ep["step"] <= sw_step]
                post = ep[ep["step"] >= sw_step]
                ax.plot(pre["robot_x"].values, pre["robot_y"].values,
                        color="#1f77b4", linewidth=1.6, zorder=5, label="conservative")
                ax.plot(post["robot_x"].values, post["robot_y"].values,
                        color="#ff7f0e", linewidth=1.8, zorder=6, label="aggressive")
                ax.scatter(float(sw_row["robot_x"]), float(sw_row["robot_y"]),
                           color="black", s=80, marker="*", zorder=7, label="switch")
            else:
                color = "steelblue" if success else "salmon"
                ax.plot(ep["robot_x"].values, ep["robot_y"].values,
                        color=color, linewidth=1.5, zorder=5, label="trajectory")

            ax.scatter(ep["robot_x"].iloc[0],  ep["robot_y"].iloc[0],
                       color="black", s=60, marker="^", zorder=7, label="start")
            if success:
                ax.scatter(ep["robot_x"].iloc[-1], ep["robot_y"].iloc[-1],
                           color="black", s=60, marker="s", zorder=7, label="end")
            else:
                ax.scatter(ep["robot_x"].iloc[-1], ep["robot_y"].iloc[-1],
                           color="salmon", s=80, marker="x", linewidths=1.5,
                           zorder=7, label="end (failure)")

            status = "SUCCESS" if success else "FAILURE"
            sw_text = ""
            if switched_now:
                sw_text = f"  sw@{sw_step}"
            ax.set_title(f"{agent} — ep {ep_idx}\n[{status}  cost={ep['cost_cumulative'].iloc[-1]:.0f}]",
                         fontsize=9)
            if sw_text:
                ax.text(0.02, 0.02, sw_text, transform=ax.transAxes,
                        fontsize=8, va="bottom", ha="left",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75})
            if row_i == 0 and col_j == 0:
                ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Individual episode trajectories", fontsize=13, weight="bold")
    _save_or_show(fig, save)


def filter_high_budget_regime(
    df: pd.DataFrame,
    selected_agents,
    q: float,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Filter to selected agents and keep only episodes with budget >= quantile q.

    Quantile is computed over episode-level budgets (mean over steps).
    Returns: filtered_step_df, threshold, episode_budget_df.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"--high_q must be in (0, 1), got {q}")

    selected_agents = list(selected_agents)
    missing = [a for a in selected_agents if a not in set(df["agent"].unique())]
    if missing:
        raise ValueError(
            "Requested agents not found in CSV: "
            f"{missing}. Available: {sorted(df['agent'].unique())}"
        )

    sub = df[df["agent"].isin(selected_agents)].copy()
    ep_budget = (
        sub.groupby(["agent", "episode_idx"], as_index=False)["budget"]
        .mean()
        .rename(columns={"budget": "episode_budget"})
    )
    threshold = float(ep_budget["episode_budget"].quantile(q))

    keep = ep_budget[ep_budget["episode_budget"] >= threshold][["agent", "episode_idx"]]
    filtered = sub.merge(keep, on=["agent", "episode_idx"], how="inner")

    return filtered, threshold, ep_budget


def plot_highregime(
    df: pd.DataFrame,
    save: Optional[str],
    compare_agents,
    high_q: float,
    max_episodes: int,
):
    """
    One figure for trajectory comparison across selected agents in the
    high-budget regime (episode budget >= quantile high_q).
    """
    df = episode_success(df)
    filtered, threshold, ep_budget = filter_high_budget_regime(df, compare_agents, high_q)

    if filtered.empty:
        raise ValueError(
            "No episodes left after high-regime filtering. "
            "Try lowering --high_q or checking --compare_agents."
        )

    agents = list(compare_agents)
    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    has_switch_cols = {"switch_event", "switched", "switch_step"}.issubset(filtered.columns)

    for ax, agent in zip(axes, agents):
        ep_budget_agent = ep_budget[ep_budget["agent"] == agent]
        kept_count = int((ep_budget_agent["episode_budget"] >= threshold).sum())
        total_count = int(len(ep_budget_agent))

        setup_map_ax(ax, title=f"{agent}\nkept {kept_count}/{total_count} eps")
        sub = filtered[filtered["agent"] == agent]
        episodes = sub["episode_idx"].unique()
        if max_episodes and len(episodes) > max_episodes:
            rng = np.random.default_rng(0)
            episodes = rng.choice(episodes, max_episodes, replace=False)

        for ep_idx in episodes:
            ep = sub[sub["episode_idx"] == ep_idx].sort_values("step")
            success = bool(ep["success"].iloc[0])

            switched_now = bool(np.any(ep["switch_event"].astype(float).values > 0.5)) if has_switch_cols else False
            if switched_now:
                sw_row = ep[ep["switch_event"].astype(float) > 0.5].sort_values("step").iloc[0]
                sw_step = int(sw_row["step"])
                pre = ep[ep["step"] <= sw_step]
                post = ep[ep["step"] >= sw_step]
                ax.plot(pre["robot_x"].values, pre["robot_y"].values,
                        color="#1f77b4", alpha=0.65, linewidth=1.2, zorder=5)
                ax.plot(post["robot_x"].values, post["robot_y"].values,
                        color="#ff7f0e", alpha=0.9, linewidth=1.3, zorder=6)
            else:
                color = "steelblue" if success else "salmon"
                ax.plot(ep["robot_x"].values, ep["robot_y"].values,
                        color=color, alpha=0.35, linewidth=0.9, zorder=5)

            ax.scatter(ep["robot_x"].iloc[0], ep["robot_y"].iloc[0],
                       color="black", s=10, alpha=0.4, zorder=7)

    fig.suptitle(
        f"Trajectory comparison in high timebudget regime (q={high_q:.2f}, threshold={threshold:.2f})",
        fontsize=12,
        weight="bold",
    )
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, save: Optional[str]):
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Plot robot trajectories from eval_exp2_trajectories.py CSV output."
    )
    p.add_argument("--csv", type=str, nargs="+", required=True,
                   help="Path(s) to the trajectory CSV file(s). Glob patterns are accepted.")
    p.add_argument("--mode", type=str,
                   choices=["overlay", "heatmap", "dist", "single", "highregime"],
                   default="overlay",
                   help="Plot mode (default: overlay).")
    p.add_argument("--save", type=str, default=None,
                   help="Output file path. If omitted the plot is shown interactively.")
    p.add_argument("--max_episodes", type=int, default=200,
                   help="Max number of trajectories to draw in overlay mode (default: 200).")
    p.add_argument("--heatmap_bins", type=int, default=60,
                   help="Number of bins per axis in heatmap mode (default: 60).")
    p.add_argument("--n_single", type=int, default=5,
                   help="Number of individual episodes in single mode (default: 5).")
    p.add_argument("--single_seed", type=int, default=0,
                   help="RNG seed for picking episodes in single mode (default: 0).")
    p.add_argument("--by_decade", action="store_true",
                   help="Group penalty agents (p<N>) into 3 decade buckets and average "
                        "their trajectories per episode. Baseline agents shown separately.")
    p.add_argument("--compare_agents", type=str, nargs="+", default=None,
                   help="Agent names to compare in --mode highregime. "
                        "Example: aggressive_baseline reward_shaped_baseline my_method")
    p.add_argument("--high_q", type=float, default=0.75,
                   help="Quantile for the high-budget regime in highregime mode (default: 0.75).")
    p.add_argument("--episode_filter_agent", type=str, default=None,
                   help="Optional: keep only episodes where this agent has final cost_cumulative > --episode_filter_cost_gt.")
    p.add_argument("--episode_filter_cost_gt", type=float, default=0.0,
                   help="Threshold used with --episode_filter_agent (default: 0.0).")
    p.add_argument("--failed_filter_agent", type=str, default=None,
                   help="Optional: keep only episodes where this agent failed (goal not reached).")
    args = p.parse_args()

    df = load_csv(args.csv)

    if args.episode_filter_agent:
        df = filter_episodes_by_agent_final_cost(
            df,
            ref_agent=args.episode_filter_agent,
            cost_gt=args.episode_filter_cost_gt,
        )
        if df.empty:
            raise ValueError(
                "No rows left after episode filtering. "
                "Try lowering --episode_filter_cost_gt or checking --episode_filter_agent."
            )

    if args.failed_filter_agent:
        df = filter_episodes_failed_by_agent(
            df,
            ref_agent=args.failed_filter_agent,
        )
        if df.empty:
            raise ValueError(
                "No rows left after failed-episode filtering. "
                "Check --failed_filter_agent and whether it has failed episodes in this CSV."
            )

    if args.by_decade:
        df = collapse_to_decades(df)
        print(f"After decade collapse  |  groups: {sorted(df['agent'].unique())}")

    if args.mode == "overlay":
        plot_overlay(df, args.save, args.max_episodes)
    elif args.mode == "heatmap":
        plot_heatmap(df, args.save, args.heatmap_bins)
    elif args.mode == "dist":
        plot_dist(df, args.save)
    elif args.mode == "single":
        plot_single(df, args.save, args.n_single, args.single_seed)
    elif args.mode == "highregime":
        if not args.compare_agents:
            raise ValueError(
                "--mode highregime requires --compare_agents with the methods to compare."
            )
        plot_highregime(df, args.save, args.compare_agents, args.high_q, args.max_episodes)


if __name__ == "__main__":
    main()
