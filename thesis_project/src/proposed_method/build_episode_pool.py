#!/usr/bin/env python3
"""
build_episode_pool.py
---------------------
Build a conservative episode pool and save it to CSV.

Output CSV columns:
    episode, seed, budget, cons_success

The script screens conservative-only episodes and keeps seeds/budgets until
it reaches the requested pool size.

Modes:
- Balanced mode: set --fail_frac in [0,1] to target a specific fail ratio.
- Natural mode: leave --fail_frac unset to keep the screened outcome ratio.
"""

import os
import sys
import csv
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from safety_gym.envs.engine import Engine

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from meta_env import MetaEnv
from common import (
    STATIC_CONFIG,
    load_policy,
    save_mujoco_state,
    counterfactual_switch_return,
    oracle_conservative_value,
    _find_best_k_zone_search,
)


def rollout_conservative_outcome(env: MetaEnv, seed: int, budget: int, max_horizon: int) -> bool:
    """Run one conservative-only episode and return terminal success."""
    env.seed(int(seed))
    try:
        env._env.seed(int(seed))
    except Exception:
        pass

    obs = env.reset().copy()
    env.B = int(budget)
    if env.irreversible_switch:
        obs[-2] = env._budget_norm()
        obs[-1] = 0.0
    else:
        obs[-1] = env._budget_norm()

    done = False
    ep_len = 0
    goal_met_final = False

    while not done and ep_len < max_horizon:
        _obs_next, _r, done, info = env.step(0)
        ep_len += info.get("n_steps_taken", 1)
        if bool(info.get("goal_met", False)):
            goal_met_final = True

    return goal_met_final


def rollout_conservative_steps(env: MetaEnv, seed: int, budget: int,
                               max_horizon: int,
                               cost_weight: float,
                               deadline_weight: float):
    """Run one conservative episode and save states/rewards needed for k* search."""
    env.seed(int(seed))
    try:
        env._env.seed(int(seed))
    except Exception:
        pass

    obs = env.reset().copy()
    env.B = int(budget)
    if env.irreversible_switch:
        obs[-2] = env._budget_norm()
        obs[-1] = 0.0
    else:
        obs[-1] = env._budget_norm()

    steps = []
    done = False
    ep_len = 0
    goal_met_final = False

    while not done and ep_len < max_horizon:
        state_t = save_mujoco_state(env)
        steps.append({
            "state": state_t,
        })

        obs_next, _r, done, info = env.step(0)
        cost_step = float(info.get("cumulative_cost", 0.0))
        ep_len += info.get("n_steps_taken", 1)

        r_step = -cost_weight * cost_step
        goal_met = bool(info.get("goal_met", False))
        budget_expired = bool(info.get("budget_expired", False))
        if goal_met:
            r_step += 1.0
            goal_met_final = True
        elif budget_expired and not goal_met:
            r_step -= deadline_weight

        steps[-1]["r_step"] = r_step
        steps[-1]["done_next"] = done

        obs = obs_next.copy()

    return steps, goal_met_final


def find_best_k_positive_delta(steps: list,
                               env: MetaEnv,
                               cost_weight: float,
                               deadline_weight: float,
                               gamma: float,
                               max_horizon: int,
                               switch_interval: int,
                               scan_interval: int,
                               n_top_zones: int):
    """Find best k maximizing delta=switch-wait and require positive delta."""
    n_steps = len(steps)
    if n_steps == 0:
        return {
            "best_k": -1,
            "best_delta": -np.inf,
            "best_switch_return": np.nan,
            "best_wait_return": np.nan,
            "has_positive": False,
        }

    def _switch_return_at(k: int) -> float:
        if "switch_return" not in steps[k]:
            steps[k]["switch_return"] = counterfactual_switch_return(
                env,
                steps[k]["state"],
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
            )
        return float(steps[k]["switch_return"])

    def _wait_return_at(k: int) -> float:
        if "wait_return" in steps[k]:
            return float(steps[k]["wait_return"])
        r_step = steps[k]["r_step"]
        if steps[k]["done_next"] or k + 1 >= n_steps:
            wr = r_step
        else:
            oracle_v = oracle_conservative_value(
                env,
                steps[k + 1]["state"],
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
                switch_interval=switch_interval,
            )
            wr = r_step + gamma * oracle_v
        steps[k]["wait_return"] = wr
        return float(wr)

    def _delta_at(k: int) -> float:
        return _switch_return_at(k) - _wait_return_at(k)

    k_zone = _find_best_k_zone_search(
        steps,
        env,
        cost_weight,
        deadline_weight,
        gamma,
        max_horizon,
        scan_interval=scan_interval,
        n_top_zones=n_top_zones,
    )

    best_k = int(k_zone)
    best_delta = float(_delta_at(best_k))

    if best_delta <= 0.0:
        for kk in range(n_steps):
            dkk = float(_delta_at(kk))
            if dkk > best_delta or (dkk == best_delta and kk > best_k):
                best_delta = dkk
                best_k = int(kk)

    best_switch = _switch_return_at(best_k)
    best_wait = _wait_return_at(best_k)

    return {
        "best_k": int(best_k),
        "best_delta": float(best_delta),
        "best_switch_return": float(best_switch),
        "best_wait_return": float(best_wait),
        "has_positive": bool(best_delta > 0.0 and best_switch > best_wait),
    }


def build_episode_pool(env: MetaEnv,
                       pool_size: int,
                       budget_values: list,
                       max_horizon: int,
                       base_seed: int,
                       fail_frac,
                       cost_weight: float,
                       deadline_weight: float,
                       gamma: float,
                       switch_interval: int,
                       scan_interval: int,
                       n_top_zones: int,
                       max_attempts_mult: int,
                       progress_every: int):
    """Build episode pool in balanced mode or natural-ratio mode."""
    pool_size = int(max(2, pool_size))
    balanced_mode = fail_frac is not None
    if balanced_mode:
        fail_frac = float(np.clip(fail_frac, 0.0, 1.0))
        target_fail = int(round(pool_size * fail_frac))
        target_fail = min(pool_size - 1, max(1, target_fail))
        target_win = pool_size - target_fail
    else:
        target_fail = None
        target_win = None

    max_attempts = int(max(1, max_attempts_mult) * pool_size)
    rng_seed = np.random.RandomState(base_seed + 101)
    rng_budget = np.random.RandomState(base_seed + 102)
    rng_shuffle = np.random.RandomState(base_seed + 103)

    win_rows = []
    fail_rows = []
    attempts = 0
    rejected_fail_no_positive = 0

    if balanced_mode:
        print(
            f"\nBuilding balanced episode pool: size={pool_size} "
            f"(win={target_win}, fail={target_fail})"
        )
    else:
        print(f"\nBuilding natural-ratio episode pool: size={pool_size}")

    while attempts < max_attempts:
        if balanced_mode:
            done = (len(win_rows) >= target_win and len(fail_rows) >= target_fail)
        else:
            done = (len(win_rows) + len(fail_rows) >= pool_size)
        if done:
            break

        attempts += 1
        seed = int(rng_seed.randint(0, 2**31 - 1))
        budget = int(rng_budget.choice(budget_values))
        cons_success = bool(
            rollout_conservative_outcome(
                env=env,
                seed=seed,
                budget=budget,
                max_horizon=max_horizon,
            )
        )
        row = {
            "seed": int(seed),
            "budget": int(budget),
            "cons_success": int(cons_success),
            "best_k": -1,
            "best_k_delta": np.nan,
            "best_k_switch_return": np.nan,
            "best_k_wait_return": np.nan,
        }

        if balanced_mode:
            if cons_success and len(win_rows) >= target_win:
                continue
            if (not cons_success) and len(fail_rows) >= target_fail:
                continue

        if cons_success:
            win_rows.append(row)
        else:
            steps, cons_success_rollout = rollout_conservative_steps(
                env=env,
                seed=seed,
                budget=budget,
                max_horizon=max_horizon,
                cost_weight=cost_weight,
                deadline_weight=deadline_weight,
            )
            if bool(cons_success_rollout):
                continue

            kinfo = find_best_k_positive_delta(
                steps=steps,
                env=env,
                cost_weight=cost_weight,
                deadline_weight=deadline_weight,
                gamma=gamma,
                max_horizon=max_horizon,
                switch_interval=switch_interval,
                scan_interval=scan_interval,
                n_top_zones=n_top_zones,
            )

            if kinfo["has_positive"]:
                row["best_k"] = int(kinfo["best_k"])
                row["best_k_delta"] = float(kinfo["best_delta"])
                row["best_k_switch_return"] = float(kinfo["best_switch_return"])
                row["best_k_wait_return"] = float(kinfo["best_wait_return"])
            else:
                rejected_fail_no_positive += 1
                print(
                    "  [warning] conservative-fail episode without optimal positive-delta k*: "
                    f"seed={seed} budget={budget}"
                )
                if balanced_mode:
                    continue

            fail_rows.append(row)

        if (
            attempts % max(1, progress_every) == 0
            or (
                balanced_mode
                and len(win_rows) == target_win
                and len(fail_rows) == target_fail
            )
            or (
                (not balanced_mode)
                and (len(win_rows) + len(fail_rows) == pool_size)
            )
        ):
            print(
                f"  pool attempts={attempts} "
                f"wins={len(win_rows)}"
                + (f"/{target_win}" if balanced_mode else "")
                + " "
                + f"fails={len(fail_rows)}"
                + (f"/{target_fail}" if balanced_mode else "")
                + " "
                f"rejected_fail_no_positive={rejected_fail_no_positive}"
            )

    if balanced_mode:
        if len(win_rows) < target_win or len(fail_rows) < target_fail:
            raise RuntimeError(
                "Unable to build balanced pool. "
                f"Reached attempts={attempts}/{max_attempts} with "
                f"wins={len(win_rows)}/{target_win}, fails={len(fail_rows)}/{target_fail}. "
                f"Rejected fail episodes without positive-delta k*: {rejected_fail_no_positive}. "
                "Increase --max_attempts_mult or revise budget range."
            )
    else:
        if len(win_rows) + len(fail_rows) < pool_size:
            raise RuntimeError(
                "Unable to build natural-ratio pool. "
                f"Reached attempts={attempts}/{max_attempts} with "
                f"size={len(win_rows) + len(fail_rows)}/{pool_size}. "
                "Increase --max_attempts_mult or revise budget range."
            )

    all_rows = win_rows + fail_rows
    all_rows = all_rows[:pool_size]
    rng_shuffle.shuffle(all_rows)

    stats = {
        "pool_size": int(pool_size),
        "mode": "balanced" if balanced_mode else "natural",
        "target_win": int(target_win) if balanced_mode else None,
        "target_fail": int(target_fail) if balanced_mode else None,
        "n_win": int(sum(r["cons_success"] for r in all_rows)),
        "n_fail": int(len(all_rows) - sum(r["cons_success"] for r in all_rows)),
        "fail_frac": float(fail_frac) if balanced_mode else None,
        "rejected_fail_no_positive": int(rejected_fail_no_positive),
        "attempts": int(attempts),
        "max_attempts": int(max_attempts),
    }
    return all_rows, stats


def main():
    p = argparse.ArgumentParser(
        description="Build conservative episode pool CSV for train_gap_switch.py"
    )

    p.add_argument("--cons_dir", type=str, required=True)
    p.add_argument("--agg_dir", type=str, required=True)
    p.add_argument("--budget_min", type=int, default=120)
    p.add_argument("--budget_max", type=int, default=220)
    p.add_argument("--budget_step", type=int, default=5)
    p.add_argument("--meta_interval", type=int, default=1)
    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max env steps per episode (0 = budget_max).")

    p.add_argument("--pool_size", type=int, default=1000,
                   help="Total pool size.")
    p.add_argument("--fail_frac", type=float, default=None,
                   help=(
                       "Optional target fraction of conservative-fail episodes in [0,1]. "
                       "If omitted, pool keeps natural screened ratio (no forced balancing)."
                   ))
    p.add_argument("--max_attempts_mult", type=int, default=30,
                   help="Max attempts multiplier: max_attempts = pool_size * multiplier.")
    p.add_argument("--progress_every", type=int, default=200,
                   help="Print progress every N screened episodes.")

    # Oracle return settings used to precompute best_k on fail episodes
    p.add_argument("--cost_weight", type=float, default=0.02)
    p.add_argument("--deadline_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--switch_interval", type=int, default=5)
    p.add_argument("--scan_interval", type=int, default=5)
    p.add_argument("--n_top_zones", type=int, default=2)

    p.add_argument("--base_seed", type=int, default=1111)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--output_stats_json", type=str, default="")

    args = p.parse_args()

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("\nLoading low-level policies ...")
    sess_cons, act_fn_cons = load_policy(args.cons_dir)
    sess_agg, act_fn_agg = load_policy(args.agg_dir)

    def env_fn():
        return Engine(STATIC_CONFIG)

    env = MetaEnv(
        env_fn=env_fn,
        act_fn_cons=act_fn_cons,
        act_fn_agg=act_fn_agg,
        meta_interval=args.meta_interval,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
        budget_step=args.budget_step,
        irreversible_switch=True,
        seed=args.base_seed + 99,
    )

    bvals = list(range(args.budget_min, args.budget_max + 1, args.budget_step))

    rows, stats = build_episode_pool(
        env=env,
        pool_size=args.pool_size,
        budget_values=bvals,
        max_horizon=max_horizon,
        base_seed=args.base_seed,
        fail_frac=args.fail_frac,
        cost_weight=args.cost_weight,
        deadline_weight=args.deadline_weight,
        gamma=args.gamma,
        switch_interval=args.switch_interval,
        scan_interval=args.scan_interval,
        n_top_zones=args.n_top_zones,
        max_attempts_mult=args.max_attempts_mult,
        progress_every=args.progress_every,
    )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "seed",
                "budget",
                "cons_success",
                "best_k",
                "best_k_delta",
                "best_k_switch_return",
                "best_k_wait_return",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({
                "episode": i,
                "seed": int(row["seed"]),
                "budget": int(row["budget"]),
                "cons_success": int(row["cons_success"]),
                "best_k": int(row["best_k"]),
                "best_k_delta": float(row["best_k_delta"]) if np.isfinite(row["best_k_delta"]) else "",
                "best_k_switch_return": float(row["best_k_switch_return"]) if np.isfinite(row["best_k_switch_return"]) else "",
                "best_k_wait_return": float(row["best_k_wait_return"]) if np.isfinite(row["best_k_wait_return"]) else "",
            })

    stats_out = {
        **stats,
        "budget_min": args.budget_min,
        "budget_max": args.budget_max,
        "budget_step": args.budget_step,
        "max_horizon": max_horizon,
        "base_seed": args.base_seed,
        "fail_frac": args.fail_frac if args.fail_frac is not None else None,
        "cost_weight": args.cost_weight,
        "deadline_weight": args.deadline_weight,
        "gamma": args.gamma,
        "switch_interval": args.switch_interval,
        "scan_interval": args.scan_interval,
        "n_top_zones": args.n_top_zones,
        "output_csv": args.output_csv,
    }

    stats_path = args.output_stats_json
    if not stats_path:
        root, _ = os.path.splitext(args.output_csv)
        stats_path = root + ".stats.json"

    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)

    print("\nSaved:")
    print(f"  {args.output_csv}")
    print(f"  {stats_path}")
    print(
        f"Pool stats: size={stats['pool_size']} wins={stats['n_win']} "
        f"fails={stats['n_fail']} attempts={stats['attempts']}"
    )

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
