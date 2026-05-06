#!/usr/bin/env python3
"""
evaluate_oracle_switch.py
-------------------------
Demonstrates that switching from conservative to aggressive at the oracle-
optimal timestep k* improves outcomes compared to conservative-only.

For each episode in a batch:
  1. Run the full conservative episode (saving MuJoCo state at every step).
  2. [optional] Re-run with pure aggressive (switch at k=0).
  3. If conservative SUCCEEDED  → oracle = conservative (no switch needed).
  4. If conservative FAILED      → find k* = argmax_k R_switch(k) via the
     same coarse-to-fine zone search used in train_q_threshold.py.
     Then replay "wait k* steps → switch" and record actual outcome.

Three policies are compared:
  - Conservative only
  - Pure aggressive   (k=0, always switch immediately)
  - Oracle switch     (k = k* found by zone search)

Outputs (in --results_dir):
  oracle_results.csv   — per-episode details
  comparison.txt       — aggregate statistics table

Usage
-----
python src/training/switching_policies/evaluate_oracle_switch.py \\
    --cons_dir  WCSAC/.../simple_save6 \\
    --agg_dir   WCSAC/.../simple_save9 \\
    --episodes  200 \\
    --results_dir results/threshold/oracle_eval_001
"""

import os
import sys
import csv
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from safety_gym.envs.engine import Engine

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from meta_env import MetaEnv
from common.policy_loader import load_policy
from common.mujoco_state import save_mujoco_state, restore_mujoco_state
from common.features import extract_7features as extract_features
from common.oracle import (
    counterfactual_switch_return,
    oracle_conservative_value,
    _find_best_k_zone_search,
)



STATIC_CONFIG = {
    "placements_extents": [-1.5, -1.5, 1.5, 1.5],


    "robot_base": "xmls/point.xml",
    "task": "goal",
    "goal_size": 0.3,
    "goal_keepout": 0.305,
    "goal_locations": [(1.1, 1.1)],
    "observe_goal_lidar": True,
    "observe_hazards": True,
    "constrain_hazards": True,
    "lidar_max_dist": 3,
    "lidar_num_bins": 16,
    "hazards_num": 1,
    "hazards_size": 0.7,
    "hazards_keepout": 0.705,
    "hazards_locations": [(0, 0)],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_env_with_seed(env, seed, budget):
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
    return obs


def _min_hazard_distance(base_env):
    try:
        robot_xy = np.array(base_env.robot_pos[:2])
        hazard_positions = base_env.hazards_pos
        hazard_size = float(base_env.hazards_size)
    except Exception:
        return float("nan")

    if not hazard_positions:
        return float("nan")

    dists = [
        np.linalg.norm(robot_xy - np.array(h[:2])) - hazard_size
        for h in hazard_positions
    ]
    return float(min(dists))


def _collect_episode_row(
    env,
    seed,
    budget,
    max_horizon,
    episode_idx,
    switch_step=None,
    capture_steps=False,
    cost_weight=0.0,
    deadline_weight=0.0,
):
    obs = _reset_env_with_seed(env, seed, budget)

    steps = [] if capture_steps else None
    done = False
    ep_len = 0
    goal_first_step = -1
    cum_cost = 0.0
    dist_sum = 0.0
    cost_cum = np.zeros(max_horizon, dtype=np.float32)
    switched = False

    while (not done) and (ep_len < max_horizon):
        if capture_steps:
            saved_state = save_mujoco_state(env)
            feats = extract_features(obs, env)
            steps.append({
                "state": saved_state,
                "feats": feats.copy(),
                "cum_cost_before": cum_cost,
            })

        if switch_step is None:
            action = 0
        else:
            action = 1 if (switched or ep_len >= int(switch_step)) else 0

        prev_ep_len = ep_len
        obs, _r, done, info = env.step(action)

        step_cost = float(info.get("cumulative_cost", 0.0))
        cum_cost += step_cost
        steps_taken = int(info.get("n_steps_taken", 1))
        ep_len += steps_taken
        fill_limit = min(ep_len, max_horizon)
        for idx in range(prev_ep_len, fill_limit):
            cost_cum[idx] = cum_cost

        dist_sum += _min_hazard_distance(env._env)

        if goal_first_step == -1 and bool(info.get("goal_met", False)):
            goal_first_step = ep_len

        if action == 1:
            switched = True

        if capture_steps:
            goal_met = bool(info.get("goal_met", False))
            budget_expired = bool(info.get("budget_expired", False))
            r_step = -cost_weight * step_cost
            if goal_met:
                r_step += 1.0
            elif budget_expired:
                r_step -= deadline_weight

            steps[-1]["cost_step"] = step_cost
            steps[-1]["r_step"] = r_step
            steps[-1]["done_next"] = done

    if ep_len < max_horizon:
        cost_cum[ep_len:] = cum_cost

    mean_dist = float(dist_sum / ep_len) if ep_len > 0 else float("nan")
    success = int(goal_first_step != -1 and goal_first_step <= int(budget))
    goal_first_capped = goal_first_step if goal_first_step == -1 else min(goal_first_step, max_horizon)

    row = {
        "budget": int(budget),
        "episode_idx": int(episode_idx),
        "seed": int(seed),
        "ep_len": int(min(ep_len, max_horizon)),
        "goal_first_step": int(goal_first_capped),
        "success": success,
        "cost_total": float(cum_cost),
        "mean_dist_hazard": mean_dist,
        "cost_cum": cost_cum.astype(float).tolist(),
    }

    return (steps if capture_steps else None), row


def _row_with_agent(row, agent_name):
    cloned = dict(row)
    cloned["cost_cum"] = list(row.get("cost_cum", []))
    cloned["agent"] = agent_name
    return cloned


def _write_traindist_csv(path, rows, max_horizon):
    base_fields = [
        "agent",
        "budget",
        "episode_idx",
        "seed",
        "ep_len",
        "goal_first_step",
        "success",
        "cost_total",
        "mean_dist_hazard",
    ]
    fieldnames = base_fields + [f"cost_cum_{t}" for t in range(1, max_horizon + 1)]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in base_fields}
            series = list(row.get("cost_cum", []))
            if len(series) < max_horizon:
                pad_value = series[-1] if series else 0.0
                series.extend([pad_value] * (max_horizon - len(series)))
            for idx in range(max_horizon):
                out[f"cost_cum_{idx + 1}"] = float(series[idx])
            writer.writerow(out)

# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_conservative_episode(env, seed, budget, max_horizon,
                              cost_weight, deadline_weight, episode_idx):
    """Run a full conservative episode, saving MuJoCo state at every step."""
    steps, row = _collect_episode_row(
        env,
        seed=seed,
        budget=budget,
        max_horizon=max_horizon,
        episode_idx=episode_idx,
        switch_step=None,
        capture_steps=True,
        cost_weight=cost_weight,
        deadline_weight=deadline_weight,
    )
    goal_met_final = bool(row["success"])
    total_cost = float(row["cost_total"])
    return steps, goal_met_final, total_cost, row


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _cvar(costs, alpha):
    arr = np.sort(np.array(costs, dtype=float))
    k = max(1, int(np.ceil(alpha * len(arr))))
    return float(np.mean(arr[-k:]))


def _stats(successes, costs):
    s = np.array(successes, dtype=float)
    c = np.array(costs,     dtype=float)
    return {
        "n":           len(s),
        "success_rate": float(np.mean(s)),
        "mean_cost":    float(np.mean(c)),
        "cvar10":       _cvar(c, 0.10),
        "cvar20":       _cvar(c, 0.20),
        "cvar30":       _cvar(c, 0.30),
    }


def _print_stats(label, d):
    print(f"  {label:25s}  "
          f"succ={d['success_rate']:.3f}  "
          f"cost={d['mean_cost']:.2f}  "
          f"CVaR10={d['cvar10']:.2f}  "
          f"CVaR20={d['cvar20']:.2f}  "
          f"CVaR30={d['cvar30']:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Oracle switch evaluation: compare conservative-only, "
                    "pure-aggressive, and oracle-switch-at-k* on a batch "
                    "of episodes."
    )
    # --- Environment ---
    p.add_argument("--cons_dir",        type=str, required=True)
    p.add_argument("--agg_dir",         type=str, required=True)
    p.add_argument("--budget_min",      type=int,   default=120)
    p.add_argument("--budget_max",      type=int,   default=220)
    p.add_argument("--budget_step",     type=int,   default=5)
    p.add_argument("--meta_interval",   type=int,   default=1)
    p.add_argument("--max_horizon",     type=int,   default=0,
                   help="Max env steps per episode (0 = budget_max).")

    # --- Evaluation ---
    p.add_argument("--episodes",        type=int,   default=200,
                   help="Number of episodes to evaluate.")
    p.add_argument("--cost_weight",     type=float, default=0.02)
    p.add_argument("--deadline_weight", type=float, default=1.0)
    p.add_argument("--gamma",           type=float, default=1.0)

    # --- Zone search ---
    p.add_argument("--scan_interval",   type=int,   default=2,
                   help="Zone width for coarse probe (default: 2).")
    p.add_argument("--n_top_zones",     type=int,   default=5,
                   help="Top zones to densely scan (default: 2).")

    # --- Output ---
    p.add_argument("--base_seed",       type=int,   default=42)
    p.add_argument("--results_dir",     type=str,
                   default="results/threshold/oracle_eval_001")
    p.add_argument("--oracle_label",    type=str, default="oracle_switch",
                   help="Display name stored in the traindist CSV for the oracle policy.")
    p.add_argument("--tag",              type=str, default="",
                   help="Optional suffix appended to the traindist CSV filename.")
    args = p.parse_args()

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    oracle_label = str(args.oracle_label)
    n_agents_total = 1

    # ----- Episode pool -----
    rng     = np.random.RandomState(args.base_seed)
    seeds   = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)
    bvals   = list(range(args.budget_min, args.budget_max + 1, args.budget_step))
    budgets = np.random.RandomState(args.base_seed + 1).choice(
        bvals, size=args.episodes, replace=True
    )

    traindist_base = (
        f"traindist_timeaware_"
        f"seed{args.base_seed}_eps{args.episodes}"
        f"_Bmin{args.budget_min}_Bmax{args.budget_max}_H{max_horizon}"
        f"_{n_agents_total}agents"
    )
    if args.tag:
        traindist_base += f"_{args.tag}"
    traindist_csv = os.path.join(args.results_dir, traindist_base + ".csv")

    # ----- Load policies -----
    print("\nLoading policies ...")
    sess_cons, act_fn_cons = load_policy(args.cons_dir)
    sess_agg,  act_fn_agg  = load_policy(args.agg_dir)

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
    print(f"  obs_dim={env.observation_space.shape[0]}")

    # ----- Per-episode collection -----
    csv_path = os.path.join(args.results_dir, "oracle_results.csv")
    fields = [
        "ep", "seed", "budget", "N_steps",
        "cons_success", "cons_cost",
        "agg_success",  "agg_cost",
        "oracle_k", "oracle_R_switch",
        "oracle_success", "oracle_cost",
        "improved",       "degraded",
    ]

    records = []
    all_rows = []
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for ep_idx in range(args.episodes):
            seed   = int(seeds[ep_idx])
            budget = int(budgets[ep_idx])

            print(f"\n[{ep_idx+1:4d}/{args.episodes}]  seed={seed}  B={budget}")

            # ---- 1. Conservative rollout ----
            steps, _, cons_cost_val, cons_row = run_conservative_episode(
                env, seed, budget, max_horizon,
                args.cost_weight, args.deadline_weight, ep_idx,
            )
            N = len(steps)
            cons_success_bool = bool(cons_row["success"])
            cons_success_int = int(cons_row["success"])
            cons_cost_val = float(cons_row["cost_total"])
            print(f"  conservative: success={cons_success_int}  "
                f"cost={cons_cost_val:.2f}  N={N}")
            oracle_episode_row = cons_row

            # ---- 2. Oracle switch at k* (zone search + rollout) ----
            oracle_row_source = cons_row
            oracle_success_bool = cons_success_bool
            oracle_cost_val = cons_cost_val
            oracle_k = -1
            oracle_R_switch = ""

            if cons_success_bool or N == 0:
                print("  oracle:       conservative succeeded → no switch")
            else:
                best_k = _find_best_k_zone_search(
                    steps, env,
                    args.cost_weight, args.deadline_weight,
                    args.gamma, max_horizon,
                    scan_interval=args.scan_interval,
                    n_top_zones=args.n_top_zones,
                )

                def _switch_return_at(k: int) -> float:
                    if "switch_return" not in steps[k]:
                        steps[k]["switch_return"] = counterfactual_switch_return(
                            env,
                            steps[k]["state"],
                            args.cost_weight,
                            args.deadline_weight,
                            args.gamma,
                            max_horizon,
                        )
                    return steps[k]["switch_return"]

                def _wait_return_at(k: int) -> float:
                    if "wait_return" in steps[k]:
                        return steps[k]["wait_return"]

                    r_step = steps[k]["r_step"]
                    if steps[k]["done_next"] or k + 1 >= N:
                        wr = r_step
                    else:
                        oracle_v = oracle_conservative_value(
                            env,
                            steps[k + 1]["state"],
                            args.cost_weight,
                            args.deadline_weight,
                            args.gamma,
                            max_horizon,
                            switch_interval=1,
                        )
                        wr = r_step + args.gamma * oracle_v

                    steps[k]["wait_return"] = wr
                    return wr

                sw_init = _switch_return_at(best_k)
                wt_init = _wait_return_at(best_k)
                if sw_init <= wt_init:
                    print("  oracle consistency: wait > switch at zone k*; dense scan full trajectory")
                    for k in range(N):
                        _switch_return_at(k)
                    best_k = max(
                        range(N),
                        key=lambda k: (steps[k]["switch_return"], k),
                    )

                k0_switch = _switch_return_at(0)
                best_switch = _switch_return_at(best_k)
                if k0_switch > best_switch:
                    best_k = 0
                    best_switch = k0_switch

                R_switch_star = best_switch
                oracle_k = best_k
                oracle_R_switch = round(R_switch_star, 4)

                _, oracle_row_source = _collect_episode_row(
                    env,
                    seed=seed,
                    budget=budget,
                    max_horizon=max_horizon,
                    episode_idx=ep_idx,
                    switch_step=best_k,
                )
                oracle_success_bool = bool(oracle_row_source["success"])
                oracle_cost_val = float(oracle_row_source["cost_total"])
                oracle_episode_row = oracle_row_source
                print(f"  oracle:       k*={best_k}  R_switch={R_switch_star:.4f}  "
                      f"success={int(oracle_success_bool)}  cost={oracle_cost_val:.2f}")

            # ---- 3. Pure aggressive baseline (switch at k=0) ----
            _, agg_row_source = _collect_episode_row(
                env,
                seed=seed,
                budget=budget,
                max_horizon=max_horizon,
                episode_idx=ep_idx,
                switch_step=0,
            )
            agg_success_bool = bool(agg_row_source["success"])
            agg_cost_val = float(agg_row_source["cost_total"])
            print(f"  aggressive:   success={int(agg_success_bool)}  cost={agg_cost_val:.2f}")

            # Safety fallback: oracle should dominate k=0 candidate.
            if (not cons_success_bool) and N > 0 and agg_success_bool and (not oracle_success_bool):
                print("  [WARN] oracle < aggressive(k=0) contradiction; forcing oracle=k0")
                oracle_k = 0
                oracle_episode_row = agg_row_source
                oracle_success_bool = agg_success_bool
                oracle_cost_val = agg_cost_val
                if "switch_return" in steps[0]:
                    oracle_R_switch = round(steps[0]["switch_return"], 4)
                else:
                    oracle_R_switch = ""

            all_rows.append(_row_with_agent(oracle_episode_row, oracle_label))

            cons_success_int = int(cons_success_bool)
            agg_success_int = int(agg_success_bool)
            oracle_success_int = int(oracle_success_bool)
            improved = oracle_success_int - cons_success_int
            degraded = cons_success_int - oracle_success_int

            row = {
                "ep":             ep_idx + 1,
                "seed":           seed,
                "budget":         budget,
                "N_steps":        N,
                "cons_success":   cons_success_int,
                "cons_cost":      round(cons_cost_val,    4),
                "agg_success":    agg_success_int,
                "agg_cost":       round(agg_cost_val,     4),
                "oracle_k":       oracle_k,
                "oracle_R_switch": oracle_R_switch,
                "oracle_success": oracle_success_int,
                "oracle_cost":    round(oracle_cost_val,  4),
                "improved":       improved,
                "degraded":       degraded,
            }
            writer.writerow(row)
            csv_file.flush()
            records.append(row)

    _write_traindist_csv(traindist_csv, all_rows, max_horizon)

    # ----- Aggregate statistics -----
    cons_succ  = [r["cons_success"]   for r in records]
    cons_costs = [r["cons_cost"]      for r in records]
    agg_succ   = [r["agg_success"]    for r in records]
    agg_costs  = [r["agg_cost"]       for r in records]
    ora_succ   = [r["oracle_success"] for r in records]
    ora_costs  = [r["oracle_cost"]    for r in records]

    # Stats restricted to episodes where conservative FAILED
    failed_idx = [i for i, r in enumerate(records) if r["cons_success"] == 0]
    n_failed   = len(failed_idx)
    n_success  = len(records) - n_failed
    n_improved = sum(r["improved"] for r in records)
    n_degraded = sum(r["degraded"] for r in records)

    s_cons  = _stats(cons_succ,  cons_costs)
    s_agg   = _stats(agg_succ,   agg_costs)
    s_ora   = _stats(ora_succ,   ora_costs)

    print(f"\n{'='*70}")
    print(f"  ORACLE SWITCH EVALUATION  —  {args.episodes} episodes total")
    print(f"    cons succeeded:   {n_success}  ({100*n_success/len(records):.1f}%)")
    print(f"    cons failed:      {n_failed}  ({100*n_failed/len(records):.1f}%)")
    print(f"    oracle improved:  {n_improved}  (cons failed → oracle succeeded)")
    print(f"    oracle degraded:  {n_degraded}  (cons succeeded → oracle failed)")
    print(f"{'='*70}")
    _print_stats("Conservative only",   s_cons)
    _print_stats("Pure aggressive",      s_agg)
    _print_stats("Oracle switch at k*",  s_ora)
    print(f"\n  Δ(oracle - cons)  succ={s_ora['success_rate']-s_cons['success_rate']:+.3f}  "
          f"cost={s_ora['mean_cost']-s_cons['mean_cost']:+.2f}  "
          f"CVaR10={s_ora['cvar10']-s_cons['cvar10']:+.2f}")
    print(f"{'='*70}")

    # Stats on failing episodes only
    if n_failed > 0:
        f_cons_succ  = [cons_succ[i]  for i in failed_idx]
        f_cons_costs = [cons_costs[i] for i in failed_idx]
        f_agg_succ   = [agg_succ[i]   for i in failed_idx]
        f_agg_costs  = [agg_costs[i]  for i in failed_idx]
        f_ora_succ   = [ora_succ[i]   for i in failed_idx]
        f_ora_costs  = [ora_costs[i]  for i in failed_idx]

        print(f"\n  Failing episodes only (N={n_failed}):")
        _print_stats("Conservative only",   _stats(f_cons_succ, f_cons_costs))
        _print_stats("Pure aggressive",      _stats(f_agg_succ,  f_agg_costs))
        _print_stats("Oracle switch at k*",  _stats(f_ora_succ,  f_ora_costs))
        print()

    # ----- Save comparison.txt -----
    comp_path = os.path.join(args.results_dir, "comparison.txt")
    with open(comp_path, "w") as f:
        f.write(f"episodes        = {args.episodes}\n")
        f.write(f"cons_succeeded  = {n_success}\n")
        f.write(f"cons_failed     = {n_failed}\n")
        f.write(f"oracle_improved = {n_improved}\n")
        f.write(f"oracle_degraded = {n_degraded}\n\n")

        for label, d in [("conservative", s_cons),
                         ("pure_aggressive", s_agg),
                         ("oracle_switch", s_ora)]:
            f.write(f"[{label}]\n")
            for k, v in d.items():
                f.write(f"  {k} = {v}\n")
            f.write("\n")

        f.write("[delta: oracle - conservative]\n")
        f.write(f"  success_rate = {s_ora['success_rate']-s_cons['success_rate']:+.4f}\n")
        f.write(f"  mean_cost    = {s_ora['mean_cost']-s_cons['mean_cost']:+.4f}\n")
        f.write(f"  cvar10       = {s_ora['cvar10']-s_cons['cvar10']:+.4f}\n")
        f.write(f"  cvar20       = {s_ora['cvar20']-s_cons['cvar20']:+.4f}\n")
        f.write(f"  cvar30       = {s_ora['cvar30']-s_cons['cvar30']:+.4f}\n")

    print(f"\nResults saved:")
    print(f"  {traindist_csv}")
    print(f"  {csv_path}")
    print(f"  {comp_path}")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
