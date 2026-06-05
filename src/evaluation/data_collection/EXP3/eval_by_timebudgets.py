#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from safety_gym.envs.engine import Engine
from wc_sac.sac.wrappers import TimeBudgetWrapper


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


def _pick_signature(meta_graph_def):
    sigs = meta_graph_def.signature_def
    for k in ("serving_default", "serve", "default"):
        if k in sigs:
            return sigs[k]
    if len(sigs) == 0:
        raise RuntimeError("No signature_def found in SavedModel.")
    return sigs[next(iter(sigs.keys()))]


def load_deterministic_policy(saved_model_dir: str):
    if not os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
        raise FileNotFoundError(f"saved_model.pb not found inside: {saved_model_dir}")

    g = tf.Graph()
    sess = tf.Session(graph=g)

    with g.as_default():
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir
        )
        sig = _pick_signature(meta_graph_def)

        x_name = sig.inputs["x"].name if "x" in sig.inputs else next(iter(sig.inputs.values())).name
        if "mu" in sig.outputs:
            out_name = sig.outputs["mu"].name
        elif "pi" in sig.outputs:
            out_name = sig.outputs["pi"].name
        else:
            out_name = next(iter(sig.outputs.values())).name

        x_t = g.get_tensor_by_name(x_name)
        a_t = g.get_tensor_by_name(out_name)

        def act_fn(obs_batch: np.ndarray) -> np.ndarray:
            return sess.run(a_t, feed_dict={x_t: obs_batch})

    return sess, act_fn


def reset_with_seed(env, seed: int):
    seed = int(seed)
    # try to seed both wrapper and underlying env
    try:
        env.seed(seed)
    except Exception:
        pass
    try:
        env.unwrapped.seed(seed)
    except Exception:
        pass
    try:
        return env.reset()
    except TypeError:
        # some gym versions accept seed=...
        return env.reset(seed=seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_timeaware_env(time_budget: int, deadline_penalty: float, eval_max_budget: int):
    """
    Always wraps with TimeBudgetWrapper, fixing B = time_budget.
    """
    base_env = Engine(STATIC_CONFIG)
    env = TimeBudgetWrapper(
        base_env,
        budget_min=int(time_budget),
        budget_max=int(time_budget),
        deadline_penalty=float(deadline_penalty),
        eval_mode=True,
        eval_max_budget=int(eval_max_budget),
    )
    return env


def write_csv(path: str, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rollout_collect_for_budget(act_fn, env, seeds, budget: int, max_horizon: int, render: bool, agent_name: str):
    """
    Runs the SAME seeds for this specific budget.
    Stores cumulative cost after each step (1..max_horizon), padded with last value.
    Note: ep_len will be <= budget because wrapper terminates at B.
    """
    out_rows = []
    for ep_idx, s in enumerate(seeds):
        o = reset_with_seed(env, int(s))

        # print robot position
        # pos = env.sim.data.get_body_xpos('robot')
        # print(f"robot_pos = {pos[:2]}, robot_z = {pos[2]:.3f}")
        
        done = False
        ep_len = 0
        goal_first_step = -1

        cost_cum = np.zeros(max_horizon, dtype=np.float32)
        cum = 0.0
        dist_hazard_sum = 0.0

        while (not done) and (ep_len < max_horizon):
            a = act_fn(o.reshape(1, -1))[0]
            o, r, done, info = env.step(a)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            ep_len += 1
            step_cost = float(info.get("cost", 0.0))
            cum += step_cost
            cost_cum[ep_len - 1] = cum

            if goal_first_step == -1 and bool(info.get("goal_met", False)):
                goal_first_step = ep_len

            # compute min distance to any hazard
            try:
                base_env = env.unwrapped
                robot_xy = np.array(base_env.robot_pos[:2])
                hazard_positions = base_env.hazards_pos
                min_dist = min(
                    np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                    for hp in hazard_positions
                )
                dist_hazard_sum += min_dist
            except Exception:
                pass

        if ep_len < max_horizon:
            cost_cum[ep_len:] = cum

        # success within budget B
        success = int(goal_first_step != -1 and goal_first_step <= int(budget))
        mean_dist_hazard = float(dist_hazard_sum / ep_len) if ep_len > 0 else float("nan")

        row = {
            "agent": agent_name,
            "budget": int(budget),
            "episode_idx": int(ep_idx),
            "seed": int(s),
            "ep_len": int(ep_len),
            "goal_first_step": int(goal_first_step),
            "success": int(success),
            "cost_total": float(cum),
            "mean_dist_hazard": mean_dist_hazard,
        }
        for t in range(1, max_horizon + 1):
            row[f"cost_cum_{t}"] = float(cost_cum[t - 1])

        out_rows.append(row)

        # print(
        #     f"Agent={agent_name} | B={budget} | ep={ep_idx} seed={s} "
        #     f"len={ep_len} success={success} goal_first_step={goal_first_step} cost_total={cum:.3f}"
        # )

    return out_rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--agent_dir", type=str, required=True)

    p.add_argument("--time_budgets", type=int, nargs="+", required=True,
                   help="List of time budgets to evaluate on, e.g. 130 150 170 200")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--base_seed", type=int, default=0)

    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max steps to store in CSV. If 0, uses max(time_budgets).")
    p.add_argument("--render", action="store_true")

    p.add_argument("--deadline_penalty", type=float, default=0.0)

    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    ensure_dir(args.results_dir)

    budgets = [int(b) for b in args.time_budgets]
    max_B = max(budgets)
    print(f"max_B = {max_B}")
    max_horizon = int(args.max_horizon) if int(args.max_horizon) > 0 else max_B
    eval_max_budget = max_B  # important for correct budget normalization in observation

    rng = np.random.RandomState(args.base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)

    sess, act = load_deterministic_policy(args.agent_dir)

    base = (
        f"fixedEpisodes_timeaware_{os.path.basename(os.path.normpath(args.agent_dir))}_"
        f"seed{args.base_seed}_eps{args.episodes}_B{min(budgets)}to{max(budgets)}_H{max_horizon}"
    )
    if args.tag:
        base += f"_{args.tag}"
    out_csv = os.path.join(args.results_dir, base + ".csv")

    all_rows = []
    try:
        for B in budgets:
            env = None
            try:
                print(f"\nEvaluating time-aware agent with time budget B={B}...")
                env = make_timeaware_env(B, args.deadline_penalty, eval_max_budget)
                rows = rollout_collect_for_budget(
                    act_fn=act,
                    env=env,
                    seeds=seeds,
                    budget=B,
                    max_horizon=max_horizon,
                    render=args.render,
                    agent_name="time_aware_agent",
                )
                all_rows.extend(rows)
            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
    finally:
        try:
            sess.close()
        except Exception:
            pass

    fieldnames = (
        ["agent", "budget", "episode_idx", "seed", "ep_len", "goal_first_step", "success", "cost_total", "mean_dist_hazard"]
        + [f"cost_cum_{t}" for t in range(1, max_horizon + 1)]
    )
    write_csv(out_csv, all_rows, fieldnames)
    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
