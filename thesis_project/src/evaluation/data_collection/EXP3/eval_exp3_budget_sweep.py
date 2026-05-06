#!/usr/bin/env python3
"""
eval_exp1_budget_sweep.py

Evaluates one or more time-aware agents over the SAME seeded episodes while
sweeping a FIXED time budget from budget_max down to budget_min.

Key behavior:
  - Episode seeds are generated once and reused for all agents and all budgets.
  - For a given sweep budget B, every episode in that pass uses exactly B.
  - Budgets are evaluated in descending order: B_max, B_max-step, ..., B_min.
  - Optional policy-switching agent is evaluated through MetaEnv.
  - All results are written to one CSV and identified by (agent, budget, episode_idx).

Example:
  python eval_exp1_budget_sweep.py \
      --agent_dirs /path/to/agent1 /path/to/agent2 \
      --agent_names agent1 agent2 \
      --episodes 1000 \
      --budget_min 120 --budget_max 220 --budget_step 5 \
      --results_dir results/
"""

import os
import csv
import argparse
from collections import deque
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import sys
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from safety_gym.envs.engine import Engine
from wc_sac.sac.wrappers import TimeBudgetWrapper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

from training.supervised_learning.common.features import (  # noqa: E402
    N_FEATURES,
    extract_7features,
    extract_features,
)
from training.supervised_learning.meta_env import MetaEnv  # noqa: E402
from training.supervised_learning.common.mujoco_state import save_mujoco_state  # noqa: E402
from training.supervised_learning.common.oracle import (  # noqa: E402
    counterfactual_switch_return,
    oracle_conservative_value,
    _find_best_k_zone_search,
)


# Identical to config1 in wcsac_timeaware.py — no fixed robot_placements / robot_keepout
TRAIN_CONFIG = {
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

TRAIN_BUDGET_MIN = 120
TRAIN_BUDGET_MAX = 220
TRAIN_BUDGET_STEP = 10


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _pick_signature(meta_graph_def):
    sigs = meta_graph_def.signature_def
    for k in ("serving_default", "serve", "default"):
        if k in sigs:
            return sigs[k]
    if len(sigs) == 0:
        raise RuntimeError("No signature_def found in SavedModel.")
    return sigs[next(iter(sigs.keys()))]


def load_deterministic_policy(saved_model_dir: str):
    print(f"Loading policy from: {saved_model_dir}")
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


class DeltaNet(nn.Module):
    """MLP used by the switch classifier checkpoint."""

    def __init__(self, input_dim: int = N_FEATURES, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """Normalize common checkpoint key prefixes (e.g. DataParallel module.)."""
    if "net.0.weight" in state_dict:
        return state_dict
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module.") :]] = v
        else:
            stripped[k] = v
    return stripped


def load_switch_classifier(checkpoint_path: str):
    """Load classifier checkpoint and infer hidden size from weights."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at: {checkpoint_path}")

    state_dict = _normalize_state_dict_keys(state_dict)
    if "net.0.weight" not in state_dict:
        keys_preview = ", ".join(list(state_dict.keys())[:5])
        raise ValueError(
            "Could not find expected classifier keys in checkpoint. "
            f"First keys: {keys_preview}"
        )

    first_weight = state_dict["net.0.weight"]
    hidden_size = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    model = DeltaNet(input_dim=input_dim, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(
        "Loaded switch classifier: "
        f"{checkpoint_path} (input_dim={input_dim}, hidden_size={hidden_size})"
    )
    return model


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def reset_with_seed(env, seed: int):
    seed = int(seed)
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
        return env.reset(seed=seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_env(budget_min: int, budget_max: int, deadline_penalty: float):
    """
    Creates the environment matching the training setup:
      - Random robot placement.
      - Wrapper present for observation format compatibility.
      - eval_mode=True to disable deadline penalty.
    """
    base_env = Engine(TRAIN_CONFIG)
    env = TimeBudgetWrapper(
        base_env,
        budget_min=budget_min,
        budget_max=budget_max,
        deadline_penalty=float(deadline_penalty),
        eval_mode=True,
        eval_max_budget=int(budget_max),
    )
    return env


def make_meta_env(
    cons_act_fn,
    agg_act_fn,
    meta_interval: int,
    budget_min: int,
    budget_max: int,
    deadline_penalty: float,
    render: bool,
):
    def env_fn():
        return Engine(TRAIN_CONFIG)

    return MetaEnv(
        env_fn=env_fn,
        act_fn_cons=cons_act_fn,
        act_fn_agg=agg_act_fn,
        meta_interval=int(meta_interval),
        budget_min=int(budget_min),
        budget_max=int(budget_max),
        budget_step=TRAIN_BUDGET_STEP,
        cost_weight=0.0,
        goal_reward=0.0,
        deadline_penalty=float(deadline_penalty),
        irreversible_switch=True,
        eval_mode=False,
        seed=None,
        render=bool(render),
    )


def build_budget_sweep(budget_min: int, budget_max: int, budget_step: int):
    if budget_step <= 0:
        raise ValueError("budget_step must be > 0")
    if budget_min > budget_max:
        raise ValueError("budget_min must be <= budget_max")
    budgets = list(range(int(budget_min), int(budget_max) + 1, int(budget_step)))
    if budgets[-1] != int(budget_max):
        raise ValueError(
            "budget_max must align with budget_step from budget_min. "
            f"Received min={budget_min}, max={budget_max}, step={budget_step}."
        )
    return list(reversed(budgets))


def reset_meta_with_seed_and_budget(meta_env, seed: int, budget: int):
    meta_env.seed(int(seed))
    try:
        meta_env._env.seed(int(seed))
    except Exception:
        pass

    obs = meta_env.reset().copy()
    meta_env.B = int(budget)
    if meta_env.irreversible_switch:
        obs[-2] = meta_env._budget_norm()
        obs[-1] = 0.0
    else:
        obs[-1] = meta_env._budget_norm()
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


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def write_csv(path: str, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_collect_fixed_budget(
    act_fn,
    env,
    seeds,
    fixed_budget: int,
    max_horizon: int,
    render: bool,
    agent_name: str,
):
    out_rows = []
    print(len(seeds), f"episodes to collect with fixed budget={fixed_budget}...")

    for ep_idx, s in enumerate(seeds):
        o = reset_with_seed(env, int(s))

        # Force fixed evaluation budget for this sweep pass.
        env.B = int(fixed_budget)
        o[-1] = env._budget_norm()

        done = False
        ep_len = 0
        goal_first_step = -1

        cost_cum = np.zeros(max_horizon, dtype=np.float32)
        cum = 0.0
        dist_hazard_sum = 0.0

        while (not done) and (ep_len < max_horizon):
            a = act_fn(o.reshape(1, -1))[0]
            o, _r, done, info = env.step(a)

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

        success = int(goal_first_step != -1 and goal_first_step <= int(fixed_budget))
        mean_dist_hazard = float(dist_hazard_sum / ep_len) if ep_len > 0 else float("nan")

        row = {
            "agent": agent_name,
            "budget": int(fixed_budget),
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

    return out_rows


def rollout_collect_switch_classifier_meta_fixed_budget(
    classifier_model,
    meta_env,
    switch_prob_threshold: float,
    seeds,
    fixed_budget: int,
    max_horizon: int,
    agent_name: str,
    feature_history: int,
):
    """Evaluate switch classifier through MetaEnv on a fixed budget per pass."""
    out_rows = []
    print(len(seeds), f"episodes to collect (MetaEnv) with fixed budget={fixed_budget}...")

    p_thr = float(np.clip(switch_prob_threshold, 1e-6, 1.0 - 1e-6))
    hist_len = max(0, int(feature_history))

    for ep_idx, seed in enumerate(seeds):
        meta_env.seed(int(seed))
        try:
            meta_env._env.seed(int(seed))
        except Exception:
            pass

        obs = meta_env.reset().copy()
        meta_env.B = int(fixed_budget)
        if meta_env.irreversible_switch:
            obs[-2] = meta_env._budget_norm()
            obs[-1] = 0.0
        else:
            obs[-1] = meta_env._budget_norm()

        done = False
        ep_len = 0
        goal_first_step = -1
        switch_step = -1
        cum_cost = 0.0
        cost_cum = np.zeros(max_horizon, dtype=np.float32)
        hist_buffer = deque(maxlen=hist_len) if hist_len > 0 else None
        base_dim = None

        while (not done) and (ep_len < max_horizon):
            if switch_step < 0:
                feats = extract_features(obs, meta_env).astype(np.float32, copy=False)
                if hist_len > 0:
                    if base_dim is None:
                        base_dim = feats.shape[0]
                        hist_buffer.extend(
                            [np.zeros(base_dim, dtype=np.float32) for _ in range(hist_len)]
                        )
                    stacked_feats = np.concatenate(list(hist_buffer) + [feats], axis=0)
                else:
                    stacked_feats = feats

                with torch.no_grad():
                    x = torch.tensor(stacked_feats, dtype=torch.float32).unsqueeze(0)
                    logit = float(classifier_model(x).item())
                    p_switch = 1.0 / (1.0 + np.exp(-logit))

                action = 1 if p_switch > p_thr else 0
                if action == 1:
                    switch_step = ep_len + 1
                if hist_len > 0:
                    hist_buffer.append(feats)
            else:
                action = 1

            prev_len = ep_len
            obs, _r, done, info = meta_env.step(action)

            step_cost = float(info.get("cumulative_cost", 0.0))
            cum_cost += step_cost
            steps_taken = int(info.get("n_steps_taken", 1))
            ep_len += steps_taken

            fill_limit = min(ep_len, max_horizon)
            for idx in range(prev_len, fill_limit):
                cost_cum[idx] = cum_cost

            if goal_first_step == -1 and bool(info.get("goal_met", False)):
                goal_first_step = min(ep_len, max_horizon)

        if ep_len < max_horizon:
            cost_cum[ep_len:] = cum_cost

        success = int(goal_first_step != -1 and goal_first_step <= int(fixed_budget))

        row = {
            "agent": agent_name,
            "budget": int(fixed_budget),
            "episode_idx": int(ep_idx),
            "seed": int(seed),
            "ep_len": int(min(ep_len, max_horizon)),
            "goal_first_step": int(goal_first_step),
            "success": success,
            "cost_total": float(cum_cost),
            "mean_dist_hazard": float("nan"),
        }
        for t in range(1, max_horizon + 1):
            row[f"cost_cum_{t}"] = float(cost_cum[t - 1])

        out_rows.append(row)

    return out_rows


def _collect_oracle_episode_row(
    meta_env,
    seed,
    budget,
    max_horizon,
    episode_idx,
    switch_step=None,
    capture_steps=False,
    cost_weight=0.02,
    deadline_weight=1.0,
):
    obs = reset_meta_with_seed_and_budget(meta_env, int(seed), int(budget))

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
            saved_state = save_mujoco_state(meta_env)
            feats = extract_7features(obs, meta_env)
            steps.append(
                {
                    "state": saved_state,
                    "feats": feats.copy(),
                    "cum_cost_before": cum_cost,
                }
            )

        if switch_step is None:
            action = 0
        else:
            action = 1 if (switched or ep_len >= int(switch_step)) else 0

        prev_ep_len = ep_len
        obs, _r, done, info = meta_env.step(action)

        step_cost = float(info.get("cumulative_cost", 0.0))
        cum_cost += step_cost
        steps_taken = int(info.get("n_steps_taken", 1))
        ep_len += steps_taken
        fill_limit = min(ep_len, max_horizon)
        for idx in range(prev_ep_len, fill_limit):
            cost_cum[idx] = cum_cost

        try:
            dist_sum += _min_hazard_distance(meta_env._env)
        except Exception:
            pass

        if goal_first_step == -1 and bool(info.get("goal_met", False)):
            goal_first_step = ep_len

        if action == 1:
            switched = True

        if capture_steps:
            goal_met = bool(info.get("goal_met", False))
            budget_expired = bool(info.get("budget_expired", False))
            r_step = -float(cost_weight) * step_cost
            if goal_met:
                r_step += 1.0
            elif budget_expired:
                r_step -= float(deadline_weight)

            steps[-1]["cost_step"] = step_cost
            steps[-1]["r_step"] = r_step
            steps[-1]["done_next"] = done

    if ep_len < max_horizon:
        cost_cum[ep_len:] = cum_cost

    mean_dist = float(dist_sum / ep_len) if ep_len > 0 else float("nan")
    success = int(goal_first_step != -1 and goal_first_step <= int(budget))
    goal_first_capped = goal_first_step if goal_first_step == -1 else min(goal_first_step, max_horizon)

    row = {
        "agent": None,
        "budget": int(budget),
        "episode_idx": int(episode_idx),
        "seed": int(seed),
        "ep_len": int(min(ep_len, max_horizon)),
        "goal_first_step": int(goal_first_capped),
        "success": int(success),
        "cost_total": float(cum_cost),
        "mean_dist_hazard": mean_dist,
    }
    for t in range(1, max_horizon + 1):
        row[f"cost_cum_{t}"] = float(cost_cum[t - 1])

    return (steps if capture_steps else None), row


def rollout_collect_oracle_meta_fixed_budget(
    meta_env,
    seeds,
    fixed_budget: int,
    max_horizon: int,
    agent_name: str,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    scan_interval: int,
    n_top_zones: int,
):
    """Evaluate oracle-switch agent via MetaEnv with fixed budget per pass."""
    out_rows = []
    details_rows = []
    print(len(seeds), f"episodes to collect (oracle MetaEnv) with fixed budget={fixed_budget}...")

    for ep_idx, seed in enumerate(seeds):
        steps, cons_row = _collect_oracle_episode_row(
            meta_env=meta_env,
            seed=int(seed),
            budget=int(fixed_budget),
            max_horizon=max_horizon,
            episode_idx=ep_idx,
            switch_step=None,
            capture_steps=True,
            cost_weight=cost_weight,
            deadline_weight=deadline_weight,
        )
        n_steps = len(steps)
        cons_success = bool(cons_row["success"])
        oracle_row = dict(cons_row)
        oracle_k = -1
        oracle_r_switch = ""

        if (not cons_success) and n_steps > 0:
            best_k = _find_best_k_zone_search(
                steps,
                meta_env,
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
                scan_interval=int(scan_interval),
                n_top_zones=int(n_top_zones),
            )

            def _switch_return_at(k: int) -> float:
                if "switch_return" not in steps[k]:
                    steps[k]["switch_return"] = counterfactual_switch_return(
                        meta_env,
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

                r_step = float(steps[k]["r_step"])
                if steps[k]["done_next"] or k + 1 >= n_steps:
                    wr = r_step
                else:
                    oracle_v = oracle_conservative_value(
                        meta_env,
                        steps[k + 1]["state"],
                        cost_weight,
                        deadline_weight,
                        gamma,
                        max_horizon,
                        switch_interval=1,
                    )
                    wr = r_step + gamma * float(oracle_v)

                steps[k]["wait_return"] = wr
                return float(wr)

            sw_init = _switch_return_at(best_k)
            wt_init = _wait_return_at(best_k)
            if sw_init <= wt_init:
                for k in range(n_steps):
                    _switch_return_at(k)
                best_k = max(range(n_steps), key=lambda k: (float(steps[k]["switch_return"]), k))

            k0_switch = _switch_return_at(0)
            best_switch = _switch_return_at(best_k)
            if k0_switch > best_switch:
                best_k = 0
                best_switch = k0_switch

            oracle_k = int(best_k)
            oracle_r_switch = round(float(best_switch), 4)

            _steps_oracle, oracle_row = _collect_oracle_episode_row(
                meta_env=meta_env,
                seed=int(seed),
                budget=int(fixed_budget),
                max_horizon=max_horizon,
                episode_idx=ep_idx,
                switch_step=int(best_k),
                capture_steps=False,
                cost_weight=cost_weight,
                deadline_weight=deadline_weight,
            )

        oracle_row["agent"] = agent_name
        out_rows.append(oracle_row)
        details_rows.append(
            {
                "agent": agent_name,
                "budget": int(fixed_budget),
                "episode_idx": int(ep_idx),
                "seed": int(seed),
                "cons_success": int(cons_row["success"]),
                "cons_cost": float(cons_row["cost_total"]),
                "oracle_k": int(oracle_k),
                "oracle_r_switch": oracle_r_switch,
                "oracle_success": int(oracle_row["success"]),
                "oracle_cost": float(oracle_row["cost_total"]),
            }
        )

    return out_rows, details_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate one or more time-aware agents on fixed-budget sweeps over the same "
            "seeded episodes. Budgets are evaluated from budget_max down to budget_min."
        )
    )
    p.add_argument(
        "--agent_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to SavedModel directories.",
    )
    p.add_argument(
        "--agent_names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional display names for each agent (must match --agent_dirs in length). "
            "Defaults to the basename of each agent_dir."
        ),
    )
    p.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes.")
    p.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Seed for generating per-episode seeds shared across all agents and budgets.",
    )
    p.add_argument(
        "--budget_min",
        type=int,
        default=TRAIN_BUDGET_MIN,
        help=f"Min sweep budget (default: {TRAIN_BUDGET_MIN}).",
    )
    p.add_argument(
        "--budget_max",
        type=int,
        default=TRAIN_BUDGET_MAX,
        help=f"Max sweep budget (default: {TRAIN_BUDGET_MAX}).",
    )
    p.add_argument(
        "--budget_step",
        type=int,
        default=TRAIN_BUDGET_STEP,
        help=f"Budget sweep step (default: {TRAIN_BUDGET_STEP}).",
    )
    p.add_argument(
        "--max_horizon",
        type=int,
        default=0,
        help="Max steps stored per episode in CSV. If 0, uses budget_max.",
    )
    p.add_argument("--render", action="store_true", help="Render during evaluation.")
    p.add_argument(
        "--deadline_penalty",
        type=float,
        default=0.0,
        help="Deadline penalty reference value.",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where output CSV is saved.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix appended to output filename.",
    )

    # Optional: add classifier-based switch agent (cons -> agg)
    p.add_argument(
        "--switch_classifier_ckpt",
        type=str,
        default="",
        help="Optional path to gap classifier checkpoint (.pt).",
    )
    p.add_argument(
        "--switch_cons_dir",
        type=str,
        default="",
        help="Conservative policy dir used by switch classifier agent.",
    )
    p.add_argument(
        "--switch_agg_dir",
        type=str,
        default="",
        help="Aggressive policy dir used by switch classifier agent.",
    )
    p.add_argument(
        "--switch_prob_threshold",
        type=float,
        default=0.5,
        help="Switch when classifier probability > threshold.",
    )
    p.add_argument(
        "--switch_prob_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of switch thresholds. If set, evaluates one switch agent per threshold.",
    )
    p.add_argument(
        "--switch_agent_name",
        type=str,
        default="gap_switch_classifier",
        help="Name written in CSV for switch classifier agent.",
    )
    p.add_argument(
        "--switch_feature_history",
        type=int,
        default=0,
        help="Past feature vectors to append when evaluating switch classifier.",
    )
    p.add_argument(
        "--switch_meta_interval",
        type=int,
        default=1,
        help="MetaEnv interval (env steps per classifier decision).",
    )

    # Optional: add oracle switch agent (cons -> agg with k* found when conservative fails)
    p.add_argument(
        "--oracle_cons_dir",
        type=str,
        default="",
        help="Optional conservative policy dir for oracle-switch agent.",
    )
    p.add_argument(
        "--oracle_agg_dir",
        type=str,
        default="",
        help="Optional aggressive policy dir for oracle-switch agent.",
    )
    p.add_argument(
        "--oracle_agent_name",
        type=str,
        default="oracle_switch",
        help="Name written in CSV for oracle-switch agent.",
    )
    p.add_argument(
        "--oracle_cost_weight",
        type=float,
        default=0.02,
        help="Oracle return cost weight.",
    )
    p.add_argument(
        "--oracle_deadline_weight",
        type=float,
        default=1.0,
        help="Oracle return deadline penalty weight.",
    )
    p.add_argument(
        "--oracle_gamma",
        type=float,
        default=1.0,
        help="Oracle return discount factor.",
    )
    p.add_argument(
        "--oracle_scan_interval",
        type=int,
        default=2,
        help="Oracle zone-search scan interval.",
    )
    p.add_argument(
        "--oracle_n_top_zones",
        type=int,
        default=5,
        help="Oracle zone-search number of top zones to refine.",
    )

    args = p.parse_args()

    use_switch_classifier = bool(str(args.switch_classifier_ckpt).strip())
    if use_switch_classifier:
        if not str(args.switch_cons_dir).strip() or not str(args.switch_agg_dir).strip():
            p.error(
                "When --switch_classifier_ckpt is set, you must also set --switch_cons_dir and --switch_agg_dir."
            )

    use_oracle_agent = bool(str(args.oracle_cons_dir).strip() or str(args.oracle_agg_dir).strip())
    if use_oracle_agent:
        if not str(args.oracle_cons_dir).strip() or not str(args.oracle_agg_dir).strip():
            p.error("When using oracle agent, set both --oracle_cons_dir and --oracle_agg_dir.")
        if int(args.oracle_scan_interval) <= 0:
            p.error("--oracle_scan_interval must be > 0")
        if int(args.oracle_n_top_zones) <= 0:
            p.error("--oracle_n_top_zones must be > 0")

    if args.switch_prob_thresholds is not None and len(args.switch_prob_thresholds) > 0:
        switch_thresholds = [float(t) for t in args.switch_prob_thresholds]
    else:
        switch_thresholds = [float(args.switch_prob_threshold)]

    for t in switch_thresholds:
        if not (0.0 <= t <= 1.0):
            p.error(f"Each switch threshold must be in [0, 1]. Invalid value: {t}")

    if args.agent_names is not None:
        if len(args.agent_names) != len(args.agent_dirs):
            p.error("--agent_names must have the same number of entries as --agent_dirs.")
        agent_names = args.agent_names
    else:
        agent_names = [os.path.basename(os.path.normpath(d)) for d in args.agent_dirs]

    ensure_dir(args.results_dir)

    budget_sweep = build_budget_sweep(args.budget_min, args.budget_max, args.budget_step)
    max_horizon = int(args.max_horizon) if int(args.max_horizon) > 0 else int(args.budget_max)

    rng = np.random.RandomState(args.base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)

    n_switch_agents = len(switch_thresholds) if use_switch_classifier else 0
    n_oracle_agents = 1 if use_oracle_agent else 0
    n_agents_total = len(args.agent_dirs) + n_switch_agents + n_oracle_agents

    print(
        f"Evaluation settings:\n"
        f"  agents             : {n_agents_total}\n"
        f"  episodes           : {args.episodes} (same seeds across agents and budgets)\n"
        f"  budget sweep       : {budget_sweep} (descending)\n"
        f"  robot placement    : random within placements_extents=[-1.5, -1.5, 1.5, 1.5]\n"
        f"  max_horizon        : {max_horizon}\n"
        f"  base_seed          : {args.base_seed}\n"
    )

    if use_switch_classifier:
        print(
            f"  switch-agent       : enabled ({n_switch_agents} variant(s))\n"
            f"    classifier       : {args.switch_classifier_ckpt}\n"
            f"    cons_dir         : {args.switch_cons_dir}\n"
            f"    agg_dir          : {args.switch_agg_dir}\n"
            f"    p_thr list       : {switch_thresholds}\n"
        )

    if use_oracle_agent:
        print(
            f"  oracle-agent       : enabled\n"
            f"    cons_dir         : {args.oracle_cons_dir}\n"
            f"    agg_dir          : {args.oracle_agg_dir}\n"
            f"    scan_interval    : {args.oracle_scan_interval}\n"
            f"    n_top_zones      : {args.oracle_n_top_zones}\n"
        )

    base = (
        f"fixedbudget_sweep_timeaware_"
        f"seed{args.base_seed}_eps{args.episodes}_"
        f"Bmin{args.budget_min}_Bmax{args.budget_max}_Bstep{args.budget_step}_"
        f"H{max_horizon}_{n_agents_total}agents"
    )
    if args.tag:
        base += f"_{args.tag}"
    out_csv = os.path.join(args.results_dir, base + ".csv")
    oracle_details_csv = os.path.join(args.results_dir, base + "_oracle_details.csv")

    all_rows = []
    oracle_details_rows = []

    for agent_dir, agent_name in zip(args.agent_dirs, agent_names):
        print(f"\n=== Agent: {agent_name} ({agent_dir}) ===")
        env = None
        sess = None
        try:
            sess, act = load_deterministic_policy(agent_dir)
            env = make_env(args.budget_min, args.budget_max, args.deadline_penalty)

            for sweep_budget in budget_sweep:
                print(f"\n--- Budget pass: {sweep_budget} ---")
                rows = rollout_collect_fixed_budget(
                    act_fn=act,
                    env=env,
                    seeds=seeds,
                    fixed_budget=int(sweep_budget),
                    max_horizon=max_horizon,
                    render=args.render,
                    agent_name=agent_name,
                )
                all_rows.extend(rows)
                print(f"    done - {len(rows)} episodes at budget={sweep_budget}.")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass

    if use_switch_classifier:
        sess_cons = None
        sess_agg = None
        try:
            classifier_model = load_switch_classifier(args.switch_classifier_ckpt)
            sess_cons, cons_act = load_deterministic_policy(args.switch_cons_dir)
            sess_agg, agg_act = load_deterministic_policy(args.switch_agg_dir)

            for thr in switch_thresholds:
                thr_label = f"{thr:.3f}".rstrip("0").rstrip(".")
                if len(switch_thresholds) == 1:
                    switch_agent_name = args.switch_agent_name
                else:
                    switch_agent_name = f"{args.switch_agent_name}_pthr{thr_label}"

                print(
                    f"\n=== Agent: {switch_agent_name} "
                    f"(classifier switch, p_thr={thr}) ==="
                )

                meta_env = None
                try:
                    meta_env = make_meta_env(
                        cons_act_fn=cons_act,
                        agg_act_fn=agg_act,
                        meta_interval=args.switch_meta_interval,
                        budget_min=args.budget_min,
                        budget_max=args.budget_max,
                        deadline_penalty=args.deadline_penalty,
                        render=args.render,
                    )

                    for sweep_budget in budget_sweep:
                        print(f"\n--- Budget pass: {sweep_budget} ---")
                        rows = rollout_collect_switch_classifier_meta_fixed_budget(
                            classifier_model=classifier_model,
                            meta_env=meta_env,
                            switch_prob_threshold=thr,
                            seeds=seeds,
                            fixed_budget=int(sweep_budget),
                            max_horizon=max_horizon,
                            agent_name=switch_agent_name,
                            feature_history=args.switch_feature_history,
                        )
                        all_rows.extend(rows)
                        print(
                            f"    done - {len(rows)} episodes at budget={sweep_budget} "
                            f"(MetaEnv)."
                        )
                finally:
                    if meta_env is not None:
                        try:
                            meta_env.close()
                        except Exception:
                            pass
        finally:
            if sess_cons is not None:
                try:
                    sess_cons.close()
                except Exception:
                    pass

    if use_oracle_agent:
        sess_cons = None
        sess_agg = None
        try:
            sess_cons, cons_act = load_deterministic_policy(args.oracle_cons_dir)
            sess_agg, agg_act = load_deterministic_policy(args.oracle_agg_dir)

            print(f"\n=== Agent: {args.oracle_agent_name} (oracle switch) ===")
            meta_env = None
            try:
                meta_env = make_meta_env(
                    cons_act_fn=cons_act,
                    agg_act_fn=agg_act,
                    meta_interval=args.switch_meta_interval,
                    budget_min=args.budget_min,
                    budget_max=args.budget_max,
                    deadline_penalty=args.deadline_penalty,
                    render=args.render,
                )

                for sweep_budget in budget_sweep:
                    print(f"\n--- Budget pass: {sweep_budget} ---")
                    rows, details = rollout_collect_oracle_meta_fixed_budget(
                        meta_env=meta_env,
                        seeds=seeds,
                        fixed_budget=int(sweep_budget),
                        max_horizon=max_horizon,
                        agent_name=args.oracle_agent_name,
                        cost_weight=float(args.oracle_cost_weight),
                        deadline_weight=float(args.oracle_deadline_weight),
                        gamma=float(args.oracle_gamma),
                        scan_interval=int(args.oracle_scan_interval),
                        n_top_zones=int(args.oracle_n_top_zones),
                    )
                    all_rows.extend(rows)
                    oracle_details_rows.extend(details)
                    print(
                        f"    done - {len(rows)} episodes at budget={sweep_budget} "
                        f"(oracle MetaEnv)."
                    )
            finally:
                if meta_env is not None:
                    try:
                        meta_env.close()
                    except Exception:
                        pass
        finally:
            if sess_cons is not None:
                try:
                    sess_cons.close()
                except Exception:
                    pass
            if sess_agg is not None:
                try:
                    sess_agg.close()
                except Exception:
                    pass
            if sess_agg is not None:
                try:
                    sess_agg.close()
                except Exception:
                    pass

    fieldnames = [
        "agent",
        "budget",
        "episode_idx",
        "seed",
        "ep_len",
        "goal_first_step",
        "success",
        "cost_total",
        "mean_dist_hazard",
    ] + [f"cost_cum_{t}" for t in range(1, max_horizon + 1)]

    write_csv(out_csv, all_rows, fieldnames)
    print(f"\nSaved CSV ({len(all_rows)} total rows): {out_csv}")

    if len(oracle_details_rows) > 0:
        oracle_fields = [
            "agent",
            "budget",
            "episode_idx",
            "seed",
            "cons_success",
            "cons_cost",
            "oracle_k",
            "oracle_r_switch",
            "oracle_success",
            "oracle_cost",
        ]
        write_csv(oracle_details_csv, oracle_details_rows, oracle_fields)
        print(
            f"Saved oracle details CSV ({len(oracle_details_rows)} rows): "
            f"{oracle_details_csv}"
        )


if __name__ == "__main__":
    main()
