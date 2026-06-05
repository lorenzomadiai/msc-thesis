#!/usr/bin/env python3
from __future__ import annotations

"""
eval_exp2_trajectories.py

Like eval_exp2_quantiles.py, but instead of per-episode aggregate stats it
stores the full (x, y) trajectory of the robot at every timestep.

The resulting CSV has one row **per step** (not per episode), with columns:
    agent, budget, episode_idx, seed, step,
    robot_x, robot_y,
    dist_to_hazard,   # signed: boundary of hazard is 0, negative = inside
    cost_step,        # per-step cost (0 or 1)
    cost_cumulative,  # running cumulative cost within the episode
    goal_met,         # 1 if goal was reached on this step, else 0

This lets you reconstruct and plot every trajectory per agent/episode and
visually compare who keeps further away from the hazard.

Example usage:
  python eval_exp2_trajectories.py \\
      --agent_dirs /path/to/agent1 /path/to/agent2 \\
      --agent_names agent1 agent2 \\
      --quantile_low 0.0 --quantile_high 1.0 \\
      --episodes 50 --results_dir results/
"""
import os
import csv
import argparse
import sys
from collections import deque
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

from training.supervised_learning.common.features import extract_features  # noqa: E402
from training.supervised_learning.meta_env import MetaEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Training config  (identical to wcsac_timeaware.py)
# ---------------------------------------------------------------------------

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
TRAIN_BUDGET_STEP = 5


# ---------------------------------------------------------------------------
# Quantile budget selection  (identical to eval_exp2_quantiles.py)
# ---------------------------------------------------------------------------

def budgets_from_quantiles(budget_min: int, budget_max: int, step: int,
                            q_low: float, q_high: float) -> list:
    all_budgets = np.arange(budget_min, budget_max + 1, step, dtype=int)
    n = len(all_budgets)
    if n == 0:
        raise ValueError("Empty budget grid — check budget_min/max/step.")
    idx_lo = max(0, min(int(np.floor(q_low  * (n - 1))), n - 1))
    idx_hi = max(0, min(int(np.ceil (q_high * (n - 1))), n - 1))
    return all_budgets[idx_lo: idx_hi + 1].tolist()


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
    if not os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
        raise FileNotFoundError(
            f"saved_model.pb not found inside: {saved_model_dir}"
        )
    g    = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir
        )
        sig = _pick_signature(meta_graph_def)
        x_name  = (sig.inputs["x"].name  if "x"  in sig.inputs
                   else next(iter(sig.inputs.values())).name)
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

    def __init__(self, input_dim: int = 36, hidden_size: int = 16):
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
            stripped[k[len("module."):]] = v
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
    return model, input_dim


def extract_switch_features(obs: np.ndarray, env) -> np.ndarray:
    """
    Build lidar+kinematics features for the switch classifier.

    Feature layout (36-D base):
      [hazard_lidar(16), goal_lidar(16), v_x, v_y, time_left_norm, budget_norm]
    """
    lidar_bins = 16

    cache = getattr(env, "_cached_switch_lidar_slices", None)
    if cache is None:
        base_env_for_obs = getattr(env, "unwrapped", env)
        goal_slice = None
        hazard_slice = None
        offset = 0
        try:
            obs_dict = base_env_for_obs.obs_space_dict
        except AttributeError:
            goal_slice = slice(0, lidar_bins)
            hazard_slice = slice(lidar_bins, 2 * lidar_bins)
        else:
            for key, space in obs_dict.items():
                size = int(np.prod(space.shape))
                key_l = str(key).lower()
                if "lidar" in key_l or key_l == "velocimeter":
                    if key == "goal_lidar":
                        goal_slice = slice(offset, offset + size)
                    elif key == "hazards_lidar":
                        hazard_slice = slice(offset, offset + size)
                    offset += size
        if goal_slice is None or hazard_slice is None:
            raise RuntimeError("Could not resolve goal/hazards lidar slices from observation.")
        cache = (goal_slice, hazard_slice)
        setattr(env, "_cached_switch_lidar_slices", cache)

    goal_slice, hazard_slice = cache
    goal_lidar = np.asarray(obs[goal_slice], dtype=np.float32)
    hazard_lidar = np.asarray(obs[hazard_slice], dtype=np.float32)

    base_env = env.unwrapped
    robot_vel = base_env.sim.data.get_body_xvelp("robot")
    v_x = float(robot_vel[0])
    v_y = float(robot_vel[1])

    # TimeBudgetWrapper appends [time_left_norm, budget_norm] as the last 2 elements.
    time_left_norm = float(obs[-2])
    budget_norm = float(obs[-1])

    return np.concatenate([
        hazard_lidar,
        goal_lidar,
        np.array([v_x, v_y, time_left_norm, budget_norm], dtype=np.float32),
    ]).astype(np.float32, copy=False)


def build_classifier_input(
    base_feats: np.ndarray,
    classifier_input_dim: int,
    feature_history: int,
    hist_buffer: deque,
) -> tuple[np.ndarray, int, deque]:
    """
    Build classifier input matching the checkpoint input dimension.

    If feature_history >= 0, enforce exactly that many past feature vectors.
    If feature_history < 0, infer history from checkpoint input dimension.
    """
    base_dim = int(base_feats.shape[0])

    if feature_history >= 0:
        history_needed = int(feature_history)
        expected = (history_needed + 1) * base_dim
        if expected != classifier_input_dim:
            raise ValueError(
                "Feature size mismatch for switch classifier: "
                f"checkpoint input_dim={classifier_input_dim}, base_feat_dim={base_dim}, "
                f"requested history={history_needed} -> expected input_dim={expected}."
            )
    else:
        if classifier_input_dim % base_dim != 0:
            raise ValueError(
                "Cannot infer history for switch classifier: "
                f"checkpoint input_dim={classifier_input_dim} is not a multiple of "
                f"base_feat_dim={base_dim}."
            )
        history_needed = classifier_input_dim // base_dim - 1
        if history_needed < 0:
            raise ValueError(
                "Invalid inferred history: "
                f"checkpoint input_dim={classifier_input_dim}, base_feat_dim={base_dim}."
            )

    if history_needed == 0:
        return base_feats, history_needed, hist_buffer

    if hist_buffer is None:
        hist_buffer = deque(maxlen=history_needed)
        for _ in range(history_needed):
            hist_buffer.append(np.zeros(base_dim, dtype=np.float32))

    stacked = np.concatenate(list(hist_buffer) + [base_feats], axis=0).astype(np.float32, copy=False)
    return stacked, history_needed, hist_buffer


# ---------------------------------------------------------------------------
# Helpers
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


def make_env(budget_min: int, budget_max: int, deadline_penalty: float,
             non_time_aware: bool = False):
    base_env = Engine(TRAIN_CONFIG)
    if non_time_aware:
        return base_env
    env = TimeBudgetWrapper(
        base_env,
        budget_min=budget_min,
        budget_max=budget_max,
        deadline_penalty=float(deadline_penalty),
        eval_mode=True,
        eval_max_budget=budget_max,
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
        budget_step=int(TRAIN_BUDGET_STEP),
        cost_weight=0.0,
        goal_reward=0.0,
        deadline_penalty=float(deadline_penalty),
        irreversible_switch=True,
        eval_mode=False,
        seed=None,
        render=bool(render),
    )


def write_csv(path: str, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Trajectory rollout
# ---------------------------------------------------------------------------

def rollout_trajectories(act_fn, env, seeds, budgets_seq, max_horizon: int,
                          render: bool, agent_name: str,
                          non_time_aware: bool = False):
    """
    One episode per seed.  Returns a list of **per-step** rows.

    Each row contains the (x, y) position of the robot at that step, the
    signed distance to the nearest hazard boundary, cumulative cost, etc.
    """
    step_rows = []

    for ep_idx, s in enumerate(seeds):
        B = int(budgets_seq[ep_idx])
        o = reset_with_seed(env, int(s))

        if not non_time_aware:
            # Override the budget that reset() set internally
            env.B = B
            o[-1] = env._budget_norm()

        done          = False
        ep_len        = 0
        cost_cum      = 0.0
        goal_met_ever = False
        # For non-time-aware agents the env has no budget termination, so cap
        # the episode at the per-episode budget B (same as what the wrapper
        # does internally for time-aware agents).
        effective_horizon = min(max_horizon, B) if non_time_aware else max_horizon

        # Record the starting position (step 0)
        base_env = env if non_time_aware else env.unwrapped
        try:
            robot_xy        = np.array(base_env.robot_pos[:2])
            hazard_positions = base_env.hazards_pos
            dist_to_hazard   = min(
                np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                for hp in hazard_positions
            )
        except Exception:
            robot_xy       = np.array([float("nan"), float("nan")])
            dist_to_hazard = float("nan")

        step_rows.append({
            "agent":          agent_name,
            "budget":         B,
            "episode_idx":    int(ep_idx),
            "seed":           int(s),
            "step":           0,
            "robot_x":        float(robot_xy[0]),
            "robot_y":        float(robot_xy[1]),
            "dist_to_hazard": float(dist_to_hazard),
            "cost_step":      0.0,
            "cost_cumulative":0.0,
            "goal_met":       0,
            "switched":       0,
            "switch_event":   0,
            "switch_step":    -1,
            "active_policy":  "fixed",
        })

        while (not done) and (ep_len < effective_horizon):
            a = act_fn(o.reshape(1, -1))[0]
            o, _, done, info = env.step(a)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            ep_len   += 1
            step_cost = float(info.get("cost", 0.0))
            cost_cum += step_cost
            gm        = int(bool(info.get("goal_met", False)))
            if gm:
                goal_met_ever = True

            # Robot position after the step
            try:
                robot_xy        = np.array(base_env.robot_pos[:2])
                hazard_positions = base_env.hazards_pos
                dist_to_hazard   = min(
                    np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                    for hp in hazard_positions
                )
            except Exception:
                robot_xy       = np.array([float("nan"), float("nan")])
                dist_to_hazard = float("nan")

            step_rows.append({
                "agent":           agent_name,
                "budget":          B,
                "episode_idx":     int(ep_idx),
                "seed":            int(s),
                "step":            int(ep_len),
                "robot_x":         float(robot_xy[0]),
                "robot_y":         float(robot_xy[1]),
                "dist_to_hazard":  float(dist_to_hazard),
                "cost_step":       step_cost,
                "cost_cumulative": float(cost_cum),
                "goal_met":        gm,
                "switched":        0,
                "switch_event":    0,
                "switch_step":     -1,
                "active_policy":   "fixed",
            })

        success = int(goal_met_ever)
        print(
            f"  ep {ep_idx:3d}  seed={s:10d}  budget={B}  "
            f"steps={ep_len}  success={success}  cost={cost_cum:.1f}"
        )

    return step_rows


def rollout_trajectories_switch_classifier(
    classifier_model,
    classifier_input_dim: int,
    meta_env,
    switch_prob_threshold: float,
    seeds,
    budgets_seq,
    max_horizon: int,
    agent_name: str,
    switch_feature_history: int,
):
    """Collect per-step trajectories for classifier-based irreversible switching via MetaEnv."""
    step_rows = []
    p_thr = float(np.clip(switch_prob_threshold, 1e-6, 1.0 - 1e-6))

    for ep_idx, s in enumerate(seeds):
        B = int(budgets_seq[ep_idx])
        meta_env.seed(int(s))
        try:
            meta_env._env.seed(int(s))
        except Exception:
            pass

        o = meta_env.reset().copy()
        meta_env.B = B
        # Align budget feature with the shared per-episode sampled budget.
        if meta_env.irreversible_switch:
            o[-2] = meta_env._budget_norm()
            o[-1] = 0.0
        else:
            o[-1] = meta_env._budget_norm()

        done = False
        ep_len = 0
        cost_cum = 0.0
        goal_met_ever = False
        switched = bool(getattr(meta_env, "_switched", False))
        switch_step = -1
        hist_buffer = None
        inferred_history = None

        base_env = meta_env._env
        try:
            robot_xy = np.array(base_env.robot_pos[:2])
            hazard_positions = base_env.hazards_pos
            dist_to_hazard = min(
                np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                for hp in hazard_positions
            )
        except Exception:
            robot_xy = np.array([float("nan"), float("nan")])
            dist_to_hazard = float("nan")

        step_rows.append({
            "agent":           agent_name,
            "budget":          B,
            "episode_idx":     int(ep_idx),
            "seed":            int(s),
            "step":            0,
            "robot_x":         float(robot_xy[0]),
            "robot_y":         float(robot_xy[1]),
            "dist_to_hazard":  float(dist_to_hazard),
            "cost_step":       0.0,
            "cost_cumulative": 0.0,
            "goal_met":        0,
            "switched":        int(switched),
            "switch_event":    0,
            "switch_step":     int(switch_step),
            "active_policy":   "conservative",
        })

        while (not done) and (ep_len < max_horizon):
            switch_event = 0
            if switch_step < 0:
                feats = extract_features(o, meta_env).astype(np.float32, copy=False)
                feats_in, used_history, hist_buffer = build_classifier_input(
                    base_feats=feats,
                    classifier_input_dim=int(classifier_input_dim),
                    feature_history=int(switch_feature_history),
                    hist_buffer=hist_buffer,
                )
                if inferred_history is None:
                    inferred_history = int(used_history)
                with torch.no_grad():
                    x = torch.tensor(feats_in, dtype=torch.float32).unsqueeze(0)
                    logit = float(classifier_model(x).item())
                    p_switch = 1.0 / (1.0 + np.exp(-logit))
                action = 1 if p_switch > p_thr else 0
                if action == 1:
                    switch_event = 1
                    switch_step = int(ep_len + 1)
                if hist_buffer is not None:
                    hist_buffer.append(feats)
            else:
                action = 1

            o, _, done, info = meta_env.step(action)

            steps_taken = int(info.get("n_steps_taken", 1))
            ep_len += steps_taken

            step_cost = float(info.get("cumulative_cost", 0.0))
            cost_cum += step_cost
            gm = int(bool(info.get("goal_met", False)))
            if gm:
                goal_met_ever = True
            switched = bool(info.get("switched", switched))
            active_policy = str(info.get("active_policy", "conservative"))

            try:
                robot_xy = np.array(base_env.robot_pos[:2])
                hazard_positions = base_env.hazards_pos
                dist_to_hazard = min(
                    np.linalg.norm(robot_xy - np.array(hp[:2])) - base_env.hazards_size
                    for hp in hazard_positions
                )
            except Exception:
                robot_xy = np.array([float("nan"), float("nan")])
                dist_to_hazard = float("nan")

            step_rows.append({
                "agent":           agent_name,
                "budget":          B,
                "episode_idx":     int(ep_idx),
                "seed":            int(s),
                "step":            int(min(ep_len, max_horizon)),
                "robot_x":         float(robot_xy[0]),
                "robot_y":         float(robot_xy[1]),
                "dist_to_hazard":  float(dist_to_hazard),
                "cost_step":       step_cost,
                "cost_cumulative": float(cost_cum),
                "goal_met":        gm,
                "switched":        int(switched),
                "switch_event":    int(switch_event),
                "switch_step":     int(switch_step),
                "active_policy":   active_policy,
            })

        success = int(goal_met_ever)
        print(
            f"  ep {ep_idx:3d}  seed={s:10d}  budget={B}  steps={ep_len}  "
            f"switch_step={switch_step}  h={inferred_history if inferred_history is not None else 0}  "
            f"success={success}  cost={cost_cum:.1f}"
        )

    return step_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate time-aware agents and record full (x,y) trajectories "
            "for trajectory plotting."
        )
    )
    p.add_argument("--agent_dirs", type=str, nargs="+", required=True,
                   help="Paths to SavedModel directories.")
    p.add_argument("--agent_names", type=str, nargs="+", default=None,
                   help="Display names (must match --agent_dirs). "
                        "Defaults to directory basename.")

    q = p.add_argument_group("Quantile budget selection")
    q.add_argument("--quantile_low",  type=float, default=0.0,
                   help="Lower quantile bound (default: 0.0).")
    q.add_argument("--quantile_high", type=float, default=1.0,
                   help="Upper quantile bound (default: 1.0).")

    p.add_argument("--budget_min",  type=int,   default=TRAIN_BUDGET_MIN)
    p.add_argument("--budget_max",  type=int,   default=TRAIN_BUDGET_MAX)
    p.add_argument("--budget_step", type=int,   default=TRAIN_BUDGET_STEP)
    p.add_argument("--episodes",    type=int,   default=50,
                   help="Episodes per agent (default: 50).")
    p.add_argument("--base_seed",   type=int,   default=0,
                   help="RNG seed for episode seeds (default: 0).")
    p.add_argument("--budget_seed", type=int,   default=99,
                   help="RNG seed for budget sampling (default: 99).")
    p.add_argument("--max_horizon", type=int,   default=0,
                   help="Max steps per episode (0 = budget_max).")
    p.add_argument("--render",      action="store_true")
    p.add_argument("--non_time_aware", action="store_true",
                   help="Evaluate agents that do NOT use the time/budget "
                        "observation (no TimeBudgetWrapper).")
    p.add_argument("--deadline_penalty", type=float, default=0.0)
    p.add_argument("--results_dir", type=str,   default="results")
    p.add_argument("--tag",         type=str,   default="",
                   help="Optional suffix in the output filename.")

    # Optional: add classifier-based switch agent (cons -> agg)
    p.add_argument("--switch_classifier_ckpt", type=str, default="",
                   help="Optional path to gap classifier checkpoint (.pt).")
    p.add_argument("--switch_cons_dir", type=str, default="",
                   help="Conservative policy dir used by the switch classifier agent.")
    p.add_argument("--switch_agg_dir", type=str, default="",
                   help="Aggressive policy dir used by the switch classifier agent.")
    p.add_argument("--switch_prob_threshold", type=float, default=0.5,
                   help="Switch when classifier probability > threshold.")
    p.add_argument("--switch_prob_thresholds", type=float, nargs="+", default=None,
                   help="Optional list of switch thresholds. If set, evaluates one switch agent per threshold.")
    p.add_argument("--switch_agent_name", type=str, default="gap_switch_classifier",
                   help="Name written in CSV for the classifier-based switch agent.")
    p.add_argument("--switch_feature_history", type=int, default=-1,
                   help="Number of past feature vectors concatenated for the switch classifier. "
                        "Use -1 to infer automatically from checkpoint input_dim (default: -1).")
    p.add_argument("--switch_meta_interval", type=int, default=1,
                   help="MetaEnv interval (env steps per classifier decision) for switch-agent evaluation.")
    args = p.parse_args()

    use_switch_classifier = bool(str(args.switch_classifier_ckpt).strip())
    if use_switch_classifier:
        if args.non_time_aware:
            p.error("--switch_classifier_ckpt is incompatible with --non_time_aware.")
        if not str(args.switch_cons_dir).strip() or not str(args.switch_agg_dir).strip():
            p.error("When --switch_classifier_ckpt is set, you must also set --switch_cons_dir and --switch_agg_dir.")
        if int(args.switch_meta_interval) != 1:
            p.error(
                "In trajectory mode, --switch_meta_interval must be 1 to preserve true per-step trajectories."
            )

    if args.switch_prob_thresholds is not None and len(args.switch_prob_thresholds) > 0:
        switch_thresholds = [float(t) for t in args.switch_prob_thresholds]
    else:
        switch_thresholds = [float(args.switch_prob_threshold)]

    for t in switch_thresholds:
        if not (0.0 <= t <= 1.0):
            p.error(f"Each switch threshold must be in [0, 1]. Invalid value: {t}")

    # Validate quantiles
    for attr, name in [("quantile_low", "--quantile_low"),
                       ("quantile_high", "--quantile_high")]:
        v = getattr(args, attr)
        if not (0.0 <= v <= 1.0):
            p.error(f"{name} must be in [0, 1].")
    if args.quantile_low > args.quantile_high:
        p.error("--quantile_low must be <= --quantile_high.")

    # Budget pool
    budget_pool = budgets_from_quantiles(
        args.budget_min, args.budget_max, args.budget_step,
        args.quantile_low, args.quantile_high,
    )
    if not budget_pool:
        p.error("Quantile range produced an empty budget pool.")

    # Agent names
    if args.agent_names is not None:
        if len(args.agent_names) != len(args.agent_dirs):
            p.error("--agent_names must have the same length as --agent_dirs.")
        agent_names = args.agent_names
    else:
        agent_names = [os.path.basename(os.path.normpath(d))
                       for d in args.agent_dirs]

    ensure_dir(args.results_dir)
    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max

    # Shared seeds & budgets across agents for fair comparison
    rng         = np.random.RandomState(args.base_seed)
    seeds       = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)
    budget_rng  = np.random.RandomState(args.budget_seed)
    budgets_seq = budget_rng.choice(budget_pool, size=args.episodes, replace=True)

    q_lo_pct = int(round(args.quantile_low  * 100))
    q_hi_pct = int(round(args.quantile_high * 100))
    q_tag = (f"q{q_lo_pct}to{q_hi_pct}"
             if args.quantile_low != args.quantile_high
             else f"q{q_lo_pct}")

    print(
        f"Trajectory evaluation\n"
        f"  fixed agents   : {len(args.agent_dirs)}\n"
        f"  quantile range : [{args.quantile_low:.3f}, {args.quantile_high:.3f}]  "
        f"→  budgets {budget_pool[0]}–{budget_pool[-1]}  ({len(budget_pool)} values)\n"
        f"  episodes       : {args.episodes}  (shared seeds across agents)\n"
        f"  max_horizon    : {max_horizon}\n"
    )

    n_switch_agents = len(switch_thresholds) if use_switch_classifier else 0
    n_agents_total = len(args.agent_dirs) + n_switch_agents

    if use_switch_classifier:
        print(
            f"  switch agents  : {n_switch_agents} variant(s)\n"
            f"    classifier   : {args.switch_classifier_ckpt}\n"
            f"    cons_dir     : {args.switch_cons_dir}\n"
            f"    agg_dir      : {args.switch_agg_dir}\n"
            f"    p_thr list   : {switch_thresholds}\n"
        )

    base = (
        f"trajectories_{q_tag}_"
        f"seed{args.base_seed}_eps{args.episodes}"
        f"_Bmin{args.budget_min}_Bmax{args.budget_max}"
        f"_{n_agents_total}agents"
    )
    if args.tag:
        base += f"_{args.tag}"
    out_csv = os.path.join(args.results_dir, base + ".csv")

    fieldnames = [
        "agent", "budget", "episode_idx", "seed", "step",
        "robot_x", "robot_y", "dist_to_hazard",
        "cost_step", "cost_cumulative", "goal_met",
        "switched", "switch_event", "switch_step", "active_policy",
    ]

    all_rows = []

    for agent_dir, agent_name in zip(args.agent_dirs, agent_names):
        print(f"\n--- Agent: {agent_name}  ({agent_dir}) ---")
        env  = None
        sess = None
        try:
            sess, act = load_deterministic_policy(agent_dir)
            env = make_env(args.budget_min, args.budget_max, args.deadline_penalty,
                          non_time_aware=args.non_time_aware)
            rows = rollout_trajectories(
                act_fn=act,
                env=env,
                seeds=seeds,
                budgets_seq=budgets_seq,
                max_horizon=max_horizon,
                render=args.render,
                agent_name=agent_name,
                non_time_aware=args.non_time_aware,
            )
            all_rows.extend(rows)
            print(f"  → {len(rows)} step-rows collected.")
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
        meta_env = None
        try:
            classifier_model, classifier_input_dim = load_switch_classifier(args.switch_classifier_ckpt)
            sess_cons, cons_act = load_deterministic_policy(args.switch_cons_dir)
            sess_agg, agg_act = load_deterministic_policy(args.switch_agg_dir)

            print(
                "  switch features : "
                f"checkpoint input_dim={classifier_input_dim}, "
                f"requested history={args.switch_feature_history}"
            )

            for thr in switch_thresholds:
                thr_label = f"{thr:.3f}".rstrip("0").rstrip(".")
                if len(switch_thresholds) == 1:
                    switch_agent_name = args.switch_agent_name
                else:
                    switch_agent_name = f"{args.switch_agent_name}_pthr{thr_label}"

                print(f"\n--- Agent: {switch_agent_name}  (classifier switch, p_thr={thr}) ---")
                meta_env = None
                try:
                    meta_env = make_meta_env(
                        cons_act_fn=cons_act,
                        agg_act_fn=agg_act,
                        meta_interval=int(args.switch_meta_interval),
                        budget_min=args.budget_min,
                        budget_max=args.budget_max,
                        deadline_penalty=args.deadline_penalty,
                        render=args.render,
                    )
                    rows = rollout_trajectories_switch_classifier(
                        classifier_model=classifier_model,
                        classifier_input_dim=int(classifier_input_dim),
                        meta_env=meta_env,
                        switch_prob_threshold=float(thr),
                        seeds=seeds,
                        budgets_seq=budgets_seq,
                        max_horizon=max_horizon,
                        agent_name=switch_agent_name,
                        switch_feature_history=int(args.switch_feature_history),
                    )
                    all_rows.extend(rows)
                    print(f"  → {len(rows)} step-rows collected.")
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

    write_csv(out_csv, all_rows, fieldnames)
    print(f"\nSaved CSV ({len(all_rows)} step-rows): {out_csv}")


if __name__ == "__main__":
    main()
