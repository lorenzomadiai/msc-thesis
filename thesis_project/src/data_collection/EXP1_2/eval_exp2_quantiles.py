#!/usr/bin/env python3
"""
eval_with_quantile_budget.py

Like eval_with_random_settings.py, but the time budget B is sampled from a
**quantile-restricted** subset of the training budget distribution.

The full training budget set is:
    {budget_min, budget_min+step, ..., budget_max}   (discrete uniform)

Given --quantile_low q_lo and --quantile_high q_hi (both in [0, 1]), only the
budget values that fall within those quantiles are kept and sampled uniformly.

Examples:
  # Sample only from the lower quartile of budgets (140..170)
  python eval_with_quantile_budget.py \\
      --agent_dirs /path/to/agent1 /path/to/agent2 \\
      --agent_names agent1 agent2 \\
      --quantile_low 0.0 --quantile_high 0.25 \\
      --episodes 300 --results_dir results/

  # Sample only from the median budget (single value)
  python eval_with_quantile_budget.py \\
      --agent_dirs /path/to/agent1 \\
      --quantile_low 0.5 --quantile_high 0.5 \\
      --episodes 300 --results_dir results/

  # Sample from the upper half of budgets (200..260)
  python eval_with_quantile_budget.py \\
      --agent_dirs /path/to/agent1 /path/to/agent2 \\
      --quantile_low 0.5 --quantile_high 1.0 \\
      --episodes 300 --results_dir results/
"""
import os
import csv
import argparse
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.training.supervised_learning.common import features as switch_features


# Identical to training config in wcsac_timeaware.py
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
# Quantile budget selection
# ---------------------------------------------------------------------------

def budgets_from_quantiles(budget_min: int, budget_max: int, step: int,
                           q_low: float, q_high: float) -> list:
    """
    Return the subset of budget values in [budget_min, budget_max] (step=step)
    that fall within the [q_low, q_high] quantile range of the full discrete
    uniform distribution.

    If q_low == q_high a single closest value is returned.
    """
    all_budgets = np.arange(budget_min, budget_max + 1, step, dtype=int)
    n = len(all_budgets)

    if n == 0:
        raise ValueError("Empty budget grid — check budget_min/max/step.")

    # Map quantile to index (numpy-style linear interpolation)
    idx_lo = int(np.floor(q_low * (n - 1)))
    idx_hi = int(np.ceil(q_high * (n - 1)))

    # Clamp
    idx_lo = max(0, min(idx_lo, n - 1))
    idx_hi = max(0, min(idx_hi, n - 1))

    selected = all_budgets[idx_lo: idx_hi + 1].tolist()
    return selected


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

    def __init__(self, hidden_size: int = 16, input_size: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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

    input_size = int(state_dict["net.0.weight"].shape[1])
    hidden_size = int(state_dict["net.0.weight"].shape[0])
    model = DeltaNet(hidden_size=hidden_size, input_size=input_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(
        f"Loaded switch classifier: {checkpoint_path} "
        f"(hidden_size={hidden_size}, input_dim={input_size})"
    )
    return model, input_size


def extract_legacy_switch_features(obs: np.ndarray, env) -> np.ndarray:
    """Build the 7-D feature vector used by the switch classifier."""
    base_env = env.unwrapped
    robot_vel = base_env.sim.data.get_body_xvelp("robot")
    v_x = float(robot_vel[0])
    v_y = float(robot_vel[1])

    goal_pos = np.array([1.1, 1.1], dtype=np.float32)
    haz_pos = np.array([0.0, 0.0], dtype=np.float32)
    d_max = float(np.linalg.norm(np.array([3.0, 3.0], dtype=np.float32)))

    robot_pos = base_env.sim.data.get_body_xpos("robot")[:2]
    d_goal = float(np.linalg.norm(robot_pos - goal_pos))
    d_haz = float(np.linalg.norm(robot_pos - haz_pos))

    vec_goal = goal_pos - robot_pos
    vec_haz = haz_pos - robot_pos
    angle_goal = np.arctan2(vec_goal[1], vec_goal[0])
    angle_haz = np.arctan2(vec_haz[1], vec_haz[0])
    delta_theta = float(angle_goal - angle_haz)
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

    d_goal_norm = d_goal / d_max
    d_haz_norm = d_haz / d_max
    delta_theta_norm = delta_theta / np.pi

    time_left_norm = float(obs[-2])
    budget_norm = float(obs[-1])

    return np.array(
        [v_x, v_y, d_goal_norm, d_haz_norm, delta_theta_norm, time_left_norm, budget_norm],
        dtype=np.float32,
    )


def get_switch_feature_fn(feature_dim: int):
    if feature_dim == 7:
        return extract_legacy_switch_features
    if feature_dim == switch_features.N_FEATURES:
        return switch_features.extract_lidar_features
    raise ValueError(
        f"Unsupported classifier input dimension {feature_dim}. "
        f"Known options: 7 (legacy), {switch_features.N_FEATURES} (lidar)."
    )


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
    Creates the base environment. The wrapper will be used only to provide the
    observation format; the actual budget per episode is overridden by the
    rollout loop using quantile-sampled values.
    """
    base_env = Engine(TRAIN_CONFIG)
    env = TimeBudgetWrapper(
        base_env,
        budget_min=budget_min,
        budget_max=budget_max,
        deadline_penalty=float(deadline_penalty),
        eval_mode=True,
        eval_max_budget=budget_max,  # max possible budget in the training distribution
    )
    return env


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

def rollout_collect(act_fn, env, seeds, budgets_seq, max_horizon: int,
                    render: bool, agent_name: str):
    """
    Runs one episode per seed.

    At each episode the time budget B is taken from budgets_seq (pre-generated
    and shared across all agents). Because the wrapper's reset() always
    overwrites env.B internally, we force our value after the reset and
    patch the last obs element (budget_norm) accordingly.
    """
    out_rows = []

    for ep_idx, s in enumerate(seeds):
        B = int(budgets_seq[ep_idx])
        o = reset_with_seed(env, int(s))
        # reset() overwrites env.B — force our shared budget after reset
        env.B = B
        # Fix budget_norm in obs (last element); time_left_norm at t=0 is always 1.0
        o[-1] = env._budget_norm()

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

        success = int(goal_first_step != -1 and goal_first_step <= B)
        mean_dist_hazard = float(dist_hazard_sum / ep_len) if ep_len > 0 else float("nan")

        # print(f"Episode {ep_idx:3d}  seed={s}  budget={B}" 
        #       f"  ep_len={ep_len}  success={success}  cost_total={cum:.2f}"
        #       f"  mean_dist_hazard={mean_dist_hazard:.3f}")
        row = {
            "agent": agent_name,
            "budget": B,
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


def rollout_collect_switch_classifier(
    classifier_model,
    cons_act_fn,
    agg_act_fn,
    switch_prob_threshold: float,
    feature_fn,
    env,
    seeds,
    budgets_seq,
    max_horizon: int,
    render: bool,
    agent_name: str,
):
    """Evaluate switch classifier with irreversible conservative->aggressive logic."""
    out_rows = []
    p_thr = float(np.clip(switch_prob_threshold, 1e-6, 1.0 - 1e-6))

    for ep_idx, s in enumerate(seeds):
        B = int(budgets_seq[ep_idx])
        o = reset_with_seed(env, int(s))
        env.B = B
        o[-1] = env._budget_norm()

        done = False
        ep_len = 0
        goal_first_step = -1
        switched = False

        cost_cum = np.zeros(max_horizon, dtype=np.float32)
        cum = 0.0
        dist_hazard_sum = 0.0

        while (not done) and (ep_len < max_horizon):
            if not switched:
                feats = feature_fn(o, env)
                with torch.no_grad():
                    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                    logit = float(classifier_model(x).item())
                    p_switch = 1.0 / (1.0 + np.exp(-logit))
                if p_switch > p_thr:
                    switched = True

            if switched:
                a = agg_act_fn(o.reshape(1, -1))[0]
            else:
                a = cons_act_fn(o.reshape(1, -1))[0]

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

        success = int(goal_first_step != -1 and goal_first_step <= B)
        mean_dist_hazard = float(dist_hazard_sum / ep_len) if ep_len > 0 else float("nan")

        row = {
            "agent": agent_name,
            "budget": B,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate time-aware agents with the budget sampled from a "
            "quantile-restricted subset of the training distribution."
        )
    )
    p.add_argument("--agent_dirs", type=str, nargs="+", required=True,
                   help="One or more paths to SavedModel directories.")
    p.add_argument("--agent_names", type=str, nargs="+", default=None,
                   help="Display names for each agent (must match --agent_dirs). "
                        "Defaults to basename of each agent_dir.")

    # Quantile parameters
    q_group = p.add_argument_group("Quantile budget selection")
    q_group.add_argument("--quantile_low", type=float, default=0.0,
                         help="Lower quantile bound for budget sampling (default: 0.0). "
                              "E.g. 0.25 = first quartile.")
    q_group.add_argument("--quantile_high", type=float, default=1.0,
                         help="Upper quantile bound for budget sampling (default: 1.0). "
                              "Set equal to --quantile_low for a single budget value.")

    p.add_argument("--budget_min", type=int, default=TRAIN_BUDGET_MIN,
                   help=f"Min budget of the full training distribution (default: {TRAIN_BUDGET_MIN}).")
    p.add_argument("--budget_max", type=int, default=TRAIN_BUDGET_MAX,
                   help=f"Max budget of the full training distribution (default: {TRAIN_BUDGET_MAX}).")
    p.add_argument("--budget_step", type=int, default=TRAIN_BUDGET_STEP,
                   help=f"Budget grid step (default: {TRAIN_BUDGET_STEP}).")
    p.add_argument("--episodes", type=int, default=300,
                   help="Number of evaluation episodes per agent (default: 300).")
    p.add_argument("--base_seed", type=int, default=0,
                   help="Seed for generating per-episode seeds (shared across agents).")
    p.add_argument("--budget_seed", type=int, default=99,
                   help="Separate seed for the budget sampler (default: 99).")
    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max steps stored per episode. If 0, uses budget_max.")
    p.add_argument("--render", action="store_true",
                   help="Render the environment during evaluation.")
    p.add_argument("--deadline_penalty", type=float, default=0.0,
                   help="Deadline penalty (not applied in eval_mode).")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Output directory for the CSV (default: results).")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix appended to the output filename.")

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
    args = p.parse_args()

    use_switch_classifier = bool(str(args.switch_classifier_ckpt).strip())
    if use_switch_classifier:
        if not str(args.switch_cons_dir).strip() or not str(args.switch_agg_dir).strip():
            p.error("When --switch_classifier_ckpt is set, you must also set --switch_cons_dir and --switch_agg_dir.")

    if args.switch_prob_thresholds is not None and len(args.switch_prob_thresholds) > 0:
        switch_thresholds = [float(t) for t in args.switch_prob_thresholds]
    else:
        switch_thresholds = [float(args.switch_prob_threshold)]

    for t in switch_thresholds:
        if not (0.0 <= t <= 1.0):
            p.error(f"Each switch threshold must be in [0, 1]. Invalid value: {t}")

    # --- Validate quantiles ---
    if not (0.0 <= args.quantile_low <= 1.0):
        p.error("--quantile_low must be in [0, 1].")
    if not (0.0 <= args.quantile_high <= 1.0):
        p.error("--quantile_high must be in [0, 1].")
    if args.quantile_low > args.quantile_high:
        p.error("--quantile_low must be <= --quantile_high.")

    # --- Build quantile-restricted budget pool ---
    budget_pool = budgets_from_quantiles(
        args.budget_min, args.budget_max, args.budget_step,
        args.quantile_low, args.quantile_high,
    )
    if not budget_pool:
        p.error("Quantile range produced an empty budget pool — adjust --quantile_low/high.")

    # --- Validate agent names ---
    if args.agent_names is not None:
        if len(args.agent_names) != len(args.agent_dirs):
            p.error("--agent_names must have the same number of entries as --agent_dirs.")
        agent_names = args.agent_names
    else:
        agent_names = [os.path.basename(os.path.normpath(d)) for d in args.agent_dirs]

    ensure_dir(args.results_dir)
    max_horizon = int(args.max_horizon) if int(args.max_horizon) > 0 else args.budget_max

    # Shared episode seeds (same across agents for fair comparison)
    rng = np.random.RandomState(args.base_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)

    # Pre-generate the shared budget sequence once — same for every agent
    budget_rng = np.random.RandomState(args.budget_seed)
    budgets_seq = budget_rng.choice(budget_pool, size=args.episodes, replace=True)

    q_lo_pct = int(round(args.quantile_low * 100))
    q_hi_pct = int(round(args.quantile_high * 100))

    n_switch_agents = len(switch_thresholds) if use_switch_classifier else 0
    n_agents_total = len(args.agent_dirs) + n_switch_agents

    print(
        f"Evaluation settings:\n"
        f"  agents            : {n_agents_total}\n"
        f"  quantile range    : [{args.quantile_low:.3f}, {args.quantile_high:.3f}]  "
        f"→  budgets {budget_pool[0]}–{budget_pool[-1]}  ({len(budget_pool)} values)\n"
        f"  full budget grid  : {args.budget_min}–{args.budget_max}  step={args.budget_step}\n"
        f"  episodes          : {args.episodes}  (shared seeds across agents)\n"
        f"  max_horizon       : {max_horizon}\n"
    )

    if use_switch_classifier:
        print(
            f"  switch-agent      : enabled ({n_switch_agents} variant(s))\n"
            f"    classifier      : {args.switch_classifier_ckpt}\n"
            f"    cons_dir        : {args.switch_cons_dir}\n"
            f"    agg_dir         : {args.switch_agg_dir}\n"
            f"    p_thr list      : {switch_thresholds}\n"
        )

    # Output filename encodes the quantile range
    q_tag = (
        f"q{q_lo_pct}to{q_hi_pct}"
        if args.quantile_low != args.quantile_high
        else f"q{q_lo_pct}"
    )
    base = (
        f"quantile_{q_tag}_"
        f"seed{args.base_seed}_eps{args.episodes}"
        f"_Bmin{args.budget_min}_Bmax{args.budget_max}"
        f"_{n_agents_total}agents"
    )
    if args.tag:
        base += f"_{args.tag}"
    out_csv = os.path.join(args.results_dir, base + ".csv")

    all_rows = []

    for agent_dir, agent_name in zip(args.agent_dirs, agent_names):
        print(f"\n--- Agent: {agent_name}  ({agent_dir}) ---")
        env = None
        sess = None
        try:
            sess, act = load_deterministic_policy(agent_dir)
            env = make_env(args.budget_min, args.budget_max, args.deadline_penalty)
            rows = rollout_collect(
                act_fn=act,
                env=env,
                seeds=seeds,
                budgets_seq=budgets_seq,
                max_horizon=max_horizon,
                render=args.render,
                agent_name=agent_name,
            )
            all_rows.extend(rows)
            print(f"    done — {len(rows)} episodes collected.")
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
        env = None
        try:
            classifier_model, classifier_in_dim = load_switch_classifier(args.switch_classifier_ckpt)
            feature_fn = get_switch_feature_fn(classifier_in_dim)
            sess_cons, cons_act = load_deterministic_policy(args.switch_cons_dir)
            sess_agg, agg_act = load_deterministic_policy(args.switch_agg_dir)

            for thr in switch_thresholds:
                thr_label = f"{thr:.3f}".rstrip("0").rstrip(".")
                if len(switch_thresholds) == 1:
                    switch_agent_name = args.switch_agent_name
                else:
                    switch_agent_name = f"{args.switch_agent_name}_pthr{thr_label}"

                print(f"\n--- Agent: {switch_agent_name}  (classifier switch, p_thr={thr}) ---")
                env = None
                try:
                    env = make_env(args.budget_min, args.budget_max, args.deadline_penalty)
                    rows = rollout_collect_switch_classifier(
                        classifier_model=classifier_model,
                        cons_act_fn=cons_act,
                        agg_act_fn=agg_act,
                        switch_prob_threshold=float(thr),
                        feature_fn=feature_fn,
                        env=env,
                        seeds=seeds,
                        budgets_seq=budgets_seq,
                        max_horizon=max_horizon,
                        render=args.render,
                        agent_name=switch_agent_name,
                    )
                    all_rows.extend(rows)
                    print(f"    done — {len(rows)} episodes collected.")
                finally:
                    if env is not None:
                        try:
                            env.close()
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

    fieldnames = (
        ["agent", "budget", "episode_idx", "seed", "ep_len", "goal_first_step",
         "success", "cost_total", "mean_dist_hazard"]
        + [f"cost_cum_{t}" for t in range(1, max_horizon + 1)]
    )
    write_csv(out_csv, all_rows, fieldnames)
    print(f"\nSaved CSV ({len(all_rows)} total rows): {out_csv}")


if __name__ == "__main__":
    main()
