#!/usr/bin/env python3
"""
train_threshold_mlp.py
----------------------
Trains an MLP-based switching policy (conservative → aggressive) using CEM.

Instead of a hand-crafted linear formula, a small neural network maps
7 geometric features to a switch probability:

    features = [v_x, v_y, d_goal, d_haz, Δθ, t_frac, budget_norm]
    p(switch) = MLP(features)          # sigmoid output ∈ [0, 1]
    action    = 1  if p > 0.5  else 0  # deterministic at eval time

Architecture:  7 → hidden_size → 1   (tanh hidden, sigmoid output)
    hidden_size=4  → 37 parameters
    hidden_size=8  → 73 parameters

Geometric features (distances + angles from sim state) are more precise
than lidar-based closeness values. The MLP can learn non-linear
interactions like:
    "hazard close + moving fast towards it + little time → switch"
    "hazard far + goal aligned                          → stay"
which a linear formula cannot express.

Optimisation: Cross-Entropy Method (CEM) on the flattened weight vector.
Fitness:      R_i = success_i - deadline_weight*(1-success_i) - cost_weight*cost_i

Usage
-----
python src/training/switching_policies/train_threshold_mlp.py \\
    --cons_dir  WCSAC/.../simple_save6 \\
    --agg_dir   WCSAC/.../simple_save9 \\
    --episodes 200 --hidden_size 4 \\
    --cem_iterations 30 --cem_pop_size 80 \\
    --deadline_weight 1.0 \\
    --budget_min 120 --budget_max 220 \\
    --results_dir results/threshold/mlp_001
"""

import os
import sys
import csv
import json
import argparse
import warnings
import multiprocessing as mp
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from safety_gym.envs.engine import Engine

# Local modules
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from riskawarerl.thesis_project.src.training.switching_policies.supervised_learning.meta_env import MetaEnv


# ---------------------------------------------------------------------------
# Safety-Gym config
# ---------------------------------------------------------------------------

STATIC_CONFIG = {
    "placements_extents": [-1.5, -1.5, 1.5, 1.5],

    # # Note: the "robot_keepout" is set to 0 here, which means the robot can
    # # start anywhere within the placements_extents.
    # "robot_placements": [(-1.5, -1.5, 0.0, 0.0)],

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
# Low-level policy loader (TF1 SavedModel)
# ---------------------------------------------------------------------------

def _pick_signature(meta_graph_def):
    sigs = meta_graph_def.signature_def
    for k in ("serving_default", "serve", "default"):
        if k in sigs:
            return sigs[k]
    if not sigs:
        raise RuntimeError("No signature_def found in SavedModel.")
    return sigs[next(iter(sigs.keys()))]


def load_policy(saved_model_dir: str):
    """Load a frozen TF1 SavedModel and return (session, act_fn)."""
    print(f"  Loading policy: {saved_model_dir}")
    pb = os.path.join(saved_model_dir, "saved_model.pb")
    if not os.path.exists(pb):
        raise FileNotFoundError(f"saved_model.pb not found in: {saved_model_dir}")
    g    = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        mgd = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir
        )
        sig    = _pick_signature(mgd)
        x_name = (sig.inputs["x"].name if "x" in sig.inputs
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


# ---------------------------------------------------------------------------
# MLP switching policy
# ---------------------------------------------------------------------------

class MLPSwitchPolicy:
    """
    Small MLP that maps 7 geometric features to a switch decision.

    Features (extracted from obs + sim state at every meta-step):
        0: v_x              = velocimeter forward     (body-frame)
        1: v_y              = velocimeter lateral      (body-frame)
        2: d_goal_norm      = dist(robot, goal) / D_MAX   ∈ [0, 1]
        3: d_haz_norm       = dist(robot, haz)  / D_MAX   ∈ [0, 1]
        4: delta_theta_norm = Δθ(goal−haz) / π             ∈ [-1, 1]
        5: t_frac           = t_rem / B                    ∈ [0, 1]
        6: budget_norm      = B / budget_max               ∈ (0, 1]

    Architecture:
        h = tanh(W1 @ x + b1)          W1: (hidden, 7), b1: (hidden,)
        p = sigmoid(W2 @ h + b2)       W2: (1, hidden),  b2: (1,)
        action = 1  if  p > 0.5

    Total parameters = 7*H + H + H + 1 = 9*H + 1
        H=4  → 37
        H=8  → 73
    """

    # Fixed environment positions
    GOAL_POS = np.array([1.1, 1.1])
    HAZ_POS  = np.array([0.0, 0.0])
    D_MAX    = float(np.linalg.norm(np.array([3.0, 3.0])))  # arena diagonal ≈ 4.24

    N_FEATURES = 7

    def __init__(self, weights: np.ndarray, hidden_size: int = 4,
                 budget_max: float = 220.0, env=None):
        """
        Parameters
        ----------
        weights : 1-D array of length 9*hidden_size + 1
        hidden_size : neurons in the single hidden layer
        budget_max : used to normalise budget_norm = B / budget_max
        env : MetaEnv instance (needed for sim access to robot position)
        """
        self.hidden_size = hidden_size
        self.budget_max  = float(budget_max)
        self.env         = env
        self._n_features = 7
        expected = self._n_features * hidden_size + hidden_size + hidden_size + 1
        assert len(weights) == expected, \
            f"Expected {expected} weights for H={hidden_size}, got {len(weights)}"
        self._unpack(weights)

    def _unpack(self, w: np.ndarray):
        """Unpack flat weight vector into W1, b1, W2, b2."""
        H = self.hidden_size
        F = self._n_features
        idx = 0
        self.W1 = w[idx: idx + F * H].reshape(H, F);  idx += F * H
        self.b1 = w[idx: idx + H];                      idx += H
        self.W2 = w[idx: idx + H].reshape(1, H);        idx += H
        self.b2 = w[idx: idx + 1];                       idx += 1

    @classmethod
    def n_params(cls, hidden_size: int = 4) -> int:
        """Number of parameters for a given hidden_size."""
        return 7 * hidden_size + hidden_size + hidden_size + 1

    def _extract_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract 7 geometric features from obs + sim state."""
        # Body-frame velocity (from meta-obs velocimeter, indices 0-2)
        v_x = float(obs[0])
        v_y = float(obs[1])

        # Geometric distances from simulation positions
        robot_pos = self.env._env.sim.data.get_body_xpos('robot')[:2]
        d_goal = float(np.linalg.norm(robot_pos - self.GOAL_POS))
        d_haz  = float(np.linalg.norm(robot_pos - self.HAZ_POS))

        # Angle between goal and hazard directions (from robot)
        vec_goal = self.GOAL_POS - robot_pos
        vec_haz  = self.HAZ_POS  - robot_pos
        angle_goal  = np.arctan2(vec_goal[1], vec_goal[0])
        # print(f"angle_goal in degrees = {np.arctan2(vec_goal[1], vec_goal[0]) * 180 / np.pi:.1f}°")
        # print(f"angle_haz in degrees = {np.arctan2(vec_haz[1], vec_haz[0]) * 180 / np.pi:.1f}°")
        angle_haz   = np.arctan2(vec_haz[1],  vec_haz[0])
        delta_theta = float(angle_goal - angle_haz)
        # Normalise to [-π, π]
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

        # Normalise to NN-friendly ranges
        d_goal_norm      = d_goal / self.D_MAX          # ∈ [0, 1]
        d_haz_norm       = d_haz  / self.D_MAX          # ∈ [0, 1]
        delta_theta_norm = delta_theta / np.pi           # ∈ [-1, 1]

        # Time features (from meta-obs tail)
        time_left_norm = float(obs[-3])
        t_frac         = (time_left_norm + 1.0) / 2.0   # t_rem/B ∈ [0, 1]
        budget_norm    = float(obs[-2])                  # B / budget_max

        return np.array([v_x, v_y, d_goal_norm, d_haz_norm,
                         delta_theta_norm, t_frac, budget_norm],
                        dtype=np.float64)

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass: features (7,) → switch probability (scalar)."""
        h = np.tanh(self.W1 @ x + self.b1)
        logit = float(self.W2 @ h + self.b2)
        return 1.0 / (1.0 + np.exp(-logit))   # sigmoid

    def act(self, obs: np.ndarray, stochastic: bool = False) -> int:
        """Return 0 (conservative) or 1 (switch to aggressive).

        stochastic=True  → Bernoulli(p): smooth fitness landscape for CEM
        stochastic=False → deterministic threshold p > 0.5 (eval time)
        """
        x = self._extract_features(obs)
        p = self._forward(x)
        if stochastic:
            temp = int(np.random.random() < p)
            if temp == 1:
                print(f"  [STOCHASTIC] p={p:.4f} → action=1 (switch)")
            return temp
        return int(p > 0.5)


# ---------------------------------------------------------------------------
# Parallel worker helpers  (one env + TF sessions per process)
# ---------------------------------------------------------------------------

_w_env  = None   # MetaEnv, created once per worker
_w_sess = []     # TF sessions to close at exit


def _worker_init(cons_dir, agg_dir, meta_interval,
                 budget_min, budget_max, budget_step):
    """Create env + TF sessions inside each Pool worker (called once)."""
    global _w_env, _w_sess
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    s_c, fn_c = load_policy(cons_dir)
    s_a, fn_a = load_policy(agg_dir)
    _w_sess = [s_c, s_a]

    def env_fn():
        return Engine(STATIC_CONFIG)

    _w_env = MetaEnv(
        env_fn=env_fn,
        act_fn_cons=fn_c,
        act_fn_agg=fn_a,
        meta_interval=meta_interval,
        budget_min=budget_min,
        budget_max=budget_max,
        budget_step=budget_step,
        irreversible_switch=True,
        seed=os.getpid(),
    )


def _eval_one(packed):
    """Evaluate one candidate — called by Pool.map().

    Returns (score, metrics_dict).
    """
    (weights, hidden_size, budget_max_f, seeds, budgets_seq,
     max_horizon, cost_weight, cvar_alpha, deadline_weight) = packed
    global _w_env

    policy  = MLPSwitchPolicy(weights, hidden_size=hidden_size,
                               budget_max=budget_max_f, env=_w_env)
    metrics = _run_episodes(policy, _w_env, seeds, budgets_seq, max_horizon)
    sc      = _score(metrics, cost_weight=cost_weight,
                     cvar_alpha=cvar_alpha, deadline_weight=deadline_weight)
    return (sc, metrics)


# ---------------------------------------------------------------------------
# Shared episode runner
# ---------------------------------------------------------------------------

def _run_episodes(policy, env: MetaEnv, seeds: np.ndarray,
                  budgets_seq: np.ndarray, max_horizon: int,
                  stochastic: bool = True,
                  debug: bool = False, debug_episodes: int = 3,
                  debug_steps: int = 5) -> dict:
    """Run one episode per seed, return aggregate metrics.

    stochastic : bool
        If True, policy.act() samples from Bernoulli(p) — used during
        CEM training for a smooth fitness landscape.
        If False, uses deterministic threshold p > 0.5 — used at eval.
    """
    successes    = []
    costs        = []
    switch_steps = []

    for ep_idx, seed in enumerate(seeds):
        budget = int(budgets_seq[ep_idx])
        env.seed(int(seed))
        try:
            env._env.seed(int(seed))
        except Exception:
            pass
        obs = env.reset()
        obs = obs.copy()
        env.B = budget
        if env.irreversible_switch:
            obs[-2] = env._budget_norm()
            obs[-1] = 0.0
        else:
            obs[-1] = env._budget_norm()

        if hasattr(policy, 'reset'):
            policy.reset(obs, float(budget))

        # --- Debug: print features at episode start ---
        _dbg = debug and ep_idx < debug_episodes
        if _dbg:
            feats = policy._extract_features(obs)
            rpos = env._env.sim.data.get_body_xpos('robot')[:2]
            print(f"\n    [DBG ep {ep_idx+1}] seed={seed}  B={budget}  "
                  f"robot=({rpos[0]:.2f},{rpos[1]:.2f})")
            feat_names = ["v_x", "v_y", "d_goal", "d_haz",
                          "Δθ", "t_frac", "budget"]
            feat_str = "  ".join(f"{n}={v:.4f}" for n, v
                                 in zip(feat_names, feats))
            print(f"    [DBG ep {ep_idx+1}] step=0  {feat_str}")

        done        = False
        ep_len      = 0
        cum_cost    = 0.0
        switch_step = -1
        switched    = False
        meta_step   = 0

        while not done and ep_len < max_horizon:
            if not switched:
                action = policy.act(obs, stochastic=stochastic)
                if action == 1:
                    switched    = True
                    switch_step = ep_len
            else:
                action = 1   # already switched — skip MLP, stay aggressive
            obs, _r, done, info = env.step(action)
            cum_cost += float(info.get("cumulative_cost", 0.0))
            ep_len   += info.get("n_steps_taken", 1)
            meta_step += 1

            # --- Debug: print features for first N steps ---
            if _dbg and meta_step <= debug_steps:
                feats = policy._extract_features(obs)
                rpos = env._env.sim.data.get_body_xpos('robot')[:2]
                feat_str = "  ".join(
                    f"{n}={v:.4f}" for n, v
                    in zip(["v_x", "v_y", "d_goal", "d_haz",
                            "Δθ", "t_frac", "budget"], feats))
                act_str = "AGG" if action == 1 else "CON"
                print(f"    [DBG ep {ep_idx+1}] step={meta_step}  "
                      f"robot=({rpos[0]:.2f},{rpos[1]:.2f})  "
                      f"act={act_str}  {feat_str}")

        success = bool(info.get("goal_met", False))
        successes.append(int(success))
        costs.append(cum_cost)
        switch_steps.append(switch_step)

    costs_arr     = np.array(costs)
    successes_arr = np.array(successes, dtype=np.float32)
    success_rate  = float(np.mean(successes))
    mean_cost     = float(np.mean(costs_arr))
    frac_switched = float(np.mean([s > 0 for s in switch_steps]))
    mean_sw_step  = float(np.mean([s for s in switch_steps if s > 0])
                          if any(s > 0 for s in switch_steps) else 0.0)

    def _cvar(alpha: float) -> float:
        k = max(1, int(np.ceil(alpha * len(costs_arr))))
        return float(np.mean(np.sort(costs_arr)[-k:]))

    return {
        "success_rate":     success_rate,
        "mean_cost":        mean_cost,
        "cvar_10":          _cvar(0.10),
        "cvar_20":          _cvar(0.20),
        "cvar_30":          _cvar(0.30),
        "frac_switched":    frac_switched,
        "mean_switch_step": mean_sw_step,
        "_costs":           costs_arr,
        "_successes":       successes_arr,
    }


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def _score(metrics: dict, cost_weight: float = 0.01,
           cvar_alpha: float = 0.0,
           deadline_weight: float = 0.0) -> float:
    """Scalar fitness for CEM.

    deadline_weight > 0:
        R_i = success_i - deadline_weight*(1-success_i) - cost_weight*cost_i
        score = mean(R_i)

    deadline_weight == 0:
        score = success_rate - cost_weight * cost_metric
    """
    if deadline_weight > 0.0:
        successes_arr = metrics["_successes"]
        costs_arr     = metrics["_costs"]
        rewards = (successes_arr
                   - deadline_weight * (1.0 - successes_arr)
                   - cost_weight * costs_arr)
        return float(np.mean(rewards))

    if cvar_alpha > 0.0:
        key = f"cvar_{int(cvar_alpha * 100)}"
        if key in metrics:
            cost_metric = metrics[key]
        else:
            costs_arr = metrics["_costs"]
            k = max(1, int(np.ceil(cvar_alpha * len(costs_arr))))
            cost_metric = float(np.mean(np.sort(costs_arr)[-k:]))
    else:
        cost_metric = metrics["mean_cost"]
    return metrics["success_rate"] - cost_weight * cost_metric


# ---------------------------------------------------------------------------
# CEM optimiser for MLP weights
# ---------------------------------------------------------------------------

def cem_search_mlp(env: MetaEnv, seeds: np.ndarray, budgets_seq: np.ndarray,
                   max_horizon: int, budget_max: int,
                   hidden_size: int = 4,
                   n_iterations: int = 30, pop_size: int = 80,
                   elite_frac: float = 0.25, rng_seed: int = 0,
                   init_std: float = 0.5,
                   cost_weight: float = 0.01,
                   cvar_alpha: float = 0.0,
                   deadline_weight: float = 0.0,
                   checkpoint_dir: str = None,
                   n_workers: int = 1,
                   env_kwargs: dict = None) -> dict:
    """
    Cross-Entropy Method over the flattened MLP weight vector.

    Architecture: 7 → hidden_size → 1  (tanh + sigmoid)
    Parameters:   9*hidden_size + 1

    When n_workers > 1, candidates are evaluated in parallel using a
    multiprocessing Pool (each worker has its own Engine + TF sessions).

    Returns the best weight vector found and its metrics.
    """
    n_params  = MLPSwitchPolicy.n_params(hidden_size)
    rng       = np.random.RandomState(rng_seed)
    n_elite   = max(2, int(pop_size * elite_frac))

    # Initialise CEM distribution: zero mean, init_std
    # (Xavier-like: std ≈ 1/sqrt(fan_in) ≈ 1/sqrt(6) ≈ 0.41 for first layer)
    mu  = np.zeros(n_params)
    std = np.full(n_params, init_std)

    best_weights = mu.copy()
    best_score   = -np.inf
    best_metrics = None
    history      = []

    # Score description for logging
    cost_label = f"CVaR({int(cvar_alpha*100)}%)" if cvar_alpha > 0 else "E[cost]"
    if deadline_weight > 0.0:
        score_desc = (f"mean(success_i - {deadline_weight}*(1-success_i) "
                      f"- {cost_weight}*cost_i)")
    else:
        score_desc = f"success_rate - {cost_weight} * {cost_label}"

    print(f"\n{'='*70}")
    print(f"  CEM-MLP search  ({n_iterations} iters × {pop_size} candidates)")
    print(f"    architecture: 7 → {hidden_size} (tanh) → 1 (sigmoid)")
    print(f"    parameters:   {n_params}")
    print(f"    features:     [v_x, v_y, d_goal, d_haz, Δθ, t_frac, budget_norm]")
    print(f"    init_std:     {init_std}")
    print(f"    elite:        top {n_elite} / {pop_size}")
    print(f"    score:        {score_desc}")
    print(f"    episodes:     {len(seeds)} per candidate")
    print(f"    workers:      {n_workers}")
    print(f"{'='*70}")

    # --- Create worker pool (if parallel) ---
    _pool = None
    if n_workers > 1:
        assert env_kwargs is not None, \
            "env_kwargs required for parallel mode (n_workers > 1)"
        ctx = mp.get_context("spawn")
        _pool = ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(env_kwargs["cons_dir"],
                      env_kwargs["agg_dir"],
                      env_kwargs["meta_interval"],
                      env_kwargs["budget_min"],
                      env_kwargs["budget_max"],
                      env_kwargs["budget_step"]),
        )
        print(f"  Pool started — {n_workers} workers ready")

    for it in range(n_iterations):
        # Sample population
        pop = rng.randn(pop_size, n_params) * std + mu

        if _pool is not None:
            print(f"  [iter {it+1:2d}/{n_iterations}]  "
                  f"Evaluating {pop_size} candidates in parallel "
                  f"with {n_workers} workers ...")
            # ---- Parallel evaluation ----
            packed = [
                (pop[i], hidden_size, float(budget_max), seeds, budgets_seq,
                 max_horizon, cost_weight, cvar_alpha, deadline_weight)
                for i in range(pop_size)
            ]
            raw = _pool.map(_eval_one, packed)
            scores_list = [r[0] for r in raw]
            results     = [r[1] for r in raw]
            # numpy arrays survive pickling, but double-check
            for m in results:
                if not isinstance(m["_costs"], np.ndarray):
                    m["_costs"]     = np.array(m["_costs"])
                    m["_successes"] = np.array(m["_successes"], dtype=np.float32)
            print(f"  [iter {it+1:2d}/{n_iterations}]  "
                  f"Evaluated {pop_size} candidates  ({n_workers} workers)")
        else:
            # ---- Sequential evaluation ----
            scores_list = []
            results     = []
            for i in range(pop_size):
                print(f"  [iter {it+1:2d}/{n_iterations}]  "
                      f"Evaluating candidate {i+1:3d}/{pop_size} ...")
                policy  = MLPSwitchPolicy(pop[i], hidden_size=hidden_size,
                                          budget_max=float(budget_max), env=env)
                metrics = _run_episodes(policy, env, seeds, budgets_seq,
                                        max_horizon)
                sc = _score(metrics, cost_weight=cost_weight,
                            cvar_alpha=cvar_alpha,
                            deadline_weight=deadline_weight)
                scores_list.append(sc)
                results.append(metrics)
                if (i + 1) % 20 == 0:
                    print(f"    [iter {it+1:2d}/{n_iterations}]  "
                          f"candidate {i+1:3d}/{pop_size}  "
                          f"score={sc:.4f}  succ={metrics['success_rate']:.3f}")

        # --- Log candidates to CSV (after all evaluated) ---
        if checkpoint_dir is not None:
            cand_csv  = os.path.join(checkpoint_dir, "cem_candidates.csv")
            write_hdr = (it == 0)
            with open(cand_csv, "a", newline="") as f_cand:
                fields = ["iteration", "candidate", "score", "success_rate",
                           "mean_cost", "cvar_10", "cvar_20", "cvar_30",
                           "frac_switched", "mean_switch_step"]
                cw = csv.DictWriter(f_cand, fieldnames=fields)
                if write_hdr:
                    cw.writeheader()
                for i in range(pop_size):
                    cw.writerow({
                        "iteration":        it + 1,
                        "candidate":        i + 1,
                        "score":            round(scores_list[i], 6),
                        "success_rate":     round(results[i]["success_rate"], 4),
                        "mean_cost":        round(results[i]["mean_cost"], 4),
                        "cvar_10":          round(results[i]["cvar_10"], 4),
                        "cvar_20":          round(results[i]["cvar_20"], 4),
                        "cvar_30":          round(results[i]["cvar_30"], 4),
                        "frac_switched":    round(results[i]["frac_switched"], 4),
                        "mean_switch_step": round(results[i]["mean_switch_step"], 1),
                    })

        scores = np.array(scores_list)

        # Elite update
        elite_idx = np.argsort(scores)[-n_elite:]
        elite_pop = pop[elite_idx]
        mu_new    = elite_pop.mean(axis=0)
        std_new   = elite_pop.std(axis=0) + 1e-6
        mu, std   = mu_new, std_new

        # Track best ever
        best_it_idx = int(np.argmax(scores))
        if scores[best_it_idx] > best_score:
            best_score   = scores[best_it_idx]
            best_weights = pop[best_it_idx].copy()
            best_metrics = results[best_it_idx]

        # Summary stats
        elite_scores = scores[elite_idx]
        mean_std     = float(np.mean(std))

        history.append({
            "iteration":    it + 1,
            "best_score":   round(float(best_score), 4),
            "elite_mean":   round(float(np.mean(elite_scores)), 4),
            "elite_min":    round(float(np.min(elite_scores)), 4),
            "pop_mean":     round(float(np.mean(scores)), 4),
            "mean_std":     round(mean_std, 6),
            "best_succ":    round(best_metrics["success_rate"], 4),
            "best_cost":    round(best_metrics["mean_cost"], 4),
            "best_cvar10":  round(best_metrics["cvar_10"], 4),
            "best_cvar20":  round(best_metrics["cvar_20"], 4),
            "best_cvar30":  round(best_metrics["cvar_30"], 4),
            "best_sw_frac": round(best_metrics["frac_switched"], 4),
            "best_sw_step": round(best_metrics["mean_switch_step"], 1),
        })

        print(f"  [iter {it+1:2d}/{n_iterations}]  "
              f"best_score={best_score:.4f}  "
              f"elite_mean={np.mean(elite_scores):.4f}  "
              f"mean_std={mean_std:.4f}  "
              f"best_succ={best_metrics['success_rate']:.3f}  "
              f"best_cost={best_metrics['mean_cost']:.3f}  "
              f"sw%={best_metrics['frac_switched']:.2f}  "
              f"sw_step={best_metrics['mean_switch_step']:.0f}")

        # --- Checkpoint: save best weights + CEM state after every iteration ---
        if checkpoint_dir is not None:
            ckpt_file = os.path.join(checkpoint_dir, "best_weights.npz")
            np.savez(ckpt_file,
                     weights=best_weights,
                     hidden_size=np.array([hidden_size]),
                     budget_max=np.array([budget_max]),
                     cem_mu=mu,
                     cem_std=std,
                     iteration=np.array([it + 1]),
                     best_score=np.array([best_score]))
            # Also append current history row to CSV incrementally
            ckpt_csv = os.path.join(checkpoint_dir, "cem_mlp_history.csv")
            write_header = (it == 0) or not os.path.exists(ckpt_csv)
            with open(ckpt_csv, "a", newline="") as f:
                w_csv = csv.DictWriter(f, fieldnames=list(history[-1].keys()))
                if write_header:
                    w_csv.writeheader()
                w_csv.writerow(history[-1])
            print(f"    checkpoint saved → {ckpt_file}  (iter {it+1})")

    # Clean up pool + metrics before returning
    if _pool is not None:
        _pool.terminate()
        _pool.join()
    best_metrics.pop("_costs", None)
    best_metrics.pop("_successes", None)
    return {
        "weights":     best_weights,
        "hidden_size": hidden_size,
        "n_params":    n_params,
        "history":     history,
        **best_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    p = argparse.ArgumentParser(
        description="CEM-MLP switching policy: 7 geometric features → hidden → switch decision."
    )
    # --- Environment ---
    p.add_argument("--cons_dir",      type=str, required=True)
    p.add_argument("--agg_dir",       type=str, required=True)
    p.add_argument("--budget_min",    type=int, default=120)
    p.add_argument("--budget_max",    type=int, default=220)
    p.add_argument("--budget_step",   type=int, default=5)
    p.add_argument("--meta_interval", type=int, default=1)

    # --- MLP architecture ---
    p.add_argument("--hidden_size", type=int, default=4,
                   help="Neurons in hidden layer. 4 → 37 params, 8 → 73 params.")

    # --- CEM hyperparams ---
    p.add_argument("--cem_iterations", type=int, default=30,
                   help="CEM iterations (default: 30).")
    p.add_argument("--cem_pop_size",   type=int, default=80,
                   help="CEM population per iteration (default: 80).")
    p.add_argument("--cem_elite_frac", type=float, default=0.25,
                   help="Fraction kept as elite (default: 0.25).")
    p.add_argument("--cem_init_std",   type=float, default=0.5,
                   help="Initial std for weight sampling (default: 0.5, Xavier-like).")

    # --- Evaluation ---
    p.add_argument("--episodes",    type=int, default=300,
                   help="Episodes per candidate policy.")
    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max env steps per episode (0 = budget_max).")

    # --- Score function ---
    p.add_argument("--deadline_weight", type=float, default=1.0,
                   help="Penalty for failure: R_i = succ - dw*(1-succ) - cw*cost. "
                        "Default 1.0: success=+1, failure=-1.")
    p.add_argument("--cost_weight", type=float, default=0.02,
                   help="Weight for cost term in fitness (default: 0.02).")
    p.add_argument("--cvar_alpha",  type=float, default=0.0,
                   help="CVaR level for cost (0 = mean cost). Only used if deadline_weight=0.")

    # --- Output ---
    p.add_argument("--base_seed",   type=int, default=42)
    p.add_argument("--results_dir", type=str, default="results/threshold/mlp_001")
    p.add_argument("--debug", action="store_true", default=False,
                   help="Run a quick debug pass (no CEM): 5 episodes with zero weights, "
                        "printing features at each meta-step.")

    # --- Parallelism ---
    p.add_argument("--n_workers", type=int, default=4,
                   help="Number of parallel workers for CEM evaluation. "
                        "1 = sequential (default). >1 uses multiprocessing.Pool "
                        "(each worker loads its own TF sessions + MuJoCo env).")
    args = p.parse_args()


    # import os

    # n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    # n_workers = max(1, n_cpus - 1)

    # print("CPUs available:", n_cpus)
    # print("Suggested n_workers:", n_workers)

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    # Save config immediately so it survives crashes
    config_file = os.path.join(args.results_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved → {config_file}")

    n_params = MLPSwitchPolicy.n_params(args.hidden_size)
    print(f"\n  MLP: 7 → {args.hidden_size} → 1  ({n_params} parameters)")
    print(f"  Budget range: [{args.budget_min}, {args.budget_max}] step {args.budget_step}")


    # Seeds & budgets
    rng         = np.random.RandomState(args.base_seed)
    seeds       = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)
    budget_rng  = np.random.RandomState(args.base_seed + 1)
    bvals       = list(range(args.budget_min, args.budget_max + 1, args.budget_step))
    budgets_seq = budget_rng.choice(bvals, size=args.episodes, replace=True)

    print("\nLoading low-level policies ...")
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

    obs_dim = env.observation_space.shape[0]
    print(f"  obs_dim = {obs_dim}\n")

    # ---------------------------------------------------------------
    # Debug mode: quick feature check, then exit
    # ---------------------------------------------------------------
    if args.debug:
        print("\n" + "="*60)
        print("  DEBUG MODE — running 5 episodes with zero-weight MLP")
        print("="*60)
        n_p = MLPSwitchPolicy.n_params(args.hidden_size)
        dummy_w = np.zeros(n_p)
        policy = MLPSwitchPolicy(dummy_w, hidden_size=args.hidden_size,
                                 budget_max=float(args.budget_max), env=env)
        _run_episodes(policy, env, seeds[:5], budgets_seq[:5], max_horizon,
                      stochastic=False, debug=True, debug_episodes=5, debug_steps=8)
        print("\n  Debug done.")
        env.close()
        sess_cons.close()
        sess_agg.close()
        return

    # ---------------------------------------------------------------
    # CEM-MLP search
    # ---------------------------------------------------------------
    env_kwargs = {
        "cons_dir":      args.cons_dir,
        "agg_dir":       args.agg_dir,
        "meta_interval": args.meta_interval,
        "budget_min":    args.budget_min,
        "budget_max":    args.budget_max,
        "budget_step":   args.budget_step,
    }
    result = cem_search_mlp(
        env=env,
        seeds=seeds,
        budgets_seq=budgets_seq,
        max_horizon=max_horizon,
        budget_max=args.budget_max,
        hidden_size=args.hidden_size,
        n_iterations=args.cem_iterations,
        pop_size=args.cem_pop_size,
        elite_frac=args.cem_elite_frac,
        rng_seed=args.base_seed,
        init_std=args.cem_init_std,
        cost_weight=args.cost_weight,
        cvar_alpha=args.cvar_alpha,
        deadline_weight=args.deadline_weight,
        checkpoint_dir=args.results_dir,
        n_workers=args.n_workers,
        env_kwargs=env_kwargs,
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    # 1. CEM history CSV
    out_csv = os.path.join(args.results_dir, "cem_mlp_history.csv")
    cem_fields = list(result["history"][0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cem_fields)
        w.writeheader()
        w.writerows(result["history"])
    print(f"\nCEM history saved → {out_csv}")

    # 2. Best weights (.npz — loadable with np.load)
    weights_file = os.path.join(args.results_dir, "best_weights.npz")
    np.savez(weights_file,
             weights=result["weights"],
             hidden_size=np.array([args.hidden_size]),
             budget_max=np.array([args.budget_max]))
    print(f"Best weights saved → {weights_file}")

    # 3. Human-readable summary
    summary_file = os.path.join(args.results_dir, "best_params_mlp.txt")
    with open(summary_file, "w") as f:
        f.write(f"hidden_size={args.hidden_size}\n")
        f.write(f"n_params={result['n_params']}\n")
        f.write(f"budget_max={args.budget_max}\n")
        f.write(f"success_rate={result['success_rate']:.6f}\n")
        f.write(f"mean_cost={result['mean_cost']:.6f}\n")
        f.write(f"cvar_10={result['cvar_10']:.6f}\n")
        f.write(f"cvar_20={result['cvar_20']:.6f}\n")
        f.write(f"cvar_30={result['cvar_30']:.6f}\n")
        f.write(f"frac_switched={result['frac_switched']:.6f}\n")
        f.write(f"mean_switch_step={result['mean_switch_step']:.1f}\n")
    print(f"Summary saved → {summary_file}")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  MLP SWITCH POLICY  ({n_params} params, H={args.hidden_size})")
    print(f"    features: [v_x, v_y, d_goal, d_haz, Δθ, t_frac, budget_norm]")
    print(f"    succ={result['success_rate']:.3f}  "
          f"cost={result['mean_cost']:.3f}  "
          f"cvar10={result['cvar_10']:.3f}")
    print(f"    sw%={result['frac_switched']:.2f}  "
          f"sw_step={result['mean_switch_step']:.0f}")
    print(f"{'='*60}\n")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
