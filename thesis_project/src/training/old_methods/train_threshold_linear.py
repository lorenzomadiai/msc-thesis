#!/usr/bin/env python3
"""
train_threshold_meta.py
-----------------------
Fits two switching policies for the irreversible conservative→aggressive switch:

1D threshold (grid search)
--------------------------
    switch  iff  t_remaining / B  <  θ
    Single parameter θ ∈ [0,1], found by coarse→fine grid search.

2D linear threshold (CEM)
--------------------------
    switch  iff  t_remaining  <  α·B  +  β·(1 - goal_closeness)  -  γ·hazard_closeness

    where:
        t_remaining     = remaining env steps  (= (time_left_norm+1)/2 * B)
        goal_closeness  = max(obs[0:16])   lidar: 0=far, 1=close to goal
        hazard_closeness= max(obs[16:32])  lidar: 0=far, 1=close to hazard
        α  ∈ [0,1]     fraction of budget reserved for aggressive
        β  ∈ [0,B_max] extra steps when far from goal
        γ  ∈ [0,B_max] step discount when close to hazard (delay switch)

    Parameters (α, β, γ) found by Cross-Entropy Method (CEM):
    sample candidates from a Gaussian, keep top-k, update distribution.

Objective (lexicographic):
    1. maximise success rate  (reached goal within budget)
    2. minimise mean cost     (total hazard violations)

Usage
-----
python src/training/switching_policies/train_threshold_meta.py \\
    --cons_dir  WCSAC/.../conservative_agent_baseline/simple_save6 \\
    --agg_dir   WCSAC/.../aggressive_agent_baseline/simple_save9   \\
    --episodes  200 \\
    --results_dir results/threshold/run_001
"""

import os
import sys
import csv
import argparse
import warnings
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
# Safety-Gym config (same as other meta scripts)
# ---------------------------------------------------------------------------

STATIC_CONFIG = {
    "placements_extents": [-1.5, -1.5, 1.5, 1.5],

    # Note: the "robot_keepout" is set to 0 here, which means the robot can start anywhere within the placements_extents.
    "robot_placements": [(-1.5, -1.5, 0.0, 0.0)],

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
# Threshold policy
# ---------------------------------------------------------------------------

class ThresholdPolicy:
    """
    Switch from conservative to aggressive when:
        t_remaining / B  <  theta
    i.e.  (time_left_norm + 1) / 2  <  theta
    i.e.   time_left_norm            <  2*theta - 1

    The MetaEnv (irreversible_switch=True) exposes time_left_norm as
    obs[-3] when velocimeter is included, or obs[-2] otherwise.
    We read it from obs[-3] (with velocimeter) or obs[-2] (without).

    After the switch the env itself locks to aggressive, so we only need
    to emit action=1 the first time the condition is true — emitting 1
    again is harmless.
    """

    def __init__(self, theta: float, time_left_idx: int = -3):
        """
        Parameters
        ----------
        theta : float
            Fraction of budget remaining at which to switch (in [0, 1]).
            theta=0  → never switch (always conservative).
            theta=1  → switch immediately (always aggressive from step 0).
        time_left_idx : int
            Index of time_left_norm in the meta observation vector.
            With irreversible_switch=True and velocimeter included: -3.
            With irreversible_switch=True and no velocimeter:        -2.
        """
        assert 0.0 <= theta <= 1.0, f"theta must be in [0,1], got {theta}"
        self.theta = theta
        self.time_left_idx = time_left_idx
        # Pre-compute threshold in time_left_norm space
        self._tln_threshold = 2.0 * theta - 1.0  # range [-1, 1]

    def act(self, obs: np.ndarray) -> int:
        """Return 0 (conservative) or 1 (switch to aggressive)."""
        time_left_norm = float(obs[self.time_left_idx])
        return int(time_left_norm < self._tln_threshold)


# ---------------------------------------------------------------------------
# 2D linear threshold policy
# ---------------------------------------------------------------------------

class LinearThresholdPolicy:
    """
    Switch from conservative to aggressive when:

        t_rem / B  <  beta*(1-goal_cl)  -  gamma*haz_cl
                    +  phi * cos(θ_goal − θ_haz)
                    +  psi * sin(θ_goal − θ_haz)
                    +  delta * budget_norm

    LHS:   t_frac = t_rem / B  ∈ [0, 1]   (fraction of this episode's budget left).
    RHS:   geometry + budget term.  A higher threshold means the switch triggers
           earlier (more of the time range satisfies t_frac < threshold).

    The delta * budget_norm term encodes budget sensitivity:
        delta < 0  →  large budget (budget_norm≈1) lowers the threshold (switch later)
                       small budget (budget_norm≈0.5) raises it (switch earlier)
        This lets the same geometry parameters produce early switches on short
        episodes and late (or no) switches on long episodes.

    α = θ_goal − θ_haz, where θ_k = k / N_BINS * 2π,  k = argmax(lidar).

    Obs layout (irreversible, with velocimeter):
        obs[0:16]  = goal_lidar   (proximity: 0=far, 1=close)
        obs[16:32] = hazard_lidar (proximity: 0=far, 1=close)
        obs[-3]    = time_left_norm = 2*(t_rem/B) - 1
        obs[-2]    = budget_norm   = B / budget_max
        obs[-1]    = is_aggressive
    """

    # Meta-env obs layout: [velocimeter(3), goal_lidar(16), hazards_lidar(16),
    #                        time_left_norm, budget_norm, is_aggressive]
    GOAL_LIDAR_SLICE   = slice(3, 19)
    HAZARD_LIDAR_SLICE = slice(19, 35)
    N_LIDAR_BINS       = 16

    def __init__(self, beta: float, gamma: float,
                 phi: float = 0.0, psi: float = 0.0, delta: float = 0.0,
                 budget_max: int = 260):
        self.beta       = float(beta)   # goal-distance weight   ∈ [0, 1]
        self.gamma      = float(gamma)  # hazard-proximity weight ∈ [0, 1]
        self.phi        = float(phi)    # cos(α) weight ∈ [-1, 1]
        self.psi        = float(psi)    # sin(α) weight ∈ [-1, 1]
        self.delta      = float(delta)  # budget_norm weight ∈ [-1, 1]
        self.budget_max = float(budget_max)

    @classmethod
    def _trig_angle_between(cls, goal_lidar: np.ndarray,
                            haz_lidar: np.ndarray) -> tuple:
        """Return (cos(α), sin(α)) where α = θ_goal − θ_haz (peak-bin angles)."""
        k_g   = int(np.argmax(goal_lidar))
        k_h   = int(np.argmax(haz_lidar))
        alpha = (k_g - k_h) / cls.N_LIDAR_BINS * 2.0 * np.pi
        return float(np.cos(alpha)), float(np.sin(alpha))

    def act(self, obs: np.ndarray) -> int:
        """Return 0 (conservative) or 1 (switch to aggressive)."""
        time_left_norm   = float(obs[-3])
        budget_norm      = float(obs[-2])
        t_frac           = (time_left_norm + 1.0) / 2.0   # t_rem/B ∈ [0, 1]

        goal_closeness   = float(np.max(obs[self.GOAL_LIDAR_SLICE]))
        hazard_closeness = float(np.max(obs[self.HAZARD_LIDAR_SLICE]))
        cos_gh, sin_gh   = self._trig_angle_between(obs[self.GOAL_LIDAR_SLICE],
                                                    obs[self.HAZARD_LIDAR_SLICE])

        threshold = (self.beta  * (1.0 - goal_closeness)
                     - self.gamma * hazard_closeness
                     + self.phi   * cos_gh
                     + self.psi   * sin_gh
                     + self.delta * budget_norm)
        return int(t_frac < threshold)


# ---------------------------------------------------------------------------
# 2D linear threshold policy  (FIXED: threshold computed once at episode start)
# ---------------------------------------------------------------------------

class LinearThresholdPolicyFixed:
    """
    Like LinearThresholdPolicy but computes the switch threshold **once at
    episode start** using the initial observation, then keeps it fixed.

    At t=0:
        switch_at = beta*(1-goal_cl_0) - gamma*haz_cl_0
                      + phi*cos(α_0) + psi*sin(α_0)
                      + delta * budget_norm_0
        where α_0 = θ_goal_0 − θ_haz_0, budget_norm_0 = B / budget_max

    Then switches when t_rem/B < switch_at (per-episode fraction).

    With delta < 0: large budget → lower switch_at (switch later / never);
                    small budget → higher switch_at (switch earlier).

    Requires calling reset(obs_0, B) before each episode.
    """

    # Meta-env obs layout: [velocimeter(3), goal_lidar(16), hazards_lidar(16),
    #                        time_left_norm, budget_norm, is_aggressive]
    GOAL_LIDAR_SLICE   = slice(3, 19)
    HAZARD_LIDAR_SLICE = slice(19, 35)
    N_LIDAR_BINS       = 16

    def __init__(self, beta: float, gamma: float,
                 phi: float = 0.0, psi: float = 0.0, delta: float = 0.0,
                 budget_max: int = 260):
        self.beta       = float(beta)
        self.gamma      = float(gamma)
        self.phi        = float(phi)    # cos(α) weight ∈ [-1, 1]
        self.psi        = float(psi)    # sin(α) weight ∈ [-1, 1]
        self.delta      = float(delta)  # budget_norm weight ∈ [-1, 1]
        self.budget_max = float(budget_max)
        self._switch_at: float = 0.0   # set by reset()

    @classmethod
    def _trig_angle_between(cls, goal_lidar: np.ndarray,
                            haz_lidar: np.ndarray) -> tuple:
        """Return (cos(α), sin(α)) where α = θ_goal − θ_haz (peak-bin angles)."""
        k_g   = int(np.argmax(goal_lidar))
        k_h   = int(np.argmax(haz_lidar))
        alpha = (k_g - k_h) / cls.N_LIDAR_BINS * 2.0 * np.pi
        return float(np.cos(alpha)), float(np.sin(alpha))

    def reset(self, obs: np.ndarray, B: float) -> None:
        """Call once per episode with the initial observation and budget."""
        goal_closeness   = float(np.max(obs[self.GOAL_LIDAR_SLICE]))
        hazard_closeness = float(np.max(obs[self.HAZARD_LIDAR_SLICE]))
        cos_gh, sin_gh   = self._trig_angle_between(obs[self.GOAL_LIDAR_SLICE],
                                                    obs[self.HAZARD_LIDAR_SLICE])
        budget_norm      = B / self.budget_max
        self._switch_at  = (self.beta  * (1.0 - goal_closeness)
                            - self.gamma * hazard_closeness
                            + self.phi   * cos_gh
                            + self.psi   * sin_gh
                            + self.delta * budget_norm)

    def act(self, obs: np.ndarray) -> int:
        """Return 0 (conservative) or 1 (switch to aggressive)."""
        time_left_norm = float(obs[-3])
        t_frac         = (time_left_norm + 1.0) / 2.0   # t_rem/B ∈ [0, 1]
        return int(t_frac < self._switch_at)


# ---------------------------------------------------------------------------
# Shared episode runner (used by both 1D and 2D evaluation)
# ---------------------------------------------------------------------------

def _run_episodes(policy, env: MetaEnv, seeds: np.ndarray,
                  budgets_seq: np.ndarray, max_horizon: int) -> dict:
    """Run one episode per seed, return aggregate metrics dict.

    Returns both mean_cost and cvar_cost at multiple alpha levels so that
    _score() can use whichever the caller prefers.
    """
    successes    = []
    costs        = []
    switch_steps = []

    for ep_idx, seed in enumerate(seeds):
        # print(f"  Running episode {ep_idx+1}/{len(seeds)} with seed={seed} and budget={budgets_seq[ep_idx]}...")
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

        # Allow policies that pre-compute a per-episode switch time
        if hasattr(policy, 'reset'):
            policy.reset(obs, float(budget))

        done        = False
        ep_len      = 0
        cum_cost    = 0.0
        switch_step = -1
        switched    = False

        while not done and ep_len < max_horizon:
            action = policy.act(obs)
            obs, _r, done, info = env.step(action)
            cum_cost += float(info.get("cumulative_cost", 0.0))
            ep_len   += info.get("n_steps_taken", 1)
            if action == 1 and not switched:
                switch_step = ep_len
                switched    = True

        success = bool(info.get("goal_met", False))
        successes.append(int(success))
        costs.append(cum_cost)
        switch_steps.append(switch_step)

    costs_arr     = np.array(costs)
    success_rate  = float(np.mean(successes))
    mean_cost     = float(np.mean(costs_arr))
    frac_switched = float(np.mean([s > 0 for s in switch_steps]))
    mean_sw_step  = float(np.mean([s for s in switch_steps if s > 0])
                          if any(s > 0 for s in switch_steps) else 0.0)

    # CVaR at common alpha levels (always computed, cheap)
    def _cvar(alpha: float) -> float:
        """Mean cost of the worst (alpha*100)% episodes."""
        k = max(1, int(np.ceil(alpha * len(costs_arr))))
        return float(np.mean(np.sort(costs_arr)[-k:]))

    successes_arr = np.array(successes, dtype=np.float32)
    return {
        "success_rate":     success_rate,
        "mean_cost":        mean_cost,
        "cvar_10":          _cvar(0.10),
        "cvar_20":          _cvar(0.20),
        "cvar_30":          _cvar(0.30),
        "frac_switched":    frac_switched,
        "mean_switch_step": mean_sw_step,
        "_costs":           costs_arr,      # kept for custom alpha in _score
        "_successes":       successes_arr,  # kept for per-episode reward in _score
    }


# ---------------------------------------------------------------------------
# Single-θ evaluation  (1D)
# ---------------------------------------------------------------------------

def evaluate_theta(theta: float, env: MetaEnv, seeds: np.ndarray,
                   budgets_seq: np.ndarray, max_horizon: int,
                   time_left_idx: int) -> dict:
    """Run one episode per seed and return aggregate metrics (1D)."""
    policy  = ThresholdPolicy(theta, time_left_idx=time_left_idx)
    metrics = _run_episodes(policy, env, seeds, budgets_seq, max_horizon)
    return {"theta": theta, **metrics}


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(thetas: np.ndarray, env: MetaEnv, seeds: np.ndarray,
                budgets_seq: np.ndarray, max_horizon: int,
                time_left_idx: int, min_success: float = 1.1) -> list:
    results = []
    n = len(thetas)
    for i, theta in enumerate(thetas):
        metrics = evaluate_theta(theta, env, seeds, budgets_seq,
                                 max_horizon, time_left_idx)
        results.append(metrics)
        target_tag = "  ✓ TARGET" if metrics['success_rate'] >= min_success else ""
        print(f"  [{i+1:3d}/{n}]  θ={theta:.4f}  "
              f"succ={metrics['success_rate']:.3f}  "
              f"cost={metrics['mean_cost']:.3f}  "
              f"switch%={metrics['frac_switched']:.2f}  "
              f"sw_step={metrics['mean_switch_step']:.1f}{target_tag}")
        if metrics['success_rate'] >= min_success:
            print(f"  [EARLY STOP] success_rate={metrics['success_rate']:.3f} "
                  f">= min_success={min_success:.3f}")
            break
    return results


def best_theta(results: list) -> dict:
    """Lexicographic: max success_rate, then min mean_cost."""
    return max(results, key=lambda r: (r["success_rate"], -r["mean_cost"]))


# ---------------------------------------------------------------------------
# CEM optimiser for 2D linear threshold  (α, β, γ)
# ---------------------------------------------------------------------------

def _score(metrics: dict, cost_weight: float = 0.01,
           cvar_alpha: float = 0.0,
           deadline_weight: float = 0.0) -> float:
    """Scalar fitness for CEM.

    Two modes:

    deadline_weight == 0  (default):
        score = success_rate - cost_weight * cost_metric
        cost_metric = mean_cost  (cvar_alpha=0)  or  CVaR_alpha  (cvar_alpha>0)

    deadline_weight > 0:
        Compute per-episode reward and average:
            R_i = success_i - deadline_weight*(1-success_i) - cost_weight*cost_i
        score = mean(R_i)
        Failing an episode adds a -deadline_weight penalty (instead of 0),
        giving a stronger signal to avoid failures.
        E.g. deadline_weight=1.0: success→+1, failure→-1 (before cost term).
    """
    if deadline_weight > 0.0:
        successes_arr = metrics["_successes"]          # per-episode float 0/1
        costs_arr     = metrics["_costs"]              # per-episode cumulative cost
        rewards       = (successes_arr
                         - deadline_weight * (1.0 - successes_arr)
                         - cost_weight * costs_arr)
        return float(np.mean(rewards))

    # --- original mode ---
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


def cem_search(env: MetaEnv, seeds: np.ndarray, budgets_seq: np.ndarray,
               max_horizon: int, budget_max: int,
               n_iterations: int = 25, pop_size: int = 40,
               elite_frac: float = 0.25, rng_seed: int = 0,
               min_success: float = 1.1,
               cost_weight: float = 0.01,
               cvar_alpha: float = 0.0,
               deadline_weight: float = 0.0,
               fixed: bool = False) -> dict:
    """
    Cross-Entropy Method over (beta, gamma, phi, psi, delta).

    Parameter space (unconstrained Gaussian, clipped after sampling):
        beta   ∈ [0, 1]   goal-distance weight
        gamma  ∈ [0, 1]   hazard-proximity weight
        phi    ∈ [-1, 1]  weight for cos(θ_goal − θ_haz)  (alignment)
        psi    ∈ [-1, 1]  weight for sin(θ_goal − θ_haz)  (laterality)
        delta  ∈ [-1, 1]  weight for budget_norm = B / budget_max
                          (CEM will likely find delta < 0: large budget → switch later)

    Comparison: t_rem/B < threshold  (LHS ∈ [0,1] per episode).

    Returns the best (beta, gamma, phi, psi, delta) found and its metrics.
    """
    rng       = np.random.RandomState(rng_seed)
    n_elite   = max(2, int(pop_size * elite_frac))

    # Initial distribution: mean and std for [beta, gamma, phi, psi, delta]
    # beta:  0.30 ± 0.20
    # gamma: 0.10 ± 0.10
    # phi:   0.00 ± 0.15  (cos weight)
    # psi:   0.00 ± 0.15  (sin weight)
    # delta: 0.00 ± 0.20  (budget weight, expected to converge to negative)
    mu  = np.array([0.30, 0.10, 0.0, 0.0, 0.0])
    std = np.array([0.20, 0.10, 0.15, 0.15, 0.20])

    # Clip bounds: beta/gamma in [0, 1], phi/psi/delta in [-1, 1]
    lo = np.array([0.0,  0.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0,  1.0,  1.0,  1.0,  1.0])

    best_params  = mu.copy()
    best_score   = -np.inf
    history      = []   # list of dicts for CSV

    policy_cls = LinearThresholdPolicyFixed if fixed else LinearThresholdPolicy
    policy_tag = "FIXED (t=0 obs)" if fixed else "per-step"

    cost_label = f"CVaR({int(cvar_alpha*100)}%)" if cvar_alpha > 0 else "E[cost]"
    if deadline_weight > 0.0:
        score_desc = (f"mean(success_i - {deadline_weight}*(1-success_i) "
                      f"- {cost_weight}*cost_i)   [deadline penalty mode]")
    else:
        score_desc = f"success_rate - {cost_weight} * {cost_label}"
    print(f"\n=== CEM search  ({n_iterations} iterations × {pop_size} candidates) ===")
    print(f"    variant: {policy_tag}")
    print(f"    policy:  switch when t_rem/B < β·(1-goal_cl) - γ·haz_cl"
          f" + φ·cos(α) + ψ·sin(α) + δ·budget_norm   [α = θ_goal − θ_haz]")
    print(f"    search:  β ∈ [0,1], γ ∈ [0,1], φ ∈ [-1,1], ψ ∈ [-1,1], δ ∈ [-1,1]")
    print(f"    score:   {score_desc}")
    for it in range(n_iterations):
        # Sample population
        pop = rng.randn(pop_size, 5) * std + mu
        pop = np.clip(pop, lo, hi)

        # Evaluate all candidates
        scores  = []
        results = []
        for i, (beta, gamma, phi, psi, delta) in enumerate(pop):
            print(f"  [iter {it+1:2d}/{n_iterations}]  Evaluating candidate {i+1:3d}/{pop_size} "
                  f"(β={beta:.2f}, γ={gamma:.2f}, φ={phi:.2f}, ψ={psi:.2f}, δ={delta:.2f})...")
            policy  = policy_cls(beta, gamma,
                               phi=phi, psi=psi, delta=delta,
                               budget_max=budget_max)
            metrics = _run_episodes(policy, env, seeds, budgets_seq,
                                    max_horizon)
            sc = _score(metrics, cost_weight=cost_weight,
                        cvar_alpha=cvar_alpha, deadline_weight=deadline_weight)
            scores.append(sc)
            results.append((beta, gamma, phi, psi, delta, metrics))

        scores = np.array(scores)

        # Elite update
        elite_idx = np.argsort(scores)[-n_elite:]
        elite_pop = pop[elite_idx]
        mu_new    = elite_pop.mean(axis=0)
        std_new   = elite_pop.std(axis=0) + 1e-6   # avoid collapse
        mu, std   = mu_new, std_new

        # Track best
        best_it_idx  = int(np.argmax(scores))
        b, g, ph, ps, dl, m = results[best_it_idx]
        if scores[best_it_idx] > best_score:
            best_score   = scores[best_it_idx]
            best_params  = np.array([b, g, ph, ps, dl])
            best_metrics = m

        history.append({
            "iteration": it + 1,
            "beta":  round(mu[0], 4), "gamma": round(mu[1], 4),
            "phi":   round(mu[2], 4), "psi":   round(mu[3], 4),
            "delta": round(mu[4], 4),
            "std_beta":  round(std[0], 4), "std_gamma": round(std[1], 4),
            "std_phi":   round(std[2], 4), "std_psi":   round(std[3], 4),
            "std_delta": round(std[4], 4),
            "best_score": round(float(best_score), 4),
            "best_succ":  round(best_metrics["success_rate"], 4),
            "best_cost":  round(best_metrics["mean_cost"], 4),
            "best_cvar10": round(best_metrics["cvar_10"], 4),
            "best_cvar20": round(best_metrics["cvar_20"], 4),
            "best_cvar30": round(best_metrics["cvar_30"], 4),
        })
        print(f"  [iter {it+1:2d}/{n_iterations}]  "
              f"μ=(β={mu[0]:.2f}, γ={mu[1]:.2f}, φ={mu[2]:.2f}, ψ={mu[3]:.2f}, δ={mu[4]:.2f})  "
              f"std=({std[0]:.2f},{std[1]:.2f},{std[2]:.2f},{std[3]:.2f},{std[4]:.2f})  "
              f"best_succ={best_metrics['success_rate']:.3f}  "
              f"best_{cost_label}={best_metrics['mean_cost'] if cvar_alpha==0 else best_metrics.get(f'cvar_{int(cvar_alpha*100)}', 0):.3f}")

        if best_metrics["success_rate"] >= min_success:
            print(f"  [EARLY STOP] success_rate={best_metrics['success_rate']:.3f} "
                  f">= min_success={min_success:.3f}")
            break

    beta_opt, gamma_opt, phi_opt, psi_opt, delta_opt = best_params
    best_metrics.pop("_costs", None)   # remove raw array before returning
    return {
        "beta":         float(beta_opt),
        "gamma":        float(gamma_opt),
        "phi":          float(phi_opt),
        "psi":          float(psi_opt),
        "delta":        float(delta_opt),
        "history":      history,
        **best_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Grid-search a threshold policy θ for the irreversible switch."
    )
    p.add_argument("--cons_dir",  type=str, required=True)
    p.add_argument("--agg_dir",   type=str, required=True)
    p.add_argument("--budget_min",  type=int,   default=140)
    p.add_argument("--budget_max",  type=int,   default=260)
    p.add_argument("--budget_step", type=int,   default=5)
    p.add_argument("--meta_interval", type=int, default=5)
    p.add_argument("--episodes",  type=int,   default=200,
                   help="Episodes per θ candidate.")
    p.add_argument("--max_horizon", type=int, default=0,
                   help="Max env steps per episode (0 = budget_max).")
    p.add_argument("--coarse_steps", type=int, default=21,
                   help="Number of θ values in coarse grid (default: 21 → step 0.05).")
    p.add_argument("--fine_steps",   type=int, default=11,
                   help="Number of θ values in fine grid around best coarse θ (0 = skip).")
    p.add_argument("--fine_radius",  type=float, default=0.05,
                   help="Half-width of fine grid around best coarse θ.")
    # CEM options
    p.add_argument("--cem_iterations", type=int, default=25,
                   help="CEM iterations for 2D policy (default: 25).")
    p.add_argument("--cem_pop_size",   type=int, default=40,
                   help="CEM population size per iteration (default: 40).")
    p.add_argument("--cem_elite_frac", type=float, default=0.25,
                   help="Fraction of population kept as elite (default: 0.25).")
    p.add_argument("--skip_1d",  action="store_true",
                   help="Skip 1D grid search, run only 2D CEM.")
    p.add_argument("--skip_2d",  action="store_true",
                   help="Skip 2D CEM search, run only 1D grid.")
    p.add_argument("--fixed_2d", action="store_true",
                   help="Use fixed-threshold 2D policy: compute switch_at once at episode "
                        "start from initial obs, instead of re-evaluating every step.")
    p.add_argument("--min_success", type=float, default=1.1,
                   help="Early-stop search when success_rate >= this value (default: 1.1 = never stop early). "
                        "E.g. --min_success 0.9 stops as soon as a policy achieves 90%% success rate.")
    p.add_argument("--cost_weight", type=float, default=0.01,
                   help="Weight of cost term in CEM fitness: score = success_rate - cost_weight * cost. "
                        "Higher = more conservative (default: 0.01). Try 0.05 or 0.1 for safer policies.")
    p.add_argument("--cvar_alpha", type=float, default=0.0,
                   help="CVaR level applied to per-episode reward R. "
                        "0.0 (default) = use E[R]. "
                        "0.2 = optimise mean R of worst 20%% episodes. "
                        "Typical values: 0.1, 0.2, 0.3.")
    p.add_argument("--deadline_weight", type=float, default=0.0,
                   help="Penalty for missing the deadline: "
                        "R_i = success_i - deadline_weight*(1-success_i) - cost_weight*cost_i. "
                        "0.0 (default) = failure is neutral (reward=0). "
                        "0.5 = failure gives reward=-0.5. "
                        "1.0 = failure gives reward=-1 (symmetric with success).")
    p.add_argument("--base_seed",  type=int, default=42)
    p.add_argument("--results_dir", type=str, default="results/threshold/run_001")
    args = p.parse_args()

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    # Shared seeds & budgets across all θ for a fair comparison
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
        irreversible_switch=True,   # always True for threshold policy
        seed=args.base_seed + 99,
    )

    # Detect time_left_norm index in obs
    # MetaEnv obs layout (irreversible): [...lidar/vel..., time_left_norm, budget_norm, is_aggressive]
    # → time_left_norm is at index -3
    time_left_idx = -3

    obs_dim = env.observation_space.shape[0]
    print(f"  obs_dim = {obs_dim}  (time_left_norm at index {time_left_idx})\n")

    # ---------------------------------------------------------------
    # 1D grid search
    # ---------------------------------------------------------------
    best_1d = None
    if not args.skip_1d:
        coarse_thetas = np.linspace(0.0, 1.0, args.coarse_steps)
        print(f"=== 1D Coarse grid search  ({len(coarse_thetas)} values, "
              f"{args.episodes} episodes each) ===")
        coarse_results = grid_search(coarse_thetas, env, seeds, budgets_seq,
                                     max_horizon, time_left_idx,
                                     min_success=args.min_success)
        best_coarse = best_theta(coarse_results)
        print(f"\nBest coarse θ = {best_coarse['theta']:.4f}  "
              f"→  succ={best_coarse['success_rate']:.3f}  "
              f"cost={best_coarse['mean_cost']:.3f}")

        all_1d = list(coarse_results)
        if args.fine_steps > 0:
            lo = max(0.0, best_coarse["theta"] - args.fine_radius)
            hi = min(1.0, best_coarse["theta"] + args.fine_radius)
            fine_thetas = np.linspace(lo, hi, args.fine_steps)
            fine_thetas = np.array([t for t in fine_thetas
                                     if not any(abs(t - r["theta"]) < 1e-9
                                                for r in coarse_results)])
            if len(fine_thetas) > 0:
                print(f"\n=== 1D Fine grid  ({len(fine_thetas)} values, "
                      f"[{lo:.3f}, {hi:.3f}]) ===")
                fine_results = grid_search(fine_thetas, env, seeds, budgets_seq,
                                           max_horizon, time_left_idx,
                                           min_success=args.min_success)
                all_1d.extend(fine_results)

        all_1d.sort(key=lambda r: r["theta"])
        best_1d = best_theta(all_1d)

        out_csv_1d = os.path.join(args.results_dir, "threshold_1d_search.csv")
        fieldnames_1d = ["theta", "success_rate", "mean_cost",
                         "frac_switched", "mean_switch_step"]
        with open(out_csv_1d, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_1d)
            w.writeheader()
            w.writerows(all_1d)
        print(f"\n1D results saved → {out_csv_1d}")

        with open(os.path.join(args.results_dir, "best_theta_1d.txt"), "w") as f:
            f.write(f"theta={best_1d['theta']:.6f}\n")
            f.write(f"success_rate={best_1d['success_rate']:.6f}\n")
            f.write(f"mean_cost={best_1d['mean_cost']:.6f}\n")
            f.write(f"time_left_idx={time_left_idx}\n")

    # ---------------------------------------------------------------
    # 2D CEM search
    # ---------------------------------------------------------------
    best_2d = None
    if not args.skip_2d:
        cem_result = cem_search(
            env=env,
            seeds=seeds,
            budgets_seq=budgets_seq,
            max_horizon=max_horizon,
            budget_max=args.budget_max,
            n_iterations=args.cem_iterations,
            pop_size=args.cem_pop_size,
            elite_frac=args.cem_elite_frac,
            rng_seed=args.base_seed,
            min_success=args.min_success,
            cost_weight=args.cost_weight,
            cvar_alpha=args.cvar_alpha,
            deadline_weight=args.deadline_weight,
            fixed=args.fixed_2d,
        )
        best_2d = cem_result

        out_csv_cem = os.path.join(args.results_dir, "cem_history.csv")
        cem_fields = ["iteration", "beta", "gamma", "phi", "psi", "delta",
                      "std_beta", "std_gamma", "std_phi", "std_psi", "std_delta",
                      "best_score", "best_succ", "best_cost",
                      "best_cvar10", "best_cvar20", "best_cvar30"]
        with open(out_csv_cem, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cem_fields)
            w.writeheader()
            w.writerows(cem_result["history"])
        print(f"\nCEM history saved → {out_csv_cem}")

        with open(os.path.join(args.results_dir, "best_params_2d.txt"), "w") as f:
            f.write(f"beta={cem_result['beta']:.6f}\n")
            f.write(f"gamma={cem_result['gamma']:.6f}\n")
            f.write(f"phi={cem_result['phi']:.6f}\n")
            f.write(f"psi={cem_result['psi']:.6f}\n")
            f.write(f"delta={cem_result['delta']:.6f}\n")
            f.write(f"success_rate={cem_result['success_rate']:.6f}\n")
            f.write(f"mean_cost={cem_result['mean_cost']:.6f}\n")
            f.write(f"cvar_10={cem_result['cvar_10']:.6f}\n")
            f.write(f"cvar_20={cem_result['cvar_20']:.6f}\n")
            f.write(f"cvar_30={cem_result['cvar_30']:.6f}\n")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    if best_1d:
        print(f"  1D THRESHOLD:  θ* = {best_1d['theta']:.4f}")
        print(f"    rule:   switch when  t_rem/B < {best_1d['theta']:.4f}")
        print(f"    succ={best_1d['success_rate']:.3f}  cost={best_1d['mean_cost']:.3f}")
    if best_2d:
        b, g = best_2d['beta'], best_2d['gamma']
        ph, ps, dl = best_2d['phi'], best_2d['psi'], best_2d['delta']
        R  = (ph**2 + ps**2) ** 0.5
        p0 = np.degrees(np.arctan2(ps, ph))
        print(f"\n  2D THRESHOLD:  (β={b:.3f}, γ={g:.3f}, φ={ph:.3f}, ψ={ps:.3f}, δ={dl:.3f})")
        print(f"    rule:   switch when  t_rem/B  <  {b:.3f}·(1-goal_cl) - {g:.3f}·haz_cl"
              f" + {ph:.3f}·cos(α) + {ps:.3f}·sin(α) + {dl:.3f}·budget_norm")
        print(f"    equiv:  geom term = {R:.3f}·cos(α − {p0:.1f}°)   [α = θ_goal − θ_haz]")
        print(f"    succ={best_2d['success_rate']:.3f}  cost={best_2d['mean_cost']:.3f}")
    print(f"{'='*60}\n")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
