#!/usr/bin/env python3
"""
train_q_threshold.py
--------------------
Trains an MLP-based switching policy (conservative → aggressive) using
DQN (Deep Q-Learning) with counterfactual switch rollouts.

The meta-controller learns Q(s, wait) and Q(s, switch) for every state
observed during the conservative phase. At deployment, the policy is:

    π(s) = switch   if Q(s, switch) > Q(s, wait)
           wait     otherwise

For each state visited while conservative, TWO transitions are collected:

  1) **wait** transition (from actual env step):
       (s_t, wait, r_t, s_{t+1}, done)
     where r_t = -cost_weight * cost_t  (+ terminal bonus/penalty if done)

  2) **switch** transition (counterfactual rollout):
     Save the MuJoCo state at s_t, run the aggressive policy until
     episode end, compute the discounted return from that point on.
       (s_t, switch, R^agg_{t:T}, terminal=True)
     Since switch is irreversible—once you switch, no more decisions—
     this is a terminal transition with target = R^agg_{t:T}.

Architecture:
    7 geometric features → hidden_size (ReLU) → 2  [Q_wait, Q_switch]

Features:
    [v_x, v_y, d_goal_norm, d_haz_norm, Δθ_norm, t_frac, budget_norm]

Usage
-----
python src/training/switching_policies/train_q_threshold.py \\
    --cons_dir  WCSAC/.../simple_save6 \\
    --agg_dir   WCSAC/.../simple_save9 \\
    --episodes 300 --n_epochs 200 --batch_size 64 \\
    --hidden_size 8 --lr 1e-3 \\
    --deadline_weight 1.0 --cost_weight 0.05 \\
    --results_dir results/threshold/dqn_001
"""

import os
import sys
import csv
import json
import copy
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from safety_gym.envs.engine import Engine

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from meta_env import MetaEnv


# ---------------------------------------------------------------------------
# Safety-Gym config  (same as CEM / REINFORCE versions)
# ---------------------------------------------------------------------------

STATIC_CONFIG = {
    "placements_extents": [-1.5, -1.5, 1.5, 1.5],

    # #         # Note: the "robot_keepout" is set to 0 here, which means the robot can start anywhere within the placements_extents.
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
# Geometric feature extraction  (same 7 features as CEM / REINFORCE)
# ---------------------------------------------------------------------------

GOAL_POS   = np.array([1.1, 1.1])
HAZ_POS    = np.array([0.0, 0.0])
D_MAX      = float(np.linalg.norm(np.array([3.0, 3.0])))   # arena diagonal ~ 4.24
N_FEATURES = 7


def extract_features(obs: np.ndarray, env: MetaEnv) -> np.ndarray:
    """Extract 7 geometric features from obs + sim state.

    Returns: [v_x, v_y, d_goal_norm, d_haz_norm, delta_theta_norm, t_frac, budget_norm]
    """
    # Get velocity from MuJoCo sim (obs velocity fields can be zero)
    robot_vel = env._env.sim.data.get_body_xvelp('robot')
    v_x = float(robot_vel[0])
    v_y = float(robot_vel[1])

    robot_pos = env._env.sim.data.get_body_xpos('robot')[:2]
    d_goal = float(np.linalg.norm(robot_pos - GOAL_POS))
    d_haz  = float(np.linalg.norm(robot_pos - HAZ_POS))

    vec_goal = GOAL_POS - robot_pos
    vec_haz  = HAZ_POS  - robot_pos
    angle_goal  = np.arctan2(vec_goal[1], vec_goal[0])
    angle_haz   = np.arctan2(vec_haz[1],  vec_haz[0])
    delta_theta = float(angle_goal - angle_haz)
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

    d_goal_norm      = d_goal / D_MAX
    d_haz_norm       = d_haz  / D_MAX
    delta_theta_norm = delta_theta / np.pi

    time_left_norm = float(obs[-3])
    t_frac         = time_left_norm  # fraction of episode remaining (0 to 1)
    budget_norm    = float(obs[-2])

    return np.array([v_x, v_y, d_goal_norm, d_haz_norm,
                     delta_theta_norm, t_frac, budget_norm],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# MuJoCo state save / restore helpers
# ---------------------------------------------------------------------------

def save_mujoco_state(env: MetaEnv) -> dict:
    """Save full simulator + MetaEnv state for counterfactual rollouts.

    Returns a dict containing everything needed to restore the state.
    """
    sim = env._env.sim
    return {
        "mj_state":    copy.deepcopy(sim.get_state()),
        "raw_obs":     env._raw_obs.copy(),
        "t":           env.t,
        "B":           env.B,
        "goal_met":    env._goal_met,
        "switched":    env._switched,
        # Engine-level state (not covered by sim.get_state())
        "engine_done":  env._env.done,
        "engine_steps": env._env.steps,
    }


def restore_mujoco_state(env: MetaEnv, state: dict):
    """Restore full simulator + MetaEnv state from a saved snapshot."""
    sim = env._env.sim
    sim.set_state(state["mj_state"])
    sim.forward()
    env._raw_obs   = state["raw_obs"].copy()
    env.t          = state["t"]
    env.B          = state["B"]
    env._goal_met  = state["goal_met"]
    env._switched  = state["switched"]
    # Engine-level state
    env._env.done  = state["engine_done"]
    env._env.steps = state["engine_steps"]


# ---------------------------------------------------------------------------
# PyTorch Q-Network
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    """
    Q-network with 2 outputs: Q(s, wait) and Q(s, switch).

    Architecture: 7 → hidden_size (ReLU) → 2
    Parameters: 9*H + 2   (H=8 → 74,  H=16 → 146)
    """

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),   # [Q_wait, Q_switch]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 7) -> Q_values: (..., 2)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class AdvantageGapBuffer:
    """Replay buffer weighted by advantage gap |R_switch - R_wait|.

    Unlike classic PER (TD-error based), priorities are fixed at insertion
    time and reflect how *critical* a transition is for the switch/wait
    decision — not how wrong the network currently is.

    Transitions where the action choice has high impact (large gap) are
    sampled more often; transitions where switch ≈ wait are deprioritized.

    Sampling probability:  P(i) = p_i^α / Σ_j p_j^α
    IS weight:             w_i = (N · P(i))^{-β} / max_j w_j
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self._max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, priority: float = None):
        data = (state, action, reward, next_state, done)
        if priority is None:
            priority = self._max_priority
        pa = (abs(priority) + 1e-6) ** self.alpha

        if self.size < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = pa
        self._max_priority = max(self._max_priority, abs(priority) + 1e-6)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.RandomState,
               beta: float = 0.4):
        n = min(batch_size, self.size)
        probs = self.priorities[:self.size].copy()
        probs /= probs.sum()

        idx = rng.choice(self.size, size=n, replace=True, p=probs)

        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in idx]
        states   = np.array([b[0] for b in batch], dtype=np.float32)
        actions  = np.array([b[1] for b in batch], dtype=np.int64)
        rewards  = np.array([b[2] for b in batch], dtype=np.float32)
        n_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones    = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, n_states, dones, weights.astype(np.float32)

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Counterfactual switch rollout
# ---------------------------------------------------------------------------

def counterfactual_switch_return(env: MetaEnv, saved_state: dict,
                                 cost_weight: float, deadline_weight: float,
                                 gamma: float, max_horizon: int) -> float:
    """From a saved state, execute aggressive policy until episode end.

    Computes the discounted return:
        R = Σ_t γ^t * r_t
    where r_t = -cost_weight * cost_t per step, and terminal reward is
        +1 (success) or -deadline_weight (failure).

    The env state is restored after this rollout (no side effects).
    """
    # Save current state to restore later
    current_state = save_mujoco_state(env)

    # Restore the target state
    restore_mujoco_state(env, saved_state)

    # Force switch to aggressive
    env._switched = True
    obs = env._augment_meta(env._raw_obs)

    discounted_return = 0.0
    discount = 1.0
    done = False
    ep_len = env.t

    while not done and ep_len < max_horizon:
        obs, _r, done, info = env.step(1)   # always aggressive
        cost_step = float(info.get("cumulative_cost", 0.0))
        r_step = -cost_weight * cost_step

        goal_met = bool(info.get("goal_met", False))
        budget_expired = bool(info.get("budget_expired", False))
        ep_len = info.get("time_step", ep_len + 1)

        # Add step reward
        discounted_return += discount * r_step
        discount *= gamma

        if goal_met:
            discounted_return += discount * 1.0
            done = True
        elif budget_expired:
            discounted_return += discount * (-deadline_weight)
            done = True

    # If neither goal nor budget triggered done (env_done from underlying)
    if not done:
        discounted_return += discount * (-deadline_weight)

    # Restore original state (no side effects)
    restore_mujoco_state(env, current_state)
    return discounted_return


# ---------------------------------------------------------------------------
# Oracle conservative value: V*(s) ≈ max_k R(cons k steps then switch)
# ---------------------------------------------------------------------------

def oracle_conservative_value(env: MetaEnv, saved_state: dict,
                              cost_weight: float, deadline_weight: float,
                              gamma: float, max_horizon: int,
                              switch_interval: int = 1) -> float:
    """Compute oracle value of a state by rolling out conservative and
    periodically evaluating switch.

    From the saved state, run the conservative policy.  Every
    ``switch_interval`` meta-steps, save the state, compute the
    counterfactual switch return, and record the cumulative return:

        candidate_k = Σ_{j=0}^{k-1} γ^j r^cons_j  +  γ^k · R^switch_k

    Also includes the "switch immediately" candidate (k=0).

    Returns  max_k candidate_k  (best achievable return from this state).

    The env state is restored after the rollout (no side effects).
    """

    # Save current state to restore later
    current_state = save_mujoco_state(env)

    # Restore the target state
    restore_mujoco_state(env, saved_state)

    # --- k=0: switch immediately ---
    switch_now = counterfactual_switch_return(
        env, saved_state, cost_weight, deadline_weight, gamma, max_horizon
    )
    best_return = switch_now
    # --- k>0: conservative for k steps, then evaluate switch ---
    cons_cum_return = 0.0
    cons_discount   = 1.0
    done = False
    meta_step = 0
    ep_len = env.t

    while not done and ep_len < max_horizon:
        obs, _r, done, info = env.step(0)   # conservative step
        cost_step = float(info.get("cumulative_cost", 0.0))
        n_taken   = info.get("n_steps_taken", 1)
        ep_len   += n_taken
        meta_step += 1

        r_step = -cost_weight * cost_step
        goal_met       = bool(info.get("goal_met", False))
        budget_expired = bool(info.get("budget_expired", False))

        if goal_met:
            r_step += 1.0
        elif budget_expired and not goal_met:
            r_step -= deadline_weight

        cons_cum_return += cons_discount * r_step
        cons_discount   *= gamma

        # Episode ended during conservative phase
        if done:
            # The conservative-only return is also a candidate
            best_return = max(best_return, cons_cum_return)
            break
        
        # Periodically evaluate switch from current state
        if meta_step % switch_interval == 0:
            checkpoint = save_mujoco_state(env)
            switch_ret = counterfactual_switch_return(
                env, checkpoint, cost_weight, deadline_weight,
                gamma, max_horizon
            )
            candidate = cons_cum_return + cons_discount * switch_ret
            best_return = max(best_return, candidate)

    # Restore original state (no side effects)
    restore_mujoco_state(env, current_state)
    return best_return


# ---------------------------------------------------------------------------
# Data collection: run one episode, collect BOTH wait and switch transitions
# ---------------------------------------------------------------------------

def collect_episode_transitions(env: MetaEnv, q_net: QNet, seed: int,
                                budget: int, max_horizon: int,
                                cost_weight: float, deadline_weight: float,
                                gamma: float, epsilon: float,
                                rng: np.random.RandomState,
                                switch_interval: int = 1,
                                force_wait: bool = False,
                                min_adv_gap: float = 0.0) -> list:
    """Run one episode with epsilon-greedy policy, collecting transitions.

    For each meta-step while still conservative, TWO **terminal**
    transitions are collected (symmetric dataset, no Q-bootstrap needed):

      1) **switch** (counterfactual): from s_t run aggressive to end.
           → (s_t, switch, R^agg_{t:T}, -, done=True)

      2) **wait** (oracle): execute one real conservative step to get
         r_t and reach s_{t+1}. Then compute oracle_conservative_value
         from s_{t+1} — i.e. the best return achievable by staying
         conservative for some number of steps and then switching.
           → (s_t, wait, r_t + γ·V*(s_{t+1}), -, done=True)

    Both transitions are terminal: the reward field holds the full
    return, so the DQN target is just the stored reward (no bootstrap).

    The epsilon-greedy action only determines whether the agent actually
    switches in the real trajectory (affecting which future states are
    visited), but BOTH transitions are always stored for every state.

    Returns list of (state, action, reward, next_state, done) tuples.
    """
    env.seed(int(seed))
    try:
        env._env.seed(int(seed))
    except Exception:
        pass
    obs = env.reset().copy()
    env.B = budget
    if env.irreversible_switch:
        obs[-2] = env._budget_norm()
        obs[-1] = 0.0
    else:
        obs[-1] = env._budget_norm()

    transitions = []
    done = False
    ep_len = 0
    switched = False

    while not done and ep_len < max_horizon:
        feats = extract_features(obs, env)
        if switched:
            # Already switched — no more decisions, just run aggressive
            obs, _r, done, info = env.step(1)
            ep_len += info.get("n_steps_taken", 1)
            continue

        # Extract features for current state
        feats = extract_features(obs, env)

        # Save MuJoCo state BEFORE acting (for counterfactual rollout)
        saved_state = save_mujoco_state(env)
        _pos_before = env._env.sim.data.get_body_xpos('robot')[:2].copy()

        # ---- 1) Counterfactual SWITCH return ----
        switch_return = counterfactual_switch_return(
            env, saved_state, cost_weight, deadline_weight, gamma, max_horizon
        )

        # # Sanity check: state restore correctness (first 3 states only)
        # if len(transitions) < 6:
        #     _pos_after = env._env.sim.data.get_body_xpos('robot')[:2]
        #     _pos_err = float(np.linalg.norm(_pos_before - _pos_after))
        #     if _pos_err > 1e-6:
        #         print(f"  [WARN] state restore drift: "
        #               f"before=({_pos_before[0]:.6f},{_pos_before[1]:.6f})  "
        #               f"after=({_pos_after[0]:.6f},{_pos_after[1]:.6f})  "
        #               f"err={_pos_err:.2e}")
        #     else:
        #         print(f"  [OK] state restore: pos_err={_pos_err:.2e}  "
        #               f"t={env.t}  done={env._env.done}  steps={env._env.steps}")

        # ---- 2) Real WAIT step + oracle value of s_{t+1} ----
        # Execute a conservative meta-step to get r_t
        obs_next, _r, done, info = env.step(0)   # always wait in env
        cost_step = float(info.get("cumulative_cost", 0.0))
        ep_len += info.get("n_steps_taken", 1)

        r_step = -cost_weight * cost_step
        goal_met = bool(info.get("goal_met", False))
        budget_expired = bool(info.get("budget_expired", False))

        if goal_met:
            r_step += 1.0
        elif budget_expired and not goal_met:
            r_step -= deadline_weight

        # Oracle bootstrap: V*(s_{t+1}) = best return from s_{t+1} by
        # following conservative for k steps then switching optimally.
        oracle_v = None
        if done:
            wait_return = r_step
        else:
            state_after_wait = save_mujoco_state(env)
            oracle_v = oracle_conservative_value(
                env, state_after_wait, cost_weight, deadline_weight,
                gamma, max_horizon, switch_interval=switch_interval,
            )
            wait_return = r_step + gamma * oracle_v

        # ---- Advantage gap ----
        adv_gap = abs(switch_return - wait_return)

        # Skip uninformative transitions (both actions look equally good/bad).
        if adv_gap < min_adv_gap:
            obs = obs_next.copy()
            continue

        # If the next state is unrecoverable under both actions, stop collecting
        # further transitions — they will all have gap ≈ 0 anyway.
        if oracle_v is not None and oracle_v <= -deadline_weight * 0.9:
            switched = True   # stop after storing the current (informative) pair

        # ---- Store both transitions with advantage-gap priority ----
        feats_next = extract_features(obs_next, env)

        transitions.append((
            feats.copy(),
            1,                   # action = switch
            switch_return,       # full discounted return
            feats.copy(),        # next_state placeholder (unused: done=True)
            True,                # done = True (terminal)
            adv_gap,             # priority hint for PER
        ))
        transitions.append((
            feats.copy(),
            0,                   # action = wait
            wait_return,         # full return (terminal)
            feats_next.copy(),   # next_state (unused: done=True)
            True,                # done = True (terminal, oracle bootstrap)
            adv_gap,             # priority hint for PER
        ))


        # ---- Epsilon-greedy: decide if we actually switch ----
        # The wait transition above is already stored; now we decide
        # whether the real trajectory continues conservative or switches.
        if force_wait:
            action = 0   # warmup: always wait to collect full episodes
        elif rng.random() < epsilon:
            action = rng.randint(2)
        else:
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                q_vals = q_net(x).squeeze(0)
                action = int(q_vals.argmax().item())

        if action == 1 and not done:
            # Agent chose to switch — from the NEXT state onwards,
            # run aggressive for the rest of the episode.
            switched = True

        obs = obs_next.copy()
    return transitions


# ---------------------------------------------------------------------------
# Zone-based k* search helper
# ---------------------------------------------------------------------------

def _find_best_k_zone_search(
    steps: list, env: MetaEnv,
    cost_weight: float, deadline_weight: float,
    gamma: float, max_horizon: int,
    scan_interval: int = 5,
    n_top_zones: int = 2,
) -> int:
    """Find k* = argmax_k R_switch(k) using a coarse-to-fine zone search.

    Strategy
    --------
    1. Divide [0, N-1] into zones of width ``scan_interval``.
    2. Probe the **midpoint** of every zone  (coarse, cheap).
    3. Rank zones by their midpoint return; keep the best ``n_top_zones``.
    4. **Dense scan** every step inside the selected zones.
    5. Return the global argmax over all evaluated k.

    Why this beats a uniform scan
    ------------------------------
    A uniform scan of step ``scan_interval`` can miss the true peak when it
    lies between two probes.  By densely scanning the most promising zones
    we guarantee we evaluate every candidate inside those zones at the cost
    of only a moderate increase over the coarse scan.

    Cost (N steps, n_zones = ceil(N/scan_interval)):
        n_zones  (coarse probes)  +  n_top_zones * scan_interval  (dense)
    vs  N  for brute-force.

    Example  N=30, scan_interval=5, n_top_zones=2:
        6 coarse + 10 dense = 16 rollout calls  (vs 30 brute-force)
    """
    N = len(steps)
    if N == 0:
        return 0

    zone_size = max(1, scan_interval)
    n_zones   = (N + zone_size - 1) // zone_size  # ceil division

    def _eval(k: int) -> float:
        """Evaluate and cache R_switch at step k."""
        if "switch_return" not in steps[k]:
            steps[k]["switch_return"] = counterfactual_switch_return(
                env, steps[k]["state"], cost_weight, deadline_weight,
                gamma, max_horizon,
            )
        return steps[k]["switch_return"]

    # ---- Phase 1: probe midpoint of every zone ----
    zone_scores = []
    for z in range(n_zones):
        k_start = z * zone_size
        k_end   = min(k_start + zone_size, N)
        k_mid   = (k_start + k_end - 1) // 2
        zone_scores.append((z, k_mid, _eval(k_mid)))

    # ---- Phase 2: select top n_top_zones by midpoint score ----
    zone_scores.sort(key=lambda x: x[2], reverse=True)
    top_zones = zone_scores[:min(n_top_zones, n_zones)]

    # ---- Phase 3: dense scan inside every top zone ----
    for z, _k_mid, _score in top_zones:
        k_start = z * zone_size
        k_end   = min(k_start + zone_size, N)
        for k in range(k_start, k_end):
            _eval(k)

    # ---- Return global argmax over all evaluated steps ----
    # Ties broken by largest k (latest step preferred).
    return max(
        (k for k in range(N) if "switch_return" in steps[k]),
        key=lambda k: (steps[k]["switch_return"], k),
    )


# ---------------------------------------------------------------------------
# Data collection: find the single most informative switch transition
# ---------------------------------------------------------------------------

def collect_best_switch_transition(
    env: MetaEnv, seed: int, budget: int, max_horizon: int,
    cost_weight: float, deadline_weight: float, gamma: float,
    rng: np.random.RandomState,
    switch_interval: int = 1,
    scan_interval: int = 5,
    n_top_zones: int = 2,
    n_anchors: int = 2,
    min_adv_gap: float = 0.0,
) -> list:
    """Find the most informative switch transitions for one episode.

    Algorithm
    ---------
    1. Run the full episode under wait-only policy, saving MuJoCo state at
       every meta-step.
    2. If the episode succeeds → wait alone is sufficient → return [].
    3. Use ``_find_best_k_zone_search`` to find k* = argmax_k R_switch(k).
    4. Build up to ``n_anchors`` transition pairs covering different regions
       of the switch/wait return landscape:

         n_anchors ≥ 1 → k*        (k where switch wins most — large positive gap)
         n_anchors ≥ 2 → k_low     (earliest evaluated k — wait wins, negative gap)
         n_anchors ≥ 3 → k_mid     (midpoint of [k_low, k*] — boundary region)

    Why three anchors?
    ------------------
    The DQN needs to learn Q(s, switch) AND Q(s, wait) accurately.  A buffer
    with only k* teaches the network that switch is always good.  Adding k_low
    teaches that wait wins early in the episode.  Adding k_mid provides a
    near-boundary example that sharpens the decision threshold.

    k_low costs 1 extra oracle call; k_mid costs at most 2 (1 switch rollout
    if not cached + 1 oracle).

    Returns [] if the conservative-only run succeeds, otherwise 2*n_anchors
    transitions (fewer if k_low or k_mid coincide with k*).
    """
    env.seed(int(seed))
    try:
        env._env.seed(int(seed))
    except Exception:
        pass
    obs = env.reset().copy()
    env.B = budget
    if env.irreversible_switch:
        obs[-2] = env._budget_norm()
        obs[-1] = 0.0
    else:
        obs[-1] = env._budget_norm()

    # ---------------------------------------------------------------- #
    # Phase 1: full conservative rollout, save state at every step     #
    # ---------------------------------------------------------------- #
    steps = []   # one dict per meta-step
    done = False
    ep_len = 0
    goal_met_final = False

    while not done and ep_len < max_horizon:
        saved_state = save_mujoco_state(env)
        feats = extract_features(obs, env)
        steps.append({
            "feats": feats.copy(),
            "state": saved_state,
        })

        obs, _r, done, info = env.step(0)   # always wait
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

        steps[-1]["r_step"]    = r_step
        steps[-1]["done_next"] = done

    # ---------------------------------------------------------------- #
    # Phase 2a: SUCCESS episode — collect "wait wins" anchors          #
    # ---------------------------------------------------------------- #
    # For these episodes the conservative policy alone reaches the goal.
    # R_wait(k) ≈ +1 at most steps; R_switch(k) is lower wherever the
    # aggressive policy incurs costs or hazard violations.
    # These transitions teach the network: Q(s, wait) >> Q(s, switch).
    N_steps = len(steps)
    if goal_met_final or not steps:
        # print(f"  [collect] seed={seed}  B={budget}  N={N_steps}  "
        #       f"conservative SUCCESS → collecting wait-wins anchors")

        def _switch_return_at_s(k: int) -> float:
            if "switch_return" not in steps[k]:
                steps[k]["switch_return"] = counterfactual_switch_return(
                    env, steps[k]["state"], cost_weight, deadline_weight,
                    gamma, max_horizon,
                )
            return steps[k]["switch_return"]

        def _wait_return_at_s(k: int) -> float:
            r = steps[k]["r_step"]
            if steps[k]["done_next"] or k + 1 >= N_steps:
                return r
            ov = oracle_conservative_value(
                env, steps[k + 1]["state"],
                cost_weight, deadline_weight, gamma, max_horizon,
                switch_interval=switch_interval,
            )
            return r + gamma * ov

        # Choose n_anchors positions: always include k=0 (episode start),
        # then distribute the remaining n_anchors-1 positions evenly.
        # k=0 is critical: the aggressive policy starts from the very
        # beginning and is most likely to incur hazard costs → large gap.
        n_pos = min(n_anchors, N_steps)
        if n_pos == 1:
            anchor_ks = [0]
        else:
            anchor_ks = [0] + [N_steps * i // n_pos for i in range(1, n_pos)]

        result = []
        for k in anchor_ks:
            sr  = _switch_return_at_s(k)
            wr  = _wait_return_at_s(k)
            gap = abs(sr - wr)
            winner = "SWITCH" if sr > wr else ("WAIT" if wr > sr else "TIE")
            # print(f"    anchor success k={k:3d}  "
            #       f"R_switch={sr:+.4f}  R_wait={wr:+.4f}  "
            #       f"gap={gap:.4f}  winner={winner}")
            if gap < min_adv_gap:
                # print(f"      skipped (gap < min_adv_gap={min_adv_gap:.4f})")
                continue
            if sr >= wr:
                # print(f"      skipped (switch wins — not a clean wait-wins example)")
                continue
            f = steps[k]["feats"]
            result.append((f.copy(), 1, sr, f.copy(), True, gap))
            result.append((f.copy(), 0, wr, f.copy(), True, gap))

        print(f"    → {len(result)} transitions saved for this episode")
        return result

    # print(f"  [collect] seed={seed}  B={budget}  N={N_steps}  "
    #       f"conservative FAILED → searching best switch...")

    # ---------------------------------------------------------------- #
    # Phase 2: zone-based search → k* = argmax_k switch_return        #
    # ---------------------------------------------------------------- #
    best_k = _find_best_k_zone_search(
        steps, env, cost_weight, deadline_weight, gamma, max_horizon,
        scan_interval=scan_interval,
        n_top_zones=n_top_zones,
    )

    # ---------------------------------------------------------------- #
    # Phase 3: build transition pairs at anchor points                 #
    # ---------------------------------------------------------------- #
    def _switch_return_at(k: int) -> float:
        """Evaluate and cache R_switch at step k."""
        if "switch_return" not in steps[k]:
            steps[k]["switch_return"] = counterfactual_switch_return(
                env, steps[k]["state"], cost_weight, deadline_weight,
                gamma, max_horizon,
            )
        return steps[k]["switch_return"]

    def _wait_return_at(k: int) -> float:
        """Oracle wait-return at step k."""
        r = steps[k]["r_step"]
        if steps[k]["done_next"] or k + 1 >= len(steps):
            return r
        ov = oracle_conservative_value(
            env, steps[k + 1]["state"],
            cost_weight, deadline_weight, gamma, max_horizon,
            switch_interval=switch_interval,
        )
        return r + gamma * ov

    def _make_pair(k: int, label: str) -> list:
        """(switch, wait) transition pair at step k, with a diagnostic print."""
        sr  = _switch_return_at(k)
        wr  = _wait_return_at(k)
        gap = abs(sr - wr)
        f   = steps[k]["feats"]
        winner = "SWITCH" if sr > wr else ("WAIT" if wr > sr else "TIE")
        # print(f"    anchor {label:6s}  k={k:3d}  "
        #       f"R_switch={sr:+.4f}  R_wait={wr:+.4f}  "
        #       f"gap={gap:.4f}  winner={winner}")
        return [
            (f.copy(), 1, sr, f.copy(), True, gap),
            (f.copy(), 0, wr, f.copy(), True, gap),
        ]

    # Anchor 0 — k*: the step where switch gives the highest return.
    #
    # Consistency check: if R_wait(k*) > R_switch(k*), it means the oracle
    # found a better switch point AFTER k* (it searched forward from s_{k*+1}).
    # The zone search missed that point.  Resolve by dense-scanning all k > k*
    # not yet evaluated and recomputing the global argmax.
    #
    # This guarantees that the stored k* satisfies R_switch(k*) ≥ R_wait(k*),
    # i.e., it is truly the step where switching is at least as good as waiting.
    main_pair = _make_pair(best_k, "k* (init)")
    if main_pair[0][2] <= main_pair[1][2]:   # R_switch ≤ R_wait → zone search missed true k*
        # print(f"    oracle found better k later — dense-scanning k={best_k+1}..{N_steps-1}")
        for k in range(best_k + 1, N_steps):
            _switch_return_at(k)   # fills cache for all remaining k
        # Recompute global argmax (tie-broken by largest k)
        best_k = max(
            (k for k in range(N_steps) if "switch_return" in steps[k]),
            key=lambda k: (steps[k]["switch_return"], k),
        )
        main_pair = _make_pair(best_k, "k* (refined)")

    sw_ret_star = main_pair[0][2]
    wt_ret_star = main_pair[1][2]
    gap_star    = main_pair[0][5]

    if sw_ret_star <= wt_ret_star:
        # Even after full scan, switch never beats wait → no useful signal
        # print(f"    → switch never beats wait anywhere, episode skipped")
        return []
    if gap_star < min_adv_gap:
        # print(f"    → gap={gap_star:.4f} < min_adv_gap ({min_adv_gap:.4f}), episode skipped")
        return []
    result = list(main_pair)

    if n_anchors >= 2:
        # Anchor 1 — k_low: the earliest evaluated k (usually k=0 or first zone midpoint)
        #   R_switch is low here, R_wait (oracle) is still high
        #   → teaches Q(s, wait) >> Q(s, switch) at the start of the episode
        k_low = min(k for k in range(len(steps)) if "switch_return" in steps[k])
        if k_low != best_k:
            result.extend(_make_pair(k_low, "k_low"))
        else:
            # print(f"    anchor k_low  skipped (same as k*={best_k})")
                pass

        if n_anchors >= 3:
            # Anchor 2 — midpoint between k_low and k* (boundary/transition region)
            #   Both R_switch and R_wait are moderate; gap is small
            #   → sharpens the decision boundary learned by the network
            k_mid_anchor = (k_low + best_k) // 2
            if k_mid_anchor not in (k_low, best_k):
                if "switch_return" not in steps[k_mid_anchor]:
                    steps[k_mid_anchor]["switch_return"] = counterfactual_switch_return(
                        env, steps[k_mid_anchor]["state"],
                        cost_weight, deadline_weight, gamma, max_horizon,
                    )
                result.extend(_make_pair(k_mid_anchor, "k_mid"))
            else:
                # print(f"    anchor k_mid  skipped (coincides with k_low or k*)")
                pass


    print(f"    → {len(result)} transitions saved for this episode")
    return result


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_policy(q_net: QNet, env: MetaEnv, seeds: np.ndarray,
                    budgets: np.ndarray, max_horizon: int) -> dict:
    """Run deterministic greedy evaluation (no exploration)."""
    q_net.eval()
    successes    = []
    costs        = []
    switch_steps = []

    for seed, budget in zip(seeds, budgets):
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
        cum_cost = 0.0
        switch_step = -1
        switched = False
        info = {}

        while not done and ep_len < max_horizon:
            if not switched:
                feats = extract_features(obs, env)
                with torch.no_grad():
                    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                    q_vals = q_net(x).squeeze(0)
                    action = int(q_vals.argmax().item())

                if action == 1:
                    switched = True
                    switch_step = ep_len + 1
            else:
                action = 1

            obs, _r, done, info = env.step(action)
            cum_cost += float(info.get("cumulative_cost", 0.0))
            ep_len += info.get("n_steps_taken", 1)

        success = float(bool(info.get("goal_met", False)))
        successes.append(success)
        costs.append(cum_cost)
        switch_steps.append(switch_step)

    costs_arr = np.array(costs)
    succ_rate = float(np.mean(successes))
    mean_cost = float(np.mean(costs_arr))

    def _cvar(alpha):
        k = max(1, int(np.ceil(alpha * len(costs_arr))))
        return float(np.mean(np.sort(costs_arr)[-k:]))

    frac_switched = float(np.mean([s > 0 for s in switch_steps]))
    mean_sw_step  = float(np.mean([s for s in switch_steps if s > 0])
                          if any(s > 0 for s in switch_steps) else 0.0)

    return {
        "success_rate":     succ_rate,
        "mean_cost":        mean_cost,
        "cvar_10":          _cvar(0.10),
        "cvar_20":          _cvar(0.20),
        "cvar_30":          _cvar(0.30),
        "frac_switched":    frac_switched,
        "mean_switch_step": mean_sw_step,
    }


# ---------------------------------------------------------------------------
# DQN training loop
# ---------------------------------------------------------------------------

def dqn_train(q_net: QNet, target_net: QNet, env: MetaEnv,
              all_seeds: np.ndarray, all_budgets: np.ndarray,
              max_horizon: int,
              cost_weight: float, deadline_weight: float,
              gamma: float, n_epochs: int, batch_size: int,
              lr: float, epsilon_start: float, epsilon_end: float,
              epsilon_decay_epochs: int,
              target_update_every: int,
              buffer_capacity: int,
              rng_seed: int, eval_every: int,
              eval_seeds: np.ndarray, eval_budgets: np.ndarray,
              checkpoint_dir: str = None,
              switch_interval: int = 1,
              episodes_per_epoch: int = 1,
              min_buffer_size: int = 500,
              warmup_epochs: int = 0,
              per_alpha: float = 0.6,
              per_beta_start: float = 0.4,
              per_beta_end: float = 1.0,
              budget_bias: float = 0.0,
              min_adv_gap: float = 0.0,
              collection_mode: str = "all",
              scan_interval: int = 5,
              n_top_zones: int = 2,
              n_anchors: int = 2,
              feature_noise: float = 0.0) -> list:
    """
    DQN training loop with counterfactual switch transitions.

    For each epoch:
      1. Sample ``episodes_per_epoch`` episodes from the pool
      2. Collect wait + switch transitions via epsilon-greedy
      3. Store transitions in replay buffer
      4. Sample a mini-batch and perform a DQN update

    Returns history: list of dicts (one per epoch).
    """
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    rng = np.random.RandomState(rng_seed)
    replay_buffer = AdvantageGapBuffer(capacity=buffer_capacity,
                                        alpha=per_alpha)
    history = []
    best_eval_score = -np.inf

    # Budget-biased episode selection weights
    if budget_bias > 0:
        b_min, b_max = float(all_budgets.min()), float(all_budgets.max())
        b_norm = (all_budgets.astype(float) - b_min) / max(1.0, b_max - b_min)
        ep_weights = np.exp(-budget_bias * b_norm)
        ep_weights /= ep_weights.sum()
    else:
        ep_weights = None

    n_params = sum(p.numel() for p in q_net.parameters())
    hidden_size = q_net.net[0].out_features

    print(f"\n{'='*70}")
    print(f"  DQN training  ({n_epochs} epochs)")
    print(f"    architecture: 7 -> {hidden_size} (ReLU) -> 2  [Q_wait, Q_switch]")
    print(f"    parameters:   {n_params}")
    print(f"    features:     [v_x, v_y, d_goal, d_haz, dtheta, t_frac, budget_norm]")
    print(f"    lr:           {lr}")
    print(f"    gamma:        {gamma}")
    print(f"    epsilon:      {epsilon_start} -> {epsilon_end}  "
          f"(decay over {epsilon_decay_epochs} epochs)")
    print(f"    batch_size:   {batch_size}")
    print(f"    target_update: every {target_update_every} epochs")
    print(f"    buffer_cap:   {buffer_capacity}")
    print(f"    reward:       r_t = -{cost_weight}*cost_t, "
          f"terminal: +1(success) / -{deadline_weight}(failure)")
    print(f"    episode pool: {len(all_seeds)}")
    print(f"    oracle_interval: switch every {switch_interval} meta-steps")
    print(f"    ep/epoch:     {episodes_per_epoch}")
    _min_buf = min_buffer_size if min_buffer_size > 0 else batch_size
    print(f"    min_buffer:   {_min_buf}  (start updates after this many transitions)")
    print(f"    warmup_epochs: {warmup_epochs}  (force wait, no switch)")
    print(f"    adv-gap alpha: {per_alpha}  beta: {per_beta_start} -> {per_beta_end}")
    print(f"    budget_bias:  {budget_bias}")
    _scan_str = (f"  (zone search: scan_interval={scan_interval}, top_zones={n_top_zones})"
                 if collection_mode == "best_only" else "")
    print(f"    collection:   {collection_mode}{_scan_str}")
    print(f"    eval every:   {eval_every} epochs  ({len(eval_seeds)} episodes)")
    print(f"{'='*70}")

    # Copy initial weights to target network
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # CSV setup
    csv_file = None
    csv_writer = None
    hist_path = None
    if checkpoint_dir:
        hist_path = os.path.join(checkpoint_dir, "dqn_history.csv")

    for epoch in range(1, n_epochs + 1):
        q_net.train()

        # Epsilon schedule (linear decay)
        if epoch <= epsilon_decay_epochs:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (
                (epoch - 1) / max(1, epsilon_decay_epochs - 1)
            )
        else:
            epsilon = epsilon_end

        # --- Collect transitions from episodes_per_epoch episodes ---
        _force_wait = epoch <= warmup_epochs
        transitions = []
        for _ep in range(episodes_per_epoch):
            if ep_weights is not None:
                ep_idx = rng.choice(len(all_seeds), p=ep_weights)
            else:
                ep_idx = rng.randint(len(all_seeds))
            seed = int(all_seeds[ep_idx])
            budget = int(all_budgets[ep_idx])

            if collection_mode == "best_only":
                ep_trans = collect_best_switch_transition(
                    env, seed, budget, max_horizon,
                    cost_weight, deadline_weight, gamma, rng,
                    switch_interval=switch_interval,
                    scan_interval=scan_interval,
                    n_top_zones=n_top_zones,
                    n_anchors=n_anchors,
                    min_adv_gap=min_adv_gap,
                )
            else:
                ep_trans = collect_episode_transitions(
                    env, q_net, seed, budget, max_horizon,
                    cost_weight, deadline_weight, gamma, epsilon, rng,
                    switch_interval=switch_interval,
                    force_wait=_force_wait,
                    min_adv_gap=min_adv_gap,
                )
            transitions.extend(ep_trans)

        for t in transitions:
            replay_buffer.push(*t)

        # --- DQN update (if enough samples) ---
        loss_val = 0.0
        _update_threshold = max(batch_size, min_buffer_size)
        if len(replay_buffer) >= _update_threshold:
            # PER: beta annealing
            beta = per_beta_start + (per_beta_end - per_beta_start) * (
                (epoch - 1) / max(1, n_epochs - 1)
            )
            states, actions, rewards, next_states, dones, is_weights = \
                replay_buffer.sample(batch_size, rng, beta=beta)

            states_t      = torch.tensor(states, dtype=torch.float32)
            actions_t     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards_t     = torch.tensor(rewards, dtype=torch.float32)
            next_states_t = torch.tensor(next_states, dtype=torch.float32)
            dones_t       = torch.tensor(dones, dtype=torch.float32)
            is_weights_t  = torch.tensor(is_weights, dtype=torch.float32)

            # Current Q values for the actions taken
            if feature_noise > 0.0:
                states_t = states_t + torch.randn_like(states_t) * feature_noise
            q_values = q_net(states_t).gather(1, actions_t).squeeze(1)

            # Target Q values (DQN target)
            with torch.no_grad():
                q_next = target_net(next_states_t).max(dim=1).values
                targets = rewards_t + gamma * (1.0 - dones_t) * q_next

            # IS-weighted MSE loss (corrects advantage-gap sampling bias)
            loss = (is_weights_t * (q_values - targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(loss.detach())

        # --- Update target network ---
        if epoch % target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        # --- Logging ---
        n_wait   = sum(1 for t in transitions if t[1] == 0)
        n_switch = sum(1 for t in transitions if t[1] == 1)

        # Count winner categories from collected pairs.
        # Transitions are stored as consecutive (switch, wait) pairs;
        # compare R_switch vs R_wait to classify each pair.
        n_pro_switch = 0  # switch wins: R_switch > R_wait
        n_pro_wait   = 0  # wait wins:   R_wait   > R_switch
        n_pro_tie    = 0  # tie:          |R_switch - R_wait| <= 1e-8
        for _i in range(0, len(transitions) - 1, 2):
            _t_sw, _t_wt = transitions[_i], transitions[_i + 1]
            if _t_sw[1] == 1 and _t_wt[1] == 0:   # correct order
                _r_sw, _r_wt = _t_sw[2], _t_wt[2]
                if _r_sw > _r_wt + 1e-8:
                    n_pro_switch += 1
                elif _r_wt > _r_sw + 1e-8:
                    n_pro_wait += 1
                else:
                    n_pro_tie += 1

        row = {
            "epoch":         epoch,
            "loss":          round(loss_val, 6),
            "epsilon":       round(epsilon, 4),
            "n_transitions": len(transitions),
            "n_wait":        n_wait,
            "n_switch":      n_switch,
            "ep_pro_switch": n_pro_switch,
            "ep_pro_wait":   n_pro_wait,
            "ep_tie":        n_pro_tie,
            "buffer_size":   len(replay_buffer),
        }

        # Periodic evaluation
        do_eval = (epoch % eval_every == 0) or (epoch == n_epochs)
        if do_eval:
            eval_m = evaluate_policy(
                q_net, env, eval_seeds, eval_budgets, max_horizon
            )
            row["eval_succ"]      = round(eval_m["success_rate"], 4)
            row["eval_mean_cost"] = round(eval_m["mean_cost"], 4)
            row["eval_cvar10"]    = round(eval_m["cvar_10"], 4)
            row["eval_cvar20"]    = round(eval_m["cvar_20"], 4)
            row["eval_cvar30"]    = round(eval_m["cvar_30"], 4)
            row["eval_sw_frac"]   = round(eval_m["frac_switched"], 4)
            row["eval_sw_step"]   = round(eval_m["mean_switch_step"], 1)

            # Checkpoint best
            eval_score = eval_m["success_rate"] - cost_weight * eval_m["mean_cost"]
            if checkpoint_dir and eval_score > best_eval_score:
                best_eval_score = eval_score
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"    * new best model -> {best_path}  "
                      f"(score={eval_score:.4f})")

            print(f"  [epoch {epoch:4d}/{n_epochs}]  "
                  f"loss={loss_val:.5f}  eps={epsilon:.3f}  "
                  f"buf={len(replay_buffer)}  "
                  f"EVAL succ={eval_m['success_rate']:.3f}  "
                  f"cost={eval_m['mean_cost']:.3f}  "
                  f"sw%={eval_m['frac_switched']:.2f}  "
                  f"sw_step={eval_m['mean_switch_step']:.0f}")
        else:
            if epoch % 10 == 0 or epoch <= 5:
                print(f"  [epoch {epoch:4d}/{n_epochs}]  "
                      f"loss={loss_val:.5f}  eps={epsilon:.3f}  "
                      f"trans={len(transitions)}  buf={len(replay_buffer)}")

        history.append(row)

        # Write CSV incrementally
        if checkpoint_dir:
            if csv_writer is None:
                all_fields = ["epoch", "loss", "epsilon",
                              "n_transitions", "n_wait", "n_switch",
                              "ep_pro_switch", "ep_pro_wait", "ep_tie",
                              "buffer_size",
                              "eval_succ", "eval_mean_cost",
                              "eval_cvar10", "eval_cvar20", "eval_cvar30",
                              "eval_sw_frac", "eval_sw_step"]
                csv_file = open(hist_path, "w", newline="")
                csv_writer = csv.DictWriter(
                    csv_file, fieldnames=all_fields, extrasaction="ignore"
                )
                csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="DQN training for per-step MLP switching policy "
                    "(7 geometric features -> Q(wait), Q(switch))."
    )
    # --- Environment ---
    p.add_argument("--cons_dir",      type=str, required=True)
    p.add_argument("--agg_dir",       type=str, required=True)
    p.add_argument("--budget_min",    type=int, default=120)
    p.add_argument("--budget_max",    type=int, default=220)
    p.add_argument("--budget_step",   type=int, default=5)
    p.add_argument("--meta_interval", type=int, default=1,
                   help="Low-level env steps per meta-step (default: 1).")
    p.add_argument("--max_horizon",   type=int, default=0,
                   help="Max env steps per episode (0 = budget_max).")

    # --- Q-Network architecture ---
    p.add_argument("--hidden_size", type=int, default=8,
                   help="Neurons in hidden layer. 8 -> 74 params, 16 -> 146 params.")

    # --- Episode pool ---
    p.add_argument("--episodes", type=int, default=30_000,
                   help="Total episodes in the training pool.")

    # --- Training ---
    p.add_argument("--n_epochs",    type=int,   default=50,
                   help="Number of DQN training epochs (1 episode + 1 update each).")
    p.add_argument("--batch_size",  type=int,   default=64,
                   help="Mini-batch size for DQN updates.")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Adam learning rate (default: 1e-3).")
    p.add_argument("--gamma",       type=float, default=1.0,
                   help="Discount factor (default: 1.0). Use 1.0 to avoid "
                        "asymmetric discounting bias between wait and switch. "
                        "Episodes are already bounded by the time budget.")
    p.add_argument("--epsilon_start", type=float, default=1.0,
                   help="Initial epsilon for exploration.")
    p.add_argument("--epsilon_end",   type=float, default=0.05,
                   help="Final epsilon after decay.")
    p.add_argument("--epsilon_decay_epochs", type=int, default=200,
                   help="Epochs over which epsilon decays linearly.")
    p.add_argument("--target_update_every", type=int, default=2,
                   help="Copy q_net -> target_net every N epochs.")
    p.add_argument("--buffer_capacity", type=int, default=100_000,
                   help="Replay buffer capacity.")
    p.add_argument("--switch_interval", type=int, default=5,
                   help="Oracle: evaluate switch every N conservative meta-steps "
                        "when computing wait bootstrap (default: 5 = every 5 steps).")
    p.add_argument("--episodes_per_epoch", type=int, default=5,
                   help="Number of episodes to collect per epoch (default: 5). "
                        "More episodes fill the buffer faster.")
    p.add_argument("--min_buffer_size", type=int, default=200,
                   help="Minimum transitions in replay buffer before first "
                        "DQN update (default: 0 = use batch_size as threshold).")
    p.add_argument("--warmup_epochs", type=int, default=1,
                   help="Force wait (no switch) for the first N epochs to "
                        "collect full-length trajectories (default: 0).")

    # --- PER & budget bias ---
    p.add_argument("--per_alpha", type=float, default=0.0,
                   help="PER priority exponent (0 = uniform, 0.6 = default PER).")
    p.add_argument("--per_beta_start", type=float, default=0.4,
                   help="PER importance-sampling beta start (default: 0.4).")
    p.add_argument("--per_beta_end", type=float, default=1.0,
                   help="PER importance-sampling beta end (1.0 = unbiased).")
    p.add_argument("--budget_bias", type=float, default=0,
                   help="Bias episode sampling toward tight budgets "
                        "(0 = uniform, 2.0 = moderate, 3.0 = strong).")
    p.add_argument("--min_adv_gap", type=float, default=0.0,
                   help="Skip transitions where |R_switch - R_wait| < this threshold "
                        "(0 = keep all, 0.05 = discard near-tie transitions).")
    p.add_argument("--collection_mode", type=str, default="best_only",
                   choices=["all", "best_only"],
                   help="'all': collect transitions at every step (old behaviour). "
                        "'best_only': run conservative once, scan every "
                        "scan_interval steps to find k*=argmax switch_return, "
                        "store ONE pair per episode (much faster).")
    p.add_argument("--scan_interval", type=int, default=5,
                   help="Zone width for the coarse probe in best_only mode "
                        "(default: 5 = zones of 5 meta-steps each).")
    p.add_argument("--n_top_zones", type=int, default=2,
                   help="Number of top-scoring zones to densely scan in "
                        "best_only mode (default: 2). Higher = more thorough "
                        "but more rollouts.")
    p.add_argument("--n_anchors", type=int, default=2,
                   choices=[1, 2, 3],
                   help="Number of anchor transitions per failing episode: "
                        "1=only k* (switch wins); "
                        "2=k*+k_low (adds early 'wait wins' anchor, +1 oracle); "
                        "3=k*+k_low+k_mid (adds boundary-region anchor, +2 oracle).")
    p.add_argument("--feature_noise", type=float, default=0.01,
                   help="Std of Gaussian noise added to features during training "
                        "forward pass only (not evaluation). "
                        "Smooths the decision boundary around seen states. "
                        "0 = disabled (default). Suggested: 0.01.")

    # --- Score function ---
    p.add_argument("--deadline_weight", type=float, default=1.0,
                   help="Terminal penalty for failure (default: 1.0).")
    p.add_argument("--cost_weight", type=float, default=0.02,
                   help="Per-step cost penalty weight (default: 0.02).")

    # --- Evaluation ---
    p.add_argument("--eval_every",    type=int, default=1,
                   help="Run deterministic eval every N epochs.")
    p.add_argument("--eval_episodes", type=int, default=200,
                   help="Episodes for periodic evaluation.")

    # --- Output ---
    p.add_argument("--base_seed",   type=int, default=2501)
    p.add_argument("--results_dir", type=str,
                   default="results/threshold/dqn_001")
    args = p.parse_args()

    max_horizon = args.max_horizon if args.max_horizon > 0 else args.budget_max
    os.makedirs(args.results_dir, exist_ok=True)

    # ----- Seeds & budgets -----
    rng         = np.random.RandomState(args.base_seed)
    all_seeds   = rng.randint(0, 2**31 - 1, size=args.episodes, dtype=np.int64)
    budget_rng  = np.random.RandomState(args.base_seed + 1)
    bvals       = list(range(args.budget_min, args.budget_max + 1, args.budget_step))
    all_budgets = budget_rng.choice(bvals, size=args.episodes, replace=True)

    eval_rng     = np.random.RandomState(args.base_seed + 2)
    eval_seeds   = eval_rng.randint(0, 2**31 - 1, size=args.eval_episodes,
                                    dtype=np.int64)
    eval_bud_rng = np.random.RandomState(args.base_seed + 3)
    eval_budgets = eval_bud_rng.choice(bvals, size=args.eval_episodes, replace=True)

    # ----- Low-level policies -----
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
    print(f"  obs_dim={obs_dim},  features=7 (geometric)\n")

    # ----- Build networks -----
    torch.manual_seed(args.base_seed)
    q_net = QNet(hidden_size=args.hidden_size)
    target_net = QNet(hidden_size=args.hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    n_params = sum(p_.numel() for p_ in q_net.parameters())
    print(q_net)
    print(f"  Trainable parameters: {n_params}\n")

    # ----- Save config -----
    config = {
        "method":          "DQN",
        "hidden_size":     args.hidden_size,
        "n_features":      N_FEATURES,
        "n_params":        n_params,
        "budget_min":      args.budget_min,
        "budget_max":      args.budget_max,
        "deadline_weight": args.deadline_weight,
        "cost_weight":     args.cost_weight,
        "gamma":           args.gamma,
        "n_epochs":        args.n_epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "epsilon_start":   args.epsilon_start,
        "epsilon_end":     args.epsilon_end,
        "epsilon_decay_epochs": args.epsilon_decay_epochs,
        "target_update_every":  args.target_update_every,
        "buffer_capacity":      args.buffer_capacity,
        "switch_interval":      args.switch_interval,
        "episodes_per_epoch":   args.episodes_per_epoch,
        "min_buffer_size":      args.min_buffer_size,
        "warmup_epochs":        args.warmup_epochs,
        "per_alpha":            args.per_alpha,
        "per_beta_start":       args.per_beta_start,
        "per_beta_end":         args.per_beta_end,
        "budget_bias":          args.budget_bias,
        "min_adv_gap":          args.min_adv_gap,
        "collection_mode":      args.collection_mode,
        "scan_interval":        args.scan_interval,
        "n_top_zones":          args.n_top_zones,
        "n_anchors":            args.n_anchors,
        "meta_interval":   args.meta_interval,
        "episodes":        args.episodes,
        "eval_every":      args.eval_every,
        "eval_episodes":   args.eval_episodes,
        "base_seed":       args.base_seed,
    }
    config_path = os.path.join(args.results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved -> {config_path}")

    # ----- Train -----
    history = dqn_train(
        q_net=q_net,
        target_net=target_net,
        env=env,
        all_seeds=all_seeds,
        all_budgets=all_budgets,
        max_horizon=max_horizon,
        cost_weight=args.cost_weight,
        deadline_weight=args.deadline_weight,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_epochs=args.epsilon_decay_epochs,
        target_update_every=args.target_update_every,
        buffer_capacity=args.buffer_capacity,
        rng_seed=args.base_seed + 7,
        eval_every=args.eval_every,
        eval_seeds=eval_seeds,
        eval_budgets=eval_budgets,
        checkpoint_dir=args.results_dir,
        switch_interval=args.switch_interval,
        episodes_per_epoch=args.episodes_per_epoch,
        min_buffer_size=args.min_buffer_size,
        warmup_epochs=args.warmup_epochs,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        budget_bias=args.budget_bias,
        min_adv_gap=args.min_adv_gap,
        collection_mode=args.collection_mode,
        scan_interval=args.scan_interval,
        n_top_zones=args.n_top_zones,
        n_anchors=args.n_anchors,
        feature_noise=args.feature_noise,
    )

    # ----- Save final model -----
    weights_path = os.path.join(args.results_dir, "final_model.pt")
    torch.save(q_net.state_dict(), weights_path)
    print(f"\nFinal model saved -> {weights_path}")

    # ----- Final deterministic evaluation -----
    print("\n=== Final deterministic evaluation ===")
    eval_m = evaluate_policy(q_net, env, eval_seeds, eval_budgets, max_horizon)
    print(f"  success_rate     : {eval_m['success_rate']:.4f}")
    print(f"  mean_cost        : {eval_m['mean_cost']:.4f}")
    print(f"  CVaR(10%)        : {eval_m['cvar_10']:.4f}")
    print(f"  CVaR(20%)        : {eval_m['cvar_20']:.4f}")
    print(f"  CVaR(30%)        : {eval_m['cvar_30']:.4f}")
    print(f"  frac_switched    : {eval_m['frac_switched']:.4f}")
    print(f"  mean_switch_step : {eval_m['mean_switch_step']:.1f}")

    eval_path = os.path.join(args.results_dir, "eval_results.txt")
    with open(eval_path, "w") as f:
        for k, v in eval_m.items():
            f.write(f"{k}={v:.6f}\n")
        f.write(f"hidden_size={args.hidden_size}\n")
        f.write(f"n_params={n_params}\n")
    print(f"\nEval results saved -> {eval_path}")

    print(f"\n{'='*60}")
    print(f"  Q-LEARNING SWITCH POLICY  (DQN, {n_params} params, H={args.hidden_size})")
    print(f"    features: [v_x, v_y, d_goal, d_haz, dtheta, t_frac, budget_norm]")
    print(f"    succ={eval_m['success_rate']:.3f}  "
          f"cost={eval_m['mean_cost']:.3f}  "
          f"cvar10={eval_m['cvar_10']:.3f}")
    print(f"    sw%={eval_m['frac_switched']:.2f}  "
          f"sw_step={eval_m['mean_switch_step']:.0f}")
    print(f"{'='*60}\n")

    env.close()
    sess_cons.close()
    sess_agg.close()


if __name__ == "__main__":
    main()
