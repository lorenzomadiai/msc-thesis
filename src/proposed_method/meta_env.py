"""
MetaEnv
-------
A gym-compatible wrapper that turns the Safety-Gym Engine into a meta-level
environment for the switching policy.

At each *meta-step* the policy chooses between two pre-trained, frozen
low-level policies (0 = conservative, 1 = aggressive) and that policy is
executed for `meta_interval` consecutive environment steps (or fewer if the
episode ends earlier).

Observation returned to the meta-policy (lidar + velocimeter + time features):
    [goal_lidar (16), hazards_lidar (16), velocimeter (3), time_left_norm, budget_norm]

    goal_lidar / hazards_lidar  spatial proximity signals for navigation/safety
    velocimeter (3)             body-frame linear velocity — signals how fast
                                the agent is moving, useful to estimate whether
                                the remaining budget will be enough
    time_left_norm  ∈ [-1, 1]   2*remaining/B - 1
    budget_norm     ∈ [ 0, 1]   B / max_budget

Irreversible-switch mode (irreversible_switch=True):
    The meta-policy can only switch ONCE per episode from conservative to
    aggressive.  Action 0 = "stay conservative" (no-op if already switched),
    action 1 = "switch to aggressive NOW — permanently".
    This turns the problem into an optimal stopping problem: learn the latest
    safe moment to commit to the aggressive policy so the goal is reached
    within budget while keeping cumulative cost low.

NOTE on observations:
  - Low-level act_fns receive the FULL augmented obs [raw_obs | time_left_norm | budget_norm]
    because they need all proprioceptive sensors (accelerometer, velocimeter, gyro,
    magnetometer) for locomotion control.
  - The meta-policy receives [lidar_obs | velocimeter (3) | time_left_norm | budget_norm]. The velocimeter is included because
    knowing the agent's speed helps estimate whether the remaining budget suffices
    with the conservative policy; gyro, accelerometer and magnetometer are excluded
    as they are irrelevant to the switching decision.

Meta-reward (simple three-term design):
    r_meta = - cost_weight * Σ cost_env     dense: penalise hazard violations
             + goal_reward                  sparse: bonus when goal is reached
             - deadline_penalty             sparse: penalty when budget expires
                                                    without reaching the goal

"""

import numpy as np
import gym


class MetaEnv(gym.Env):
    """
    Parameters
    ----------
    env_fn : callable
        Zero-argument factory that returns a fresh Safety-Gym Engine instance.
    act_fn_cons : callable  (obs_batch [1, D]) -> action_batch [1, A]
        Deterministic conservative policy.
    act_fn_agg : callable   (obs_batch [1, D]) -> action_batch [1, A]
        Deterministic aggressive policy.
    meta_interval : int
        Number of env steps per meta-step (n).
    budget_min, budget_max : int
        Range for the time budget sampled at each reset.
    budget_step : int
        Granularity of budget sampling (same as TimeBudgetWrapper default=5).
    cost_weight : float
        Weight on accumulated hazard cost subtracted every meta-step.
        Keep small (e.g. 0.1) so cost is secondary to goal completion.
    goal_reward : float
        Bonus when goal is reached (e.g. 1.0).
    deadline_penalty : float
        Penalty when the budget expires without reaching the goal.
        Should be > goal_reward so missing the deadline is always bad
        (e.g. 5.0).
    eval_mode : bool
        If True, deterministically use the max budget instead of sampling.
    seed : int or None
        RNG seed for budget sampling.
    render : bool
        Whether to call env.render() at each env step.
    """

    def __init__(
        self,
        env_fn,
        act_fn_cons,
        act_fn_agg,
        meta_interval: int = 1,
        budget_min: int = 120,
        budget_max: int = 220,
        budget_step: int = 5,
        cost_weight: float = 0.1,
        goal_reward: float = 1.0,
        deadline_penalty: float = 5.0,
        irreversible_switch: bool = False,
        eval_mode: bool = False,
        seed: int = None,
        render: bool = False,
    ):
        super().__init__()

        self._env_fn = env_fn
        self.act_fn = [act_fn_cons, act_fn_agg]
        self.meta_interval = int(meta_interval)
        self.budget_min = int(budget_min)
        self.budget_max = int(budget_max)
        self.budget_step = int(budget_step)
        self.cost_weight = float(cost_weight)
        self.goal_reward = float(goal_reward)
        self.deadline_penalty = float(deadline_penalty)
        self.irreversible_switch = bool(irreversible_switch)
        self.eval_mode = eval_mode
        self._render = render
        self._rng = np.random.RandomState(seed)

        # Build the underlying env once to introspect its observation space
        self._env = env_fn()
        base_space = self._env.observation_space
        assert isinstance(base_space, gym.spaces.Box) and len(base_space.shape) == 1, (
            "MetaEnv expects a 1-D Box observation space from the base env."
        )
        self._base_obs_dim = base_space.shape[0]

        # Compute which base-obs indices to expose to the meta-policy.
        # We keep lidar observations (goal_lidar, hazards_lidar, …) and the
        # velocimeter (3-D body-frame linear velocity), dropping other
        # proprioceptive sensors (accelerometer, gyro, magnetometer).
        # Low-level policies still receive the full augmented obs.
        self._meta_obs_indices = self._compute_meta_obs_indices()

        # Meta observation = lidar + velocimeter + time_left_norm + budget_norm [+ is_aggressive]
        # is_aggressive is appended ONLY when irreversible_switch=True.
        extra_low  = [-1.0, 0.0, 0.0] if self.irreversible_switch else [-1.0, 0.0]
        extra_high = [ 1.0, 1.0, 1.0] if self.irreversible_switch else [ 1.0, 1.0]
        low  = np.concatenate([base_space.low[self._meta_obs_indices],  extra_low ]).astype(np.float32)
        high = np.concatenate([base_space.high[self._meta_obs_indices], extra_high]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Binary action: 0=conservative (or "keep current"), 1=aggressive (or "switch now")
        self.action_space = gym.spaces.Discrete(2)

        # Episode state (initialised at reset)
        self._raw_obs = None
        self.B = None       # current time budget
        self.t = 0          # current env step within the episode
        self._goal_met = False
        self._switched = False  # True once irreversible switch fires

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_left_norm(self) -> float:
        rem = max(self.B - self.t, 0)
        return 2.0 * (rem / float(self.B)) - 1.0

    def _budget_norm(self) -> float:
        return float(self.B) / float(self.budget_max)

    def _compute_meta_obs_indices(self) -> np.ndarray:
        """
        Return the indices of raw_obs to expose to the meta-policy.
        Keeps lidar observations (goal_lidar, hazards_lidar, …) and the
        velocimeter (body-frame linear velocity, 3-D), which gives the
        meta-policy a sense of how fast the agent is moving without
        exposing low-level locomotion sensors (accelerometer, gyro,
        magnetometer).
        Falls back to skipping the first 12 proprioceptive values if
        obs_space_dict is not available.
        """
        try:
            obs_dict = self._env.obs_space_dict
        except AttributeError:
            # Safety-Gym Point robot: first 12 are proprioceptive sensors
            return np.arange(12, self._base_obs_dim, dtype=np.int64)

        indices = []
        offset = 0
        for key, space in obs_dict.items():
            size = int(np.prod(space.shape))
            if "lidar" in key.lower() or key == "velocimeter":
                indices.extend(range(offset, offset + size))
            offset += size

        if not indices:
            # No matching keys found — fall back to all indices
            return np.arange(self._base_obs_dim, dtype=np.int64)

        return np.array(indices, dtype=np.int64)

    def _augment(self, raw_obs: np.ndarray) -> np.ndarray:
        """Full augmented obs (raw + time features) — used by low-level policies."""
        return np.concatenate([
            raw_obs,
            np.array([self._time_left_norm(), self._budget_norm()], dtype=np.float32),
        ]).astype(np.float32)

    def _augment_meta(self, raw_obs: np.ndarray) -> np.ndarray:
        """Filtered obs for the meta-policy (lidar + velocimeter + time features [+ is_aggressive])."""
        features = [self._time_left_norm(), self._budget_norm()]
        if self.irreversible_switch:
            features.append(1.0 if self._switched else 0.0)
        return np.concatenate([
            raw_obs[self._meta_obs_indices],
            np.array(features, dtype=np.float32),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        # Reset the underlying env (seed-agnostic: Safety-Gym resets randomly)
        self._raw_obs = np.array(self._env.reset(), dtype=np.float32)

        # Sample a fresh time budget
        budgets = np.arange(self.budget_min, self.budget_max + 1, self.budget_step)
        if self.eval_mode:
            self.B = self.budget_max
        else:
            self.B = int(self._rng.choice(budgets))

        self.t = 0
        self._goal_met = False
        self._switched = False
        return self._augment_meta(self._raw_obs)

    def step(self, action: int):
        """
        Execute the chosen low-level policy for `meta_interval` env steps (or
        until the episode terminates).

        Returns
        -------
        obs      : augmented observation at the END of the meta-step
        r_meta   : scalar meta-reward
        done     : whether the episode has ended
        info     : dict with cumulative_cost, goal_met, time_budget, time_step,
                   active_policy, n_steps_taken
        """
        assert action in (0, 1), f"Action must be 0 or 1, got {action}"

        # Irreversible-switch mode: once switched to aggressive, ignore future actions.
        # Action 1 fires the switch; subsequent steps are always aggressive.
        if self.irreversible_switch:
            if action == 1:
                self._switched = True
            effective_action = 1 if self._switched else 0
        else:
            effective_action = action

        act_fn = self.act_fn[effective_action]

        r_meta = 0.0
        cum_cost = 0.0
        done = False
        goal_met_this_step = False
        budget_expired = False
        n_steps_taken = 0

        for _ in range(self.meta_interval):
            # Low-level policy receives the full augmented obs (raw + time features)
            a_ll = act_fn(self._augment(self._raw_obs).reshape(1, -1))[0]

            raw_next, r_env, env_done, info = self._env.step(a_ll)
            self._raw_obs = np.array(raw_next, dtype=np.float32)
            self.t += 1
            n_steps_taken += 1

            # Raw env reward intentionally ignored at the meta level
            cum_cost += float(info.get("cost", 0.0))

            if bool(info.get("goal_met", False)) and not self._goal_met:
                self._goal_met = True
                goal_met_this_step = True

            # Check deadline
            if self.t >= self.B:
                budget_expired = True

            if env_done or budget_expired or goal_met_this_step:
                done = True
                break

        # Cost shaping: subtract weighted cost from meta-reward
        r_meta -= self.cost_weight * cum_cost

        # Goal bonus
        if goal_met_this_step:
            r_meta += self.goal_reward

        # Deadline penalty: only if budget expired without goal
        if budget_expired and not self._goal_met:
            r_meta -= self.deadline_penalty

        return (
            self._augment_meta(self._raw_obs),
            r_meta,
            done,
            {
                "cumulative_cost": cum_cost,
                "goal_met": self._goal_met,
                "goal_met_this_step": goal_met_this_step,
                "budget_expired": budget_expired,
                "time_budget": self.B,
                "time_step": self.t,
                "active_policy": "conservative" if effective_action == 0 else "aggressive",
                "switched": self._switched,
                "n_steps_taken": n_steps_taken,
            },
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
