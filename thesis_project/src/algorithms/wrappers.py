import gym
import numpy as np

class TimeBudgetWrapper(gym.Wrapper):
    """
    - Campiona un time budget B a ogni reset
    - Aggiunge:
        * time_left normalizzato in [-1,1]
        * budget_norm = B / max_budget in [0,1]
      come ultimi 2 elementi dell'osservazione (in quest'ordine)
    - Termina l'episodio quando t >= B
    - Applica una penalty SOLO se scade il budget e il goal non è stato raggiunto
    """
    def __init__(self, env, budget_min, budget_max, deadline_penalty=0.0, eval_mode=False, eval_max_budget=None):
        super().__init__(env)

        self.deadline_penalty = float(deadline_penalty)
        self.eval_mode = eval_mode
        self.t = 0
        self.B = None
        self.min_budget = int(budget_min)
        self.max_budget = int(budget_max)
        # self.budget_step = 10
        self.budget_step = 5


        if eval_mode:
            self.eval_max_budget = int(eval_max_budget)

        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        assert len(old_space.shape) == 1

        low = old_space.low
        high = old_space.high

        low = np.concatenate([low,  np.array([-1.0], dtype=old_space.dtype)])
        high = np.concatenate([high, np.array([ 1.0], dtype=old_space.dtype)])

        low = np.concatenate([low,  np.array([0.0], dtype=old_space.dtype)])
        high = np.concatenate([high, np.array([1.0], dtype=old_space.dtype)])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=old_space.dtype)

    def _time_left_norm(self):
        rem = max(self.B - self.t, 0)
        frac = rem / float(self.B)      # [0,1]
        return 2.0 * frac - 1.0         # [-1,1]

    def _budget_norm(self):
        if self.eval_mode:
            return float(self.B) / float(self.eval_max_budget)  # [0,1]
        
        return float(self.B) / float(self.max_budget)  # [0,1]

    def _augment_obs(self, obs):
        parts = [obs]
        parts.append(np.array([self._time_left_norm()], dtype=obs.dtype))
        parts.append(np.array([self._budget_norm()], dtype=obs.dtype))
        return np.concatenate(parts, axis=0)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.t = 0
        self.B = np.random.choice(np.arange(self.min_budget, self.max_budget + 1, self.budget_step))
        # print(f"New episode with time budget B={self.B}")
        return self._augment_obs(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.t += 1
        info = dict(info)  # copia sicura
        # if not self.eval_mode:
        #     rem = max(self.B - self.t, 0)
        #     frac = rem / float(self.B)      # [0,1]
        #     # w = 1.0 + frac   # 2 all’inizio, 1 alla fine
        #     w = frac    # 1 all’inizio, 0 alla fine
        #     info["cost"] = info.get("cost", 0.0) * w
        
        # termina alla deadline se non è già done
        if (not done) and (self.t >= self.B):
            done = True
            goal_met = bool(info.get("goal_met", False))
            time_budget_fail = not goal_met
            info["time_budget_fail"] = time_budget_fail
            # penalty SOLO se fallisci
            if time_budget_fail and not self.eval_mode:
                r -= self.deadline_penalty


        info["time_budget"] = self.B
        info["time_step"] = self.t

        return self._augment_obs(obs), r, done, info
