"""Evaluation routines for learned and fixed switch policies."""

from collections import deque

import numpy as np
import torch

from common.features import extract_features


def evaluate_gap_policy(
    model,
    env,
    seeds: np.ndarray,
    budgets: np.ndarray,
    max_horizon: int,
    switch_prob_threshold: float = 0.5,
    feature_history: int = 0,
):
    """Evaluate learned switch policy with probability threshold rule."""
    model.eval()
    successes, costs, switch_steps = [], [], []
    p_thr = float(np.clip(switch_prob_threshold, 1e-6, 1.0 - 1e-6))
    hist_len = max(0, int(feature_history))

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
        switched = False
        switch_step = -1
        cum_cost = 0.0
        info = {}
        hist_buffer = deque(maxlen=hist_len) if hist_len > 0 else None
        base_dim = None

        while not done and ep_len < max_horizon:
            if not switched:
                feats = extract_features(obs, env)
                feats = feats.astype(np.float32)
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
                    logit = float(model(x).item())
                    p_switch = 1.0 / (1.0 + np.exp(-logit))
                action = 1 if p_switch > p_thr else 0
                if action == 1:
                    switched = True
                    switch_step = ep_len + 1
                if hist_len > 0:
                    hist_buffer.append(feats)
            else:
                action = 1

            obs, _r, done, info = env.step(action)
            cum_cost += float(info.get("cumulative_cost", 0.0))
            ep_len += info.get("n_steps_taken", 1)

        successes.append(float(bool(info.get("goal_met", False))))
        costs.append(cum_cost)
        switch_steps.append(switch_step)

    costs_arr = np.array(costs, dtype=float)

    def _cvar(alpha):
        k = max(1, int(np.ceil(alpha * len(costs_arr))))
        return float(np.mean(np.sort(costs_arr)[-k:]))

    frac_switched = float(np.mean([s > 0 for s in switch_steps]))
    mean_sw_step = float(
        np.mean([s for s in switch_steps if s > 0]) if any(s > 0 for s in switch_steps) else 0.0
    )

    return {
        "success_rate": float(np.mean(successes)),
        "mean_cost": float(np.mean(costs_arr)),
        "cvar_10": _cvar(0.10),
        "cvar_20": _cvar(0.20),
        "cvar_30": _cvar(0.30),
        "frac_switched": frac_switched,
        "mean_switch_step": mean_sw_step,
    }


def evaluate_fixed_policy(env, seeds: np.ndarray, budgets: np.ndarray, max_horizon: int, action_mode: str):
    """Evaluate fixed baselines: conservative-only or aggressive-only."""
    successes, costs, switch_steps = [], [], []

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
        info = {}

        while not done and ep_len < max_horizon:
            if action_mode == "conservative":
                action = 0
            elif action_mode == "aggressive":
                action = 1
                if switch_step < 0:
                    switch_step = ep_len + 1
            else:
                raise ValueError(f"Unknown action_mode: {action_mode}")

            obs, _r, done, info = env.step(action)
            cum_cost += float(info.get("cumulative_cost", 0.0))
            ep_len += info.get("n_steps_taken", 1)

        successes.append(float(bool(info.get("goal_met", False))))
        costs.append(cum_cost)
        switch_steps.append(switch_step)

    costs_arr = np.array(costs, dtype=float)

    def _cvar(alpha):
        k = max(1, int(np.ceil(alpha * len(costs_arr))))
        return float(np.mean(np.sort(costs_arr)[-k:]))

    frac_switched = float(np.mean([s > 0 for s in switch_steps]))
    mean_sw_step = float(
        np.mean([s for s in switch_steps if s > 0]) if any(s > 0 for s in switch_steps) else 0.0
    )

    return {
        "success_rate": float(np.mean(successes)),
        "mean_cost": float(np.mean(costs_arr)),
        "cvar_10": _cvar(0.10),
        "cvar_20": _cvar(0.20),
        "cvar_30": _cvar(0.30),
        "frac_switched": frac_switched,
        "mean_switch_step": mean_sw_step,
    }
