"""Sampling and labeling logic for supervised switch dataset collection.

Contains conservative rollouts, k selection strategies, per-episode sampling,
and full dataset assembly with optional logging.
"""

import csv
import json
from typing import List

import numpy as np

from common.features import extract_features
from common.mujoco_state import save_mujoco_state, restore_mujoco_state
from common.oracle import (
    counterfactual_switch_return,
    oracle_conservative_value,
    _find_best_k_zone_search,
)


def _stack_feature_history(steps: List[dict], idx: int, history: int) -> np.ndarray:
    """Concatenate current feature vector with up to `history` previous ones."""
    base_feats = steps[idx]["feats"].astype(np.float32)
    if history <= 0:
        return base_feats.copy()

    base_dim = base_feats.shape[0]
    zeros = np.zeros(base_dim, dtype=np.float32)
    segments = []
    for offset in range(history, 0, -1):
        past_idx = idx - offset
        if past_idx >= 0:
            segments.append(steps[past_idx]["feats"].astype(np.float32))
        else:
            segments.append(zeros)
    segments.append(base_feats)
    return np.concatenate(segments, axis=0).astype(np.float32, copy=False)


def _rollout_conservative_steps(
    env,
    seed: int,
    budget: int,
    max_horizon: int,
    cost_weight: float,
    deadline_weight: float,
):
    """Run one conservative episode and save per-step state/metadata."""
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

    steps = []
    done = False
    ep_len = 0
    goal_met_final = False

    while not done and ep_len < max_horizon:
        feats = extract_features(obs, env)
        state_t = save_mujoco_state(env)
        steps.append({
            "feats": feats.astype(np.float32),
            "state": state_t,
        })

        obs_next, _r, done, info = env.step(0)
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

        steps[-1]["r_step"] = r_step
        steps[-1]["done_next"] = done
        obs = obs_next.copy()

    return steps, goal_met_final


def _sample_k_indices(
    steps: list,
    rng: np.random.RandomState,
    sampling_mode: str,
    samples_per_episode: int,
    uniform_frac: float,
    focus_window: int,
    forced_ks: list,
    k_max_allowed: int,
    best_k_override: int,
    env,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    max_horizon: int,
    scan_interval: int,
    n_top_zones: int,
    force_k0: bool = True,
):
    """Choose timestep indices k according to the selected sampling strategy."""
    n_steps = len(steps)
    if n_steps == 0:
        return [], None
    if k_max_allowed is None:
        k_max_allowed = n_steps - 1
    k_max_allowed = max(0, min(k_max_allowed, n_steps - 1))
    candidate_ks = list(range(k_max_allowed + 1))
    forced_set = set(int(k) for k in forced_ks if 0 <= int(k) < n_steps)

    if sampling_mode == "all":
        return sorted(forced_set.union(candidate_ks)), None

    target_n = max(1, min(samples_per_episode, len(candidate_ks)))
    sampled = set()
    if force_k0 and 0 <= k_max_allowed:
        sampled.add(0)
    sampled.update(forced_set)
    best_k = None if best_k_override is None else int(best_k_override)

    if sampling_mode == "hybrid" and best_k is None:
        best_k = _find_best_k_zone_search(
            steps,
            env,
            cost_weight,
            deadline_weight,
            gamma,
            max_horizon,
            scan_interval=scan_interval,
            n_top_zones=n_top_zones,
        )
        if int(best_k) <= k_max_allowed:
            sampled.add(int(best_k))

    n_uniform = target_n
    n_focus = 0
    if sampling_mode == "hybrid":
        n_uniform = int(round(target_n * uniform_frac))
        n_uniform = max(1, min(target_n, n_uniform))
        n_focus = target_n - n_uniform

    available = [k for k in candidate_ks if k not in sampled]
    need_uniform = max(0, min(n_uniform - len(sampled), len(available)))
    if need_uniform > 0:
        picks = rng.choice(available, size=need_uniform, replace=False)
        sampled.update(int(k) for k in picks)

    if sampling_mode == "hybrid" and n_focus > 0 and best_k is not None:
        low = max(0, int(best_k) - max(0, focus_window))
        high = min(k_max_allowed, int(best_k) + max(0, focus_window))
        focus_pool = [k for k in range(low, high + 1) if k not in sampled]
        need_focus = max(0, min(n_focus, target_n - len(sampled), len(focus_pool)))
        if need_focus > 0:
            picks = rng.choice(focus_pool, size=need_focus, replace=False)
            sampled.update(int(k) for k in picks)

    if len(sampled) < target_n:
        remaining = [k for k in candidate_ks if k not in sampled]
        need = min(target_n - len(sampled), len(remaining))
        if need > 0:
            picks = rng.choice(remaining, size=need, replace=False)
            sampled.update(int(k) for k in picks)

    return sorted(sampled), best_k


def collect_gap_episode(
    env,
    seed: int,
    budget: int,
    max_horizon: int,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    switch_interval: int,
    rng: np.random.RandomState,
    cons_success_hint: int = None,
    best_k_hint: int = None,
    sampling_mode: str = "hybrid",
    samples_per_episode: int = 6,
    uniform_frac: float = 0.5,
    uniform_frac_failed: float = 0.0,
    focus_window: int = 10,
    force_pos_prev_k: int = 3,
    scan_interval: int = 5,
    n_top_zones: int = 2,
    min_abs_gap: float = 0.0,
    feature_history: int = 0,
):
    """Collect datapoints with per-episode sampling policy."""
    steps, cons_success_rollout = _rollout_conservative_steps(
        env=env,
        seed=seed,
        budget=budget,
        max_horizon=max_horizon,
        cost_weight=cost_weight,
        deadline_weight=deadline_weight,
    )
    cons_success = bool(cons_success_rollout) if cons_success_hint is None else bool(cons_success_hint)
    n_steps = len(steps)
    if n_steps == 0:
        return [], {
            "cons_success": False,
            "best_k": -1,
            "sampling_mode_used": "uniform",
            "sampled_ks": [],
            "forced_ks": [],
            "forced_success_ks": [],
            "n_steps": 0,
        }

    def _switch_return_at_local(k: int) -> float:
        if "switch_return" not in steps[k]:
            steps[k]["switch_return"] = counterfactual_switch_return(
                env,
                steps[k]["state"],
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
            )
        return float(steps[k]["switch_return"])

    def _wait_return_at_local(k: int) -> float:
        if "wait_return" in steps[k]:
            return float(steps[k]["wait_return"])
        r_step = steps[k]["r_step"]
        if steps[k]["done_next"] or k + 1 >= n_steps:
            wr = r_step
        else:
            oracle_v = oracle_conservative_value(
                env,
                steps[k + 1]["state"],
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
                switch_interval=switch_interval,
            )
            wr = r_step + gamma * oracle_v
        steps[k]["wait_return"] = wr
        return float(wr)

    def _delta_at_local(k: int) -> float:
        return _switch_return_at_local(k) - _wait_return_at_local(k)

    best_k = None
    best_k_delta = None
    forced_positive_ks = []
    k_max_allowed = n_steps - 1
    episode_sampling_mode = "uniform" if cons_success else "hybrid"

    if not cons_success:
        if best_k_hint is not None and int(best_k_hint) >= 0:
            best_k = int(np.clip(int(best_k_hint), 0, n_steps - 1))
        else:
            best_k = _find_best_k_zone_search(
                steps,
                env,
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
                scan_interval=scan_interval,
                n_top_zones=n_top_zones,
            )

        best_k_delta = _delta_at_local(int(best_k))
        if best_k_delta <= 0.0:
            best_k_all = int(best_k)
            best_delta_all = float(best_k_delta)
            for kk in range(n_steps):
                dkk = _delta_at_local(kk)
                if dkk > best_delta_all or (dkk == best_delta_all and kk > best_k_all):
                    best_delta_all = float(dkk)
                    best_k_all = int(kk)
            best_k = best_k_all
            best_k_delta = best_delta_all

        k_max_allowed = int(best_k)

        if best_k_delta > 0.0:
            forced_positive_ks = [
                k for k in [best_k - d for d in range(0, max(0, force_pos_prev_k) + 1)] if k >= 0
            ]
            if forced_positive_ks:
                k_max_allowed = int(min(forced_positive_ks))

    effective_uniform_frac = uniform_frac if cons_success else uniform_frac_failed

    sampled_ks, _ = _sample_k_indices(
        steps=steps,
        rng=rng,
        sampling_mode=episode_sampling_mode,
        samples_per_episode=samples_per_episode,
        uniform_frac=effective_uniform_frac,
        focus_window=focus_window,
        forced_ks=forced_positive_ks,
        k_max_allowed=k_max_allowed,
        best_k_override=best_k,
        env=env,
        cost_weight=cost_weight,
        deadline_weight=deadline_weight,
        gamma=gamma,
        max_horizon=max_horizon,
        scan_interval=scan_interval,
        n_top_zones=n_top_zones,
        force_k0=bool(cons_success),
    )

    def _switch_return_at(k: int) -> float:
        if "switch_return" not in steps[k]:
            steps[k]["switch_return"] = counterfactual_switch_return(
                env,
                steps[k]["state"],
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
            )
        return steps[k]["switch_return"]

    def _wait_return_at(k: int) -> float:
        if "wait_return" in steps[k]:
            return steps[k]["wait_return"]
        r_step = steps[k]["r_step"]
        if steps[k]["done_next"] or k + 1 >= n_steps:
            wr = r_step
        else:
            oracle_v = oracle_conservative_value(
                env,
                steps[k + 1]["state"],
                cost_weight,
                deadline_weight,
                gamma,
                max_horizon,
                switch_interval=switch_interval,
            )
            wr = r_step + gamma * oracle_v
        steps[k]["wait_return"] = wr
        return wr

    def _switch_success_at(k: int) -> bool:
        if "switch_success" in steps[k]:
            return bool(steps[k]["switch_success"])

        current_state = save_mujoco_state(env)
        try:
            restore_state = steps[k]["state"]
            restore_mujoco_state(env, restore_state)
            env._switched = True

            done_local = False
            ep_len_local = env.t
            goal_met_local = False
            while not done_local and ep_len_local < max_horizon:
                _obs, _r, done_local, info_local = env.step(1)
                ep_len_local += info_local.get("n_steps_taken", 1)
                if bool(info_local.get("goal_met", False)):
                    goal_met_local = True
                    done_local = True
                elif bool(info_local.get("budget_expired", False)):
                    done_local = True
            steps[k]["switch_success"] = bool(goal_met_local)
        finally:
            restore_mujoco_state(env, current_state)

        return bool(steps[k]["switch_success"])

    forced_success_ks = []
    if not cons_success:
        for kf in forced_positive_ks:
            if _switch_success_at(kf):
                forced_success_ks.append(int(kf))

    samples = []
    datapoint_info = []
    forced_positive_set = set(forced_positive_ks)
    forced_success_set = set(forced_success_ks)
    for k in sampled_ks:
        switch_ret = _switch_return_at(k)
        wait_ret = _wait_return_at(k)
        delta = switch_ret - wait_ret
        keep = abs(delta) >= min_abs_gap
        label = int(delta > 0.0)

        if (not cons_success) and (k in forced_positive_set) and (k in forced_success_set):
            keep = True
            label = 1
            delta = max(float(delta), 1e-3)

        if keep:
            samples.append(
                {
                    "x": _stack_feature_history(steps, k, feature_history),
                    "delta": float(delta),
                    "label": int(label),
                    "k": int(k),
                    "switch_ret": float(switch_ret),
                    "wait_ret": float(wait_ret),
                    "forced_positive": int(k in forced_positive_set),
                    "forced_success": int(k in forced_success_set),
                }
            )
            datapoint_info.append(
                {
                    "k": int(k),
                    "delta": float(delta),
                    "label": int(label),
                    "switch_ret": float(switch_ret),
                    "wait_ret": float(wait_ret),
                    "forced_positive": int(k in forced_positive_set),
                    "forced_success": int(k in forced_success_set),
                }
            )

    episode_info = {
        "cons_success": bool(cons_success),
        "best_k": int(best_k) if best_k is not None else -1,
        "sampling_mode_used": episode_sampling_mode,
        "sampled_ks": [int(k) for k in sampled_ks],
        "forced_ks": [int(k) for k in forced_positive_ks],
        "forced_success_ks": [int(k) for k in forced_success_ks],
        "n_steps": int(n_steps),
        "datapoints": datapoint_info,
    }
    return samples, episode_info


def collect_dataset(
    env,
    seeds: np.ndarray,
    budgets: np.ndarray,
    max_horizon: int,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    switch_interval: int,
    rng: np.random.RandomState,
    cons_outcomes: np.ndarray = None,
    best_k_hints: np.ndarray = None,
    sampling_mode: str = "hybrid",
    samples_per_episode: int = 6,
    uniform_frac: float = 0.5,
    uniform_frac_failed: float = 0.0,
    focus_window: int = 10,
    force_pos_prev_k: int = 3,
    scan_interval: int = 5,
    n_top_zones: int = 2,
    min_abs_gap: float = 0.0,
    sampling_log_path: str = None,
    print_sampling: bool = True,
    print_datapoints: bool = False,
    feature_history: int = 0,
):
    """Collect a full supervised dataset from many episodes."""
    all_x, all_y, all_lbl = [], [], []

    log_file = None
    log_writer = None
    if sampling_log_path:
        log_file = open(sampling_log_path, "w", newline="")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "episode",
                "seed",
                "budget",
                "cons_success",
                "sampling_mode_used",
                "n_steps",
                "best_k",
                "sampled_ks",
                "forced_ks",
                "forced_success_ks",
                "n_samples_kept",
            ],
        )
        log_writer.writeheader()

    for i, (seed, budget) in enumerate(zip(seeds, budgets), start=1):
        cons_success_hint = None
        if cons_outcomes is not None:
            cons_success_hint = int(cons_outcomes[i - 1])
        best_k_hint = None
        if best_k_hints is not None:
            bkh = int(best_k_hints[i - 1])
            best_k_hint = bkh if bkh >= 0 else None

        ep_samples, ep_info = collect_gap_episode(
            env=env,
            seed=int(seed),
            budget=int(budget),
            max_horizon=max_horizon,
            cost_weight=cost_weight,
            deadline_weight=deadline_weight,
            gamma=gamma,
            switch_interval=switch_interval,
            rng=rng,
            cons_success_hint=cons_success_hint,
            best_k_hint=best_k_hint,
            sampling_mode=sampling_mode,
            samples_per_episode=samples_per_episode,
            uniform_frac=uniform_frac,
            uniform_frac_failed=uniform_frac_failed,
            focus_window=focus_window,
            force_pos_prev_k=force_pos_prev_k,
            scan_interval=scan_interval,
            n_top_zones=n_top_zones,
            min_abs_gap=min_abs_gap,
            feature_history=feature_history,
        )

        if print_sampling:
            print(
                f"  [sampling {i:4d}/{len(seeds)}] "
                f"cons_success={int(ep_info['cons_success'])} "
                f"mode={ep_info['sampling_mode_used']} "
                f"k*={ep_info['best_k']} "
                f"sampled_ks={ep_info['sampled_ks']} "
                f"forced_ks={ep_info['forced_ks']} "
                f"forced_success_ks={ep_info['forced_success_ks']} "
                f"kept={len(ep_samples)}"
            )
            if print_datapoints:
                for dp in ep_info["datapoints"]:
                    print(
                        "    "
                        f"k={dp['k']:>3d} "
                        f"label={dp['label']} "
                        f"delta={dp['delta']:+.5f} "
                        f"wait_ret={dp['wait_ret']:+.5f} "
                        f"switch_ret={dp['switch_ret']:+.5f} "
                        f"forced_pos={dp['forced_positive']} "
                        f"forced_success={dp['forced_success']}"
                    )

        if log_writer is not None:
            log_writer.writerow(
                {
                    "episode": i,
                    "seed": int(seed),
                    "budget": int(budget),
                    "cons_success": int(ep_info["cons_success"]),
                    "sampling_mode_used": ep_info["sampling_mode_used"],
                    "n_steps": int(ep_info["n_steps"]),
                    "best_k": int(ep_info["best_k"]),
                    "sampled_ks": json.dumps(ep_info["sampled_ks"]),
                    "forced_ks": json.dumps(ep_info["forced_ks"]),
                    "forced_success_ks": json.dumps(ep_info["forced_success_ks"]),
                    "n_samples_kept": len(ep_samples),
                }
            )
            log_file.flush()

        if i % 25 == 0 or i == len(seeds):
            print(f"  dataset episodes: {i}/{len(seeds)}")

        for sample in ep_samples:
            all_x.append(sample["x"])
            all_y.append(sample["delta"])
            all_lbl.append(sample["label"])

    if log_file is not None:
        log_file.close()

    X = np.array(all_x, dtype=np.float32)
    Y = np.array(all_y, dtype=np.float32)
    L = np.array(all_lbl, dtype=np.int64)
    return X, Y, L
