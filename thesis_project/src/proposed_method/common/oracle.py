"""Oracle return and switch-timing search helpers.

Provides counterfactual aggressive return, oracle conservative value, and
coarse-to-fine search for the best switch timestep k*.
"""

from .mujoco_state import save_mujoco_state, restore_mujoco_state


def counterfactual_switch_return(
    env,
    saved_state: dict,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    max_horizon: int,
) -> float:
    """From a saved state, execute aggressive policy until episode end."""
    current_state = save_mujoco_state(env)
    restore_mujoco_state(env, saved_state)

    env._switched = True
    obs = env._augment_meta(env._raw_obs)

    discounted_return = 0.0
    discount = 1.0
    done = False
    ep_len = env.t

    while not done and ep_len < max_horizon:
        obs, _r, done, info = env.step(1)
        cost_step = float(info.get("cumulative_cost", 0.0))
        r_step = -cost_weight * cost_step

        goal_met = bool(info.get("goal_met", False))
        budget_expired = bool(info.get("budget_expired", False))
        ep_len = info.get("time_step", ep_len + 1)

        discounted_return += discount * r_step
        discount *= gamma

        if goal_met:
            discounted_return += discount * 1.0
            done = True
        elif budget_expired:
            discounted_return += discount * (-deadline_weight)
            done = True

    if not done:
        discounted_return += discount * (-deadline_weight)

    restore_mujoco_state(env, current_state)
    return discounted_return


def oracle_conservative_value(
    env,
    saved_state: dict,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    max_horizon: int,
    switch_interval: int = 1,
) -> float:
    """Compute oracle value by conservative rollout and periodic switch probes."""
    current_state = save_mujoco_state(env)
    restore_mujoco_state(env, saved_state)

    switch_now = counterfactual_switch_return(
        env, saved_state, cost_weight, deadline_weight, gamma, max_horizon
    )
    best_return = switch_now

    cons_cum_return = 0.0
    cons_discount = 1.0
    done = False
    meta_step = 0
    ep_len = env.t

    while not done and ep_len < max_horizon:
        obs, _r, done, info = env.step(0)
        cost_step = float(info.get("cumulative_cost", 0.0))
        n_taken = info.get("n_steps_taken", 1)
        ep_len += n_taken
        meta_step += 1

        r_step = -cost_weight * cost_step
        goal_met = bool(info.get("goal_met", False))
        budget_expired = bool(info.get("budget_expired", False))

        if goal_met:
            r_step += 1.0
        elif budget_expired and not goal_met:
            r_step -= deadline_weight

        cons_cum_return += cons_discount * r_step
        cons_discount *= gamma

        if done:
            best_return = max(best_return, cons_cum_return)
            break

        if meta_step % switch_interval == 0:
            checkpoint = save_mujoco_state(env)
            switch_ret = counterfactual_switch_return(
                env, checkpoint, cost_weight, deadline_weight, gamma, max_horizon
            )
            candidate = cons_cum_return + cons_discount * switch_ret
            best_return = max(best_return, candidate)

    restore_mujoco_state(env, current_state)
    return best_return


def _find_best_k_zone_search(
    steps: list,
    env,
    cost_weight: float,
    deadline_weight: float,
    gamma: float,
    max_horizon: int,
    scan_interval: int = 5,
    n_top_zones: int = 2,
) -> int:
    """Find k* = argmax_k R_switch(k) with coarse-to-fine zone search."""
    n_steps = len(steps)
    if n_steps == 0:
        return 0

    zone_size = max(1, scan_interval)
    n_zones = (n_steps + zone_size - 1) // zone_size

    def _eval(k: int) -> float:
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

    zone_scores = []
    for z in range(n_zones):
        k_start = z * zone_size
        k_end = min(k_start + zone_size, n_steps)
        k_mid = (k_start + k_end - 1) // 2
        zone_scores.append((z, k_mid, _eval(k_mid)))

    zone_scores.sort(key=lambda x: x[2], reverse=True)
    top_zones = zone_scores[: min(n_top_zones, n_zones)]

    for z, _k_mid, _score in top_zones:
        k_start = z * zone_size
        k_end = min(k_start + zone_size, n_steps)
        for k in range(k_start, k_end):
            _eval(k)

    return max(
        (k for k in range(n_steps) if "switch_return" in steps[k]),
        key=lambda k: (steps[k]["switch_return"], k),
    )
