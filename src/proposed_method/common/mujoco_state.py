"""MuJoCo and meta-environment state snapshot helpers.

These utilities save and restore complete simulator state so counterfactual
rollouts can be evaluated without side effects.
"""

import copy

def save_mujoco_state(env) -> dict:
    """Save full simulator + MetaEnv state for counterfactual rollouts."""
    sim = env._env.sim
    return {
        "mj_state": copy.deepcopy(sim.get_state()),
        "raw_obs": env._raw_obs.copy(),
        "t": env.t,
        "B": env.B,
        "goal_met": env._goal_met,
        "switched": env._switched,
        "engine_done": env._env.done,
        "engine_steps": env._env.steps,
    }


def restore_mujoco_state(env, state: dict):
    """Restore full simulator + MetaEnv state from a saved snapshot."""
    sim = env._env.sim
    sim.set_state(state["mj_state"])
    sim.forward()
    env._raw_obs = state["raw_obs"].copy()
    env.t = state["t"]
    env.B = state["B"]
    env._goal_met = state["goal_met"]
    env._switched = state["switched"]
    env._env.done = state["engine_done"]
    env._env.steps = state["engine_steps"]
