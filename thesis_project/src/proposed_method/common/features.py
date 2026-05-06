"""Feature extraction utilities for the supervised switch classifier.

Provides both the new 36-D lidar+kinematics features used by the neural
classifier and the legacy 7-D geometric features relied upon by earlier
scripts.
"""

import numpy as np

LIDAR_BINS = 16
N_FEATURES = 2 * LIDAR_BINS + 4  # hazards lidar, goal lidar, v_xy, t_frac, budget_norm
_GOAL_LIDAR_KEY = "goal_lidar"
_HAZARDS_LIDAR_KEY = "hazards_lidar"
_SLICE_CACHE_ATTR = "_cached_lidar_slices"

GOAL_POS = np.array([1.1, 1.1])
HAZ_POS = np.array([0.0, 0.0])
D_MAX = float(np.linalg.norm(np.array([3.0, 3.0])))


def _unwrap_env(env):
    """Return the base Safety-Gym env, unwrapping gym.Wrappers as needed."""
    base = getattr(env, "unwrapped", env)
    inner = getattr(base, "_env", None)
    return inner if inner is not None else base


def _resolve_lidar_slices(env) -> tuple:
    """Return (goal_slice, hazard_slice) over the meta-observation."""
    cached = getattr(env, _SLICE_CACHE_ATTR, None)
    if cached is not None:
        return cached

    base_env = _unwrap_env(env)

    goal_slice = None
    hazard_slice = None
    meta_offset = 0

    try:
        obs_dict = base_env.obs_space_dict
    except AttributeError:
        goal_slice = slice(0, LIDAR_BINS)
        hazard_slice = slice(LIDAR_BINS, 2 * LIDAR_BINS)
    else:
        for key, space in obs_dict.items():
            size = int(np.prod(space.shape))
            if "lidar" in key.lower() or key == "velocimeter":
                if key == _GOAL_LIDAR_KEY:
                    goal_slice = slice(meta_offset, meta_offset + size)
                elif key == _HAZARDS_LIDAR_KEY:
                    hazard_slice = slice(meta_offset, meta_offset + size)
                meta_offset += size

    if goal_slice is None or hazard_slice is None:
        raise RuntimeError("Meta observation missing goal/hazard lidar entries.")

    cache = (goal_slice, hazard_slice)
    setattr(env, _SLICE_CACHE_ATTR, cache)
    return cache


def extract_lidar_features(obs: np.ndarray, env) -> np.ndarray:
    """Extract lidar (goal + hazards), planar velocity, and time features."""
    goal_slice, hazard_slice = _resolve_lidar_slices(env)
    goal_lidar = np.asarray(obs[goal_slice], dtype=np.float32)
    hazard_lidar = np.asarray(obs[hazard_slice], dtype=np.float32)

    base_env = _unwrap_env(env)
    robot_vel = base_env.sim.data.get_body_xvelp("robot")
    v_x = float(robot_vel[0])
    v_y = float(robot_vel[1])

    time_left_norm = float(obs[-3])
    t_frac = time_left_norm
    budget_norm = float(obs[-2])

    features = np.concatenate([
        hazard_lidar,
        goal_lidar,
        np.array([v_x, v_y, t_frac, budget_norm], dtype=np.float32),
    ])
    return features.astype(np.float32, copy=False)


def extract_7features(obs: np.ndarray, env) -> np.ndarray:
    """Legacy 7-D geometric features (kept for backwards compatibility)."""
    base_env = _unwrap_env(env)
    robot_vel = base_env.sim.data.get_body_xvelp("robot")
    v_x = float(robot_vel[0])
    v_y = float(robot_vel[1])

    robot_pos = base_env.sim.data.get_body_xpos("robot")[:2]
    d_goal = float(np.linalg.norm(robot_pos - GOAL_POS))
    d_haz = float(np.linalg.norm(robot_pos - HAZ_POS))

    vec_goal = GOAL_POS - robot_pos
    vec_haz = HAZ_POS - robot_pos
    angle_goal = np.arctan2(vec_goal[1], vec_goal[0])
    angle_haz = np.arctan2(vec_haz[1], vec_haz[0])
    delta_theta = float(angle_goal - angle_haz)
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

    d_goal_norm = d_goal / D_MAX
    d_haz_norm = d_haz / D_MAX
    delta_theta_norm = delta_theta / np.pi

    time_left_norm = float(obs[-3])
    t_frac = time_left_norm
    budget_norm = float(obs[-2])

    return np.array(
        [v_x, v_y, d_goal_norm, d_haz_norm, delta_theta_norm, t_frac, budget_norm],
        dtype=np.float32,
    )


# Primary entry point for the supervised classifier
extract_features = extract_lidar_features
